"""Fuse all channels from stored global transforms into one OME-NGFF store."""

import gc
import multiprocessing as mp
import warnings
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from multiview_stitcher import fusion, misc_utils, msi_utils
from multiview_stitcher import spatial_image_utils as si_utils
from tqdm import tqdm

from merfish3danalysis.DataRegistration import (
    _direct_zarr_fusion_kwargs,
    _local_fiducial_path,
    _zarr_array_dims,
)
from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.decode_warping import (
    compose_decode_warp_transform_zyx_um,
    load_bit_round_transform_zyx_um,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
mp.set_start_method("spawn", force=True)

_STAGE_TRANSFORM_KEY = "stage_metadata"
_GLOBAL_TRANSFORM_KEY = "global_registered"


def _require_global_transforms(datastore: qi2labDataStore) -> None:
    """Raise before writing any output unless every tile has a global transform."""
    missing = []
    for tile_id in datastore.tile_ids:
        affine_zyx_um, _origin_zyx_um, _spacing_zyx_um = (
            datastore.load_global_coord_xforms_um(tile=tile_id)
        )
        if affine_zyx_um is None:
            missing.append(tile_id)
    if missing:
        raise RuntimeError(
            "Stored global transforms are required for fuseall; missing transforms "
            f"for tiles: {missing}."
        )


def _lazy_zyx_data(*, array: Any, da_module: Any) -> Any:
    """Return a lazy ZYX view, dropping only singleton non-spatial axes."""
    dims = _zarr_array_dims(array)
    if not {"z", "y", "x"}.issubset(dims):
        raise ValueError(f"Expected z, y, and x axes; got {dims}.")

    selection: list[int | slice] = []
    retained_dims = []
    for axis, dim in enumerate(dims):
        if dim in {"z", "y", "x"}:
            selection.append(slice(None))
            retained_dims.append(dim)
        else:
            if int(array.shape[axis]) != 1:
                raise ValueError(
                    f"Expected singleton non-spatial axis {dim!r}; "
                    f"got shape {array.shape}."
                )
            selection.append(0)

    data = da_module.from_array(array, chunks=array.chunks)[tuple(selection)]
    spatial_order = tuple(retained_dims.index(dim) for dim in ("z", "y", "x"))
    if spatial_order != (0, 1, 2):
        data = data.transpose(spatial_order)
    return data


def _read_readout_data(
    *,
    datastore: qi2labDataStore,
    tile_id: str,
    bit_id: str,
    zarr_module: Any,
    da_module: Any,
) -> Any:
    """Build one lazy feature-weighted readout volume in ZYX order."""
    entity_root = datastore._readouts_root_path / Path(tile_id) / Path(bit_id)
    image_path = datastore._image_store_path(entity_root / Path("decon_data"))
    if not image_path.exists():
        image_path = datastore._image_store_path(entity_root / Path("corrected_data"))
    predictor_path = datastore._image_store_path(
        entity_root / Path(f"{datastore.feature_predictor_folder_name}_data")
    )

    image_array = zarr_module.open_array(image_path / Path("0"), mode="r")
    predictor_array = zarr_module.open_array(predictor_path / Path("0"), mode="r")
    if image_array.shape != predictor_array.shape:
        raise ValueError(
            f"Readout and feature-predictor shapes differ for {tile_id} {bit_id}: "
            f"{image_array.shape} != {predictor_array.shape}."
        )

    image_data = _lazy_zyx_data(array=image_array, da_module=da_module)
    predictor_data = _lazy_zyx_data(array=predictor_array, da_module=da_module)
    if image_data.shape != predictor_data.shape:
        raise ValueError(
            f"ZYX readout and feature-predictor shapes differ for "
            f"{tile_id} {bit_id}: {image_data.shape} != {predictor_data.shape}."
        )
    return (
        image_data.astype(np.float32) * predictor_data.astype(np.float32).clip(0.25, 1)
    ).astype(np.uint16)


def _translation_affine_zyx_um(translation_zyx_um: Any) -> np.ndarray:
    """Return a homogeneous translation in physical ZYX coordinates."""
    affine = np.eye(4, dtype=np.float32)
    affine[:3, 3] = np.asarray(translation_zyx_um, dtype=np.float32)
    return affine


def _channel_global_transforms_zyx_um(
    *,
    datastore: qi2labDataStore,
    tile_id: str,
    bit_ids: list[str],
    tile_position_zyx_um: Any,
    stage_transform_zyx_um: np.ndarray,
    global_transform_zyx_um: np.ndarray,
) -> np.ndarray:
    """Compose native-channel coordinates directly into the global frame.

    Stored local round transforms map round-1 reference coordinates into a
    bit's native moving coordinates. The decode-time chromatic convention is
    composed with that round transform, then inverted for fusion's
    native-to-reference mapping. The local transform is conjugated around the
    tile's physical coordinate origin before applying the stage and stored
    global transforms. SOFIMA fields are intentionally not included.
    """
    tile_origin = _translation_affine_zyx_um(tile_position_zyx_um)
    inverse_tile_origin = _translation_affine_zyx_um(
        -np.asarray(tile_position_zyx_um, dtype=np.float32)
    )
    global_stage = (
        np.asarray(global_transform_zyx_um, dtype=np.float32)
        @ np.asarray(stage_transform_zyx_um, dtype=np.float32)
    )
    channel_transforms = [global_stage]
    for bit_id in bit_ids:
        _round_id, reference_to_native_zyx_um = load_bit_round_transform_zyx_um(
            datastore,
            tile=tile_id,
            bit_id=bit_id,
        )
        wavelengths_um = datastore.load_local_wavelengths_um(
            tile=tile_id,
            bit=bit_id,
        )
        if wavelengths_um is None:
            raise RuntimeError(
                f"Missing wavelength metadata for tile={tile_id} bit={bit_id}."
            )
        emission_wavelength_um = float(wavelengths_um[1])
        chromatic_transform_zyx_um = (
            datastore.load_chromatic_affine_transform_zyx_um(
                wavelength_um=emission_wavelength_um,
            )
        )
        reference_to_native_zyx_um = compose_decode_warp_transform_zyx_um(
            round_transform_zyx_um=reference_to_native_zyx_um,
            chromatic_transform_zyx_um=chromatic_transform_zyx_um,
        )
        native_to_reference_zyx_um = np.linalg.inv(reference_to_native_zyx_um)
        native_to_reference_at_tile = (
            tile_origin @ native_to_reference_zyx_um @ inverse_tile_origin
        )
        channel_transforms.append(global_stage @ native_to_reference_at_tile)
    return np.asarray(channel_transforms, dtype=np.float32)


def _load_tile_multichannel_msim(
    *,
    datastore: qi2labDataStore,
    tile_id: str,
    bit_ids: list[str],
    zarr_module: Any = zarr,
    da_module: Any = da,
    msi_utils_module: Any = msi_utils,
    si_utils_module: Any = si_utils,
) -> Any:
    """Build one lazy CZYX tile ordered as fiducial followed by readout bits."""
    reference_round_id = datastore.round_ids[0]
    voxel_zyx_um = datastore.voxel_size_zyx_um
    scale = {
        "z": float(voxel_zyx_um[0]),
        "y": float(voxel_zyx_um[1]),
        "x": float(voxel_zyx_um[2]),
    }
    tile_position_zyx_um, affine_zyx_px = datastore.load_local_stage_position_zyx_um(
        tile_id, reference_round_id
    )
    translation = {
        "z": float(np.round(tile_position_zyx_um[0], 2)),
        "y": float(np.round(tile_position_zyx_um[1], 2)),
        "x": float(np.round(tile_position_zyx_um[2], 2)),
    }

    fiducial_path = _local_fiducial_path(
        datastore=datastore,
        tile_id=tile_id,
        round_id=reference_round_id,
    )
    fiducial_array = zarr_module.open_array(fiducial_path / Path("0"), mode="r")
    fiducial_data = _lazy_zyx_data(array=fiducial_array, da_module=da_module).astype(
        np.uint16
    )
    channel_data = [fiducial_data]
    for bit_id in bit_ids:
        readout_data = _read_readout_data(
            datastore=datastore,
            tile_id=tile_id,
            bit_id=bit_id,
            zarr_module=zarr_module,
            da_module=da_module,
        )
        if readout_data.shape != fiducial_data.shape:
            raise ValueError(
                f"Fiducial and readout shapes differ for {tile_id} {bit_id}: "
                f"{fiducial_data.shape} != {readout_data.shape}."
            )
        channel_data.append(readout_data)

    multichannel_data = da_module.stack(channel_data, axis=0)
    channel_ids = ["fiducial", *bit_ids]
    sim = si_utils_module.get_sim_from_array(
        multichannel_data,
        dims=("c", "z", "y", "x"),
        scale=scale,
        translation=translation,
        affine=affine_zyx_px,
        transform_key=_STAGE_TRANSFORM_KEY,
        c_coords=channel_ids,
        t_coords=None,
    )

    msim = msi_utils_module.get_msim_from_sim(sim, scale_factors=[])
    affine_zyx_um, _origin_zyx_um, _spacing_zyx_um = (
        datastore.load_global_coord_xforms_um(tile=tile_id)
    )
    if affine_zyx_um is None:
        raise RuntimeError(f"Stored global transform is missing for tile {tile_id!r}.")
    stage_transform = msi_utils_module.get_transform_from_msim(
        msim,
        transform_key=_STAGE_TRANSFORM_KEY,
    )
    channel_transforms = _channel_global_transforms_zyx_um(
        datastore=datastore,
        tile_id=tile_id,
        bit_ids=bit_ids,
        tile_position_zyx_um=tile_position_zyx_um,
        stage_transform_zyx_um=np.asarray(stage_transform).squeeze(),
        global_transform_zyx_um=affine_zyx_um,
    )
    channel_transform_data = xr.DataArray(
        channel_transforms[:, None, :, :],
        dims=("c", "t", "x_in", "x_out"),
        coords={
            "c": channel_ids,
            "t": stage_transform.coords["t"],
            "x_in": stage_transform.coords["x_in"],
            "x_out": stage_transform.coords["x_out"],
        },
    )
    msi_utils_module.set_affine_transform(
        msim,
        channel_transform_data,
        transform_key=_GLOBAL_TRANSFORM_KEY,
    )
    return msim


def _write_fused_metadata(*, fused_msim: Any, output_path: Path) -> None:
    """Persist the physical transform metadata produced by direct fusion."""
    affine = msi_utils.get_transform_from_msim(
        fused_msim,
        transform_key=_GLOBAL_TRANSFORM_KEY,
    ).data.squeeze()
    fused_scale0 = msi_utils.get_sim_from_msim(fused_msim)
    origin = si_utils.get_origin_from_sim(fused_scale0, asarray=True)
    spacing = si_utils.get_spacing_from_sim(fused_scale0, asarray=True)
    qi2labDataStore._write_extra_attributes(
        image_path=output_path,
        extra_attributes={
            "affine_zyx_um": np.asarray(affine, dtype=np.float32).tolist(),
            "origin_zyx_um": np.asarray(origin, dtype=np.float32).tolist(),
            "spacing_zyx_um": np.asarray(spacing, dtype=np.float32).tolist(),
        },
        merge=True,
    )


def fuse_all_channels(root_path: Path) -> None:
    """Fuse fiducial then numerically ordered bits into one OME-Zarr.

    Parameters
    ----------
    root_path : pathlib.Path
        Experiment root containing ``qi2labdatastore``. Output is written to
        ``fused/fused_all_channels.ome.zarr``.
    """
    print("\nInitializing datastore...")
    datastore_path = root_path / Path("qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    _require_global_transforms(datastore)

    bit_ids = sorted(
        (str(bit_id) for bit_id in datastore.bit_ids),
        key=lambda bit_id: int(bit_id.removeprefix("bit")),
    )

    fused_path = root_path / Path("fused")
    fused_path.mkdir(exist_ok=True)
    fusion_kwargs = _direct_zarr_fusion_kwargs(misc_utils=misc_utils)

    print("\nLazy loading fiducial and ordered bit channels...")
    print(
        "Applying linked-round and chromatic affine transforms; "
        "SOFIMA fields are ignored."
    )
    tile_msims = [
        _load_tile_multichannel_msim(
            datastore=datastore,
            tile_id=tile_id,
            bit_ids=bit_ids,
        )
        for tile_id in tqdm(datastore.tile_ids, desc="tile")
    ]

    output_path = fused_path / Path("fused_all_channels.ome.zarr")
    print(f"Fusing all channels directly to {output_path!s}...")
    fused_msim = fusion.fuse(
        images=tile_msims,
        transform_key=_GLOBAL_TRANSFORM_KEY,
        output_zarr_url=str(output_path),
        **fusion_kwargs,
    )
    _write_fused_metadata(fused_msim=fused_msim, output_path=output_path)
    del fused_msim, tile_msims
    gc.collect()


if __name__ == "__main__":
    root_path = Path(r"/mnt/data2/bioprotean/20250220_Bartelle_control_smFISH_TqIB")
    fuse_all_channels(root_path)
