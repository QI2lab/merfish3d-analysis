"""Decode-time image warping helpers.

These helpers keep the transform composition used by pixel decoding in one
place. The actual interpolation remains in
``merfish3danalysis.utils.multiview_registration`` so registration and decoding
share the same GPU warp convention.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def load_bit_round_transform_zyx_um(
    datastore: Any,
    *,
    tile: int | str,
    bit_id: str,
) -> tuple[str | None, np.ndarray]:
    """
    Load the local fiducial-round transform for one readout bit.

    Parameters
    ----------
    datastore : Any
        qi2lab datastore-like object.
    tile : int or str
        Tile index or identifier.
    bit_id : str
        Readout bit identifier.

    Returns
    -------
    tuple[str | None, numpy.ndarray]
        Round identifier and physical Z, Y, X transform. Bits from the
        reference round return ``(None, identity)``.
    """

    round_index = datastore.load_local_round_linker(tile=tile, bit=bit_id) - 1
    if round_index <= 0:
        return None, np.eye(4, dtype=np.float32)

    round_id = datastore.round_ids[round_index]
    round_transform_zyx_um = datastore.load_local_round_transform_zyx_um(
        tile=tile,
        round=round_id,
    )
    if round_transform_zyx_um is None:
        raise RuntimeError(
            f"Missing local round transform for tile={tile} round={round_id}."
        )
    return round_id, np.asarray(round_transform_zyx_um, dtype=np.float32)


def compose_decode_warp_transform_zyx_um(
    *,
    round_transform_zyx_um: np.ndarray,
    chromatic_transform_zyx_um: np.ndarray,
) -> np.ndarray:
    """
    Compose chromatic and fiducial transforms for decode-time image loading.

    Parameters
    ----------
    round_transform_zyx_um : numpy.ndarray
        Physical transform from reference-round Z, Y, X coordinates into the
        bit's native fiducial-round coordinates.
    chromatic_transform_zyx_um : numpy.ndarray
        Chromatic calibration transform mapping the bit wavelength toward the
        reference wavelength in physical Z, Y, X coordinates.

    Returns
    -------
    numpy.ndarray
        Physical transform mapping reference-round output coordinates to the
        native readout image coordinates, suitable for
        :func:`warp_array_to_reference_gpu`.
    """

    return np.linalg.inv(np.asarray(chromatic_transform_zyx_um, dtype=np.float32)) @ (
        np.asarray(round_transform_zyx_um, dtype=np.float32)
    )


def warp_bit_image_to_reference(
    image: np.ndarray,
    *,
    datastore: Any,
    tile: int | str,
    bit_id: str,
    emission_wavelength_um: float,
    gpu_id: int = 0,
) -> np.ndarray:
    """
    Warp one native bit image into the round-1 local reference frame.

    Parameters
    ----------
    image : numpy.ndarray
        Readout image in the bit's native Z, Y, X grid.
    datastore : Any
        qi2lab datastore-like object used to load round, chromatic, and SOFIMA
        metadata.
    tile : int or str
        Tile index or identifier.
    bit_id : str
        Readout bit identifier.
    emission_wavelength_um : float
        Emission wavelength for the bit in microns.
    gpu_id : int, default=0
        CUDA device ID used for interpolation.

    Returns
    -------
    numpy.ndarray
        Image sampled in the round-1 local reference frame.
    """

    round_id, round_transform_zyx_um = load_bit_round_transform_zyx_um(
        datastore,
        tile=tile,
        bit_id=bit_id,
    )
    chromatic_transform_zyx_um = datastore.load_chromatic_affine_transform_zyx_um(
        wavelength_um=emission_wavelength_um,
    )
    transform_zyx_um = compose_decode_warp_transform_zyx_um(
        round_transform_zyx_um=round_transform_zyx_um,
        chromatic_transform_zyx_um=chromatic_transform_zyx_um,
    )
    spacing_zyx_um = datastore.voxel_size_zyx_um

    loaded_flow_field = None
    if round_id is not None:
        loaded_flow_field = datastore.load_local_sofima_flow_field(
            tile=tile,
            round=round_id,
            return_future=False,
        )

    if loaded_flow_field is None and np.allclose(
        transform_zyx_um,
        np.eye(4, dtype=np.float32),
    ):
        return np.asarray(image, dtype=np.float32)

    if loaded_flow_field is not None:
        sofima_flow_field, flow_attrs = loaded_flow_field
        return warp_image_with_sofima_metadata(
            image,
            transform_zyx_um=transform_zyx_um,
            spacing_zyx_um=spacing_zyx_um,
            sofima_flow_field_xyz_px=sofima_flow_field,
            flow_attrs=flow_attrs,
            gpu_id=gpu_id,
        ).astype(np.float32, copy=False)

    from merfish3danalysis.utils.multiview_registration import (
        warp_array_to_reference_gpu,
    )

    return warp_array_to_reference_gpu(
        image,
        transform_zyx_um=transform_zyx_um,
        spacing_zyx_um=spacing_zyx_um,
        reference_shape=image.shape,
        gpu_id=gpu_id,
    ).astype(np.float32, copy=False)


def warp_image_with_sofima_metadata(
    image: np.ndarray,
    *,
    transform_zyx_um: np.ndarray,
    spacing_zyx_um: Sequence[float],
    sofima_flow_field_xyz_px: np.ndarray,
    flow_attrs: Mapping[str, Any],
    reference_shape: Sequence[int] | None = None,
    gpu_id: int = 0,
) -> np.ndarray:
    """
    Warp an image using a SOFIMA flow field and datastore flow metadata.

    Parameters
    ----------
    image : numpy.ndarray
        Native moving image in Z, Y, X order.
    transform_zyx_um : numpy.ndarray
        Physical affine transform from reference output coordinates into the
        moving image coordinate system.
    spacing_zyx_um : Sequence[float]
        Reference image voxel spacing in microns, in Z, Y, X order.
    sofima_flow_field_xyz_px : numpy.ndarray
        SOFIMA flow field in X, Y, Z channel order with map axes Z, Y, X.
    flow_attrs : Mapping[str, Any]
        Metadata returned by ``load_local_sofima_flow_field`` or by the SOFIMA
        estimator. This must include ``map_stride_zyx_px`` and
        ``map_box_start_xyz_px``. If ``reference_shape`` is omitted it must also
        include ``reference_shape_zyx_px``.
    reference_shape : Sequence[int] or None, default=None
        Output image shape in Z, Y, X order. When omitted, the shape stored in
        ``flow_attrs`` is used.
    gpu_id : int, default=0
        CUDA device ID used for interpolation.

    Returns
    -------
    numpy.ndarray
        Image sampled in the reference frame as float32.
    """

    from merfish3danalysis.utils.multiview_registration import (
        warp_array_to_reference_with_affine_and_sofima_flow_gpu,
    )

    if reference_shape is None:
        reference_shape = flow_attrs["reference_shape_zyx_px"]
    output_shape = tuple(int(v) for v in reference_shape)
    return warp_array_to_reference_with_affine_and_sofima_flow_gpu(
        image,
        transform_zyx_um=transform_zyx_um,
        spacing_zyx_um=spacing_zyx_um,
        reference_shape=output_shape,
        sofima_flow_field_xyz_px=sofima_flow_field_xyz_px,
        flow_field_stride_zyx_px=flow_attrs["map_stride_zyx_px"],
        flow_field_box_start_xyz_px=flow_attrs["map_box_start_xyz_px"],
        gpu_id=gpu_id,
    ).astype(np.float32, copy=False)


__all__ = [
    "compose_decode_warp_transform_zyx_um",
    "load_bit_round_transform_zyx_um",
    "warp_bit_image_to_reference",
    "warp_image_with_sofima_metadata",
]
