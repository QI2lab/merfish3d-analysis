"""
Convert raw qi2lab WF MERFISH data to qi2labdatastore.

This is an example on how to convert a qi2lab experiment to the datastore
object that the qi2lab "merfish3d-analysis" package uses. Most of the
parameters are automatically extracted from the metadata written by qi2lab
microscopes. For another microscope, you will need to write new code on how to
extract the correct parameters.

Shepherd 2025/10 - change to CLI.
Shepherd 2024/12 - added more NDTIFF metadata extraction for camera and binning.
Shepherd 2024/12 - refactor
Shepherd 2024/11 - rework script to accept parameters.
Shepherd 2024/08 - rework script to utilize qi2labdatastore object.
"""

import gc
import io
from contextlib import redirect_stdout
from itertools import compress
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from tifffile import imread
from tqdm import tqdm

from merfish3danalysis.cli.qi2lab_microscopes._common import qi2lab_datastore_path
from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.dataio import read_metadatafile
from merfish3danalysis.utils.imageprocessing import (
    estimate_shading,
    replace_hot_pixels,
)
from merfish3danalysis.utils.psf import (
    QI2LAB_DEFAULT_IMMERSION_RI,
    QI2LAB_DEFAULT_NA,
    QI2LAB_EMISSION_WAVELENGTHS_UM,
    QI2LAB_EXCITATION_WAVELENGTHS_UM,
    generate_qi2lab_psf,
)

app = typer.Typer()
app.pretty_exceptions_enable = False


def _first_dataset_dir(
    root_path: Path, root_name: str, round_idx: int, tile_idx: int
) -> Path:
    """Return the first raw NDTiff directory for a round/tile.

    Parameters
    ----------
    root_path : Path
        Experiment root directory.
    root_name : str
        Raw acquisition root name from microscope metadata.
    round_idx : int
        Zero-based imaging round index.
    tile_idx : int
        Zero-based tile index.

    Returns
    -------
    Path
        Existing NDTiff directory ending in ``_1`` or ``_2``.
    """
    base = f"{root_name}_r{round_idx + 1:04d}_tile{tile_idx:04d}"
    dataset_path_1 = root_path / f"{base}_1"
    dataset_path_2 = root_path / f"{base}_2"
    if dataset_path_1.exists():
        return dataset_path_1
    if dataset_path_2.exists():
        return dataset_path_2
    raise FileNotFoundError(
        f"Could not find raw dataset at either path:\n"
        f"- {dataset_path_1}\n- {dataset_path_2}"
    )


def _load_dataset_silently(dataset_path: Path) -> Any:
    """Load an NDTiff dataset while suppressing ndstorage console output."""
    from ndstorage import Dataset

    with redirect_stdout(io.StringIO()):
        return Dataset(str(dataset_path))


def _first_stack_path(
    root_path: Path, root_name: str, round_idx: int, tile_idx: int
) -> Path:
    """Return the first raw NDTiff stack file for a round/tile."""
    base = f"{root_name}_r{round_idx + 1:04d}_tile{tile_idx:04d}"
    return (
        _first_dataset_dir(root_path, root_name, round_idx, tile_idx)
        / f"{base}_NDTiffStack.tif"
    )


def _camera_parameters(ndtiff_metadata: dict) -> tuple[str, float, float]:
    """Return camera model, gain, and offset from NDTiff metadata."""
    camera_id = ndtiff_metadata.get("Camera-CameraName")
    camera_id_alt = ndtiff_metadata.get("Core-Camera")

    if camera_id == "C13440-20CU" or camera_id_alt == "C13440-20CU":
        return (
            "orcav3",
            float(ndtiff_metadata["Camera-CONVERSION FACTOR COEFF"]),
            float(ndtiff_metadata["Camera-CONVERSION FACTOR OFFSET"]),
        )

    if (
        camera_id == "Blackfly S BFS-U3-200S6M"
        or camera_id_alt == "Blackfly S BFS-U3-200S6M"
    ):
        return "flir", 0.03, 0.0

    raise ValueError(f"Unsupported camera metadata: {camera_id!r}/{camera_id_alt!r}.")


def _camera_binning(metadata: dict, ndtiff_metadata: dict, camera: str) -> int:
    """Return camera binning from microscope or NDTiff metadata."""
    try:
        return int(metadata["binning"])
    except (KeyError, TypeError, ValueError):
        pass

    if camera == "orcav3":
        binning_str = ndtiff_metadata["Camera-Binning"]
        if binning_str == "1x1":
            return 1
        if binning_str == "2x2":
            return 2
    elif camera == "flir":
        binning_str = ndtiff_metadata["Binning"]
        if binning_str == "1":
            return 1
        if binning_str == "2":
            return 2

    raise ValueError(f"Unsupported {camera} binning metadata: {binning_str!r}.")


def _correct_channel_image(
    raw_image: np.ndarray,
    channel_idx: int,
    illuminations: np.ndarray | None,
) -> np.ndarray:
    """Return one corrected channel image as uint16."""
    channel_image = np.squeeze(raw_image[channel_idx, :]).astype(np.float32)
    if illuminations is not None:
        channel_image = channel_image / illuminations[channel_idx, :]
    return channel_image.clip(0, 2**16 - 1).astype(np.uint16)


def _readout_bit_ids(
    experiment_order: np.ndarray,
    channel_idx: int,
    bit_ids: list[str],
) -> list[str]:
    """Return bit IDs acquired through one readout channel."""
    bit_numbers = np.asarray(experiment_order[:, channel_idx], dtype=np.int64)
    unique_bit_numbers = dict.fromkeys(int(bit_number) for bit_number in bit_numbers)

    channel_bit_ids = []
    for bit_number in unique_bit_numbers:
        if bit_number < 1 or bit_number > len(bit_ids):
            raise ValueError(
                f"Invalid bit number {bit_number} in readout channel {channel_idx}."
            )
        channel_bit_ids.append(bit_ids[bit_number - 1])
    return channel_bit_ids


def _sample_readout_tile_bit_pairs(
    bit_ids: list[str],
    num_tiles: int,
    max_images: int,
    rng: np.random.Generator,
) -> list[tuple[int, str]]:
    """Assign unique sampled tile IDs across the bits for one dye."""
    if max_images < 1:
        raise ValueError("max_images must be at least 1.")
    if num_tiles < 1 or not bit_ids:
        raise ValueError("At least one tile and bit are required.")

    sample_count = min(num_tiles, max_images)
    tile_indices = rng.choice(num_tiles, size=sample_count, replace=False)
    bit_cycle = np.asarray(bit_ids, dtype=object)
    rng.shuffle(bit_cycle)
    sample_bit_ids = np.resize(bit_cycle, sample_count)
    rng.shuffle(sample_bit_ids)
    return [
        (int(tile_idx), str(bit_id))
        for tile_idx, bit_id in zip(tile_indices, sample_bit_ids, strict=True)
    ]


def _stage_position_zyx_um(
    position_list: np.ndarray,
    tile_idx: int,
    ndtiff_metadata: dict,
) -> np.ndarray:
    """Return stage position in zyx microns for one tile."""
    mirror_x = int(ndtiff_metadata["XYStage-TransposeMirrorX"]) == 1
    mirror_y = int(ndtiff_metadata["XYStage-TransposeMirrorY"]) == 1

    if mirror_x or mirror_y:
        corrected_y = np.max(position_list[:, 2]) - position_list[tile_idx, 2]
        corrected_x = np.max(position_list[:, 1]) - position_list[tile_idx, 1]
    else:
        corrected_y = position_list[tile_idx, 1]
        corrected_x = position_list[tile_idx, 2]

    return np.asarray(
        [
            np.round(position_list[tile_idx, 0], 2),
            np.round(corrected_y, 2),
            np.round(corrected_x, 2),
        ],
        dtype=np.float32,
    )


@app.command()
def convert_data(
    root_path: Path,
    use_illuminations: bool = False,
    save_illuminations: bool = True,
    channel_names: list[str] | None = None,
    hot_pixel_image_path: Path | None = None,
    output_path: Path | None = None,
    codebook_path: Path | None = None,
    bit_order_path: Path | None = None,
    fallback_na: float = QI2LAB_DEFAULT_NA,
    fallback_ri: float = QI2LAB_DEFAULT_IMMERSION_RI,
    excitation_wavelengths_um: tuple[float, float, float] = (
        QI2LAB_EXCITATION_WAVELENGTHS_UM
    ),
    emission_wavelengths_um: tuple[float, float, float] = (
        QI2LAB_EMISSION_WAVELENGTHS_UM
    ),
    default_tile_overlap: float = 0.2,
    noise_map_shape_yx: tuple[int, int] = (2048, 2048),
    hot_pixel_threshold: int = 100,
    max_flatfield_images: int = 100,
) -> None:
    """Convert qi2lab microscope data to qi2lab datastore.

    Parameters
    ----------
    root_path : Path
        Experiment root directory.
    use_illuminations : bool, default=False
        Apply illumination correction from ``illuminations.ome.tif`` when present.
    save_illuminations : bool, default=True
        Save estimated illuminations to ``illuminations.ome.tif`` for reuse.
    channel_names : list[str] | None, default=None
        Dye names in ascending wavelength order. When omitted, use the qi2lab
        defaults.
    hot_pixel_image_path : Path | None, default=None
        Hot-pixel map path. When omitted, use a zero-valued camera offset map.
    output_path : Path | None, default=None
        Datastore output directory. When omitted, write ``qi2labdatastore``
        under ``root_path``.
    codebook_path : Path | None, default=None
        Codebook CSV path. When omitted, read ``codebook.csv`` under
        ``root_path``.
    bit_order_path : Path | None, default=None
        Bit-order CSV path. This file defines the bits present in each imaging
        round, in channel order. When omitted, read ``bit_order.csv`` under
        ``root_path``.
    fallback_na : float, default=1.35
        Numerical aperture used when microscope metadata omit ``na``.
    fallback_ri : float, default=1.51
        Immersion refractive index used when microscope metadata omit ``ri``.
    excitation_wavelengths_um : tuple[float, float, float]
        Excitation wavelengths for blue, yellow, and red channels.
    emission_wavelengths_um : tuple[float, float, float]
        Emission wavelengths for blue, yellow, and red channels.
    default_tile_overlap : float, default=0.2
        Tile overlap used when microscope metadata omit ``tile_overlap``.
    noise_map_shape_yx : tuple[int, int], default=(2048, 2048)
        Shape of the generated camera-offset noise map when no hot-pixel image
        is supplied.
    hot_pixel_threshold : int, default=100
        Threshold passed to hot-pixel detection when estimating illuminations.
    max_flatfield_images : int, default=100
        Maximum number of unique tile IDs sampled to estimate each readout
        flatfield. The fiducial flatfield always uses every tile from round 1.
    """
    # load illuminations if requested
    # -----------------------------------
    illuminations = None
    if use_illuminations:
        illuminations_path = root_path / "illuminations.ome.tif"
        if illuminations_path.exists():
            illuminations = imread(illuminations_path)
            save_illuminations = False
        else:
            use_illuminations = False

    # load codebook
    # --------------
    if channel_names is None:
        channel_names = ["alexa488", "atto565", "alexa647"]
    if codebook_path is None:
        codebook = pd.read_csv(root_path / "codebook.csv")
    else:
        codebook = pd.read_csv(codebook_path)

    # load experimental order
    # -----------------------
    if bit_order_path is None:
        df_experiment_order = pd.read_csv(root_path / "bit_order.csv")
        experiment_order = df_experiment_order.values
    else:
        df_experiment_order = pd.read_csv(bit_order_path)
        experiment_order = df_experiment_order.values

    # load experiment metadata
    # ------------------------
    metadata_path = root_path / "scan_metadata.csv"
    metadata = read_metadatafile(metadata_path)
    root_name = metadata["root_name"]
    num_rounds = metadata["num_r"]
    num_tiles = metadata["num_xyz"]
    num_ch = metadata["num_ch"]

    # load first tile to get experimental metadata
    dataset_path = _first_dataset_dir(root_path, root_name, 0, 0)
    dataset = _load_dataset_silently(dataset_path)
    channel_to_test = dataset.get_image_coordinates_list()[0]["channel"]
    ndtiff_metadata = dataset.read_metadata(channel=channel_to_test, z=0)
    camera, e_per_ADU, offset = _camera_parameters(ndtiff_metadata)
    binning = _camera_binning(metadata, ndtiff_metadata, camera)
    channels_active = [
        metadata["blue_active"],
        metadata["yellow_active"],
        metadata["red_active"],
    ]
    # this entry was not contained in pre-v8 microscope csv, it was instead stored
    # in the imaging data itself. We added it to > v8 qi2lab-scope metadata csv to make the
    # access pattern easier.
    try:
        channel_order_bool = metadata["channels_reversed"]
        if channel_order_bool:
            channel_order = "reversed"
        else:
            channel_order = "forward"
    except KeyError:
        if (dataset.get_image_coordinates_list()[0]["channel"]) == "F-Blue":
            channel_order = "forward"
        else:
            channel_order = "reversed"

    # this entry was not contained in pre-v8 microscope csv, it was instead stored
    # in the imaging data itself. We added it to > v8 qi2lab-scope metadata csv to make the
    # access pattern easier.
    try:
        z_pixel_um = float(metadata["z_step_um"])
        yx_pixel_um = float(metadata["yx_pixel_um"])
        voxel_size_zyx_um = [z_pixel_um, yx_pixel_um, yx_pixel_um]
    except (KeyError, TypeError, ValueError):
        yx_pixel_um = np.round(float(ndtiff_metadata["PixelSizeUm"]), 3)
        next_ndtiff_metadata = dataset.read_metadata(channel=channel_to_test, z=1)
        z_pixel_um = np.round(
            np.abs(
                float(next_ndtiff_metadata["ZPosition_um_Intended"])
                - float(ndtiff_metadata["ZPosition_um_Intended"])
            ),
            3,
        )
        voxel_size_zyx_um = [z_pixel_um, yx_pixel_um, yx_pixel_um]

        del next_ndtiff_metadata

    # this entry was not contained in pre-v8 metadata csv, it was instead stored
    # in the imaging data itself. We added it to > v8 qi2lab-scope metadata csv to make the
    # access pattern easier.
    try:
        na = metadata["na"]
    except (KeyError, TypeError, ValueError):
        na = fallback_na

    # this entry was not contained in pre-v8 microscope csv, it was instead stored
    # in the imaging data itself. We added it to > v8 qi2lab-scope metadata csv to make the
    # access pattern easier.
    try:
        ri = metadata["ri"]
    except (KeyError, TypeError, ValueError):
        ri = fallback_ri

    ex_wavelengths_um = list(excitation_wavelengths_um)
    em_wavelengths_um = list(emission_wavelengths_um)
    channel_idxs = list(range(num_ch))
    channels_in_data = list(compress(channel_idxs, channels_active))

    # load camera specific stage vs camera vs computer orientation
    # parameters.
    #
    # these entries were not contained in pre-v8 microscope csv. There were
    # instead stored in the imaging data itself.
    #
    # We added it to > v8 qi2lab-scope metadata csv to make the access pattern easier.
    # The defaults are the "known" defaults for this camera configuration.

    if hot_pixel_image_path is None:
        noise_map = offset * np.ones(
            tuple(int(v) for v in noise_map_shape_yx),
            dtype=np.uint16,
        )
    else:
        noise_map = imread(hot_pixel_image_path)

    stage_affine_str = ndtiff_metadata["PixelSizeAffine"]
    stage_affine_values = np.asarray(
        list(map(float, stage_affine_str.split(";"))), dtype=np.float32
    )
    stage_affine_values = np.round(
        stage_affine_values / float(ndtiff_metadata["PixelSizeUm"]), 2
    )
    affine_zyx_px = np.array(
        [
            [1, 0, 0, 0],
            [0, stage_affine_values[4], stage_affine_values[3], 0],
            [0, stage_affine_values[1], stage_affine_values[0], 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    # generate PSFs
    # --------------
    raw_image = imread(_first_stack_path(root_path, root_name, 0, 0))
    if camera == "orcav3":
        raw_image = np.swapaxes(raw_image, 0, 1)
    psf_z = int(raw_image.shape[1]) if raw_image.ndim == 4 else 1
    del raw_image

    channel_psfs = []
    for channel_id in channels_in_data:
        psf = generate_qi2lab_psf(
            z_depth=psf_z,
            voxel_size_zyx_um=tuple(voxel_size_zyx_um),
            emission_wavelength_um=em_wavelengths_um[channel_id],
            na=na,
            immersion_ri=ri,
        )
        channel_psfs.append(psf)

    # initialize datastore
    datastore_path = (
        qi2lab_datastore_path(root_path) if output_path is None else output_path
    )
    existing_store = datastore_path.exists()
    datastore = qi2labDataStore(datastore_path)

    if not (existing_store):
        # required user parameters
        datastore.channels_in_data = channel_names

        # parameters from qi2lab microscope metadata
        datastore.num_rounds = num_rounds
        datastore.codebook = codebook
        datastore.experiment_order = experiment_order
        datastore.num_tiles = num_tiles
        try:
            datastore.microscope_type = metadata["experiment_type"]
        except (KeyError, TypeError, ValueError):
            if z_pixel_um < 0.5:
                datastore.microscope_type = "3D"
            else:
                datastore.microscope_type = "2D"
        datastore.camera_model = camera
        try:
            datastore.tile_overlap = metadata["tile_overlap"]
        except (KeyError, TypeError, ValueError):
            datastore.tile_overlap = default_tile_overlap
        datastore.e_per_ADU = e_per_ADU
        datastore.na = na
        datastore.ri = ri
        datastore.binning = binning
        datastore.noise_map = noise_map
        datastore._shading_maps = np.ones(
            (len(channel_names), *tuple(int(v) for v in noise_map_shape_yx)),
            dtype=np.float32,
        )  # not used yet
        datastore.voxel_size_zyx_um = voxel_size_zyx_um
        datastore.channel_psfs = channel_psfs

        # Update datastore state to note that calibrations are done
        datastore_state = datastore.datastore_state
        datastore_state.update({"Calibrations": True})
        datastore.datastore_state = datastore_state

        # Loop over data and create datastore.
        for round_idx in tqdm(range(num_rounds), desc="rounds"):
            # Get all stage positions for this round
            position_list = []
            for tile_idx in range(num_tiles):
                dataset_path = _first_dataset_dir(
                    root_path, root_name, round_idx, tile_idx
                )
                dataset = _load_dataset_silently(dataset_path)
                x_pos_um = np.round(
                    float(
                        dataset.read_metadata(channel=channel_to_test, z=0)[
                            "XPosition_um_Intended"
                        ]
                    ),
                    2,
                )
                y_pos_um = np.round(
                    float(
                        dataset.read_metadata(channel=channel_to_test, z=0)[
                            "YPosition_um_Intended"
                        ]
                    ),
                    2,
                )
                z_pos_um = np.round(
                    float(
                        dataset.read_metadata(channel=channel_to_test, z=0)[
                            "ZPosition_um_Intended"
                        ]
                    ),
                    2,
                )
                temp = [z_pos_um, y_pos_um, x_pos_um]
                position_list.append(np.asarray(temp))
                del dataset
            position_list = np.asarray(position_list)

            for tile_idx in tqdm(range(num_tiles), desc="tile", leave=False):
                # initialize datastore tile
                # this creates the directory structure and links fiducial rounds <-> readout bits
                if round_idx == 0:
                    datastore.initialize_tile(tile_idx)

                raw_image = imread(
                    _first_stack_path(root_path, root_name, round_idx, tile_idx)
                )
                if camera == "orcav3":
                    raw_image = np.swapaxes(raw_image, 0, 1)
                    if tile_idx == 0 and round_idx == 0:
                        correct_shape = raw_image.shape
                elif camera == "flir":
                    if tile_idx == 0 and round_idx == 0:
                        correct_shape = raw_image.shape
                if raw_image is None or raw_image.shape != correct_shape:
                    if raw_image.shape[0] < correct_shape[0]:
                        print(
                            "\nround="
                            + str(round_idx + 1)
                            + "; tile="
                            + str(tile_idx + 1)
                        )
                        print("Found shape: " + str(raw_image.shape))
                        print("Correct shape: " + str(correct_shape))
                        print("Replacing data with zeros.\n")
                        raw_image = np.zeros(correct_shape, dtype=np.uint16)
                    else:
                        size_to_trim = raw_image.shape[1] - correct_shape[1]
                        raw_image = raw_image[:, size_to_trim:, :].copy()

                # Correct if channels were acquired in reverse order (red->purple)
                if channel_order == "reversed":
                    raw_image = np.flip(raw_image, axis=0)

                # Correct for known camera gain and offset
                raw_image = (raw_image.astype(np.float32) - offset) * e_per_ADU
                raw_image[raw_image < 0.0] = 0.0
                raw_image = raw_image.astype(np.uint16)
                gain_corrected = True

                # Correct for known hot pixel map
                if camera == "flir":
                    raw_image = replace_hot_pixels(noise_map, raw_image)
                    raw_image = replace_hot_pixels(
                        np.max(raw_image, axis=0),
                        raw_image,
                        threshold=hot_pixel_threshold,
                    )
                    hot_pixel_corrected = True
                else:
                    hot_pixel_corrected = False

                stage_pos_zyx_um = _stage_position_zyx_um(
                    position_list, tile_idx, ndtiff_metadata
                )

                datastore.save_local_stage_position_zyx_um(
                    stage_pos_zyx_um, affine_zyx_px, tile=tile_idx, round=round_idx
                )

                for channel_idx in range(num_ch):
                    data_camera_corrected = _correct_channel_image(
                        raw_image, channel_idx, illuminations
                    )
                    wavelengths_um = (
                        ex_wavelengths_um[channel_idx],
                        em_wavelengths_um[channel_idx],
                    )
                    if channel_idx == 0:
                        datastore.save_local_corrected_image(
                            data_camera_corrected,
                            tile=tile_idx,
                            psf_idx=0,
                            gain_correction=gain_corrected,
                            hotpixel_correction=hot_pixel_corrected,
                            shading_correction=False,
                            round=round_idx,
                        )
                        datastore.save_local_wavelengths_um(
                            wavelengths_um,
                            tile=tile_idx,
                            round=round_idx,
                        )
                    else:
                        bit_idx = int(experiment_order[round_idx, channel_idx]) - 1
                        datastore.save_local_corrected_image(
                            data_camera_corrected,
                            tile=tile_idx,
                            psf_idx=channel_idx,
                            gain_correction=gain_corrected,
                            hotpixel_correction=hot_pixel_corrected,
                            shading_correction=False,
                            bit=bit_idx,
                        )
                        datastore.save_local_wavelengths_um(
                            wavelengths_um,
                            tile=tile_idx,
                            bit=bit_idx,
                        )

        datastore_state = datastore.datastore_state
        datastore_state.update({"Corrected": True})
        datastore.datastore_state = datastore_state

    # Calculate and apply flatfield corrections
    if not (use_illuminations):
        # reload datastore
        del datastore
        datastore = qi2labDataStore(datastore_path)

        data_camera_corrected = []

        # Calculate the fiducial correction from every tile in round 1.
        for tile_idx in tqdm(
            range(datastore.num_tiles),
            desc="fiducial flatfield data",
            leave=False,
        ):
            data_camera_corrected.append(
                datastore.load_local_corrected_image(
                    tile=tile_idx,
                    round=0,
                )
            )
        fiducial_illumination = estimate_shading(data_camera_corrected)
        del data_camera_corrected
        gc.collect()

        if save_illuminations:
            illuminations = np.zeros(
                (
                    num_ch,
                    fiducial_illumination.shape[0],
                    fiducial_illumination.shape[1],
                ),
                dtype=np.float32,
            )
            illuminations[0, :] = fiducial_illumination

        for round_idx in tqdm(range(datastore.num_rounds), desc="rounds"):
            for tile_idx in tqdm(range(datastore.num_tiles), desc="tile", leave=False):
                data_camera_corrected = datastore.load_local_corrected_image(
                    tile=tile_idx, round=round_idx, return_future=False
                )
                data_camera_corrected = (
                    (data_camera_corrected.astype(np.float32) / fiducial_illumination)
                    .clip(0, 2**16 - 1)
                    .astype(np.uint16)
                )
                datastore.save_local_corrected_image(
                    data_camera_corrected,
                    tile=tile_idx,
                    psf_idx=0,
                    gain_correction=True,
                    hotpixel_correction=False,
                    shading_correction=True,
                    round=round_idx,
                )

        rng = np.random.default_rng(0)
        datastore_bit_ids = list(datastore.bit_ids)
        for channel_idx in tqdm(
            range(1, num_ch),
            desc="readout flatfields",
            leave=True,
        ):
            channel_bit_ids = _readout_bit_ids(
                experiment_order,
                channel_idx,
                datastore_bit_ids,
            )
            sample_pairs = _sample_readout_tile_bit_pairs(
                channel_bit_ids,
                datastore.num_tiles,
                max_flatfield_images,
                rng,
            )
            data_camera_corrected = []
            for tile_idx, bit_id in tqdm(
                sample_pairs,
                desc=f"readout {channel_idx} flatfield data",
                leave=False,
            ):
                data_camera_corrected.append(
                    datastore.load_local_corrected_image(
                        tile=tile_idx,
                        bit=bit_id,
                    )
                )
            readout_illumination = estimate_shading(data_camera_corrected)
            del data_camera_corrected

            if save_illuminations:
                illuminations[channel_idx, :] = readout_illumination

            for bit_id in tqdm(channel_bit_ids, desc="bit", leave=False):
                for tile_idx in tqdm(
                    range(datastore.num_tiles),
                    desc="tile",
                    leave=False,
                ):
                    data_camera_corrected = datastore.load_local_corrected_image(
                        tile=tile_idx,
                        bit=bit_id,
                        return_future=False,
                    )
                    data_camera_corrected = (
                        (
                            data_camera_corrected.astype(np.float32)
                            / readout_illumination
                        )
                        .clip(0, 2**16 - 1)
                        .astype(np.uint16)
                    )
                    datastore.save_local_corrected_image(
                        data_camera_corrected,
                        tile=tile_idx,
                        psf_idx=channel_idx,
                        gain_correction=True,
                        hotpixel_correction=False,
                        shading_correction=True,
                        bit=bit_id,
                    )
                    del data_camera_corrected
            del readout_illumination
            gc.collect()

        if save_illuminations:
            from tifffile import TiffWriter

            illuminations_path = root_path / "illuminations.ome.tif"

            with TiffWriter(illuminations_path, bigtiff=True) as tif:
                metadata = {
                    "axes": "CYX",
                    "SignificantBits": 32,
                    "PhysicalSizeX": float(datastore.voxel_size_zyx_um[2]),
                    "PhysicalSizeXUnit": "µm",
                    "PhysicalSizeY": float(datastore.voxel_size_zyx_um[1]),
                    "PhysicalSizeYUnit": "µm",
                }
                options = {
                    "compression": "zlib",
                    "compressionargs": {"level": 8},
                    "predictor": True,
                    "photometric": "minisblack",
                    "resolutionunit": "CENTIMETER",
                }
                tif.write(
                    illuminations,
                    resolution=(
                        1e4 / float(datastore.voxel_size_zyx_um[2]),
                        1e4 / float(datastore.voxel_size_zyx_um[1]),
                    ),
                    **options,
                    metadata=metadata,
                )


def main() -> None:
    """Run the Typer app."""
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    app()


if __name__ == "__main__":
    main()
