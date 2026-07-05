"""
Convert synthetic statphysbio MERFISH data to qi2labdatastore.

This is an example on how to convert a synthetic experiment to the datastore
object that the qi2lab "merfish3d-analysis" package uses.

Required user parameters for system dependent variables are at end of script.

Shepherd 2025/08 - update for new BiFISH simulations.
Shepherd 2025/08 - update for v0.7
Shepherd 2024/12 - added more NDTIFF metadata extraction for camera and binning.
Shepherd 2024/12 - refactor
Shepherd 2024/11 - rework script to accept parameters.
Shepherd 2024/08 - rework script to utilize qi2labdatastore object.
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
import multiprocessing as mp

mp.set_start_method("spawn", force=True)

from itertools import compress
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from psfmodels import make_psf
from tifffile import imread
from tqdm.auto import tqdm

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.dataio import read_metadatafile

app = typer.Typer()
app.pretty_exceptions_enable = False


def synthetic_chromatic_affines_zyx_um(
    image_shape_zyx: tuple[int, int, int],
    voxel_size_zyx_um: tuple[float, float, float] | list[float],
    emission_wavelengths_um: tuple[float, ...] | list[float],
    shift_scale: float = 1.0,
) -> dict[float, np.ndarray]:
    """
    Return deterministic synthetic chromatic correction affines.

    The returned matrices map a chromatically shifted channel coordinate back
    into the lowest supplied wavelength coordinate system. The reference
    wavelength gets an identity transform. Longer wavelengths receive XY radial
    affine scaling about the image center plus a Z translation. The lateral
    shift is scaled directly for stress testing. The axial shift is capped so
    all synthetic cases remain below three z pixels, because larger axial
    displacements stop representing a plausible chromatic aberration.

    Parameters
    ----------
    image_shape_zyx : tuple[int, int, int]
        Image shape in Z, Y, X order.
    voxel_size_zyx_um : tuple[float, float, float] or list[float]
        Voxel size in microns in Z, Y, X order.
    emission_wavelengths_um : tuple[float, ...] or list[float]
        Emission wavelengths in microns.
    shift_scale : float, default=1.0
        Multiplier applied to the deterministic synthetic shift amplitudes.

    Returns
    -------
    dict[float, numpy.ndarray]
        Mapping from emission wavelength to channel-to-reference affine in
        Z/Y/X microns.
    """

    spacing = np.asarray(voxel_size_zyx_um, dtype=np.float32)
    shape = np.asarray(image_shape_zyx, dtype=np.float32)
    center_um = (shape - 1.0) * spacing / 2.0
    yx_radius_px = max(float(shape[1] - 1.0), float(shape[2] - 1.0)) / 2.0
    affines = {}
    sorted_wavelengths = sorted(float(v) for v in emission_wavelengths_um)
    reference_wavelength = sorted_wavelengths[0]
    edge_shifts_px = [0.0, 0.5, 1.0]
    z_shifts_px = [0.0, 0.25, 0.5]
    max_z_shifts_px = [0.0, 1.25, 2.5]
    for wavelength_index, wavelength in enumerate(sorted_wavelengths):
        wavelength = float(wavelength)
        shift_index = min(wavelength_index, len(edge_shifts_px) - 1)
        edge_shift_px = edge_shifts_px[shift_index] * float(shift_scale)
        z_shift_px = min(
            z_shifts_px[shift_index] * float(shift_scale),
            max_z_shifts_px[shift_index],
        )
        if np.isclose(wavelength, reference_wavelength):
            edge_shift_px = 0.0
            z_shift_px = 0.0

        source_scale = 1.0 + edge_shift_px / yx_radius_px
        correction_scale = 1.0 / source_scale
        affine = np.eye(4, dtype=np.float32)
        affine[1, 1] = correction_scale
        affine[2, 2] = correction_scale
        affine[1, 3] = center_um[1] * (1.0 - correction_scale)
        affine[2, 3] = center_um[2] * (1.0 - correction_scale)
        affine[0, 3] = -z_shift_px * spacing[0]
        affines[wavelength] = affine
    return affines


def _apply_synthetic_chromatic_aberration(
    image_zyx: np.ndarray,
    *,
    emission_wavelength_um: float,
    voxel_size_zyx_um: tuple[float, float, float] | list[float],
    emission_wavelengths_um: tuple[float, ...] | list[float],
    shift_scale: float,
) -> np.ndarray:
    """
    Apply synthetic chromatic aberration to one simulation readout image.

    Parameters
    ----------
    image_zyx : numpy.ndarray
        Input readout image in Z, Y, X order.
    emission_wavelength_um : float
        Emission wavelength in microns for the channel being written.
    voxel_size_zyx_um : tuple[float, float, float] or list[float]
        Voxel size in microns in Z, Y, X order.
    emission_wavelengths_um : tuple[float, ...] or list[float]
        Emission wavelengths in microns for all simulation channels.
    shift_scale : float
        Multiplier applied to the deterministic synthetic shift amplitudes.

    Returns
    -------
    numpy.ndarray
        Chromatically shifted image with the same shape and dtype as input.
    """

    from merfish3danalysis.utils.multiview_registration import (
        warp_array_to_reference_gpu,
    )

    affines = synthetic_chromatic_affines_zyx_um(
        tuple(int(v) for v in image_zyx.shape),
        voxel_size_zyx_um,
        emission_wavelengths_um,
        shift_scale=shift_scale,
    )
    affine = affines[float(emission_wavelength_um)]
    if np.allclose(affine, np.eye(4, dtype=np.float32)):
        return image_zyx

    warped = warp_array_to_reference_gpu(
        image_zyx.astype(np.float32, copy=False),
        transform_zyx_um=affine,
        spacing_zyx_um=voxel_size_zyx_um,
        reference_shape=image_zyx.shape,
        mode="constant",
        cval=0.0,
        order=1,
    )
    warped = np.clip(np.rint(warped), 0, np.iinfo(image_zyx.dtype).max)
    return warped.astype(image_zyx.dtype, copy=False)


@app.command()
def convert_data(
    root_path: Path,
    channel_names: list[str] | None = None,
    hot_pixel_image_path: Path | None = None,
    output_path: Path | None = None,
    codebook_path: Path | None = None,
    bit_order_path: Path | None = None,
    z_step: int = 1,
    synthetic_chromatic_aberration: bool = False,
    synthetic_chromatic_aberration_scale: float = 1.0,
) -> None:
    """Convert qi2lab microscope data to qi2lab datastore.

    Parameters
    ----------
    root_path: Path
        path to dataset
    channel_names: list[str], default ["alexa488", "atto565", "alexa647"]
        name of dye molecules used in ascending order of wavelength
    hot_pixel_image_path: Optional[Path], default None
        path to hot pixel map. Default of `None` will set it to all zeros.
    output_path: Optional[Path], default None
        path to output directory. Default of `None` and will be created
        within the root_path
    codebook_path: Optional[Path], default None
        path to codebook. Default of `None` assumes the file is in
        the root_path.
    bit_order_path: Optional[Path], default None
        path to bit order file. This file defines what bits are present in each
        imaging round, in channel order. Default of `None` assumes
        the file is in the root_path.
    z_step : int, default=1
        Keep every z_step-th axial plane when writing the datastore. The
        datastore z voxel size and generated PSFs are updated to match the
        retained axial sampling.
    synthetic_chromatic_aberration : bool, default=False
        If True, apply deterministic synthetic chromatic aberration to readout
        channels. This is intended only for simulation regression tests.
    synthetic_chromatic_aberration_scale : float, default=1.0
        Multiplier for the deterministic synthetic chromatic aberration.
    """
    if z_step <= 0:
        raise ValueError("z_step must be greater than 0.")

    # load codebook
    # --------------
    if channel_names is None:
        channel_names = ["alexa488", "atto565", "alexa647"]
    if codebook_path is None:
        codebook = pd.read_csv(root_path / Path("codebook.csv"))
    else:
        codebook = pd.read_csv(codebook_path)

    # load experimental order
    # -----------------------
    if bit_order_path is None:
        df_experiment_order = pd.read_csv(root_path / Path("bit_order.csv"))
        experiment_order = df_experiment_order.values
    else:
        df_experiment_order = pd.read_csv(bit_order_path)
        experiment_order = df_experiment_order.values

    # load experiment metadata
    # ------------------------
    metadata_path = root_path / Path("scan_metadata.csv")
    metadata = read_metadatafile(metadata_path)
    root_name = metadata["root_name"]
    num_rounds = metadata["num_r"]
    num_tiles = metadata["num_xyz"]
    num_ch = metadata["num_ch"]
    num_z = metadata["num_z"]
    output_z_planes = len(range(0, int(num_z), int(z_step)))
    if output_z_planes <= 0:
        raise ValueError(
            f"z_step={z_step} does not retain any planes from num_z={num_z}."
        )

    camera = "synthetic"
    e_per_ADU = metadata["gain"]
    offset = metadata["offset"]
    binning = 1
    channels_active = [
        metadata["blue_active"],
        metadata["yellow_active"],
        metadata["red_active"],
    ]

    channel_order_bool = metadata["channels_reversed"]
    if channel_order_bool:
        channel_order = "reversed"
    else:
        channel_order = "forward"

    voxel_size_zyx_um = [
        metadata["z_step_um"] * z_step,
        metadata["yx_pixel_um"],
        metadata["yx_pixel_um"],
    ]
    na = metadata["na"]
    ri = metadata["ri"]

    ex_wavelengths_um = [0.488, 0.561, 0.635]  # selected by channel IDs
    em_wavelengths_um = [0.520, 0.580, 0.670]  # selected by channel IDs
    channel_idxs = list(range(num_ch))
    channels_in_data = list(compress(channel_idxs, channels_active))

    if hot_pixel_image_path is None:
        pass
    else:
        imread(hot_pixel_image_path)

    affine_zyx_px = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )

    # initialize datastore
    if output_path is None:
        datastore_path = root_path / Path(r"qi2labdatastore")
        datastore = qi2labDataStore(datastore_path)
    else:
        datastore = qi2labDataStore(output_path)

    # required user parameters
    datastore.channels_in_data = channel_names

    # parameters from qi2lab microscope metadata
    datastore.num_rounds = num_rounds
    datastore.codebook = codebook
    datastore.experiment_order = experiment_order
    datastore.num_tiles = num_tiles
    try:
        datastore.microscope_type = metadata["experiment_type"]
    except Exception:
        if voxel_size_zyx_um[0] < 0.5:
            datastore.microscope_type = "3D"
        else:
            datastore.microscope_type = "2D"

    datastore.camera_model = camera
    try:
        datastore.tile_overlap = metadata["tile_overlap"]
    except Exception:
        datastore.tile_overlap = 0.2
    datastore.e_per_ADU = e_per_ADU
    datastore.na = na
    datastore.ri = ri
    datastore.binning = binning
    datastore.voxel_size_zyx_um = voxel_size_zyx_um

    # generate PSFs
    # --------------
    psf_z = int(output_z_planes)
    channel_psfs = []
    for channel_id in channels_in_data:
        psf = make_psf(
            z=psf_z,
            nx=51,
            dxy=voxel_size_zyx_um[1],
            dz=voxel_size_zyx_um[0],
            NA=na,
            wvl=em_wavelengths_um[channel_id],
            ns=1.47,
            ni=ri,
            ni0=ri,
            model="vectorial",
        ).astype(np.float32)
        psf = psf / np.sum(psf, axis=(0, 1, 2))
        channel_psfs.append(psf)
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
            stage_position_path = root_path / Path(
                root_name
                + "_r"
                + str(round_idx + 1).zfill(4)
                + "_tile"
                + str(tile_idx).zfill(4)
                + "_stage_positions.csv"
            )
            stage_positions = read_metadatafile(stage_position_path)
            stage_x = np.round(float(stage_positions["stage_x"]), 2)
            stage_y = np.round(float(stage_positions["stage_y"]), 2)
            stage_z = np.round(float(stage_positions["stage_y"]), 2)
            temp = [stage_z, stage_y, stage_x]
            position_list.append(np.asarray(temp))
        position_list = np.asarray(position_list)

        for tile_idx in tqdm(range(num_tiles), desc="tile", leave=False):
            # initialize datastore tile
            # this creates the directory structure and links fiducial rounds <-> readout bits
            if round_idx == 0:
                datastore.initialize_tile(tile_idx)

            # load raw image
            image_path = (
                root_path
                / Path(
                    root_name
                    + "_r"
                    + str(round_idx + 1).zfill(4)
                    + "_tile"
                    + str(tile_idx).zfill(4)
                    + "_1"
                )
                / Path(
                    "data_r"
                    + str(round_idx + 1).zfill(4)
                    + "_tile"
                    + str(tile_idx).zfill(4)
                    + ".tif"
                )
            )

            # load raw data and make sure it is the right shape. If not, write
            # zeros for this round/stage position.
            raw_image = imread(image_path)
            if tile_idx == 0 and round_idx == 0:
                raw_correct_shape = raw_image.shape
                correct_shape = (
                    raw_image.shape[0],
                    output_z_planes,
                    raw_image.shape[2],
                    raw_image.shape[3],
                )
            else:
                if raw_image is None or raw_image.shape != raw_correct_shape:
                    if raw_image.shape[0] < raw_correct_shape[0]:
                        print(
                            "\nround="
                            + str(round_idx + 1)
                            + "; tile="
                            + str(tile_idx + 1)
                        )
                        print("Found shape: " + str(raw_image.shape))
                        print("Correct shape: " + str(raw_correct_shape))
                        print("Replacing data with zeros.\n")
                        raw_image = np.zeros(raw_correct_shape, dtype=np.uint16)
                    else:
                        # print("\nround=" + str(round_idx + 1) + "; tile=" + str(tile_idx + 1))
                        # print("Found shape: " + str(raw_image.shape))
                        size_to_trim = raw_image.shape[1] - raw_correct_shape[1]
                        raw_image = raw_image[:, size_to_trim:, :].copy()
                        # print("Correct shape: " + str(correct_shape))
                        # print("Corrected to shape: " + str(raw_image.shape) + "\n")

            # Correct if channels were acquired in reverse order (red->purple)
            if channel_order == "reversed":
                raw_image = np.flip(raw_image, axis=0)
            raw_image = raw_image[:, ::z_step, :, :]

            # Correct for known camera gain and offset
            raw_image = (raw_image.astype(np.float32) - offset) * e_per_ADU
            raw_image[raw_image < 0.0] = 0.0
            raw_image = raw_image.astype(np.uint16)
            gain_corrected = True
            hot_pixel_corrected = False

            # load stage position
            corrected_y = position_list[tile_idx, 1]
            corrected_x = position_list[tile_idx, 2]

            corrected_x = np.round(corrected_x, 2)
            corrected_y = np.round(corrected_y, 2)
            stage_z = np.round(position_list[tile_idx, 0], 2)

            stage_pos_zyx_um = np.asarray(
                [stage_z, corrected_y, corrected_x], dtype=np.float32
            )

            # write fiducial data (ch_idx = 0) and metadata
            datastore.save_local_corrected_image(
                np.squeeze(raw_image[0, :]).astype(np.uint16),
                tile=tile_idx,
                psf_idx=0,
                gain_correction=gain_corrected,
                hotpixel_correction=hot_pixel_corrected,
                shading_correction=False,
                round=round_idx,
            )

            datastore.save_local_stage_position_zyx_um(
                stage_pos_zyx_um, affine_zyx_px, tile=tile_idx, round=round_idx
            )

            datastore.save_local_wavelengths_um(
                (ex_wavelengths_um[0], em_wavelengths_um[0]),
                tile=tile_idx,
                round=round_idx,
            )

            # write first readout channel (ch_idx = 1) and metadata
            readout_one = np.squeeze(raw_image[1, :]).astype(np.uint16)
            if synthetic_chromatic_aberration:
                readout_one = _apply_synthetic_chromatic_aberration(
                    readout_one,
                    emission_wavelength_um=em_wavelengths_um[1],
                    voxel_size_zyx_um=voxel_size_zyx_um,
                    emission_wavelengths_um=em_wavelengths_um[1:],
                    shift_scale=synthetic_chromatic_aberration_scale,
                )
            datastore.save_local_corrected_image(
                readout_one,
                tile=tile_idx,
                psf_idx=1,
                gain_correction=gain_corrected,
                hotpixel_correction=hot_pixel_corrected,
                shading_correction=False,
                bit=int(experiment_order[round_idx, 1]) - 1,
            )
            datastore.save_local_wavelengths_um(
                (ex_wavelengths_um[1], em_wavelengths_um[1]),
                tile=tile_idx,
                bit=int(experiment_order[round_idx, 1]) - 1,
            )

            # write second readout channel (ch_idx = 2) and metadata
            readout_two = np.squeeze(raw_image[2, :]).astype(np.uint16)
            if synthetic_chromatic_aberration:
                readout_two = _apply_synthetic_chromatic_aberration(
                    readout_two,
                    emission_wavelength_um=em_wavelengths_um[2],
                    voxel_size_zyx_um=voxel_size_zyx_um,
                    emission_wavelengths_um=em_wavelengths_um[1:],
                    shift_scale=synthetic_chromatic_aberration_scale,
                )
            datastore.save_local_corrected_image(
                readout_two,
                tile=tile_idx,
                psf_idx=2,
                gain_correction=gain_corrected,
                hotpixel_correction=hot_pixel_corrected,
                shading_correction=False,
                bit=int(experiment_order[round_idx, 2]) - 1,
            )
            datastore.save_local_wavelengths_um(
                (ex_wavelengths_um[2], em_wavelengths_um[2]),
                tile=tile_idx,
                bit=int(experiment_order[round_idx, 2]) - 1,
            )

    datastore.noise_map = np.zeros(
        (3, correct_shape[1], correct_shape[2]), dtype=np.float32
    )
    datastore._shading_maps = np.ones(
        (3, correct_shape[1], correct_shape[2]), dtype=np.float32
    )
    datastore_state = datastore.datastore_state
    datastore_state.update({"Corrected": True})
    datastore.datastore_state = datastore_state


def main() -> None:
    """
    Main.

    Returns
    -------
    None
        Function result.
    """
    app()


if __name__ == "__main__":
    main()
