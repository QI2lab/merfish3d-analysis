"""Convert Zhuang MERFISH MOP data to qi2labdatastore.

Data found here: https://download.brainimagelibrary.org/cf/1c/cf1c1a431ef8d021/

Download the "additional_files", "mouse1_sample1_raw", and
"dataset_metadata.xslx" folders.

Shepherd 2025/01 - rework script to accept parameters
Shepherd 2024/08 - rework script to utilized qi2labdatastore object.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from natsort import natsorted
from psfmodels import make_psf
from tifffile import imread, imwrite
from tqdm import tqdm

from merfish3danalysis.cli.qi2lab_microscopes.create_datastore import (
    _readout_bit_ids,
    _sample_readout_tile_bit_pairs,
)
from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.imageprocessing import estimate_shading, image_has_signal

app = typer.Typer(pretty_exceptions_enable=False)


def convert_data(
    root_path: Path,
    channel_names: list[str] | None = None,
    output_path: Path | None = None,
    codebook_path: Path | None = None,
    max_flatfield_images: int = 100,
) -> None:
    """Convert Zhuang images and apply qi2lab channel illumination correction.

    Parameters
    ----------
    root_path: Path
        path to dataset
    channel_names: list[str], default ["alexa488", "atto565", "alexa647"]
        name of dye molecules used in ascending order of wavelength
    output_path: Optional[Path], default None
        path to output datastore. Default of `None` writes
        ``root_path.parents[1] / "qi2labdatastore"``.
    codebook_path: Optional[Path], default None
        path to codebook. Default of `None` uses
        ``root_path / "additional_files" / "codebook.csv"``.
    max_flatfield_images: int, default 100
        Target number of distinct tiles with fiducial signal, as in the qi2lab
        CLI. Sample without replacement until enough pass or all are checked.
        Fiducial estimation samples round 1. Estimated CYX
        flatfields are saved to ``root_path / "illuminations.ome.tif"``.
    """

    if max_flatfield_images < 1:
        raise ValueError("max_flatfield_images must be at least 1.")

    # codebook
    if channel_names is None:
        channel_names = ["alexa488", "atto565", "alexa647"]
    if codebook_path is None:
        codebook_path = root_path / Path("additional_files") / Path("codebook.csv")
    codebook = pd.read_csv(codebook_path)
    codebook.drop(columns=["id"], inplace=True)
    codebook.rename(columns={"name": "gene_id"}, inplace=True)
    # Identify "RS" columns and rename them to "bitXX"
    rs_columns = [col for col in codebook.columns if col.startswith("RS")]
    bit_mapping = {
        rs_columns[i]: f"bit{str(i + 1).zfill(2)}" for i in range(len(rs_columns))
    }
    codebook.rename(columns=bit_mapping, inplace=True)

    # experimental order. 19 rounds with two readouts per round. The 20th round is fiducial and DAPI.
    # The actual experiment is more complicated, but the BIL dataset has already parsed the data.
    experiment_order = np.zeros((19, 3))
    for i in range(19):
        experiment_order[i, :] = [(i + 1), ((i + 1) * 2) - 1, (i + 1) * 2]

    # wavelengths from metadata and paper
    # https://www.nature.com/articles/s41586-021-03705-x
    wavelengths_um = np.zeros((3, 2))
    wavelengths_um[0, :] = [0.488, 0.520]
    wavelengths_um[1, :] = [0.650, 0.690]
    wavelengths_um[2, :] = [0.750, 0.790]

    # voxel size from metadata and paper
    # https://www.nature.com/articles/s41586-021-03705-x
    voxel_zyx_um = [1.5, 0.108, 0.108]

    # NA and RI from metadata and paper
    # https://www.nature.com/articles/s41586-021-03705-x
    na = 1.45
    ri = 1.51

    # gain and offset based on camera model (orca flash v3)
    # https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/static/sys/en/manual/C13440-20CU_IM_En.pdf
    e_per_ADU = 0.46  # from hamamatsu manual
    offset = 100.0  # from hamamats manual

    # stage positions from metadata
    stage_position_path = (
        root_path
        / Path("additional_files")
        / Path("fov_positions")
        / Path("mouse1_sample1.txt")
    )
    stage_position_df = pd.read_csv(stage_position_path, header=None)
    stage_positions = stage_position_df.values

    # num tiles back on number of stage positions
    num_tiles = stage_positions.shape[0]

    # generate 2D PSFs for each channel from metadata
    psfs = []
    for psf_idx in range(3):
        psf = make_psf(
            z=1,
            nx=51,
            dxy=voxel_zyx_um[1],
            NA=na,
            ni=ri,
            wvl=wavelengths_um[psf_idx, 1],
        )
        psf = psf / np.sum(psf)
        psfs.append(psf)
    psfs = np.asarray(psfs, dtype=np.float32)

    # initialize datastore
    if output_path is None:
        datastore_path = root_path.parents[1] / Path("qi2labdatastore")
    else:
        datastore_path = output_path

    # setup global datastore properties
    datastore = qi2labDataStore(datastore_path)
    datastore.num_rounds = 19
    datastore.codebook = codebook
    datastore.channels_in_data = channel_names
    datastore.experiment_order = experiment_order
    datastore.num_tiles = num_tiles
    # Required throughout this workflow: Z planes are spaced 1.5 microns apart.
    datastore.microscope_type = "2D"
    datastore.camera_model = "zhuang_orcav3"
    datastore.tile_overlap = 0.2
    datastore.e_per_ADU = e_per_ADU
    datastore.na = na
    datastore.ri = ri
    datastore.binning = 1
    datastore.noise_map = offset * (np.ones((2048, 2048), dtype=np.float32))
    datastore.channel_psfs = psfs
    datastore.voxel_size_zyx_um = voxel_zyx_um

    # Update datastore state to note that calibrations are done
    datastore_state = datastore.datastore_state.copy()
    datastore_state.update({"Calibrations": True, "Corrected": False})
    datastore.datastore_state = datastore_state

    # generate natural sorted list of raw data files
    raw_images_files_path = root_path / Path("mouse1_sample1_raw")
    raw_image_files = natsorted(list(raw_images_files_path.glob("*.tif")))

    affine_zyx_px = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )

    for tile_idx, raw_image_file in enumerate(tqdm(raw_image_files, desc="tile")):
        # initialize datastore tile
        # this creates the directory structure and links fiducial rounds <-> readout bits
        datastore.initialize_tile(tile_idx)

        # load raw image
        # some Zhuang tif files appear to be corrupted when downloaded with wget,
        # so we catch those errors and notify user.
        try:
            raw_image = imread(raw_image_file).astype(np.uint16)
            good_shape = raw_image.shape
        except Exception:
            print("Error reading: " + raw_image_file + "; Please re-download")
            raw_image = np.zeros((good_shape), dtype=np.uint16)

        # Correct for gain and offset
        raw_image = (raw_image).astype(np.float32) - offset
        raw_image[raw_image < 0.0] = 0.0
        raw_image = (raw_image * e_per_ADU).astype(np.uint16)

        # write fidicual data first.
        # Write the same fiducial for each round, as the data is already locally registered.
        # The metadata tells us fiducial is the 39th entry
        # The Zhuang data is both transposed and flipped, which we fix when writing the data
        psf_idx = 0
        for _round_idx, round_id in enumerate(
            tqdm(datastore.round_ids, desc="round", leave=False)
        ):
            datastore.save_local_corrected_image(
                np.squeeze(np.swapaxes(raw_image[38, :], 1, 2)),
                tile=tile_idx,
                psf_idx=psf_idx,
                gain_correction=True,
                hotpixel_correction=False,
                shading_correction=False,
                round=round_id,
            )
            datastore.save_local_stage_position_zyx_um(
                stage_positions[tile_idx, :],
                affine_zyx_px,
                tile=tile_idx,
                round=round_id,
            )
            datastore.save_local_wavelengths_um(
                (wavelengths_um[psf_idx, 0], wavelengths_um[psf_idx, 1]),
                tile=tile_idx,
                round=round_id,
            )

        # write all readouts
        # The bits go in order of the codebook
        # The Zhuang data is both transposed and flipped, which we fix when writing the data
        psf_idx = 1
        for bit_idx, bit_id in enumerate(
            tqdm(datastore.bit_ids, desc="bit", leave=False)
        ):
            datastore.save_local_corrected_image(
                np.squeeze(np.swapaxes(raw_image[bit_idx, :], 1, 2)),
                tile=tile_idx,
                psf_idx=psf_idx,
                gain_correction=True,
                hotpixel_correction=False,
                shading_correction=False,
                bit=bit_id,
            )
            datastore.save_local_wavelengths_um(
                (wavelengths_um[psf_idx, 0], wavelengths_um[psf_idx, 1]),
                tile=tile_idx,
                bit=bit_idx,
            )
            if psf_idx == 2:
                psf_idx = 1
            else:
                psf_idx = 2

    # Match qi2lab sampling and BaSiC estimation in the stored image orientation.
    illuminations = []
    rng = np.random.default_rng(0)
    for channel_idx in tqdm(range(3), desc="channel flatfields"):
        if channel_idx == 0:
            image_kind = "round"
            image_ids = datastore.round_ids
            candidate_tiles = (
                np.random.default_rng(0)
                .choice(num_tiles, size=num_tiles, replace=False)
                .tolist()
            )
            sample_pairs = [(tile, image_ids[0]) for tile in candidate_tiles]
            fiducial_tiles = []
        else:
            image_kind = "bit"
            image_ids = _readout_bit_ids(
                experiment_order, channel_idx, list(datastore.bit_ids)
            )
            sample_pairs = _sample_readout_tile_bit_pairs(
                image_ids, len(fiducial_tiles), max_flatfield_images, rng
            )
            sample_pairs = [
                (fiducial_tiles[tile], image_id) for tile, image_id in sample_pairs
            ]
        data_camera_corrected = []
        for tile, image_id in sample_pairs:
            image = datastore.load_local_corrected_image(
                tile=tile, **{image_kind: image_id}
            )
            if channel_idx != 0 or image_has_signal(image.result()):
                data_camera_corrected.append(image)
                if channel_idx == 0:
                    fiducial_tiles.append(tile)
                if len(data_camera_corrected) == max_flatfield_images:
                    break
            del image
        if not data_camera_corrected:
            raise ValueError(
                f"No images in channel {channel_idx} contain signal above background."
            )
        if channel_idx == 0 and len(data_camera_corrected) < min(
            num_tiles, max_flatfield_images
        ):
            warnings.warn(
                f"Only {len(data_camera_corrected)} images in channel {channel_idx} "
                "contain signal; using all available signal tiles.",
                stacklevel=2,
            )
        illumination = estimate_shading(data_camera_corrected)
        del data_camera_corrected
        illuminations.append(illumination)

        for image_id in tqdm(image_ids, desc=image_kind, leave=False):
            for tile_idx in range(num_tiles):
                image = datastore.load_local_corrected_image(
                    tile=tile_idx,
                    return_future=False,
                    **{image_kind: image_id},
                )
                image = (
                    (image.astype(np.float32) / illumination)
                    .clip(0, 2**16 - 1)
                    .astype(np.uint16)
                )
                datastore.save_local_corrected_image(
                    image,
                    tile=tile_idx,
                    psf_idx=channel_idx,
                    gain_correction=True,
                    hotpixel_correction=False,
                    shading_correction=True,
                    **{image_kind: image_id},
                )

    imwrite(
        root_path / "illuminations.ome.tif",
        np.asarray(illuminations, dtype=np.float32),
        bigtiff=True,
        compression="zlib",
        compressionargs={"level": 8},
        predictor=True,
        photometric="minisblack",
        resolutionunit="CENTIMETER",
        resolution=(1e4 / voxel_zyx_um[2], 1e4 / voxel_zyx_um[1]),
        metadata={
            "axes": "CYX",
            "SignificantBits": 32,
            "PhysicalSizeX": voxel_zyx_um[2],
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": voxel_zyx_um[1],
            "PhysicalSizeYUnit": "µm",
        },
    )

    # Only mark corrected_data complete after illumination correction is saved.
    datastore_state = datastore.datastore_state.copy()
    datastore_state.update({"Corrected": True})
    datastore.datastore_state = datastore_state


@app.command()
def main(root_path: Path) -> None:
    """Convert the Zhuang experiment using its channel names and codebook."""
    root_path = root_path.expanduser().resolve()

    convert_data(
        root_path=root_path,
        channel_names=["alexa488", "cy5", "alexa750"],
        codebook_path=root_path / Path("additional_files") / Path("codebook.csv"),
    )


if __name__ == "__main__":
    app()
