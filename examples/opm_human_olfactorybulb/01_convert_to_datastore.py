"""
Convert raw qi2lab OPM MERFISH data to qi2labdatastore.

This is an example on how to convert a qi2lab OPM experiment to the datastore
object that the qi2lab "merfish3d-analysis" package uses. Most of the
parameters are automatically extracted from the metadata written by qi2lab
microscopes.

Required user parameters for system dependent variables are at end of script.

Shepherd 2024/05 - Adapt for qi2lab OPM data
"""

import multiprocessing as mp
mp.set_start_method('forkserver', force=True)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)

from merfish3danalysis.qi2labDataStore import qi2labDataStore
import tensorstore as ts
from opm_processing.imageprocessing.opmpsf import generate_skewed_psf
from opm_processing.imageprocessing.rlgc import chunked_rlgc
from opm_processing.imageprocessing.opmtools import orthogonal_deskew, deskew_shape_estimator
from opm_processing.dataio.metadata import extract_channels, find_key, extract_stage_positions
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tifffile import imread, TiffWriter
from tqdm import tqdm
from typing import Optional
from ryomen import Slicer
import gc

def convert_data(
    root_path: Path,
    baysor_binary_path: Path,
    baysor_options_path: Path,
    julia_threads: int,
    flatfield_correction: bool = True,
    channel_names: Optional[list[str]] = ["alexa488", "atto565", "alexa647"],
    hot_pixel_image_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    codebook_path: Optional[Path] = None,
    bit_order_path: Optional[Path] = None,
    z_downsample_level: int = 2,
    deconvolve: bool = True
):
    """Convert qi2lab OPM microscope data to qi2lab datastore.

    Parameters
    ----------
    root_path: Path
        path to dataset
    baysor_binary_path: Path
        path to baysor binary
    baysor_options_path: Path
        path to baysor options toml
    julia_threads: int
        number of threads to use for Julia
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
    z_downsample_level: int, default 2
        downsample factor for deskewed z-axis.
    deconvolve: bool, default True
        whether to deconvolve the raw data before deskewing.
    """

    # load codebook
    # --------------
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

    # open raw datastore
    # -----------------------
    spec = {
        "driver" : "zarr",
        "kvstore" : {
            "driver" : "file",
            "path" : str(root_path)
        }
    }
    opm_datastore = ts.open(spec).result()

    # load experiment metadata
    # ------------------------
    zattrs_path = root_path / Path(".zattrs")
    with open(zattrs_path, "r") as f:
        zattrs = json.load(f)

    opm_mode = str(find_key(zattrs, "mode"))
    if "mirror" in opm_mode:
        scan_axis_step_um = float(find_key(zattrs,"image_mirror_step_um"))
        excess_scan_positions = 0
    elif "stage" in opm_mode:
        scan_axis_step_um = float(find_key(zattrs,"scan_axis_step_um")) 
        excess_scan_positions = int(find_key(zattrs,"excess_scan_positions"))
    pixel_size_um = float(find_key(zattrs,"pixel_size_um"))
    opm_tilt_deg = float(find_key(zattrs,"angle_deg"))
    camera_offset = float(find_key(zattrs,"offset"))
    camera_conversion = float(find_key(zattrs,"e_to_ADU"))

    channels = extract_channels(zattrs)
    stage_positions = extract_stage_positions(zattrs)
    # TO DO: start writing these in metadata!
    stage_x_flipped = False
    stage_y_flipped = True
    stage_z_flipped = True

    # flip x positions w.r.t. camera <-> stage orientation
    # TO DO: this axis is probably affected by the scan_flip flag, need to think
    #        about that.
    if stage_x_flipped:
        stage_x_max = np.max(stage_positions[:,2])
        for pos_idx, _ in enumerate(stage_positions):
            stage_positions[pos_idx,2] = stage_x_max - stage_positions[pos_idx,2]
    
    # flip y positions w.r.t. camera <-> stage orientation
    if stage_y_flipped:
        stage_y_max = np.max(stage_positions[:,1])
        for pos_idx, _ in enumerate(stage_positions):
            stage_positions[pos_idx,1] = stage_y_max - stage_positions[pos_idx,1]
    
    # flip z positions w.r.t. camera <-> stage orientation
    if stage_z_flipped:
        stage_z_max = np.max(stage_positions[:,0])
        for pos_idx, _ in enumerate(stage_positions):
            stage_positions[pos_idx,0] = stage_z_max - stage_positions[pos_idx,0]
    
    # # estimate shape of one deskewed volume
    deskewed_shape, _, _ = deskew_shape_estimator(
        [opm_datastore.shape[-3]-excess_scan_positions,opm_datastore.shape[-2],opm_datastore.shape[-1]], # type: ignore
        theta=opm_tilt_deg,
        distance=scan_axis_step_um, # type: ignore
        pixel_size=pixel_size_um
    )

    num_rounds = opm_datastore.shape[0]
    pos_shape = opm_datastore.shape[1]
    deskewed = np.zeros(
        (deskewed_shape[0]//z_downsample_level,deskewed_shape[1],deskewed_shape[2]),
        dtype=np.uint16
    )

    if flatfield_correction:
        flatfield_path = root_path.parents[0] / Path(str(root_path.stem)+"_flatfield.ome.tif")
        if flatfield_path.exists():
            flatfields = imread(flatfield_path).astype(np.float32)
        else:
            flatfields = call_estimate_illuminations(opm_datastore, camera_offset, camera_conversion)
            with TiffWriter(flatfield_path, bigtiff=True) as tif:
                metadata={
                    'axes': "CYX",
                    'SignificantBits': 32,
                    'PhysicalSizeX': pixel_size_um,
                    'PhysicalSizeXUnit': 'µm',
                    'PhysicalSizeY': pixel_size_um,
                    'PhysicalSizeYUnit': 'µm',
                }
                options = dict(
                    photometric='minisblack',
                    resolutionunit='CENTIMETER',
                )
                tif.write(
                    flatfields,
                    resolution=(
                        1e4 / pixel_size_um,
                        1e4 / pixel_size_um
                    ),
                    **options,
                    metadata=metadata
                )
    else:
        flatfields = np.ones((opm_datastore.shape[2],opm_datastore.shape[-2],opm_datastore.shape[-1]),dtype=np.float32)
    
    pad = (deskewed.shape[1]) % deskewed.shape[2]
    if not(pad == 0):
        pad_width = [(0, 0)] * deskewed.ndim
        pad_width[1] = (0, pad)
        deskewed_padded = np.pad(deskewed, pad_width, mode='constant', constant_values=0)

    crop_size = (deskewed.shape[0],deskewed.shape[2], deskewed.shape[2])
    overlap = (0,.1*deskewed.shape[2], 0)
    slices = Slicer(deskewed_padded, crop_size=crop_size, overlap=overlap)
    n_tiles_in_deskew = sum(1 for _ in slices)
    num_tiles = n_tiles_in_deskew * pos_shape
    del deskewed_padded
    gc.collect()

    channel_psfs = []
    for psf_idx in range(opm_datastore.shape[2]):
        psf = generate_skewed_psf(
            em_wvl=float(int(str(channels[psf_idx]).rstrip("nm")) / 1000),
            pixel_size_um=pixel_size_um,
            scan_axis_step_um=scan_axis_step_um,
            pz=0.0,
            plot=False
        )
        channel_psfs.append(psf)

    # initialize datastore
    if output_path is None:
        datastore_path = root_path / Path(r"qi2labdatastore")
        datastore = qi2labDataStore(datastore_path)
    else:
        datastore = qi2labDataStore(output_path)

    # required user parameters
    datastore.channels_in_data = channel_names
    datastore.baysor_path = baysor_binary_path
    datastore.baysor_options = baysor_options_path
    datastore.julia_threads = julia_threads

    # parameters from qi2lab microscope metadata
    datastore.num_rounds = num_rounds
    datastore.codebook = codebook
    datastore.experiment_order = experiment_order
    datastore.num_tiles = num_tiles
    datastore.microscope_type = "3D"
    datastore.camera_model = "OrcaFusionBT"
    try:
        datastore.tile_overlap = metadata["tile_overlap"]
    except Exception:
        datastore.tile_overlap = 0.2
    datastore.e_per_ADU = camera_conversion
    datastore.na = 1.35
    datastore.ri = 1.4
    datastore.binning = 1
    datastore.noise_map = np.zeros((opm_datastore.shape[-2], opm_datastore.shape[-1]), dtype=np.uint16)  # not used yet
    datastore._shading_maps = flatfields
    datastore.channel_psfs = None
    datastore.voxel_size_zyx_um = [
        z_downsample_level * pixel_size_um,
        pixel_size_um,
        pixel_size_um
    ]

    # Update datastore state to note that calibrations are done
    datastore_state = datastore.datastore_state
    datastore_state.update({"Calibrations": True})
    datastore.datastore_state = datastore_state
    tile_idx = 0

    ex_wavelengths_um = [0.488, 0.561, 0.635]  # selected by channel IDs
    em_wavelengths_um = [0.520, 0.580, 0.670]  # selected by channel IDs

    # Loop over data and create datastore.
    for round_idx in tqdm(range(num_rounds), desc="rounds"):
        for pos_idx in tqdm(range(pos_shape), desc="position", leave=False):
            current_pos_tile_idx = tile_idx
            for chan_idx in tqdm(range(opm_datastore.shape[2]), desc="channels", leave=False):

                # initialize datastore tile
                # this creates the directory structure and links fiducial rounds <-> readout bits
                if round_idx == 0:
                    datastore.initialize_tile(pos_idx)

                # load raw image
                camera_corrected_data = (((np.squeeze(opm_datastore[round_idx,pos_idx,chan_idx,:].read().result()).astype(np.float32)-camera_offset)*camera_conversion)/flatfields[chan_idx,:].astype(np.float32)).clip(0,2**16-1).astype(np.uint16)
                gain_corrected = True
                hot_pixel_corrected = False
                shading_corrected = True
                
                if "stage" in opm_mode:
                    flip_scan = True
                else:
                    flip_scan = False
                
                if flip_scan:
                    camera_corrected_data = np.flip(camera_corrected_data,axis=0)

                if deconvolve:
                    if camera_corrected_data.shape[1]==256:
                        chunk_size = 384
                    elif camera_corrected_data.shape[1]==512:
                        chunk_size = 196
                    deconvolved_data = chunked_rlgc(
                        camera_corrected_data[excess_scan_positions:,:,:],
                        np.asarray(channel_psfs[chan_idx]),
                        scan_chunk_size=chunk_size,
                        scan_overlap_size=32
                    )
                    
                    deskewed = orthogonal_deskew(
                        deconvolved_data,
                        theta = opm_tilt_deg,
                        distance = scan_axis_step_um,
                        pixel_size = pixel_size_um
                    )
                    deskewed_padded = np.pad(deskewed, pad_width, mode='constant', constant_values=0)
                else:
                    deskewed_padded = np.pad(camera_corrected_data, pad_width, mode='constant', constant_values=0)
                
                crop_size = (deskewed.shape[0],deskewed.shape[2], deskewed.shape[2])
                overlap = (0,.1*deskewed.shape[2], 0)
                slices = Slicer(deskewed_padded, crop_size=crop_size, overlap=overlap)
                local_tile_counter = current_pos_tile_idx
                current_tile_position_y = stage_positions[pos_idx,1]
                for crop, _, _ in slices:
                    stage_pos_zyx_um = np.asarray(
                        [
                            stage_positions[pos_idx,0], 
                            current_tile_position_y, 
                            stage_positions[pos_idx,2]
                        ],
                        dtype=np.float32
                    )
                    if channels[chan_idx] == "488nm":
                        datastore.save_local_corrected_image(
                            np.squeeze(crop).astype(np.uint16),
                            tile=local_tile_counter,
                            psf_idx=chan_idx,
                            gain_correction=gain_corrected,
                            hotpixel_correction=hot_pixel_corrected,
                            shading_correction=shading_corrected,
                            round=round_idx,
                        )
                        datastore.save_local_wavelengths_um(
                            (ex_wavelengths_um[chan_idx], em_wavelengths_um[chan_idx]),
                            tile=tile_idx,
                            round=round_idx,
                        )
                        datastore.save_local_stage_position_zyx_um(
                            stage_pos_zyx_um, tile=local_tile_counter, round=round_idx
                        )
                    else:
                        if channels[chan_idx] == "561nm":
                            this_chan_idx = 1
                        elif channels[chan_idx] == "637nm":
                            this_chan_idx = 2
                        datastore.save_local_corrected_image(
                            np.squeeze(crop).astype(np.uint16),
                            tile=tile_idx,
                            psf_idx=this_chan_idx,
                            gain_correction=gain_corrected,
                            hotpixel_correction=hot_pixel_corrected,
                            shading_correction=shading_corrected,
                            bit=int(experiment_order[round_idx, this_chan_idx]) - 1,
                        )
                        datastore.save_local_wavelengths_um(
                            (ex_wavelengths_um[this_chan_idx], em_wavelengths_um[this_chan_idx]),
                            tile=tile_idx,
                            bit=int(experiment_order[round_idx, this_chan_idx]) - 1,
                        )
                    local_tile_counter += 1
                    current_tile_position_y = current_tile_position_y + crop.shape[1]*pixel_size_um
           
    datastore_state = datastore.datastore_state
    datastore_state.update({"Corrected": True})
    datastore.datastore_state = datastore_state


def run_estimate_illuminations(datastore, camera_offset, camera_conversion, conn):
    """Helper function to run estimate_illuminations in a subprocess.
    
    This is necessary because jaxlib does not release GPU memory until the
    process exists. So we need to isolate it so that the GPU can be used for
    other processing tasks.
    
    Parameters
    ----------
    datastore: TensorStore
        TensorStore object containing the data.
    camera_offset: float
        Camera offset value.
    camera_conversion: float
        Camera conversion value.
    conn: Pipe
        Pipe connection to send the result back to the main process.
    """
    from opm_processing.imageprocessing.flatfield import estimate_illuminations

    try:
        flatfields = estimate_illuminations(datastore, camera_offset, camera_conversion)
        conn.send(flatfields)
    except Exception as e:
        conn.send(e)
    finally:
        conn.close()
    

def call_estimate_illuminations(datastore, camera_offset, camera_conversion):
    """Helper function to call estimate_illuminations in a subprocess.
    
    This is necessary because jaxlib does not release GPU memory until the
    process exists. So we need to isolate it so that the GPU can be used for
    other processing tasks.
    
    Parameters
    ----------
    datastore: TensorStore
        TensorStore object containing the data.
    camera_offset: float
        Camera offset value.
    camera_conversion: float
        Camera conversion value.
    
    Returns
    -------
    flatfields: np.ndarray
        Estimated illuminations.
    """
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(
        target=run_estimate_illuminations,
        args=(datastore, camera_offset, camera_conversion, child_conn)
    )
    p.start()
    result = parent_conn.recv()
    p.join()

    if p.exitcode != 0:
        raise RuntimeError("Subprocess failed")

    if isinstance(result, Exception):
        raise result

    return result


if __name__ == "__main__":
    root_path = Path(r"/mnt/server1/")
    baysor_binary_path = Path(
        r"/home/dps/Documents/github/Baysor/bin/baysor/bin/./baysor"
    )
    baysor_options_path = Path(
        r"/home/qi2lab/Documents/github/merfish3d-analysis/examples/human_olfactorybulb/qi2lab_humanOB.toml"
    )
    julia_threads = 20

    hot_pixel_image_path = None

    convert_data(
        root_path=root_path,
        baysor_binary_path=baysor_binary_path,
        baysor_options_path=baysor_options_path,
        julia_threads=julia_threads,
        deconvolve=False
    )