"""
Convert statphysbio simulation into a fake acquisition

This is an example on how to convert a statphysbio simulation into a fake 
qi2lab acquisition. The simulation comes as a flat tiff file with all z planes
for yellow, red, blue channels in a given round, then repeat for the next round.

Required user parameters for system dependent variables are at end of script.

Shepherd 2024/12 - create script based on metadata from Max in statphysbio lab.
"""

from pathlib import Path
from tifffile import imread, imwrite
from merfish3danalysis.utils.dataio import read_metadatafile, write_metadata
from typing import Optional
import numpy as np
import shutil

def convert_simulation(
    root_path: Path,
    output_path: Optional[Path] = None
):
    """Convert statphysbio simulation into a fake acquisition.
    
    Parameters
    ----------
    root_path: Path
        path to simulation
    output_path: Optional[Path]
        path to save fake acquisition. Default = None
    """
 
    # load metadata
    metadata_path = root_path / Path("scan_metadata.csv")
    metadata = read_metadatafile(metadata_path)
    root_name = metadata["root_name"]
    num_rounds = metadata["num_r"]
    num_ch = metadata["num_ch"]
    num_z = metadata["planes per bit"]
    yx_pixel_um = metadata["pixel_size [micron]"]
    z_pixel_um = metadata["axial_step_size [micron]"]
    gain = metadata["mean gain"]
    offset = metadata["mean offset"]
    
    
    # load simulated data
    simulation_data_path = root_path / Path("aligned_1.tiff")
    simulation_data = imread(simulation_data_path)
    
    # reshape simulation to match experimental design
    reshaped_simulation_data = simulation_data.reshape(
        num_rounds,
        num_ch,
        num_z,
        simulation_data.shape[-2],
        simulation_data.shape[-1]
    )
    
    # swap yellow and red channel to match how the microscope acquires data (red, yellow, blue)
    reshaped_simulation_data[:,[0,1],:,:,:] = reshaped_simulation_data[:,[1,0],:,:,:]
    reshaped_simulation_data = np.swapaxes(reshaped_simulation_data,1,2)
    
    fake_stage_position_zyx_um = [
        0.0,
        -1*yx_pixel_um*(reshaped_simulation_data.shape[-2]//2),
        -1*yx_pixel_um*(reshaped_simulation_data.shape[-1]//2)
    ]
    fake_tile_id = 0
    simulated_acq_path = root_path / Path("sim_acquisition")
    simulated_acq_path.mkdir(exist_ok=True)
    
    # execute fake experiment. Don't write all metadata to images, just what we need.
    for r_idx in range(num_rounds):
        tile_path = simulated_acq_path / Path("data_r"+str(r_idx+1).zfill(4)+"_tile"+str(fake_tile_id).zfill(4)+"_1")
        tile_path.mkdir(exist_ok=True)
        image_path = tile_path / Path("data_r"+str(r_idx+1).zfill(4)+"_tile"+str(fake_tile_id).zfill(4)+".tif")
        imwrite(
            image_path,
            np.squeeze(reshaped_simulation_data[r_idx,:,:,:]).astype(np.uint16)
        )
        
        stage_metadata_path = simulated_acq_path / Path("data_r"+str(r_idx+1).zfill(4)+"_tile"+str(fake_tile_id).zfill(4)+"_stage_positions.csv")
        current_stage_data = [{'stage_x': float(fake_stage_position_zyx_um[2]),
                                'stage_y': float(fake_stage_position_zyx_um[1]),
                                'stage_z': float(fake_stage_position_zyx_um[0]),
                                'offset_z': float(0.0),
                                'blue_active': True,
                                'yellow_active': True,
                                'red_active': True}]
        write_metadata(current_stage_data[0], stage_metadata_path)
        
    scan_param_data = [{'root_name': str(root_name),
                        'scan_type': "synthetic",
                        "exp_type" : "3D",
                        'camera' : "simulated",
                        "channels_reversed" : True,
                        'stage_flipped_x' : False,
                        'stage_flipped_y' : False,
                        'image_rotated' : False,
                        'image_flipped_y' : False,
                        'image_flipped_x' : False,
                        'num_t': int(1),
                        'num_r': int(num_rounds),
                        'num_xyz': int(1),
                        'num_ch': int(num_ch),
                        'na' : float(1.35),
                        'ri' : float(1.51),
                        'z_step_um' : float(z_pixel_um),
                        'yx_pixel_um' : float(yx_pixel_um),
                        'binning' : int(1),
                        'gain': float(gain),
                        'offset' : float(offset),
                        'overlap' : float(0.2),
                        'blue_active': True,
                        'yellow_active': True,
                        'red_active': True,
                        'blue_exposure': 200,
                        'yellow_exposure': 200,
                        'red_exposure': 200}]
    scan_metadata_path = simulated_acq_path / Path('scan_metadata.csv')
    write_metadata(scan_param_data[0], scan_metadata_path)
    
    # copy codebook and bit_order files to simulated acquisition folder
    sim_codebook_path = root_path / Path("codebook.csv")
    sim_acq_codebook_path = simulated_acq_path / Path("codebook.csv")
    shutil.copy(sim_codebook_path, sim_acq_codebook_path)
    
    sim_bitorder_path = root_path / Path("bit_order.csv")
    sim_acq_bitorder_path = simulated_acq_path / Path("bit_order.csv")
    shutil.copy(sim_bitorder_path, sim_acq_bitorder_path)
    
if __name__ == "__main__":
    root_path = Path(r"/mnt/opm3/20241218_statphysbio")
    convert_simulation(root_path=root_path)