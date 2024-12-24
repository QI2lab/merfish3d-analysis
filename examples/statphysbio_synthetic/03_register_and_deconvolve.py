"""
Perform registration on simulated statphysbio data.

Shepherd 2024/12 - create script to run on simulation.
"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.postprocess.DataRegistration import DataRegistration
from pathlib import Path
import numpy as np
from tifffile import TiffWriter
from typing import Optional

def local_register_data(root_path: Path):
    """Register each tile across rounds in local coordinates.

    Parameters
    ----------
    root_path: Path
        path to experiment
    """

    # initialize datastore
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    
    # initialize registration class
    registration_factory = DataRegistration(
        datastore=datastore, perform_optical_flow=False, overwrite_registered=False
    )

    # run local registration across rounds
    registration_factory.register_all_tiles()

    # update datastore state
    datastore_state = datastore.datastore_state
    datastore_state.update({"LocalRegistered": True})
    datastore.datastore_state = datastore_state


def global_register_data(
    root_path : Path, 
    create_max_proj_tiff: Optional[bool] = True
):
    """Register all tiles in first round in global coordinates. 
    
    Because there is only one tile in this simulation, we fake the registration.

    Parameters
    ----------
    root_path: Path
        path to experiment
    
    create_max_proj_tiff: Optional[bool]
        create max projection tiff in the segmentation/cellpose directory. 
        Default = True
    """
    
    # initialize datastore
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    
    datastore.save_global_coord_xforms_um(
        affine_zyx_um=np.identity(4),
        origin_zyx_um=datastore.load_local_stage_position_zyx_um(tile=0,round=0),
        spacing_zyx_um=np.asarray(
            (datastore.voxel_size_zyx_um[0],
            datastore.voxel_size_zyx_um[1],
            datastore.voxel_size_zyx_um[1])
        ),
        tile=0,
    )
    
    im_data = datastore.load_local_registered_image(
        tile=0, round=0, return_future=False
    )

    datastore.save_global_fidicual_image(
        fused_image=im_data,
        affine_zyx_um=np.identity(4),
        origin_zyx_um=np.zeros(3),
        spacing_zyx_um=np.asarray(
            (datastore.voxel_size_zyx_um[0],
            datastore.voxel_size_zyx_um[1],
            datastore.voxel_size_zyx_um[1])
        ),
    )

    # update datastore state
    datastore_state = datastore.datastore_state
    datastore_state.update({"GlobalRegistered": True})
    datastore_state.update({"Fused": True})
    datastore.datastore_state = datastore_state

    # write max projection OME-TIFF for cellpose GUI
    if create_max_proj_tiff:
        # load downsampled, fused polyDT image and coordinates 
        polyDT_fused, _, _, spacing_zyx_um = datastore.load_global_fidicual_image(return_future=False)
        
        # create max projection
        polyDT_max_projection = np.max(np.squeeze(polyDT_fused),axis=0)
        del polyDT_fused
        
        filename = 'polyDT_max_projection.ome.tiff'
        filename_path = datastore._datastore_path / Path("segmentation") / Path("cellpose") / Path(filename)
        with TiffWriter(filename_path, bigtiff=True) as tif:
            metadata={
                'axes': 'YX',
                'SignificantBits': 16,
                'PhysicalSizeX': spacing_zyx_um[2],
                'PhysicalSizeXUnit': 'µm',
                'PhysicalSizeY': spacing_zyx_um[1],
                'PhysicalSizeYUnit': 'µm',
            }
            options = dict(
                compression='zlib',
                compressionargs={'level': 8},
                predictor=True,
                photometric='minisblack',
                resolutionunit='CENTIMETER',
            )
            tif.write(
                polyDT_max_projection,
                resolution=(
                    1e4 / spacing_zyx_um[1],
                    1e4 / spacing_zyx_um[2]
                ),
                **options,
                metadata=metadata
            )
    
if __name__ == "__main__":
    root_path = Path(r"/mnt/opm3/20241218_statphysbio/sim_acquisition")
    local_register_data(root_path)
    global_register_data(root_path,create_max_proj_tiff=False)