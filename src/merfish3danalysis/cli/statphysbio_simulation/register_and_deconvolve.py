"""
Perform registration on simulated statphysbio data.

Shepherd 2025/08 - update for new BiFISH simulations.
Shepherd 2024/12 - create script to run on simulation.
"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.DataRegistration import DataRegistration
from pathlib import Path
import numpy as np
from tifffile import TiffWriter
from typing import Optional
import typer

app = typer.Typer()
app.pretty_exceptions_enable = False

@app.command()
def manage_data_registration_states(root_path: Path):
    local_register_data(root_path)
    global_register_data(root_path,create_max_proj_tiff=False)

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
        datastore=datastore, 
        perform_optical_flow=False, 
        overwrite_registered=True,
        save_all_polyDT_registered=False
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

    affine_zyx_px = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ],dtype=np.float32)

    origin = np.asarray([0.,0.,0.],dtype=np.float32)

    spacing = np.asarray(datastore.voxel_size_zyx_um.copy(), dtype=np.float32)

    datastore.save_global_coord_xforms_um(
        affine_zyx_um=affine_zyx_px,
        origin_zyx_um=origin,
        spacing_zyx_um=spacing,
        tile=0,
    )

    datastore.save_global_fidicual_image(
        fused_image=datastore.load_local_registered_image(tile=0,round=0,return_future=False),
        affine_zyx_um=affine_zyx_px,
        origin_zyx_um=origin,
        spacing_zyx_um=spacing,
    )
    
    # write max projection OME-TIFF for cellpose GUI
    if create_max_proj_tiff:
        # load downsampled, fused polyDT image and coordinates 
        polyDT_fused, _, _, spacing_zyx_um = datastore.load_global_fidicual_image(return_future=False)
        
        # create max projection
        polyDT_max_projection = np.max(np.squeeze(polyDT_fused),axis=0)
        del polyDT_fused
        
        filename = 'polyDT_max_projection.ome.tiff'
        cellpose_path = datastore._datastore_path / Path("segmentation") / Path("cellpose")
        cellpose_path.mkdir(exist_ok=True)
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

    # update datastore state
    datastore_state = datastore.datastore_state
    datastore_state.update({"GlobalRegistered": True})
    datastore_state.update({"Fused": True})
    datastore.datastore_state = datastore_state

def main():
    app()

if __name__ == "__main__":
    main()