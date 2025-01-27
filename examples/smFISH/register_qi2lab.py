"""
Perform registration on iterative smFISH qi2labdatastore.

Shepherd 2024/08 - rework script to utilized qi2labdatastore object.
"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.DataRegistration import DataRegistration
from pathlib import Path
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher import msi_utils, registration, fusion
import numpy as np
import gc
import dask.diagnostics
from tqdm import tqdm

def local_register_data():
    # root data folder
    root_path = Path(r"/data/smFISH/12062024_Bartelle24hrcryo_sample2")

    # initialize datastore
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    registration_factory = DataRegistration(
        datastore=datastore,
        perform_optical_flow=True,
    )

    registration_factory.register_all_tiles()
    
    datastore_state = datastore.datastore_state
    datastore_state.update({"LocalRegistered": True})
    datastore.datastore_state = datastore_state
    
def global_register_data():
    
    # root data folder
    root_path = Path(r"/data/smFISH/12062024_Bartelle24hrcryo_sample2")

    # initialize datastore
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    
    
    # find max y & x stage position. Need to reverse yx positions to match
    # stage motion.
    for tile_idx, tile_id in enumerate(datastore.tile_ids):
        round_id = datastore.round_ids[0]
        tile_position_zyx_um = datastore.load_local_stage_position_zyx_um(
            tile_id, round_id
        )
        
        if tile_idx == 0:
            max_y = np.round(tile_position_zyx_um[1],2)
            max_x = np.round(tile_position_zyx_um[2],2)
        else:
            if max_y < np.round(tile_position_zyx_um[1],2):
                max_y = np.round(tile_position_zyx_um[1],2)
            if max_x < np.round(tile_position_zyx_um[2],2):
                max_x = np.round(tile_position_zyx_um[2],2)
        
    msims = []
    for tile_idx, tile_id in enumerate(tqdm(datastore.tile_ids, desc="tile")):
        round_id = datastore.round_ids[0]

        voxel_zyx_um = datastore.voxel_size_zyx_um

        scale = {"z": voxel_zyx_um[0], "y": voxel_zyx_um[1], "x": voxel_zyx_um[2]}

        tile_position_zyx_um = datastore.load_local_stage_position_zyx_um(
            tile_id, round_id
        )

        tile_grid_positions = {
            "z": np.round(tile_position_zyx_um[0],2),
            "y": max_y - np.round(tile_position_zyx_um[1],2), # reverse y axis position
            "x": max_x -np.round(tile_position_zyx_um[2],2), # reverse x  axis position
        }

        im_data = []
        im_data.append(datastore.load_local_registered_image(
            tile=tile_id, round=round_id, return_future=False
        ))
        for bit_id in datastore.bit_ids:
            im_data.append(datastore.load_local_registered_image(
                tile=tile_id, bit=bit_id,return_futre = False)
            )
        im_data = np.asarray(im_data,dtype=np.uint16)

        sim = si_utils.get_sim_from_array(
            im_data,
            dims=("c", "z", "y", "x"),
            scale=scale,
            translation=tile_grid_positions,
            transform_key="stage_metadata",
        )

        msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
        msims.append(msim)
        del im_data
        gc.collect()
        
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        with dask.diagnostics.ProgressBar():
            _ = registration.register(
                msims,
                reg_channel_index=0,
                transform_key="stage_metadata",
                new_transform_key="translation_registered",
                registration_binning={"z": 4, "y": 4, "x": 4},
                plot_summary=False,
            )

    for tile_idx, msim in enumerate(msims):
        affine = msi_utils.get_transform_from_msim(
            msim, transform_key="translation_registered"
        ).data.squeeze()
        affine = np.round(affine, 2)
        origin = si_utils.get_origin_from_sim(
            msi_utils.get_sim_from_msim(msim), asarray=True
        )
        spacing = si_utils.get_spacing_from_sim(
            msi_utils.get_sim_from_msim(msim), asarray=True
        )

        datastore.save_global_coord_xforms_um(
            affine_zyx_um=affine,
            origin_zyx_um=origin,
            spacing_zyx_um=spacing,
            tile=tile_idx,
        )
        
        
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        fused_sim = fusion.fuse(
            [msi_utils.get_sim_from_msim(msim, scale='scale0') for msim in msims],
            transform_key='translation_registered',
            output_spacing={
                'z': voxel_zyx_um[0], 
                'y': voxel_zyx_um[1], 
                'x': voxel_zyx_um[2],
            },
            output_chunksize=1024,
            overlap_in_pixels=256,
        )
            
        fused_msim = msi_utils.get_msim_from_sim(fused_sim, scale_factors=[])
        affine = msi_utils.get_transform_from_msim(fused_msim, transform_key='translation_registered').data.squeeze()
        origin = si_utils.get_origin_from_sim(msi_utils.get_sim_from_msim(fused_msim), asarray=True)
        spacing = si_utils.get_spacing_from_sim(msi_utils.get_sim_from_msim(fused_msim), asarray=True)
        
        del fused_msim
        
        datastore.save_global_fidicual_image(
            fused_image=fused_sim.data.compute(),
            affine_zyx_um=affine,
            origin_zyx_um=origin,
            spacing_zyx_um=spacing,
            fusion_type='all'
        )
    
        del fused_sim
        gc.collect()
        
    datastore_state = datastore.datastore_state
    datastore_state.update({"GlobalRegistered": True})
    datastore_state.update({"Fused": True})
    datastore.datastore_state = datastore_state
                    
if __name__ == "__main__":
    local_register_data()
    global_register_data()