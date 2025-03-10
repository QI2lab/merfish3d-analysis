import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
from merfish3danalysis.qi2labDataStore import qi2labDataStore
from pathlib import Path
import gc
from tqdm import tqdm
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher import msi_utils, registration, fusion, ngff_utils
import dask.diagnostics
import dask.array as da
import dask
import numpy as np
import zarr
import ngff_zarr as nz
from ngff_zarr import Methods

def register_all_channels(root_path : Path):
    """Register all channels across all tiles.
    
    Registration is performed using the polyDT channel.

    Parameters
    ----------
    root_path: Path
        path to experiment
    """

    # initialize datastore
    print("\nInitializing datastore...")
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    gene_ids = list(datastore.codebook['gene_id'])
    channel_ids = ["polyDT"] + gene_ids

    im_data = datastore.load_local_registered_image(
            tile=0, round=0, return_future=False
        )
    
    im_shape = im_data.shape
    del im_data

    # convert local tiles from first round to multiscale spatial images
    print("\nLazy loading fiducial channel...")
    msims = []
    for tile_idx, tile_id in enumerate(tqdm(datastore.tile_ids, desc="tile")):

        # load voxel size
        voxel_zyx_um = datastore.voxel_size_zyx_um

        # format voxel size for multiview-stitcher
        scale = {
            "z": voxel_zyx_um[0], 
            "y": voxel_zyx_um[1], 
            "x": voxel_zyx_um[2]
        }

        # load stage positions and camera <-> stage mapping from first round of imaging
        # all tiles are already mapped to round 0, so we use this as the coordinate system
        tile_position_zyx_um, affine_zyx_px = datastore.load_local_stage_position_zyx_um(
            tile_id, datastore.round_ids[0]
        )

        # format tile positions for multiview-stitcher
        tile_grid_positions = {
            "z": np.round(tile_position_zyx_um[0], 2),
            "y": np.round(tile_position_zyx_um[1], 2),
            "x": np.round(tile_position_zyx_um[2], 2),
        }

        # create empty array to hold all channels for this tile
        im_data = da.zeros(
            (
                1,
                im_shape[0],
                im_shape[1],
                im_shape[2]
            ),
            dtype=np.uint16)
        
        input_path = datastore_path / Path("polyDT") / Path(tile_id) / Path("round001.zarr")
        store = zarr.DirectoryStore(str(input_path))
        im_data[0,:] = da.from_zarr(store, component="registered_decon_data").astype("uint16")
        

        # create spatial image for all channels in current tile
        sim = si_utils.get_sim_from_array(
            im_data,
            dims=("c", "z", "y", "x"),
            scale=scale,
            translation=tile_grid_positions,
            affine=affine_zyx_px,
            transform_key="stage_metadata",
        )

        # convert to multiscale spatial image object and append to list for registration
        msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
        msims.append(msim)
        del im_data
        gc.collect()
    
    # perform registration
    print("\nPerforming registration...")
    with dask.diagnostics.ProgressBar():
        _ = registration.register(
            msims,
            reg_channel_index=0,
            transform_key="stage_metadata",
            new_transform_key="affine_registered",
            pre_registration_pruning_method="keep_axis_aligned",
            registration_binning={"z": 3, "y": 6, "x": 6},
            post_registration_do_quality_filter=True,
        )
        
        

    print("\nLazy loading and fusing full-resolution polyDT and readouts...")
    tile_ids = datastore.tile_ids
    for ch_idx in tqdm(range(len(channel_ids)),desc="channel"):
        msims_full = []
        print("Loading data for "+str(channel_ids[ch_idx])+"...")
        for tile_idx, msim in enumerate(msims):
        
            # parse the registered fidicual channel to get the registration metadata
            affine = msi_utils.get_transform_from_msim(
                msim, transform_key="affine_registered"
            ).data.squeeze()
            affine = np.round(affine, 2)
            origin = si_utils.get_origin_from_sim(
                msi_utils.get_sim_from_msim(msim), asarray=False
            )
            scale = si_utils.get_spacing_from_sim(
                msi_utils.get_sim_from_msim(msim), asarray=False
            )

            # temporary variable for channel data
            im_data = da.zeros((1,im_shape[0],im_shape[1],im_shape[2]),dtype=np.uint16)

            # lazy load tile data
            tile_id = tile_ids[tile_idx]
        
            # lazy load deconvolved polyDT
            if ch_idx == 0:
                input_path = datastore_path / Path("polyDT") / Path(tile_id) / Path("round001.zarr")
                store = zarr.DirectoryStore(str(input_path))
                im_data[0,:] = da.from_zarr(store, component="registered_decon_data").astype("uint16")
            # lazy load deconvolved * (u-fish prediction>0.25) readout bits
            else:
                input_path = datastore_path / Path("readouts") / Path(tile_id) / Path("bit"+str(ch_idx).zfill(3)+".zarr")
                store = zarr.DirectoryStore(str(input_path))
                im_data[0,:] = (da.from_zarr(store, component="registered_decon_data").astype(np.float32) *\
                    da.from_zarr(store, component = "registered_ufish_data").astype(np.float32).clip(0.25,1)).astype(np.uint16)

            # create spatial image for all channels in current tile using registration metadata instead of stage metadata
            sim_full = si_utils.get_sim_from_array(
                im_data,
                dims=("c", "z", "y", "x"),
                scale=scale,
                translation=origin,
                affine=affine,
                transform_key="affine_registered"
            )

            # convert to multiscale spatial image object and append to list for fusion
            msim_full = msi_utils.get_msim_from_sim(sim_full, scale_factors=[])
            msims_full.append(msim_full)
            del im_data
            gc.collect()
        
        # create fused image object using previously calculated registration metadata and all channels
        print("Constructing fusion...")
        with dask.diagnostics.ProgressBar():
            fused = fusion.fuse(
                [msi_utils.get_sim_from_msim(msim_full) for msim_full in msims_full],
                transform_key='affine_registered',
                output_chunksize=512,
                overlap_in_pixels=64,
                )
        
        # Create zarr for full-resolution data
        full_res_path = root_path / Path(r"full_resolution") / Path("ch"+str(ch_idx).zfill(3)+".zarr")
        if not((full_res_path).exists()):
            full_res_path.mkdir()
        
        output_zarr_arr = zarr.open(
            full_res_path,
            shape=fused.data.shape,
            chunks=fused.data.chunksize,
            dtype=fused.data.dtype,
            write_empty_chunks=False,
            dimension_separator="/",
            fill_value=0,
            mode="w",
        )
                
        # fuse all channels and all tiles to disk
        print("Saving full-resolution fusion...")
        ngff_utils

        with dask.config.set({'optimization.fuse.active': True}):
            fused.data.to_zarr(
                output_zarr_arr,
                overwrite=True,
                dimension_separator="/",
                return_stored=False,
                compute=True,
                scheduler="threads"
            )
        del output_zarr_arr, fused
    
    # THIS DOES NOT WORK YET!
    # HERE BE DRAGONS
    
    # # fuse all channels and all tiles to disk into one big zarr
    # print("\nConverting full dataset fusion to multiscale ome.zarr...")
    
    # print("\nLoading full-scale resolutions...")
    # im_data = []
    # for ch_idx in tqdm(range(len(channel_ids)),desc="channel"):
    #     full_res_path = root_path / Path(r"full_resolution") / Path("ch"+str(ch_idx).zfill(3)+".zarr")
    #     im_data.append(da.from_zarr(full_res_path))
        
    # im_data = da.stack(im_data,axis=2)
    
    # image = nz.to_ngff_image(
    #     da.squeeze(im_data),
    #     dims=['c','z','y', 'x'],
    #     scale={'z' : .32,'y' : .098,'x': .098},
    #     translation={'z' : 1.0,'y' : 1.0,'x': 1.0}
    # )
    
    # print("\nCreating multiscale mapping...")
    # with dask.diagnostics.ProgressBar():
    #     multiscales = nz.to_multiscales(
    #         image,
    #         scale_factors=[2,4],
    #         method=Methods.ITK_BIN_SHRINK,
    #         cache=True
    #     )
    
    # print("\nWriting to disk...")
    # ome_output_path = root_path / Path("full_exp.ome.zarr")
    
    # # set compressor options
    # kwargs = {
    #     "compressor": {
    #         "name": "blosc",
    #         "configuration": {
    #             "cname": "zstd",
    #             "clevel": 5,
    #             "shuffle": "bitshuffle"
    #         }
    #     }
    # }
    
    # with dask.diagnostics.ProgressBar():
    #     nz.to_ngff_zarr(str(ome_output_path), multiscales,version=0.4,**kwargs)
    
if __name__ == "__main__":
    root_path = Path(r"/mnt/data2/bioprotean/20250220_Bartelle_control_smFISH_TqIB")
    register_all_channels(root_path)