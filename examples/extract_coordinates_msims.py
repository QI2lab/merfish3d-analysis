from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher import msi_utils, vis_utils, fusion, registration
import dask.diagnostics
import dask.array as da
from pathlib import Path
import zarr
from tqdm import tqdm
import numpy as np
import gc

def test_fusion():
    fuse_readouts = False
    write_fused_zarr = False

    output_dir_path = Path('/mnt/opm3/20240317_OB_MERFISH_7/processed_v2')

    fused_dir_path = output_dir_path / Path('fused')
    fused_dir_path.mkdir(parents=True, exist_ok=True)

    polyDT_dir_path = output_dir_path / Path('polyDT')
    readout_dir_path = output_dir_path / Path('readouts')

    tile_ids = sorted([entry.name for entry in polyDT_dir_path.iterdir() if entry.is_dir()],
                        key=lambda x: int(x.split('tile')[1].split('.zarr')[0]))

    msims = []
    for tile_idx, tile_id in enumerate(tqdm(tile_ids,desc='tile')):

        polyDT_current_path = polyDT_dir_path / Path(tile_id) / Path("round000.zarr")
        polyDT_current_tile = zarr.open(polyDT_current_path,mode='r')

        voxel_zyx_um = np.asarray(polyDT_current_tile.attrs['voxel_zyx_um'],
                                            dtype=np.float32)

        scale = {'z': voxel_zyx_um[0],
                'y': voxel_zyx_um[1],
                'x': voxel_zyx_um[2]}
            
        tile_position_zyx_um = np.asarray(polyDT_current_tile.attrs['stage_zyx_um'],
                                            dtype=np.float32)
        
        tile_grid_positions = {
            'z': tile_position_zyx_um[0],
            'y': tile_position_zyx_um[1],
            'x': tile_position_zyx_um[2],
        }

        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            im_data = []
            im_data.append(da.from_array(polyDT_current_tile['registered_decon_data']))
            
            readout_current_tile_path = readout_dir_path / Path(tile_id)
            bit_ids = sorted([entry.name for entry in readout_current_tile_path.iterdir() if entry.is_dir()],
                                    key=lambda x: int(x.split('bit')[1].split('.zarr')[0]))
            
            for bit_id in bit_ids:
                bit_current_tile = readout_current_tile_path / Path(bit_id)
                bit_current_tile = zarr.open(bit_current_tile,mode='r')
                im_data.append(da.from_array((bit_current_tile['registered_decon_data'])))

            im_data = da.stack(im_data)
            
        if ~fuse_readouts:
            im_data = im_data[0,:]
            sim = si_utils.get_sim_from_array(da.expand_dims(im_data,axis=0),
                                            dims=('c','z', 'y', 'x'),
                                            scale=scale,
                                            translation=tile_grid_positions,
                                            transform_key='stage_metadata')
        
        else:
            sim = si_utils.get_sim_from_array(im_data,
                                            dims=('c', 'z', 'y', 'x'),
                                            scale=scale,
                                            translation=tile_grid_positions,
                                            transform_key='stage_metadata')
            
        msim = msi_utils.get_msim_from_sim(sim,scale_factors=[])
        msims.append(msim)
        del im_data
        gc.collect()
            
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        with dask.diagnostics.ProgressBar():
            params = registration.register(
                msims,
                reg_channel_index=0,
                transform_key='stage_metadata',
                new_transform_key='translation_registered',
                registration_binning={'z': 4, 'y': 4, 'x': 4},
                pre_registration_pruning_method='otsu_threshold_on_overlap',
                plot_summary=False,
            )

    for tile_idx, param in enumerate(params):
        polyDT_current_path = polyDT_dir_path / Path(tile_ids[tile_idx]) / Path("round000.zarr")
        polyDT_current_tile = zarr.open(polyDT_current_path,mode='a')
        translation_zyx_um = np.round(np.array(param).squeeze()[:-1, -1],3).tolist()
        
        polyDT_current_tile.attrs['translation_zyx_um'] = translation_zyx_um
        del polyDT_current_tile
        
        fused_output_path = fused_dir_path / Path("fused.zarr")

    with dask.config.set(**{'array.slicing.split_large_chunks': False}):

        fused_sim = fusion.fuse(
            [msi_utils.get_sim_from_msim(msim) for msim in msims],
            transform_key='translation_registered',
            output_spacing={'z': voxel_zyx_um[0], 'y': voxel_zyx_um[1]*3.5, 'x': voxel_zyx_um[2]*3.5},
            output_chunksize=512,
            overlap_in_pixels=256,
        ).sel(t=0,c=0)
        
    for msim_idx, msim in enumerate(msims):
        affine = msi_utils.get_transform_from_msim(msim, transform_key='translation_registered').data.squeeze()
        origin = si_utils.get_origin_from_sim(msi_utils.get_sim_from_msim(msim), asarray=True)
        spacing = si_utils.get_spacing_from_sim(msi_utils.get_sim_from_msim(msim), asarray=True)

        print(tile_ids[msim_idx])
        print(affine)
        print(origin)
        print(spacing)
        
    fused_msim = msi_utils.get_msim_from_sim(fused_sim, scale_factors=[])

    affine = msi_utils.get_transform_from_msim(fused_msim, transform_key='translation_registered').data.squeeze()
    origin = si_utils.get_origin_from_sim(msi_utils.get_sim_from_msim(fused_msim), asarray=True)
    spacing = si_utils.get_spacing_from_sim(msi_utils.get_sim_from_msim(fused_msim), asarray=True)

    print('fused')
    print(affine)
    print(origin)
    print(spacing)

    with dask.diagnostics.ProgressBar():
        fused_data = da.squeeze(fused_sim).compute('single-threaded')
        
    if write_fused_zarr:
        fused_zarr = zarr.open(str(fused_output_path), mode="a")
        if fuse_readouts:
            try:
                fused_data = fused_zarr.zeros('fused_iso_zyx',
                                            shape=(fused_data.shape[0],fused_data.shape[1],fused_data.shape[2],fused_data.shape[3]),
                                            chunks=(1,1,256,256),
                                            compressor=None,
                                            dtype=np.uint16)
            except:
                fused_data = fused_zarr['fused_iso_zyx']
        else:
            try:
                fused_data = fused_zarr.zeros('fused_iso_zyx',
                                        shape=(fused_data.shape[0],fused_data.shape[1],fused_data.shape[2]),
                                        chunks=(1,256,256),
                                        compressor=None,
                                        dtype=np.uint16)
            except:
                fused_data = fused_zarr['fused_iso_zyx']
        
        fused_data[:] = fused_data
        
        fused_data.attrs['voxel_zyx_um'] = np.array([voxel_zyx_um[0], 
                                                    voxel_zyx_um[1]*3.5, 
                                                    voxel_zyx_um[2]*3.5]).tolist()

    del fused_data, fused_sim
    gc.collect()


if __name__ == '__main__':
    test_fusion()