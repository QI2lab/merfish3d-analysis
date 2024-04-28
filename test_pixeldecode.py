from wf_merfish.postprocess.PixelDecoder import PixelDecoder
from pathlib import Path
from tqdm import tqdm
import gc
import cupy as cp

dataset_path = Path('/mnt/opm3/20240317_OB_MERFISH_7/processed_v2/')

decode_factory = PixelDecoder(dataset_path=dataset_path,
                              global_normalization_limits=[.1,80.0],
                              overwrite_normalization=True,
                              exp_type='3D',
                              merfish_bits=16)

tile_ids = decode_factory._tile_ids

del decode_factory
gc.collect()
cp.get_default_memory_pool().free_all_blocks()

for tile_idx, tile_id in enumerate(tqdm(tile_ids,desc='tile',leave=True)):
    
    decode_factory = PixelDecoder(dataset_path=dataset_path,
                                  overwrite_normalization=False,
                                  tile_idx=tile_idx,
                                  exp_type='3D',
                                  merfish_bits=16)
    decode_factory.run_decoding(lowpass_sigma=(3,1,1),
                                distance_threshold=0.8,
                                magnitude_threshold=.3,
                                minimum_pixels=27,
                                skip_extraction=False)
                                
    if tile_idx == 0:
        import napari

        viewer = napari.Viewer()
        
        viewer.add_image(decode_factory._scaled_pixel_images,
                        scale=[.31,.088,.088],
                            name='pixels')
        
        viewer.add_image(decode_factory._decoded_image,
                            scale=[.31,.088,.088],
                            name='decoded')

        viewer.add_image(decode_factory._magnitude_image,
                            scale=[.31,.088,.088],
                            name='magnitude')

        viewer.add_image(decode_factory._distance_image,
                            scale=[.31,.088,.088],
                            name='distance')

        napari.run()
    
    
    decode_factory.save_barcodes()
    decode_factory.cleanup()
    
    del decode_factory
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

decode_factory = PixelDecoder(dataset_path=dataset_path,
                              overwrite_normalization=False,
                              exp_type='3D',
                              merfish_bits=16,
                              verbose=2)
    
decode_factory.load_all_barcodes()
decode_factory.filter_all_barcodes(fdr_target=.05)
decode_factory.assign_cells()
decode_factory.save_barcodes(format='parquet')
decode_factory.save_all_barcodes_for_baysor()

del decode_factory
gc.collect()
cp.get_default_memory_pool().free_all_blocks()