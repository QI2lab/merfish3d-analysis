from wf_merfish.postprocess.PixelDecoder import PixelDecoder
from pathlib import Path
from tqdm import tqdm
import gc
import cupy as cp

#data_dir_path = Path('/home/qi2lab/Documents/github/wf-merfish/examples/simulated_images/cylinder/images/jitter-0_shift_amp-0_prop_fn-0_prop_fp-0/processed')
dataset_path = Path('/mnt/opm3/20240214_MouseBrain_UA_NewRO_RK/processed_v2')


decode_factory = PixelDecoder(dataset_path=dataset_path,
                            global_normalization_limits = (80,99.95),
                            overwrite_normalization=False)

tile_ids = decode_factory._tile_ids

del decode_factory
gc.collect()
cp.get_default_memory_pool().free_all_blocks()

for tile_idx, tile_id in enumerate(tqdm(tile_ids,desc='tile',leave=True)):
    
    decode_factory = PixelDecoder(dataset_path=dataset_path,
                                tile_idx=tile_idx)
    decode_factory.run_decoding(lowpass_sigma=(3,2,2),
                                distance_threshold=0.78,
                                magnitude_threshold=0.4,
                                skip_extraction=False)
    decode_factory.save_barcodes()
    decode_factory.cleanup()
    
    del decode_factory
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    
decode_factory.load_all_barcodes()
decode_factory.filter_all_barcodes(fdr_target=.25)
decode_factory.save_barcodes()
    
#     if False:
#         import napari

#         viewer = napari.Viewer()
        
#         viewer.add_image(decode_factory._scaled_pixel_images,
#                         scale=[.31,.088,.088],
#                             name='pixels')
        
#         viewer.add_image(decode_factory._decoded_image,
#                             scale=[.31,.088,.088],
#                             name='decoded')

#         viewer.add_image(decode_factory._magnitude_image,
#                             scale=[.31,.088,.088],
#                             name='magnitude')

#         viewer.add_image(decode_factory._distance_image,
#                             scale=[.31,.088,.088],
#                             name='distance')

#         napari.run()