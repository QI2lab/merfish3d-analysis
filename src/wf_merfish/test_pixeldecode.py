from wf_merfish.postprocess.PixelDecoder import PixelDecoder
#from wf_merfish.postprocess.BarcodeFilter import BarcodeFilter
from pathlib import Path
import napari

#data_dir_path = Path('/home/qi2lab/Documents/github/wf-merfish/examples/simulated_images/cylinder/images/jitter-0_shift_amp-0_prop_fn-0_prop_fp-0/processed')
data_dir_path = Path('/mnt/opm3/20240214_MouseBrain_UA_NewRO_RK/processed_v2')

decode_factory = PixelDecoder(data_dir_path=data_dir_path)
#decode_factory._hp_filter()
decode_factory._lp_filter()
decode_factory._decode_pixels(distance_threshold=1.7, # DPS note: why is distance threshold greater than it should be? normalization issue likely..
                              magnitude_threshold=2.5) 

if True:
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