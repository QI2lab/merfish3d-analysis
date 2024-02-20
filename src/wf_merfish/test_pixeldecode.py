from wf_merfish.postprocess.PixelDecoder import PixelDecoder
from wf_merfish.postprocess.BarcodeFilter import BarcodeFilter
from pathlib import Path
import napari

data_dir_path = Path('/home/qi2lab/Documents/github/wf-merfish/examples/simulated_images/cylinder/images/jitter-0_shift_amp-0.2_prop_fn-0.1_prop_fp-0.7/processed')

# decode_factory = PixelDecoder(data_dir_path=data_dir_path)
# decode_factory._load_bit_data()
# decode_factory._load_filtered_images()
# decode_factory._intialize_scale_factors()

# for i in range(5):
#     decode_factory._decode_pixels()
#     if i==0 and True:
#         viewer = napari.Viewer()

#         viewer.add_image(decode_factory._pixel_magnitudes,
#                          scale=[.31,.088,.088],
#                          name='magnitudes')

#         viewer.add_image(decode_factory._l2_distances,
#                          scale=[.31,.088,.088],
#                          name='l2 distances')

#         napari.run()
#     if i==1:
#         decode_factory._overwrite = True
#     decode_factory._extract_refactors(extract_backgrounds=True)

# decode_factory._overwrite = False
# decode_factory._decode_pixels()

# if True:
#     viewer = napari.Viewer()

#     viewer.add_image(decode_factory._pixel_magnitudes,
#                     scale=[.31,.088,.088],
#                     name='magnitudes')

#     viewer.add_image(decode_factory._decoded_image,
#                     scale=[.31,.088,.088],
#                     name='decoded')
    
#     viewer.add_image(decode_factory._l2_distances,
#                     scale=[.31,.088,.088],
#                     name='distances')

#     napari.run()


# decode_factory._extract_barcodes()
# decode_factory._save_barcodes()

import matplotlib.pyplot as plt

optimizer = BarcodeFilter(data_dir_path)
optimizer.print_entry_counts()
optimizer.train_model()
optimizer.find_optimal_threshold()
df_filtered = optimizer.filter_dataframe()
optimizer.print_entry_counts(df_filtered)


# Assuming df is your DataFrame
x = df_filtered['area']
y = df_filtered['min_dispersion']

plt.hist2d(x, y, bins=10, cmap='Blues')
plt.colorbar()  # Adds a colorbar to indicate the scale
plt.xlabel('Area')
plt.ylabel('Intensity')
plt.title('2D Histogram of Area and Min Distance')
plt.show()
