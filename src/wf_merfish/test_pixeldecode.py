from wf_merfish.postprocess.PixelDecoder import PixelDecoder
#from wf_merfish.postprocess.BarcodeFilter import BarcodeFilter
from pathlib import Path
import napari

data_dir_path = Path('/home/qi2lab/Documents/github/wf-merfish/examples/simulated_images/cylinder/images/jitter-0_shift_amp-0_prop_fn-0_prop_fp-0/processed')

decode_factory = PixelDecoder(data_dir_path=data_dir_path)
print(decode_factory._scale_factors)
decode_factory._lp_filter()
decode_factory._decode_pixels()

if True:
    viewer = napari.Viewer()
    
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


decode_factory._extract_barcodes(minimum_area=4)
decode_factory._save_barcodes()

import matplotlib.pyplot as plt
import numpy as np

optimizer = BarcodeFilter(data_dir_path)
#optimizer._find_threshold_wollman()
optimizer.print_entry_counts(df=optimizer.full_df)
# optimizer.train_model()
# optimizer.find_optimal_threshold_v2()
# df_filtered = optimizer.filter_dataframe_v2()
# optimizer.print_entry_counts(df=df_filtered)

# target = 'called_distance_cosine'
# # Categorize each row based on whether ID starts with "Track" or "Song"
# df_filtered['Category'] = np.where(df_filtered['gene_id'].str.startswith('Blank'), 'noncoding', 'coding')

# # Plotting
# fig, ax = plt.subplots()

# # Colors for each category
# colors = {'noncoding': 'blue', 'coding': 'orange'}

# bin_edges = np.linspace(0.4, 0.5, 21)

# # Group the DataFrame by category and plot each group
# for category, group_data in df_filtered.groupby('Category'):
#     ax.hist(group_data[target], label=category, color=colors[category], alpha=0.6, bins=bin_edges, edgecolor='black')

# ax.legend()
# plt.xlabel(target)
# plt.ylabel('Frequency')
# plt.title('Histogram of area Colored by gene_id category')
# plt.show()