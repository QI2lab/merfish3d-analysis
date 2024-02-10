import numpy as np
import pandas as pd
from scipy.optimize import newton, brentq
from pathlib import Path
import zarr

data_dir_path = Path('/mnt/opm3/20240124_OB_Full_MERFISH_UA_3_allrds/processed_v2')
calibration_dir_path = data_dir_path / Path("calibrations.zarr")
polyDT_dir_path = data_dir_path / Path('polyDT')
readout_dir_path = data_dir_path / Path('readouts')
localization_dir_path = data_dir_path / Path('localizations')
tile_ids = [entry.name for entry in readout_dir_path.iterdir() if entry.is_dir()]
tile_dir_path = readout_dir_path / Path(tile_ids[0])
bit_ids = [entry.name for entry in tile_dir_path.iterdir() if entry.is_dir()]

physical_spacing=[.31,.088,.088]

decode_results_path = localization_dir_path / Path(tile_ids[0]) / Path("decoded_nomask.parquet")
df_decoded = pd.read_parquet(decode_results_path)

nB = 21
nC = 122 

uniq_decoded_species, spec_counts = np.unique(df_decoded['species'], return_counts=True)
# Initialize sums
sum_blank = 0
sum_non_blank = 0

# Iterate through the genes list and sum occurrences
for gene, occurrence in zip(uniq_decoded_species, spec_counts):
    if "Blank" in gene:
        sum_blank += occurrence
    else:
        sum_non_blank += occurrence

print('Summary before filtering')
print('------------------------')
print(f"Sum of occurrences for 'Blank': {sum_blank}")
print(f"Sum of occurrences for non-'Blank': {sum_non_blank}")
print(f"Misidentification rate': {np.round((sum_blank/nB)/(sum_non_blank/nC),3)}")


def preprocess_data(df):
    """
    Applies log transformation to 'mean_intensity' and normalization to 'dispersion_xy' and 'dispersion_z'.
    """

    #df['log_amplitude'] = np.log10(df['amplitude'] + 1)  # Log transformation
    df['log_amplitude'] = (df['amplitude'] - df['amplitude'].min()) / (df['amplitude'].max() - df['amplitude'].min())
    df['normalized_dispersion_xyz'] = (df['z/x/y std'] - df['z/x/y std'].min()) / (df['z/x/y std'].max() - df['z/x/y std'].min())
    df['normalized_amplitude_std'] = (df['amplitude std'] - df['amplitude std'].min()) / (df['amplitude std'].max() - df['amplitude std'].min())
    df['normalized_error'] = (df['error'] - df['error'].min()) / (df['error'].max() - df['error'].min())
    return df

def calculate_adaptive_threshold(df, nB, nC, target_misid_rate, tolerance=0.001):
    """
    Calculates the adaptive threshold to achieve the target misidentification rate.
    """
    # Assuming 'species' column categorizes barcodes into coding and blank
    coding_barcodes = df[~df['species'].str.startswith('Blank')]
    blank_barcodes = df[df['species'].str.startswith('Blank')]
    
    # Define bin edges based on preprocessed data
    ampltiude_bins = np.linspace(df['log_amplitude'].min(), df['log_amplitude'].max(), num=21)
    dispersion_xyz_bins = np.linspace(df['normalized_dispersion_xyz'].min(), df['normalized_dispersion_xyz'].max(), num=21)
    ampltiude_std_bins = np.linspace(df['normalized_amplitude_std'].min(), df['normalized_amplitude_std'].max(), num=21)
    error_bins = np.linspace(df['normalized_error'].min(), df['normalized_error'].max(), num=21)
    
    # Calculate 3D histograms for coding and blank barcodes
    coding_hist, _ = np.histogramdd(coding_barcodes[['log_amplitude', 'normalized_dispersion_xyz', 'normalized_amplitude_std', 'normalized_error']].values, bins=(ampltiude_bins, dispersion_xyz_bins, ampltiude_std_bins, error_bins))
    blank_hist, _ = np.histogramdd(blank_barcodes[['log_amplitude', 'normalized_dispersion_xyz', 'normalized_amplitude_std', 'normalized_error']].values, bins=(ampltiude_bins, dispersion_xyz_bins, ampltiude_std_bins, error_bins))

    #total_hist = coding_hist
    #blank_fraction = blank_hist/coding_hist
    #blank_fraction[coding_hist == 0] = np.finfo(blank_fraction.dtype).max
    blank_fraction = np.divide(blank_hist, coding_hist, out=np.ones_like(blank_hist)*np.finfo(blank_hist.dtype).max, where=coding_hist!=0)
    blank_fraction = (blank_fraction / nB) / (nB + nC)
    
    def misid_rate(threshold):
        selected_bins = blank_fraction < threshold
        selected_coding = np.sum(coding_hist[selected_bins])
        selected_blank = np.sum(blank_hist[selected_bins])
        return (selected_blank / nB) / (selected_coding / nC) - target_misid_rate
    
    optimal_threshold = np.round(newton(misid_rate, x0=1e-6, x1=1e-4, tol=tolerance),5)

    return optimal_threshold, ampltiude_bins, dispersion_xyz_bins, ampltiude_std_bins, error_bins

def filter_dataframe_using_threshold(df, optimal_threshold, ampltiude_bins, dispersion_xyz_bins, ampltiude_std_bins, error_bins):
    """
    Filters the DataFrame based on the optimal threshold determined from the blank fraction histogram.
    """
    
    # Calculate the blank fraction histogram as before
    total_hist, _ = np.histogramdd(df[~df['species'].str.startswith('Blank')][['log_amplitude', 'normalized_dispersion_xyz', 'normalized_amplitude_std', 'normalized_error']].values, bins=(ampltiude_bins, dispersion_xyz_bins, ampltiude_std_bins, error_bins))
    blank_hist, _ = np.histogramdd(df[df['species'].str.startswith('Blank')][['log_amplitude', 'normalized_dispersion_xyz', 'normalized_amplitude_std', 'normalized_error']].values, bins=(ampltiude_bins, dispersion_xyz_bins, ampltiude_std_bins, error_bins))
    blank_fraction = np.divide(blank_hist, total_hist, out=np.zeros_like(blank_hist), where=total_hist!=0)
    
    # Identify bins below the threshold
    bins_below_threshold = blank_fraction < optimal_threshold
    
    # Map each data point to its corresponding bin index
    df['amplitude_bin_index'] = np.digitize(df['log_amplitude'], ampltiude_bins[:-1]) - 1
    df['dispersion_xyz_bin_index'] = np.digitize(df['normalized_dispersion_xyz'], dispersion_xyz_bins[:-1]) - 1
    df['amplitude_std_bin_index'] = np.digitize(df['normalized_amplitude_std'], ampltiude_std_bins[:-1]) - 1
    df['normalized_error'] = np.digitize(df['normalized_error'], ampltiude_std_bins[:-1]) - 1
    

    # Filter data points based on whether their corresponding bin is below the threshold
    filtered_df = df.apply(lambda row: bins_below_threshold[row['amplitude_bin_index'], row['dispersion_xyz_bin_index'], row['amplitude_std_bin_index'],row['normalized_error']], axis=1)
    
    return df[filtered_df]

df_decoded = preprocess_data(df_decoded)  # Apply preprocessing
target_misid_rate = 0.05  # Define your target misidentification rate
optimal_threshold, ampltiude_bins, dispersion_xyz_bins, ampltiude_std_bins, error_bins = calculate_adaptive_threshold(df_decoded, nB, nC, target_misid_rate)
print(f"\nOptimal Threshold: {optimal_threshold}")

# Applying the filtering function
filtered_df = filter_dataframe_using_threshold(df_decoded, optimal_threshold, ampltiude_bins, dispersion_xyz_bins, ampltiude_std_bins, error_bins)

uniq_decoded_species, spec_counts = np.unique(filtered_df['species'], return_counts=True)
# Initialize sums
sum_blank = 0
sum_non_blank = 0

# Iterate through the genes list and sum occurrences
for gene, occurrence in zip(uniq_decoded_species, spec_counts):
    if "Blank" in gene:
        sum_blank += occurrence
    else:
        sum_non_blank += occurrence

print('\nSummary after filtering')
print('------------------------')
print(f"Sum of occurrences for 'Blank': {sum_blank}")
print(f"Sum of occurrences for non-'Blank': {sum_non_blank}")
print(f"Misidentification rate': {np.round((sum_blank/nB)/(sum_non_blank/nC),3)}")

decoded_coding_filtered = df_decoded[~df_decoded['species'].str.startswith('Blank')].copy()
decoded_species = decoded_coding_filtered['species']
decoded_coords = decoded_coding_filtered[['z', 'y', 'x']]
decoded_pixels = decoded_coding_filtered.copy()
decoded_pixels['z'] = (decoded_pixels['z'] / physical_spacing[0]).round().astype(int)
decoded_pixels['y'] = (decoded_pixels['y'] / physical_spacing[1]).round().astype(int)
decoded_pixels['x'] = (decoded_pixels['x'] / physical_spacing[2]).round().astype(int)


decoded_pixels_to_display = decoded_pixels[['z', 'y', 'x']]

uniq_decoded_species, spec_counts = np.unique(decoded_coding_filtered['species'], return_counts=True)

import matplotlib as mpl
import napari
cmap = mpl.colormaps['tab20']
colors_dic = {spec: list(cmap(i%cmap.N)) for i, spec in enumerate(decoded_species)}

colors_decoded = np.zeros((len(decoded_pixels), 4))
for spec in uniq_decoded_species:
    # colors for decoded spots
    col = colors_dic[spec]
    select = np.array([x == spec for x in decoded_species])
    colors_decoded[select, :] = col
    
   
polyDT_current_tile_path = polyDT_dir_path / Path(tile_ids[0]) / Path("round000.zarr")
polyDT_current_tile = zarr.open(polyDT_current_tile_path)
im_data = np.asarray(polyDT_current_tile['raw_data'],dtype=np.uint16)

viewer = napari.Viewer()
viewer.add_image(im_data,scale=physical_spacing)
viewer.add_points(
    decoded_pixels_to_display, 
    name='decoded mRNA',
    size=10,
    face_color=colors_decoded, 
    edge_color='transparent', 
    visible=True,
    scale=physical_spacing,
    )

napari.run()