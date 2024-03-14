import numpy as np
from pathlib import Path
import pandas as pd
import zarr
import re
import wf_merfish.postprocess._decode as decode
from tifffile import imread
import gc
import matplotlib.pyplot as plt

#data_dir_path = Path('/home/qi2lab/Documents/github/wf-merfish/examples/simulated_images/cylinder/images/jitter-0_shift_amp-0_prop_fn-0_prop_fp-0/processed')
data_dir_path = Path('/mnt/opm3/20240214_MouseBrain_UA_NewRO_RK/processed_v2')

calibration_dir_path = data_dir_path / Path("calibrations.zarr")
polyDT_dir_path = data_dir_path / Path('polyDT')
readout_dir_path = data_dir_path / Path('readouts')
localization_dir_path = data_dir_path / Path('localizations')
tile_ids = sorted([entry.name for entry in readout_dir_path.iterdir() if entry.is_dir()],
                  key=lambda x: int(x.split('tile')[1].split('.zarr')[0]))
tile_dir_path = readout_dir_path / Path(tile_ids[0])
bit_ids = sorted([entry.name for entry in tile_dir_path.iterdir() if entry.is_dir()],
                 key=lambda x: int(x.split('bit')[1].split('.zarr')[0]))

try:
    mask_path = polyDT_dir_path / Path(tile_ids[0]) / Path("round000_mask.tif")
    binary_mask = imread(mask_path)
    binary_mask[binary_mask>0]=1
    binary_mask[0:10,:] = 0
except:
    binary_mask = None  
    
calibration_zarr = zarr.open(calibration_dir_path,mode='a')
# calibration_zarr.attrs['codebook'] = codebook_df.values.tolist()
df_codebook = pd.DataFrame(calibration_zarr.attrs['codebook'])
codebook = {}

physical_spacing=[.31,.088,.088]

# Iterate through each row in the DataFrame
for index, row in df_codebook.iterrows():
    # Use the entry in column 0 as the key
    key = row[0]
    
    # Concatenate the values of columns 1 to 16 (inclusive) as a string
    # Here, we use .astype(str) to ensure all values are treated as strings before concatenation
    value = ''.join(row[1:17].astype(str))
    
    # Add the key-value pair to the dictionary
    codebook[key] = value
   
candidates_file_id = 'localization_candidates_localization_tile_coords.parquet'
spot_file_id = 'localized_spots_localization_tile_coords.parquet'
fit_vars_file_id = 'fitted_variables_localization_tile_coords.parquet'

df_candidates = []
df_spots = []
df_variables = [] 

for bit_id in bit_ids:
    
    registered_candidates_tile_bit_path = localization_dir_path / Path(tile_ids[0]) / Path(bit_id).stem / Path(candidates_file_id)
    df_candidate = pd.read_parquet(registered_candidates_tile_bit_path)
    
    registered_localization_tile_bit_path = localization_dir_path / Path(tile_ids[0]) / Path(bit_id).stem / Path(spot_file_id)
    df_localization = pd.read_parquet(registered_localization_tile_bit_path)

    
    s = str(Path(bit_id).stem)
    numbers = re.findall(r'\d+', s)
    if numbers:  # Check if the list is not empty
        extracted_bit = int(numbers[0])
        
    df_localization['bit'] = extracted_bit - 1
    df_spots.append(df_localization)
    
    df_candidate['bit'] = extracted_bit - 1
    df_candidates.append(df_candidate)
    
    registered_fitted_variables_tile_bit_path = localization_dir_path / Path(tile_ids[0]) / Path(bit_id).stem / Path(fit_vars_file_id)
    df_variables.append(pd.read_parquet(registered_fitted_variables_tile_bit_path))
   
df_all_candidates= pd.concat(df_candidates, ignore_index=True)
df_all_localizations = pd.concat(df_spots, ignore_index=True)
df_all_variables = pd.concat(df_variables,ignore_index=True)

df_filtered_localizations = df_all_localizations[df_all_localizations['select']].copy()
df_filtered_variables = df_all_variables.loc[df_filtered_localizations.index].copy()
df_filtered_candidates = df_all_candidates.loc[df_filtered_localizations.index].copy()
df_filtered_localizations.replace([np.inf, -np.inf], np.nan, inplace=True)
df_filtered_localizations.dropna(inplace=True)
df_filtered_variables = df_filtered_variables.loc[df_filtered_localizations.index].copy()
df_filtered_candidates = df_filtered_candidates.loc[df_filtered_candidates.index].copy()

if binary_mask is not None:
    df_filtered_candidates['pixel_z'] = (df_filtered_candidates['z'] / physical_spacing[0]).round().astype(int).clip(0, binary_mask.shape[0] - 1)
    df_filtered_candidates['pixel_y'] = (df_filtered_candidates['y'] / physical_spacing[1]).round().astype(int).clip(0, binary_mask.shape[1] - 1)
    df_filtered_candidates['pixel_x'] = (df_filtered_candidates['x'] / physical_spacing[2]).round().astype(int).clip(0, binary_mask.shape[2] - 1)

    df_filtered_localizations['pixel_z'] = (df_filtered_localizations['z'] / physical_spacing[0]).round().astype(int).clip(0, binary_mask.shape[0] - 1)
    df_filtered_localizations['pixel_y'] = (df_filtered_localizations['y'] / physical_spacing[1]).round().astype(int).clip(0, binary_mask.shape[1] - 1)
    df_filtered_localizations['pixel_x'] = (df_filtered_localizations['x'] / physical_spacing[2]).round().astype(int).clip(0, binary_mask.shape[2] - 1)

    mask_values = binary_mask[df_filtered_localizations['pixel_z'], df_filtered_localizations['pixel_y'], df_filtered_localizations['pixel_x']]
    df_localizations_within_mask = df_filtered_localizations[mask_values == 1].copy()
    df_variables_within_mask = df_filtered_variables[mask_values == 1].copy()
    df_localizations_within_mask['spot_id'] = range(len(df_localizations_within_mask))
    
    df_candidates_within_mask = df_filtered_candidates[mask_values == 1].copy()
    df_candidates_within_mask['spot_id'] = range(len(df_candidates_within_mask))
    del mask_values
else:
    df_filtered_candidates['pixel_z'] = (df_filtered_candidates['z'] / physical_spacing[0]).round().astype(int)
    df_filtered_candidates['pixel_y'] = (df_filtered_candidates['y'] / physical_spacing[1]).round().astype(int)
    df_filtered_candidates['pixel_x'] = (df_filtered_candidates['x'] / physical_spacing[2]).round().astype(int)

    df_filtered_localizations['pixel_z'] = (df_filtered_localizations['z'] / physical_spacing[0]).round().astype(int)
    df_filtered_localizations['pixel_y'] = (df_filtered_localizations['y'] / physical_spacing[1]).round().astype(int)
    df_filtered_localizations['pixel_x'] = (df_filtered_localizations['x'] / physical_spacing[2]).round().astype(int)
    
    df_localizations_within_mask = df_filtered_localizations.copy()
    df_variables_within_mask = df_filtered_variables.copy()
    df_candidates_within_mask= df_filtered_candidates.copy()
    df_localizations_within_mask['spot_id'] = range(len(df_localizations_within_mask))
    df_candidates_within_mask['spot_id'] = range(len(df_candidates_within_mask))

del df_localization, df_spots, df_variables, df_all_localizations, df_all_variables, df_filtered_localizations, df_filtered_variables

gc.collect()

# Step 1: Find the minimum and maximum amplitude for each bit
min_amplitude_per_bit = df_candidates_within_mask.groupby('bit')['amplitude'].transform('min')
max_amplitude_per_bit = df_candidates_within_mask.groupby('bit')['amplitude'].transform('max')

# Step 2: Rescale the amplitude for each row from 0 to 1
df_candidates_within_mask['rescaled_amplitude'] = (df_candidates_within_mask['amplitude'] - min_amplitude_per_bit) / (max_amplitude_per_bit - min_amplitude_per_bit)


coords = df_candidates_within_mask[['z', 'y', 'x']].to_numpy()
fit_vars = df_candidates_within_mask[['amplitude']].to_numpy().reshape(-1, 1)
spot_ids = df_candidates_within_mask['spot_id'].to_numpy()
spot_rounds = df_candidates_within_mask['bit'].to_numpy()

# z dispersion, x/y dispersion, mean amplitude, std amplitude, sequence error, selection size
radius_coef = 1.0
# (z dispersion,) x/y dispersion, mean amplitude, std amplitude, sequence error, selection size
# weights = np.array([1, 1, 1, 1, 0])
weights = np.array([10, 10, 1, 20, 1, 0])
min_spot_sep_xy = 0.250
min_spot_sep_z = .900
# min_spot_sep = np.array(localization_params[condi_name]['min_spot_sep'])
dist_params = [min_spot_sep_z * radius_coef, min_spot_sep_xy * radius_coef]

optim_results = decode.optimize_spots(
    coords=coords, 
    fit_vars=fit_vars, 
    spot_ids=spot_ids, 
    spot_rounds=spot_rounds, 
    dist_params=dist_params, 
    codebook=codebook,
    weights=weights,
    err_corr_dist=0,
    max_positive_bits=16,
    max_bcd_per_spot=None,
    history=True, 
    return_extra=True,
    return_contribs=True,
    rescale_used_spots=True,  # no iterations
    trim_network=False,
    # propose_method='iter_bcd',
    propose_method='single_step',
    verbose=2,
    )


decoded = optim_results['stats']
select = ~decoded['species'].isna()
decoded = decoded.loc[select, :]
decoded_species = decoded.loc[:, 'species']

# Convert the dictionary to a DataFrame
df_results = pd.DataFrame(decoded)
decode_results_path = localization_dir_path / Path(tile_ids[0]) / Path("decoded_nomask.parquet")
df_results.to_parquet(decode_results_path)

decoded_species = decoded['species']
decoded_coords = decoded[['z', 'y', 'x']]
decoded_pixels = decoded_coords.copy()
decoded_pixels['z'] = (decoded['z'] / physical_spacing[0]).round().astype(int)
decoded_pixels['y'] = (decoded['y'] / physical_spacing[1]).round().astype(int)
decoded_pixels['x'] = (decoded['x'] / physical_spacing[2]).round().astype(int)

uniq_decoded_species, spec_counts = np.unique(decoded['species'], return_counts=True)
# Initialize sums
sum_blank = 0
sum_non_blank = 0

# Iterate through the genes list and sum occurrences
for gene, occurrence in zip(uniq_decoded_species, spec_counts):
    print(gene,occurrence)
    if "Blank" in gene:
        sum_blank += occurrence
    else:
        sum_non_blank += occurrence

print(f"Sum of occurrences for 'Blank': {sum_blank}")
print(f"Sum of occurrences for non-'Blank': {sum_non_blank}")
print(f"Misidentification rate': {np.round((sum_blank/21.)/(sum_non_blank/122.),3)}")

