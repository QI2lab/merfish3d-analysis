import numpy as np
import pandas as pd
from scipy.optimize import newton, brentq
from pathlib import Path
import zarr
from typing import Union, Optional
from numpy.typing import NDArray


class BarcodeFilter():
    def __init__(self,
                 data_dir_path: Union[Path,str],
                 target_misid_rate = 0.05,
                 data_type: str = 'csv'):
        
        
        self._data_dir_path = data_dir_path
        self._target_misid_rate = target_misid_rate
        
        self._load_codebook()
        self._load_decoded_features(data_type)
        
    def _load_codebook(self):

        calibration_dir_path = self._data_dir_path / Path('calibrations.zarr')
        self._calibration_zarr = zarr.open(calibration_dir_path,mode='r')
        self._df_codebook = pd.DataFrame(self._calibration_zarr.attrs['codebook'])
        self._df_codebook.fillna(0, inplace=True)
        
        self._codebook_matrix = self._df_codebook.iloc[:, 1:17].to_numpy().astype(int)
        self._gene_ids = self._df_codebook.iloc[:, 0].tolist()
        
        self._num_blank_codebook = 0
        self._num_coding_codebook = 0
        for gene in self._gene_ids:
            if "Blank" in gene:
                self._num_blank_codebook += 1
            else:
                self._num_coding_codebook += 1
        
    def _load_decoded_features(self,
                               data_type: str ='csv'):
        
        decoded_dir_path = self._data_dir_path / Path('decoded')
        decoded = []
        if data_type == 'csv':
            decoded_files = decoded_dir_path.glob('*.csv')
            for decoded_file in decoded_files:
                decoded.append(pd.read_csv(decoded_file))
        else:
            decoded_files = decoded_dir_path.glob('*.parquet')
            for decoded_file in decoded_files:
                decoded.append(pd.read_parquet(decoded_file))
                
        self._decoded_features_df = pd.concat(decoded, ignore_index=True)
        
    def calculate_misidentification_rate(self,
                                          target: str = 'raw',
                                          verbose: int = 0):
        
        if target == 'raw':
            uniq_decoded_species, spec_counts = np.unique(self._decoded_features_df['gene_id'], return_counts=True)
        elif target == 'filtered':
            uniq_decoded_species, spec_counts = np.unique(self._filtered_features_df['gene_id'], return_counts=True)
        # Initialize sums
        sum_blank = 0
        sum_non_blank = 0

        # Iterate through the genes list and sum occurrences
        for gene, occurrence in zip(uniq_decoded_species, spec_counts):
            if "Blank" in gene:
                sum_blank += occurrence
            else:
                sum_non_blank += occurrence
                
        self._misident_rate = np.round((sum_blank/self._num_blank_codebook)/(sum_non_blank/self._num_coding_codebook),3)
        
        if verbose == 1:
            print(f"Sum of occurrences for 'Blank': {sum_blank}")
            print(f"Sum of occurrences for non-'Blank': {sum_non_blank}")
            print(f"Misidentification rate': {self._misident_rate}")
        
    def _preprocess_data(self):

        self._decoded_features_df['log10_mean_intensity'] = np.log10(self._decoded_features_df['mean_intensity'])
        self._decoded_features_df['normalized_min_dispersion'] = (self._decoded_features_df['min_dispersion'] - self._decoded_features_df['min_dispersion'].min()) / (self._decoded_features_df['min_dispersion'].max() - self._decoded_features_df['min_dispersion'].min())
        self._decoded_features_df['normalized_l2_mean_distance'] = (self._decoded_features_df['mean_distance_l2'] - self._decoded_features_df['mean_distance_l2'].min()) / (self._decoded_features_df['mean_distance_l2'].max() - self._decoded_features_df['mean_distance_l2'].min())
        self._decoded_features_df['normalized_cosine_mean_distance'] = (self._decoded_features_df['mean_distance_cosine'] - self._decoded_features_df['mean_distance_cosine'].min()) / (self._decoded_features_df['mean_distance_cosine'].max() - self._decoded_features_df['mean_distance_cosine'].min())
        self._decoded_features_df['normalized_area'] = (self._decoded_features_df['area'] - self._decoded_features_df['area'].min()) / (self._decoded_features_df['area'].max() - self._decoded_features_df['area'].min())

    def _calculate_adaptive_threshold(self,
                                      tolerance=0.001):
        """
        Calculates the adaptive threshold to achieve the target misidentification rate.
        """
        # Assuming 'species' column categorizes barcodes into coding and blank
        coding_barcodes = self._decoded_features_df[~self._decoded_features_df['gene_id'].str.startswith('Blank')]
        blank_barcodes = self._decoded_features_df[self._decoded_features_df['gene_id'].str.startswith('Blank')]
        
        # Define bin edges based on preprocessed data
        self._intensity_bins = np.linspace(self._decoded_features_df['log10_mean_intensity'].min(), self._decoded_features_df['log10_mean_intensity'].max(), num=21)
        self._dispersion_bins = np.linspace(self._decoded_features_df['normalized_min_dispersion'].min(), self._decoded_features_df['normalized_min_dispersion'].max(), num=21)
        self._l2_distance_bins = np.linspace(self._decoded_features_df['normalized_l2_mean_distance'].min(), self._decoded_features_df['normalized_l2_mean_distance'].max(), num=21)
        self._cosine_distance_bins = np.linspace(self._decoded_features_df['normalized_cosine_mean_distance'].min(), self._decoded_features_df['normalized_cosine_mean_distance'].max(), num=21)
        self._area_bins = np.linspace(self._decoded_features_df['normalized_area'].min(), self._decoded_features_df['normalized_area'].max(), num=21)
        
        # Calculate 3D histograms for coding and blank barcodes
        coding_hist, _ = np.histogramdd(coding_barcodes[['log10_mean_intensity', 'normalized_min_dispersion', 'normalized_l2_mean_distance', 'normalized_cosine_mean_distance', 'normalized_area']].values, bins=(self._intensity_bins , self._dispersion_bins, self._l2_distance_bins, self._cosine_distance_bins, self._area_bins))
        blank_hist, _ = np.histogramdd(blank_barcodes[['log10_mean_intensity', 'normalized_min_dispersion', 'normalized_l2_mean_distance', 'normalized_cosine_mean_distance', 'normalized_area']].values, bins=(self._intensity_bins , self._dispersion_bins, self._l2_distance_bins, self._cosine_distance_bins, self._area_bins))

        blank_fraction = np.divide(blank_hist, coding_hist, out=np.ones_like(blank_hist)*np.finfo(np.float32).max, where=coding_hist!=0)
        blank_fraction = (blank_fraction / self._num_blank_codebook) / (self._num_blank_codebook + self._num_coding_codebook)
        
        def misid_rate(threshold):
            selected_bins = blank_fraction < threshold
            selected_coding = np.sum(coding_hist[selected_bins])
            selected_blank = np.sum(blank_hist[selected_bins])
            return (selected_blank / self._num_blank_codebook) / (selected_coding / self._num_coding_codebook) - self._target_misid_rate
        
        self._optimal_threshold = np.round(newton(misid_rate, x0=1e-5, x1=1e-3, tol=tolerance),5)
        
    def _filter_dataframe_using_threshold(self):
        """
        Filters the DataFrame based on the optimal threshold determined from the blank fraction histogram.
        """
        
        # Calculate the blank fraction histogram as before
        total_hist, _ = np.histogramdd(self._decoded_features_df[~self._decoded_features_df['gene_id'].str.startswith('Blank')][['log10_mean_intensity', 'normalized_min_dispersion', 'normalized_l2_mean_distance', 'normalized_cosine_mean_distance',  'normalized_area']].values, bins=(self._intensity_bins , self._dispersion_bins, self._l2_distance_bins, self._cosine_distance_bins, self._area_bins))
        blank_hist, _ = np.histogramdd(self._decoded_features_df[self._decoded_features_df['gene_id'].str.startswith('Blank')][['log10_mean_intensity', 'normalized_min_dispersion', 'normalized_l2_mean_distance', 'normalized_cosine_mean_distance', 'normalized_area']].values, bins=(self._intensity_bins , self._dispersion_bins, self._l2_distance_bins, self._cosine_distance_bins, self._area_bins))
        blank_fraction = np.divide(blank_hist, total_hist, out=np.zeros_like(blank_hist), where=total_hist!=0)
        
        # Identify bins below the threshold
        bins_below_threshold = blank_fraction < self._optimal_threshold
        
        # Map each data point to its corresponding bin index
        self._decoded_features_df['intensity_bin_index'] = np.digitize(self._decoded_features_df['log10_mean_intensity'], self._intensity_bins[:-1]) - 1
        self._decoded_features_df['dispersion_bin_index'] = np.digitize(self._decoded_features_df['normalized_min_dispersion'], self._dispersion_bins[:-1]) - 1
        self._decoded_features_df['l2_distance_bin_index'] = np.digitize(self._decoded_features_df['normalized_l2_mean_distance'], self._dispersion_bins[:-1]) - 1
        self._decoded_features_df['cosine_distance_bin_index'] = np.digitize(self._decoded_features_df['normalized_cosine_mean_distance'], self._dispersion_bins[:-1]) - 1
        self._decoded_features_df['area_bin_index'] = np.digitize(self._decoded_features_df['normalized_area'], self._area_bins[:-1]) - 1
        

        # Filter data points based on whether their corresponding bin is below the threshold
        selection = self._decoded_features_df.apply(lambda row: bins_below_threshold[row['intensity_bin_index'], row['distance_bin_index'], row['l2_distance_bin_index'], row['cosine_distance_bin_index'], row['area_bin_index']], axis=1)
        self._filtered_features_df = self._decoded_features_df[selection]