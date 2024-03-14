"""
PixelDecoder: Perform pixel-based decoding for qi2lab widefield MERFISH data using GPU acceleration.

Shepherd 2024/03 - rework of GPU logic to reduce out-of-memory crashes
Shepherd 2024/01 - updates for qi2lab MERFISH file format v1.0
"""

import numpy as np
from pathlib import Path
import zarr
import gc
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter
from cucim.skimage.measure import label
from cucim.skimage.morphology import remove_small_objects
from cupyx.scipy.spatial.distance import cdist
from skimage.measure import regionprops_table
from typing import Union, Optional, Sequence, Tuple
import pandas as pd
from random import sample
from tqdm import tqdm

class PixelDecoder():
    def __init__(self,
                 dataset_path: Union[Path,str],
                 tile_idx: int = 0,
                 scale_factors: Optional[np.ndarray] = None,
                 global_normalization_limits: Optional[Sequence[float]] = None,
                 overwrite_normalization: bool = False,
                 verbose: int = 1):
        """
        Retrieve and process one tile from qi2lab 3D widefield zarr structure.
        Normalize codebook and data, perform plane-by-plane pixel decoding,
        extract barcode features, and save to disk.

        Parameters
        ----------
        dataset_path : Union[str, Path]
            Path to Zarr dataset
        tile_idx : int
            tile index to retrieve
        scale_factors: Optional[np.ndarray] = None
            optional user supplied scale factors. nbits x 2 array, with the
            first dimension being value to subtract for background and second
            being normalization by division factor
        global_normalization_limits: Optional[Sequence[float]] = None
            rescales per bit data to [0,1], using [lower_normalization,upper_normalization] limits
        overwrite_normalization: bool = False
            if true, will recalculate and save new per bit normalization factors 
        verbose: int = 1
            control verbosity. 0 - no output, 1 - tqdm bars, 2 - diagnostic outputs
        """

        self._dataset_path = dataset_path
        self._tile_idx = tile_idx
        self._verbose = verbose
        self._barcodes_filtered = False
        
        self._parse_dataset()
        self._load_experiment_parameters()
        self._load_codebook()
        self._decoding_matrix_no_errors = self._normalize_codebook(include_errors=False)
        self._decoding_matrix = self._decoding_matrix_no_errors.copy()
        self._barcode_count = self._decoding_matrix.shape[0]
        self._bit_count = self._decoding_matrix.shape[1]
        
        if scale_factors is not None:
            self._background_vector = scale_factors[:,0]
            self._normalization_vector = scale_factors[:,1]
        else:
            if global_normalization_limits is not None:
                self._global_normalization_factors(low_percentile_cut=global_normalization_limits[0],
                                                   high_percentile_cut=global_normalization_limits[1],
                                                   overwrite=overwrite_normalization)
            else:
                self._global_normalization_factors(overwrite=overwrite_normalization)
               
        self._codebook_style = 1        
        self._filter_type = None

    def _parse_dataset(self):
        self._readout_dir_path = self._dataset_path / Path('readouts')
        self._tile_ids = sorted([entry.name for entry in self._readout_dir_path.iterdir() if entry.is_dir()],
                           key=lambda x: int(x.split('tile')[1].split('.zarr')[0]))
        tile_dir_path = self._readout_dir_path / Path(self._tile_ids[self._tile_idx])
        self._bit_ids = sorted([entry.name for entry in tile_dir_path.iterdir() if entry.is_dir()],
                                key=lambda x: int(x.split('bit')[1].split('.zarr')[0]))
        
    def _load_experiment_parameters(self):
        
        calibration_dir_path = self._dataset_path / Path("calibrations.zarr")
        self._calibration_zarr = zarr.open(calibration_dir_path,mode='a')
        
        # TO DO: fix in calibration.zarr creation
        self._na = 1.35
        self._ri = 1.4
        self._num_on_bits = 4
    
    def _load_codebook(self):

        self._df_codebook = pd.DataFrame(self._calibration_zarr.attrs['codebook'])
        self._df_codebook.fillna(0, inplace=True)
        
        self._codebook_matrix = self._df_codebook.iloc[:, 1:].to_numpy().astype(int)
        self._gene_ids = self._df_codebook.iloc[:, 0].tolist()
        
    def _normalize_codebook(self, include_errors: bool = False):
        self._barcode_set = cp.asarray(self._codebook_matrix)
        magnitudes = cp.linalg.norm(self._barcode_set, axis=0, keepdims=True)

        if not include_errors:
            # Normalize directly using broadcasting
            normalized_barcodes = self._barcode_set / magnitudes
            return cp.asnumpy(normalized_barcodes)
        else:
            # Pre-compute the normalized barcodes
            normalized_barcodes = self._barcode_set / magnitudes

            # Initialize an empty list to hold all barcodes with single errors
            barcodes_with_single_errors = [normalized_barcodes]

            # Generate single-bit errors
            for bit_index in range(self._barcode_set.shape[1]):
                flipped_barcodes = self._barcode_set.copy()
                flipped_barcodes[:, bit_index] = 1 - flipped_barcodes[:, bit_index]
                flipped_magnitudes = cp.sqrt(cp.sum(flipped_barcodes**2, axis=1))
                flipped_magnitudes = cp.where(flipped_magnitudes == 0, 1, flipped_magnitudes)
                normalized_flipped = flipped_barcodes / flipped_magnitudes
                barcodes_with_single_errors.append(normalized_flipped)

            # Stack all barcodes (original normalized + with single errors)
            all_barcodes = cp.vstack(barcodes_with_single_errors)
            return cp.asnumpy(all_barcodes)
         
    def _global_normalization_factors(self,
                                      low_percentile_cut: float = 20.0,
                                      high_percentile_cut: float = 99.995,
                                      camera_background: int = 0,
                                      hot_pixel_threshold: int = 65000,
                                      overwrite: bool = False):
    
        try:
            normalization_vector = cp.asarray(self._calibration_zarr.attrs['global_normalization'])
            background_vector = cp.asarray(self._calibration_zarr.attrs['global_background'])
            data_not_found = False
        except:         
            data_not_found = True
            
        if data_not_found or overwrite:
            
            if len(self._tile_ids) > 7:
                random_tiles = sample(self._tile_ids,5)
            else:
                random_tiles = self._tile_ids
                
            normalization_vector = cp.ones(len(self._bit_ids),dtype=cp.float32)
            background_vector = cp.zeros(len(self._bit_ids),dtype=cp.float32)
            if self._verbose > 1:
                print('calculate normalizations')
            elif self._verbose >= 1:
                iterable_bits = enumerate(tqdm(self._bit_ids,desc='bit',leave=False))
            else:
                iterable_bits = enumerate(self._bit_ids)
            
            for bit_idx, bit_id in iterable_bits:
                all_images = []
                
                if self._verbose >= 1:
                    iterable_tiles = tqdm(random_tiles,desc='loading tiles',leave=False)
                else:
                    iterable_tiles = random_tiles
                
                for tile_id in iterable_tiles:
                    tile_dir_path = self._readout_dir_path / Path(tile_id)
                    bit_dir_path = tile_dir_path / Path(bit_id)
                    current_bit = zarr.open(bit_dir_path,mode='r')
                    current_image = cp.asarray(current_bit["registered_dog_data"], dtype=cp.uint16)
                    current_image[current_image<camera_background] = cp.median(current_image[current_image.shape[0]//2,:,:]).astype(cp.uint16)
                    current_image[current_image>hot_pixel_threshold] = cp.median(current_image[current_image.shape[0]//2,:,:]).astype(cp.uint16)
                    all_images.append(cp.asnumpy(current_image).astype(np.uint16))
                    del current_image
                    cp.get_default_memory_pool().free_all_blocks()
                    gc.collect()
                    
                all_images = np.array(all_images)

                if self._verbose >= 1:
                    iterable_tiles = enumerate(tqdm(random_tiles,desc='background est.',leave=False))
                else:
                    iterable_tiles = random_tiles

                low_pixels = []    
                for tile_idx, tile_id in iterable_tiles:
                    
                    current_image = cp.asarray(all_images[tile_idx,:],dtype=cp.float32)
                    low_cutoff = cp.percentile(current_image, low_percentile_cut)
                    low_pixels.append(current_image[current_image < low_cutoff].flatten().astype(cp.float32))                                        
                    del current_image
                    cp.get_default_memory_pool().free_all_blocks()
                    gc.collect()

                low_pixels = cp.concatenate(low_pixels,axis=0)
                if low_pixels.shape[0]>0:
                    background_vector[bit_idx] = cp.median(low_pixels)
                else:
                    background_vector[bit_idx] = 0
                
                del low_pixels
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()
                
                if self._verbose >= 1:
                    iterable_tiles = enumerate(tqdm(random_tiles,desc='normalization est.',leave=False))
                else:
                    iterable_tiles = random_tiles
                
                high_pixels = []
                for tile_idx, tile_id in iterable_tiles:
                    
                    current_image = cp.asarray(all_images[tile_idx,:],dtype=cp.float32) - background_vector[bit_idx]
                    current_image[current_image<0] = 0
                    high_cutoff = cp.percentile(current_image, high_percentile_cut)
                    high_pixels.append(current_image[current_image > high_cutoff].flatten().astype(cp.float32))
                    
                    del current_image
                    cp.get_default_memory_pool().free_all_blocks()
                    gc.collect()
                    
                high_pixels = cp.concatenate(high_pixels,axis=0)
                if high_pixels.shape[0]>0:
                    normalization_vector[bit_idx] = cp.median(high_pixels)
                else:
                    normalization_vector[bit_idx] = 1
                    
                del high_pixels
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()
                                                   
            self._calibration_zarr.attrs['global_normalization'] = cp.asnumpy(normalization_vector).astype(np.float32).tolist()
            self._calibration_zarr.attrs['global_background'] = cp.asnumpy(background_vector).astype(np.float32).tolist()
                      
        self._background_vector = background_vector
        self._normalization_vector = normalization_vector
        
    def _load_bit_data(self):   
        if self._verbose > 1:    
            print('load raw data')
            iterable_bits = tqdm(self._bit_ids,desc='bit',leave=False)
        elif self._verbose >= 1:
            iterable_bits = tqdm(self._bit_ids,desc='loading',leave=False)
        else:
            iterable_bits = self._bit_ids
        
        images = []
        self._em_wvl = []
        for bit_id in iterable_bits:
            tile_dir_path = self._readout_dir_path / Path(self._tile_ids[self._tile_idx])
            bit_dir_path = tile_dir_path / Path(bit_id)
            current_bit = zarr.open(bit_dir_path,mode='r')
            images.append(np.asarray(current_bit["registered_dog_data"], dtype=np.uint16))
            self._em_wvl.append(current_bit.attrs['emission_um'])
            
        self._image_data = np.stack(images,axis=0)
        voxel_size_zyx_um = np.asarray(current_bit.attrs['voxel_zyx_um'],dtype=np.float32)
        self._pixel_size = voxel_size_zyx_um[1]
        self._axial_step = voxel_size_zyx_um[0]
        
        del images, current_bit
        gc.collect()
               
    def _lp_filter(self,
                   sigma = (2,1,1)):
        
        self._image_data_lp = self._image_data.copy()
    
        if self._verbose > 1:
            print('lowpass filter')
            iterable_lp = tqdm(range(self._image_data_lp.shape[0]),desc='bit',leave=False)
        elif self._verbose >= 1:
            iterable_lp = tqdm(range(self._image_data_lp.shape[0]),desc='lowpass',leave=False)
        else:
            iterable_lp = self._image_data_lp
        
        for i in iterable_lp:
            image_data_cp = cp.asarray(self._image_data[i,:],dtype=cp.float32)
            self._image_data_lp[i,:,:,:] = cp.asnumpy(gaussian_filter(image_data_cp,sigma=sigma)).astype(np.float32)
            
        self._filter_type = 'lp'
        
        del image_data_cp
        del self._image_data
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
                
    @staticmethod
    def _scale_pixel_traces(pixel_traces: Union[np.ndarray,cp.ndarray],
                            background_vector : Union[np.ndarray,cp.ndarray],
                            normalization_vector: Union[np.ndarray,cp.ndarray]) -> cp.ndarray:
        
        if isinstance(pixel_traces, np.ndarray):
            pixel_traces = cp.asarray(pixel_traces,dtype=cp.float32)
        if isinstance(background_vector, np.ndarray):
            background_vector = cp.asarray(background_vector,dtype=cp.float32)
        if isinstance(normalization_vector, np.ndarray):
            normalization_vector = cp.asarray(normalization_vector,dtype=cp.float32)
        
        return (pixel_traces - background_vector[:,cp.newaxis])  / normalization_vector[:,cp.newaxis]
    
    @staticmethod
    def _clip_pixel_traces(pixel_traces: Union[np.ndarray,cp.ndarray],
                            clip_lower: float = 0.0,
                            clip_upper: float = 1.0) -> cp.ndarray:
        
        return cp.clip(pixel_traces,clip_lower,clip_upper,pixel_traces)
    
    @staticmethod
    def _normalize_pixel_traces(pixel_traces: Union[np.ndarray,cp.ndarray]) -> Tuple[cp.ndarray,cp.ndarray]:
  
        if isinstance(pixel_traces, np.ndarray):
            pixel_traces = cp.asarray(pixel_traces,dtype=cp.float32)
        
        norms = cp.linalg.norm(pixel_traces, axis=0)
        norms = cp.where(norms == 0, np.inf, norms)
        normalized_traces = pixel_traces / norms
        norms = cp.where(norms == np.inf, -1, norms)
        
        del pixel_traces
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        
        return normalized_traces, norms
    
    @staticmethod
    def _calculate_distances(pixel_traces: Union[np.ndarray,cp.ndarray],
                             codebook_matrix: Union[np.ndarray,cp.ndarray]) -> Tuple[cp.ndarray,cp.ndarray]:
  
        if isinstance(pixel_traces, np.ndarray):
            pixel_traces = cp.asarray(pixel_traces,dtype=cp.float32)
        if isinstance(codebook_matrix, np.ndarray):
            codebook_matrix = cp.asarray(codebook_matrix,dtype=cp.float32)
                
        distances = cdist(cp.ascontiguousarray(pixel_traces.T),cp.ascontiguousarray(codebook_matrix),metric='euclidean')
        min_indices = cp.argmin(distances, axis=1)
        min_distances = cp.min(distances, axis=1)

        del pixel_traces, codebook_matrix
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        
        return min_distances, min_indices
 
    def _decode_pixels(self,
                       distance_threshold: float = .5172,
                       magnitude_threshold: float = 1.0):
        
        if self._filter_type == 'lp':
            original_shape = self._image_data_lp.shape
            self._decoded_image = np.zeros((original_shape[1:]),dtype=np.int16)
            self._magnitude_image = np.zeros((original_shape[1:]),dtype=np.float16)
            self._scaled_pixel_images = np.zeros((original_shape),dtype=np.float16)
            self._distance_image = np.zeros((original_shape[1:]),dtype=np.float16)
        else:
            original_shape = self._image_data.shape
            self._decoded_image = np.zeros((original_shape[1:]),dtype=np.int16)
            self._magnitude_image = np.zeros((original_shape[1:]),dtype=np.float16)
            self._scaled_pixel_images = np.zeros((original_shape),dtype=np.float16)
            self._distance_image = np.zeros((original_shape[1:]),dtype=np.float16)
        
        if self._verbose > 1:
            print("decode pixels")
            iterable_z = tqdm(range(original_shape[1]),desc="z",leave=False)
        elif self._verbose >= 1:
            iterable_z = tqdm(range(original_shape[1]),desc="decoding",leave=False)
        else:
            iterable_z = range(original_shape[1])
            
        for z_idx in iterable_z:
            if self._filter_type == 'lp':
                z_plane_shape = self._image_data_lp[:,z_idx,:].shape
                scaled_pixel_traces = cp.asarray(self._image_data_lp[:,z_idx,:]).reshape(self._bit_count, -1).astype(cp.float32)
            else:
                z_plane_shape = self._image_data[:,z_idx,:].shape
                scaled_pixel_traces = cp.asarray(self._image_data[:,z_idx,:]).reshape(self._bit_count, -1).astype(cp.float32)
                
            scaled_pixel_traces = self._scale_pixel_traces(scaled_pixel_traces,
                                                    self._background_vector,
                                                    self._normalization_vector)
            scaled_pixel_traces = self._clip_pixel_traces(scaled_pixel_traces)
            normalized_pixel_traces, pixel_magnitude_trace = self._normalize_pixel_traces(scaled_pixel_traces)
            distance_trace, codebook_index_trace = self._calculate_distances(normalized_pixel_traces,self._decoding_matrix)
            
            del normalized_pixel_traces
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

            decoded_trace = cp.full((distance_trace.shape[0],), -1, dtype=cp.int16)
            mask_trace = distance_trace < distance_threshold
            decoded_trace[mask_trace] = codebook_index_trace[mask_trace].astype(cp.int16)
            decoded_trace[pixel_magnitude_trace <= magnitude_threshold] = -1
    
            self._decoded_image[z_idx,:] = cp.asnumpy(cp.reshape(cp.round(decoded_trace,3), z_plane_shape[1:]))
            self._magnitude_image[z_idx,:] = cp.asnumpy(cp.reshape(cp.round(pixel_magnitude_trace,3), z_plane_shape[1:]))
            self._scaled_pixel_images[:,z_idx,:] = cp.asnumpy(cp.reshape(cp.round(scaled_pixel_traces,3), z_plane_shape))
            self._distance_image[z_idx,:] = cp.asnumpy(cp.reshape(cp.round(distance_trace,3), z_plane_shape[1:]))
            
            del decoded_trace, pixel_magnitude_trace, scaled_pixel_traces, distance_trace
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
            
    def _extract_barcodes(self, 
                          minimum_pixels: int = 9,
                          maximum_pixels: int = 200):
        
        if self._verbose > 1:
            print('extract barcodes')
        elif self._verbose >= 1:
            iterable_barcode = tqdm(range(self._barcode_count), desc='barcode', leave=False)
        else:
            iterable_barcode = range(self._barcode_count)
        decoded_image = cp.asarray(self._decoded_image, dtype=cp.int16)
        intensity_image = np.concatenate([np.expand_dims(self._distance_image,axis=0),
                                         self._scaled_pixel_images],axis=0).transpose(1,2,3,0)

        for barcode_index in iterable_barcode:
            
            on_bits_indices = np.where(self._codebook_matrix[barcode_index])[0]
            
            if self._verbose > 1:
                print('')
                print('label image')
            labeled_image = label(decoded_image == barcode_index, 
                                  connectivity=3)
            
            if self._verbose > 1:
                    print('remove large')
            pixel_counts = cp.bincount(labeled_image.ravel())
            large_labels = cp.where(pixel_counts > maximum_pixels)[0]
            large_label_mask = cp.zeros_like(labeled_image, dtype=bool)
            large_label_mask = cp.isin(labeled_image, large_labels)
            labeled_image[large_label_mask] = 0

            if self._verbose > 1:
                print('remove small')
            labeled_image = remove_small_objects(labeled_image,
                                                min_size=minimum_pixels)
            if self._verbose > 1:
                print('regionprops table')
            props = regionprops_table(cp.asnumpy(labeled_image).astype(np.int32),
                                      intensity_image=intensity_image,
                                      properties=['label',
                                                  'area',
                                                  'centroid',
                                                  'intensity_mean',
                                                  'intensity_max',
                                                  'moments_normalized',
                                                  'inertia_tensor_eigvals'])
            
            del labeled_image
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        
            df_barcode = pd.DataFrame(props)

            df_barcode['on_bit_1'] = on_bits_indices[0] + 1
            df_barcode['on_bit_2'] = on_bits_indices[1] + 1
            df_barcode['on_bit_3'] = on_bits_indices[2] + 1
            df_barcode['on_bit_4'] = on_bits_indices[3] + 1
            df_barcode['barcode_id'] = df_barcode.apply(lambda x: (barcode_index + 1),axis=1)
            df_barcode['gene_id'] = df_barcode.apply(lambda x: self._gene_ids[barcode_index], axis=1) 
            df_barcode['tile_idx'] = self._tile_idx
            
            df_barcode.rename(columns={'centroid-0': 'z'}, inplace=True)
            df_barcode.rename(columns={'centroid-1': 'y'}, inplace=True)
            df_barcode.rename(columns={'centroid-2': 'x'}, inplace=True)
            
            df_barcode.rename(columns={'intensity_min-0': 'distance_min'}, inplace=True)
            df_barcode.rename(columns={'intensity_mean-0': 'distance_mean'}, inplace=True)
            df_barcode.rename(columns={'intensity_max-0': 'distance_max'}, inplace=True)
            for i in range(1,self._bit_count+1):
                df_barcode.rename(columns={'intensity_min-'+str(i): 'bit'+str(i).zfill(2)+'_min_intensity'}, inplace=True)
                df_barcode.rename(columns={'intensity_mean-'+str(i): 'bit'+str(i).zfill(2)+'_mean_intensity'}, inplace=True)
                df_barcode.rename(columns={'intensity_max-'+str(i): 'bit'+str(i).zfill(2)+'_max_intensity'}, inplace=True)
   
            on_bits = on_bits_indices+np.ones(4)
                
            signal_mean_columns = [f'bit{int(bit):02d}_mean_intensity' for bit in on_bits]
            bkd_mean_columns = [f'bit{int(bit):02d}_mean_intensity' for bit in range(1, self._bit_count+1) if bit not in on_bits]
            
            signal_max_columns = [f'bit{int(bit):02d}_max_intensity' for bit in on_bits]
            bkd_max_columns = [f'bit{int(bit):02d}_max_intensity' for bit in range(1, self._bit_count+1) if bit not in on_bits]
                            
            df_barcode['signal_mean'] = df_barcode[signal_mean_columns].mean(axis=1)
            df_barcode['signal_max'] = df_barcode[signal_max_columns].mean(axis=1)
            df_barcode['bkd_mean'] = df_barcode[bkd_mean_columns].mean(axis=1)
            df_barcode['bkd_max'] = df_barcode[bkd_max_columns].mean(axis=1)
            df_barcode['s-b_mean'] = df_barcode['signal_mean'] - df_barcode['bkd_mean']
            df_barcode['s-b_max'] = df_barcode['signal_max'] - df_barcode['bkd_max']
       
            #df_barcode = df_barcode.round(3)
            
            del props
            gc.collect()
            
            if self._verbose > 1:
                print('dataframe aggregation')
            if barcode_index == 0:
                self._df_barcodes = df_barcode.copy()
            else:
                self._df_barcodes = pd.concat([self._df_barcodes,df_barcode])
                self._df_barcodes.reset_index(drop=True, inplace=True)
                
            del df_barcode
            gc.collect()
                
        del decoded_image, intensity_image
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        
    def save_barcodes(self,
                      format: str = 'csv'):
        
        if self._verbose > 1:
            print('save barcodes')
        
        readout_dir_path = self._dataset_path / Path('readouts')
        tile_ids = [entry.name for entry in readout_dir_path.iterdir() if entry.is_dir()]
        
        decoded_dir_path = self._dataset_path / Path('decoded')
        decoded_dir_path.mkdir(exist_ok=True)
        
        if not(self._barcodes_filtered):
        
            barcode_path = decoded_dir_path / Path(tile_ids[self._tile_idx]+'_decoded_features.' + format)
            self._barcode_path = barcode_path
            if format == 'csv':
                self._df_barcodes.to_csv(barcode_path)
            else:
                self._df_barcodes.to_parquet(barcode_path)
                
        else:
            
            barcode_path = decoded_dir_path / Path('all_tiles_filtered_decoded_features.' + format)
            self._barcode_path = barcode_path
            if format == 'csv':
                self._df_filtered_barcodes.to_csv(barcode_path)
            else:
                self._df_filtered_barcodes.to_parquet(barcode_path)
                
    def load_all_barcodes(self,
                          format: str = 'csv'):
        
        decoded_dir_path = self._dataset_path / Path('decoded')
        tile_files = decoded_dir_path.glob('*.'+format)
        tile_files = sorted(tile_files, key=lambda x: x.name)
        
        if self._verbose >=1:
            iterable_csv = tqdm(tile_files)
        else:
            iterable_csv = tile_files
                     
        tile_data = [pd.read_csv(csv_file) for csv_file in iterable_csv]      

        self._df_barcodes_loaded = pd.concat(tile_data)
        
        
    @staticmethod
    def calculate_fdr(df, 
                      threshold,
                      blank_count,
                      barcode_count):

        df['prediction'] = df['predicted_probability'] > threshold

        coding = df[(~df['gene_id'].str.startswith('Blank')) & (df['predicted_probability'] > threshold)].shape[0]
        noncoding = df[(df['gene_id'].str.startswith('Blank')) & (df['predicted_probability'] > threshold)].shape[0]

        fdr = (noncoding/blank_count) / (coding/(barcode_count-blank_count))
        
        return fdr
    
    def filter_all_barcodes(self,
                            fdr_target: float = .15):
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import classification_report
        from imblearn.over_sampling import SMOTE 
        
        self._df_barcodes_loaded['X'] = ~self._df_barcodes_loaded['gene_id'].str.startswith('Blank')
        columns = ['X', 'area', 's-b_mean', 's-b_max', 'distance_mean',
                   'moments_normalized-0-0-2', 'moments_normalized-0-0-3', 'moments_normalized-0-1-1',
                   'moments_normalized-0-1-2', 'moments_normalized-0-1-3', 'moments_normalized-0-2-0',
                   'moments_normalized-0-2-1', 'moments_normalized-0-2-3', 'moments_normalized-0-3-0',
                   'moments_normalized-0-3-1', 'moments_normalized-0-3-2', 'moments_normalized-0-3-3',
                   'inertia_tensor_eigvals-0', 'inertia_tensor_eigvals-1', 'inertia_tensor_eigvals-2']
        df_true = self._df_barcodes_loaded[self._df_barcodes_loaded['X'] == True][columns]
        df_false = self._df_barcodes_loaded[self._df_barcodes_loaded['X'] == False][columns]
        df_true_sampled = df_true.sample(n=len(df_false), random_state=42)
        
        df_combined = pd.concat([df_true_sampled, df_false])
        x = df_combined.drop('X', axis=1)
        y = df_combined['X']
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
        
        if self._verbose > 1:
            print('generating synthetic samples for class balance')
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        if self._verbose > 1:
            print('scaling features')
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_test_scaled = scaler.transform(X_test)
        
        if self._verbose > 1:
            print('training classifier')
        mlp = MLPClassifier(solver='adam', max_iter=10000, random_state=42)
        mlp.fit(X_train_scaled, y_train_resampled)
        predictions = mlp.predict(X_test_scaled)
        
        if self._verbose >1:
            print(classification_report(y_test,predictions))
            
        if self._verbose > 1:
            print('predicting on full data')
        
        full_data_scaled = scaler.transform(self._df_barcodes_loaded[columns[1:]])
        self._df_barcodes_loaded['predicted_probability'] = mlp.predict_proba(full_data_scaled)[:, 1]
        
        if self._verbose > 1:
            print('filtering blanks')
        
        coarse_threshold = 0
        for threshold in np.arange(0, 1, 0.1):  # Coarse step: 0.1
            fdr = self.calculate_fdr(self._df_barcodes_loaded, 
                                     threshold,
                                     22,
                                     self._barcode_count)
            if fdr <= fdr_target:
                coarse_threshold = threshold
                break
            
        fine_threshold = coarse_threshold
        for threshold in np.arange(coarse_threshold-0.1, coarse_threshold+0.1, 0.001):
            fdr = self.calculate_fdr(self._df_barcodes_loaded, 
                                     threshold,
                                     22,
                                     self._barcode_count)
            if fdr <= fdr_target:
                fine_threshold = threshold
                break
            
        df_above_threshold = self._df_barcodes_loaded[self._df_barcodes_loaded['predicted_probability'] > fine_threshold]
        self._df_filtered_barcodes = df_above_threshold[['tile_idx', 'gene_id', 'z', 'y', 'x']].copy()
        self._df_filtered_barcodes['global_z'] = self._df_filtered_barcodes['z']
        self._df_filtered_barcodes['global_y'] = self._df_filtered_barcodes['y']
        self._df_filtered_barcodes['global_x'] = self._df_filtered_barcodes['x']
        self._df_filtered_barcodes['cell_id'] = -1
        self._barcodes_filtered = True
        
        if self._verbose > 1:
            print(f"fdr : {fdr}")
            print(f"retained barcodes: {len(self._df_filtered_barcodes)}")
       
        del df_above_threshold, full_data_scaled
        del mlp, predictions, X_train, X_test, y_test, y_train, X_train_scaled, X_test_scaled
        del df_true, df_false, df_true_sampled, df_combined
        gc.collect()
                                                         
    def cleanup(self):
        
        if self._filter_type == 'lp':
            del self._image_data_lp
        else:
            del self._image_data
            
        del self._scaled_pixel_images, self._decoded_image, self._distance_image, self._magnitude_image
        del self._codebook_matrix, self._df_codebook, self._decoding_matrix
        del self._df_barcodes, self._gene_ids
        
        if self._barcodes_filtered:
            del self._df_filtered_barcodes
        
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    
    def run_decoding(self,
                     lowpass_sigma: Sequence[float] = (2,1,1),
                     distance_threshold: float =.5172,
                     magnitude_threshold: float = 0.4,
                     minimum_pixels: int = 21,
                     skip_extraction: bool = False):

        self._load_bit_data()
        if not(np.any(lowpass_sigma==0)):
            self._lp_filter(sigma=lowpass_sigma)
        self._decode_pixels(distance_threshold = distance_threshold,
                            magnitude_threshold = magnitude_threshold)
        if not(skip_extraction):
            self._extract_barcodes(minimum_pixels)