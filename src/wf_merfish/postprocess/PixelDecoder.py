"""
Perform pixel-based decoding for 2D/3D MERFISH data using GPU acceleration.
"""

import numpy as np
from pathlib import Path
import zarr
import gc
import cupy as cp
from cupyx.scipy.spatial.distance import cdist
from cupyx.scipy.ndimage import gaussian_filter
from cucim.skimage.measure import label, regionprops
from typing import Union, Optional, Sequence, Tuple
import pandas as pd
from random import sample
from tqdm import tqdm

class PixelDecoder():
    def __init__(self,
                 data_dir_path: Union[Path,str],
                 tile_idx: int = 0,
                 scale_factors: Optional[np.ndarray] = None,
                 global_normalization_limits: Optional[Sequence[float]] = None):

        self._dataset_path = data_dir_path
        self._tile_idx = tile_idx
        
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
                                                   overwrite=False)
            else:
                self._global_normalization_factors(overwrite=False)
        
        self._load_bit_data()        
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
        
        self._codebook_matrix = self._df_codebook.iloc[:, 1:17].to_numpy().astype(int)
        self._gene_ids = self._df_codebook.iloc[:, 0].tolist()
        
    def _normalize_codebook(self, include_errors: bool = False):
        self._barcode_set = cp.asarray(self._codebook_matrix)
        magnitudes = cp.sqrt(cp.sum(self._barcode_set**2, axis=0,keepdims=True))

        # Avoid division by zero by ensuring magnitudes are not zero
        magnitudes = cp.where(magnitudes == 0, 1, magnitudes)

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
                                      overwrite: bool = False):
    
        try:
            normalization_vector = cp.asarray(self._calibrations_zarr.attrs['global_normalization'])
            background_vector = cp.asarray(self._calibrations_zarr.attrs['global_background'])
            data_not_found = False
        except:         
            data_not_found = True
            
        if data_not_found or overwrite:
            
            if len(self._tile_ids) > 10:
                random_tiles = sample(self._tile_ids,10)
            else:
                random_tiles = self._tile_ids

            normalization_vector = cp.ones(len(self._bit_ids),dtype=cp.float32)
            background_vector = cp.zeros(len(self._bit_ids),dtype=cp.float32)           
            for bit_idx, bit_id in enumerate(tqdm(self._bit_ids,desc='bit',leave=True)):
                all_images = []
                for tile_id in tqdm(random_tiles,desc='loading tiles',leave=False):
                    tile_dir_path = self._readout_dir_path / Path(tile_id)
                    bit_dir_path = tile_dir_path / Path(bit_id)
                    current_bit = zarr.open(bit_dir_path,mode='r')
                    current_image = cp.asarray(current_bit["registered_data"], dtype=cp.uint16)
                    current_image[current_image<100] = cp.median(current_image[current_image.shape[0]//2,:,:]).astype(cp.uint16)
                    current_image[current_image>65000] = cp.median(current_image[current_image.shape[0]//2,:,:]).astype(cp.uint16)
                    all_images.append(cp.asnumpy(current_image).astype(np.uint16))
                    del current_image
                    cp.get_default_memory_pool().free_all_blocks()
                    gc.collect()
                    
                all_images = np.array(all_images)
                    
                low_pixels = []    
                for tile_idx, tile_id in enumerate(tqdm(random_tiles,desc='background',leave=False)):
                    
                    current_image = cp.asarray(all_images[tile_idx,:],dtype=cp.float32)
                    low_cutoff = cp.percentile(current_image, low_percentile_cut)
                    low_pixels.append(current_image[current_image < low_cutoff].flatten().astype(cp.float32))
                                        
                    del current_image
                    cp.get_default_memory_pool().free_all_blocks()
                    gc.collect()    

                low_pixels = cp.concatenate(low_pixels,axis=0)
                background_vector[bit_idx] = cp.median(low_pixels)
                
                del low_pixels
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()
                
                high_pixels = []
                for tile_idx, tile_id in enumerate(tqdm(random_tiles,desc='normalize',leave=False)):
                    
                    current_image = cp.asarray(all_images[tile_idx,:],dtype=cp.float32) - background_vector[bit_idx]
                    current_image[current_image<0] = 0
                    high_cutoff = cp.percentile(current_image, high_percentile_cut)
                    high_pixels.append(current_image[current_image > high_cutoff].flatten().astype(cp.float32))
                    
                    del current_image
                    cp.get_default_memory_pool().free_all_blocks()
                    gc.collect()
                    
                high_pixels = cp.concatenate(high_pixels,axis=0)
                normalization_vector[bit_idx] = cp.median(high_pixels)
                    
                del high_pixels
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()
                                                   
            self._calibrations_zarr.attrs['global_normalization'] = cp.asnumpy(normalization_vector).astype(np.float32).tolist()
            self._calibrations_zarr.attrs['global_background'] = cp.asnumpy(background_vector).astype(np.float32).tolist()
                      
        self._background_vector = background_vector
        self._normalization_vector = normalization_vector
        
    def _load_bit_data(self):       
        print('loading raw data')
        images = []
        self._em_wvl = []
        for bit_id in tqdm(self._bit_ids,desc='bit',leave=False):
            tile_dir_path = self._readout_dir_path / Path(self._tile_ids[self._tile_idx])
            bit_dir_path = tile_dir_path / Path(bit_id)
            current_bit = zarr.open(bit_dir_path,mode='r')
            images.append(np.asarray(current_bit["registered_data"], dtype=np.uint16))
            self._em_wvl.append(current_bit.attrs['emission_um'])
            
        self._image_data = np.stack(images,axis=0)
        voxel_size_zyx_um = np.asarray(current_bit.attrs['voxel_zyx_um'],dtype=np.float32)
        self._pixel_size = voxel_size_zyx_um[1]
        self._axial_step = voxel_size_zyx_um[0]
        
        del images, current_bit
        gc.collect()
        
    def _hp_filter(self,
                   sigma= (5,3,3)):
        
        if isinstance(self._image_data, np.np.ndarray):
            image_data_cp = cp.asarray(self._image_data,dtype=cp.float32)

        for i in tqdm(range(image_data_cp.shape[0]),desc='bit'):
            lowpass = gaussian_filter(image_data_cp[i,:,:,:], sigma=sigma)
            gauss_highpass = image_data_cp[i,:,:,:] - lowpass
            gauss_highpass[lowpass > image_data_cp[i,:,:,:]] = 0
            image_data_cp[i,:,:,:] = gauss_highpass
            
        self._filter_type = 'hp'   
        self._image_data_hp = cp.asnumpy(image_data_cp).astype(np.float32)
        
        del image_data_cp, lowpass, gauss_highpass
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        
        return cp.asnumpy(gauss_highpass).astype(np.uint16)
        
    def _lp_filter(self,
                   sigma = (2,1,1)):
        
        if self._filter_type is None:
            if isinstance(self._image_data, np.np.ndarray):
                image_data_cp = cp.asarray(self._image_data,dtype=cp.float32)
        else:
            if isinstance(self._image_data_hp, np.np.ndarray):
                image_data_cp = cp.asarray(self._image_data_hp,dtype=cp.float32)
            
        print('lowpass filter')
        for i in tqdm(range(image_data_cp.shape[0]),desc='bit'):
            image_data_cp[i,:,:,:] = gaussian_filter(image_data_cp[i,:,:,:],sigma=sigma)
            
        self._filter_type = 'lp'
        self._image_data_lp = cp.asnumpy(image_data_cp).astype(np.float32)
        
        del image_data_cp
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
                
    @staticmethod
    def _scale_pixel_traces(pixel_traces: Union[np.ndarray,cp.ndarray],
                            background_vector : Union[np.ndarray,cp.ndarray],
                            normalization_vector: Union[np.ndarray,cp.ndarray]) -> cp.ndarray:
        
        if isinstance(pixel_traces, np.np.ndarray):
            pixel_traces = cp.asarray(pixel_traces,dtype=cp.float32)
        if isinstance(scale_factors, np.np.ndarray):
            scale_factors = cp.asarray(scale_factors,dtype=cp.float32)
        
        return (pixel_traces - background_vector)  / normalization_vector
    
    @staticmethod
    def _clip_pixel_traces(pixel_traces: Union[np.ndarray,cp.ndarray],
                            clip_lower: float = 0.0,
                            clip_upper: float = 1.0) -> cp.ndarray:
        
        return cp.clip(pixel_traces,clip_lower,clip_upper,pixel_traces)
    
    @staticmethod
    def _normalize_pixel_traces(pixel_traces: Union[np.ndarray,cp.ndarray]) -> Tuple[cp.ndarray,cp.ndarray]:
  
        if isinstance(pixel_traces, np.np.ndarray):
            pixel_traces = cp.asarray(pixel_traces,dtype=cp.float32)
        
        norms = cp.linalg.norm(pixel_traces, axis=0)
        norms = cp.where(norms > 0, norms, 1)
        normalized_traces = pixel_traces / norms[cp.newaxis, :, :, :]
        
        return normalized_traces, norms
    
    @staticmethod
    def _calculate_distances(pixel_traces: Union[np.ndarray,cp.ndarray],
                             codebook_matrix: Union[np.ndarray,cp.ndarray]) -> Tuple[cp.ndarray,cp.ndarray]:
  
        if isinstance(pixel_traces, np.np.ndarray):
            pixel_traces = cp.asarray(pixel_traces,dtype=cp.float32)
        if isinstance(codebook_matrix, np.np.ndarray):
            codebook_matrix = cp.asarray(codebook_matrix,dtype=cp.float32)
                
        distances = cdist(pixel_traces, codebook_matrix, metric='euclidean')
        min_distances = cp.min(distances, axis=1)
        min_indices = cp.argmin(distances, axis=1)
        
        return min_distances, min_indices
 
    def _decode_pixels(self,
                       distance_threshold: float = .5172,
                       magnitude_threshold: float = 1.0):
        
        original_shape = self._image_data_lp.shape
        pixel_traces = cp.asarray(self._image_data_lp).reshape(self._bit_count, -1).astype(cp.float32)
        pixel_traces = self._scale_pixel_traces(pixel_traces,
                                                self._background_vector,
                                                self._normalization_vector)
        scaled_pixel_traces = self._clip_pixel_traces(scaled_pixel_traces)
        normalized_pixel_traces, pixel_magnitude_trace = self._normalize_pixel_traces(scaled_pixel_traces)
        distance_trace, codebook_index_trace = self._calculate_distances(normalized_pixel_traces,self._codebook_matrix)
        
        del normalized_pixel_traces
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

        decoded_trace = cp.full((distance_trace.shape[0],), -1, dtype=cp.int16)
        mask_trace = distance_trace < distance_threshold
        decoded_trace[mask_trace] = codebook_index_trace[mask_trace,0].astype(cp.int16)
        decoded_trace[pixel_magnitude_trace <= magnitude_threshold] = -1

        self._decoded_image = cp.asnumpy(cp.reshape(decoded_trace, original_shape[1:]))
        self._magnitude_image = cp.asnumpy(cp.reshape(pixel_magnitude_trace[0,:], original_shape[1:]))
        self._scaled_pixel_images = cp.asnumpy(cp.reshape(scaled_pixel_traces, original_shape))
        self._distance_image = cp.asnumpy(cp.reshape(distance_trace, original_shape[1:]))
        
        self._decoded_image = cp.asnumpy(cp.reshape(decoded_trace, original_shape[1:]))
        del decoded_trace, pixel_magnitude_trace, scaled_pixel_traces, distance_trace
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        
    @staticmethod
    def _find_objects_cupy(label_image: cp.ndarray, 
                           max_label: int):
        ndim = label_image.ndim
        labels = cp.arange(1, max_label + 1)[:, None, None]
        
        # Initialize the start and end arrays for each dimension
        start = cp.full((max_label, ndim), cp.inf, dtype=cp.float32)
        end = cp.full((max_label, ndim), -cp.inf, dtype=cp.float32)
        
        # Find the indices of each dimension
        grid = cp.ogrid[[slice(0, x) for x in label_image.shape]]
        
        for dim in range(ndim):
            # Use broadcasting to compare labels across dimensions
            mask = label_image == labels
            dim_indices = grid[dim]
            
            # Update start and end positions for each label in this dimension
            valid_mask = mask.any(axis=tuple(range(dim)) + tuple(range(dim + 1, ndim)))
            min_indices = cp.where(valid_mask, dim_indices, cp.inf).min(axis=dim)
            max_indices = cp.where(valid_mask, dim_indices, -cp.inf).max(axis=dim) + 1
            
            start[:, dim] = cp.minimum(start[:, dim], min_indices)
            end[:, dim] = cp.maximum(end[:, dim], max_indices)
        
        # Convert start and end positions to slices
        result = [tuple(slice(int(s), int(e)) for s, e in zip(start_row, end_row))
                if not cp.isinf(start_row[0]) else None
                for start_row, end_row in zip(start, end)]
        
        return result
            
    def _extract_barcodes(self,
                          minimum_pixels: int = 3,
                          small_feature_threshold = 0.8,
                          large_feature_threshold = 0.4):
        
        scaled_pixel_images = cp.asarray(self._scaled_pixel_images,dtype=cp.float32)
        distance_image = cp.asarray(self._distance_image,dtype=cp.float32)
        decoded_image = cp.asarray(self._decoded_image,dtype=cp.int16)
        
        accumulated_barcodes = []

        print('extracting barcodes')
        column_names = ['barcode_id', 'gene_id', 'tile_idx', 'min_on_intensity', 'min_off_intensity', 
                        'area', 'mean_dispersion', 'min_dispersion', 'distance', 'z', 'y', 'x', 
                        'global_z', 'global_y', 'global_x', 'cell_index']
        on_intensity_columns = [f'on_intensity_bit_{i+1}' for i in range(self._num_on_bits)]
        off_intensity_columns = [f'off_intensity_bit_{i+1}' for i in range(self._bit_count - self._num_on_bits)]
        for barcode_index in tqdm(range(self._barcode_count),desc='barcode',leave=True):
            labeled_image = label(decoded_image==barcode_index,connectivity=3)
            scaled_pixel_on_bits = scaled_pixel_images[self._codebook_matrix[barcode_index]==1]
            scaled_pixel_off_bits = scaled_pixel_images[self._codebook_matrix[barcode_index]==0]
            
            spot_slices = self._find_objects_cupy(labeled_image,max_label=cp.amax(labeled_image))
            
            if spot_slices is None:
                continue
            
            for spot_slice in spot_slices:
                spot_mask = (labeled_image[spot_slice] == 1)
                num_pixels = cp.sum(spot_mask)
                indices = cp.nonzero(spot_mask)
                
                zyx_coords = [indices[dim] + spot_slice[dim].start for dim in range(3)]
                zyx_centroid = [cp.mean(coords) for coords in zyx_coords]
                on_bit_intensities = cp.round(cp.max(scaled_pixel_on_bits[:, zyx_coords[:, 0], zyx_coords[:, 1], zyx_coords[:, 2]], axis=1),2).get()
                off_bit_intensities = cp.round(cp.max(scaled_pixel_off_bits[:, zyx_coords[:, 0], zyx_coords[:, 1], zyx_coords[:, 2]], axis=1),2).get()
                min_on_bit_intensity = cp.min(scaled_pixel_on_bits[:,spot_mask],axis=0)
                min_off_bit_intensity = cp.min(scaled_pixel_off_bits[:,spot_mask],axis=0)
                min_distance = cp.min(distance_image[spot_mask])
                
                disperson = cp.linalg.norm(zyx_coords - zyx_centroid, axis=1)
                
                if num_pixels >= minimum_pixels:
                    
                    if large_feature_threshold is None:
                        accept_spot = True
                    else:
                        if min_on_bit_intensity > large_feature_threshold:
                            accept_spot = True
                        else:
                            accept_spot = False
                        
                else:
                    if small_feature_threshold is None:
                        accept_spot = False
                    else:
                        if min_on_bit_intensity > small_feature_threshold:
                            accept_spot = True
                        else:
                            accept_spot = False
                            
                if accept_spot:
                    # barcode_id, gene_id, tile_idx, min_on, max_on, num_pixels, dispersion, distance, z, y, x 
                    df = pd.DataFrame(np.zeros((1, len(column_names))), columns=column_names)
                    df['barcode_id'] = barcode_index + 1
                    df['gene_id'] = self._gene_ids[barcode_index]
                    df['tile_idx'] = self._tile_idx
                    df.loc[:, ['min_on_bit_intensity', 'min_off_bit_intensity', 'area']] = [np.round(min_on_bit_intensity.get(),2), np.round(min_off_bit_intensity.get(),2), num_pixels.get()]
                    df.loc[:, ['mean_dispersion', 'min_dispersion']] = [np.round(cp.mean(disperson).get(),2), np.round(cp.min(disperson).get(),2)]
                    df.loc[:, ['min_distance']] = np.round(min_distance.get(),2)
                    df.loc[:, ['z', 'y', 'x']] = np.round(zyx_centroid.get(),2) 
                    df.loc[:, ['global_z', 'global_y', 'global_x']] = np.round(zyx_centroid.get(),2)
                    df['cell_index'] = -1
                    
                    df_on_intensities = pd.DataFrame([on_bit_intensities], columns=on_intensity_columns)
                    df_off_intensities = pd.DataFrame([off_bit_intensities], columns=off_intensity_columns)
                    
                    df_full = pd.concat([df, df_on_intensities,df_off_intensities], axis=1)
                    accumulated_barcodes.append(df_full)
                    
                else:
                    decoded_image[zyx_coords] = 0

        self._df_barcodes = pd.concat(accumulated_barcodes, ignore_index=True)
        
        del scaled_pixel_images, distance_image, decoded_image, df, df_full, labeled_image, scaled_pixel_on_bits, scaled_pixel_off_bits
        del spot_slices, spot_mask, num_pixels, indices, zyx_coords, zyx_centroid, on_bit_intensities, off_bit_intensities
        del min_on_bit_intensity, min_off_bit_intensity, min_distance, disperson
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        
    def save_barcodes(self):
        
        readout_dir_path = self._dataset_path / Path('readouts')
        tile_ids = [entry.name for entry in readout_dir_path.iterdir() if entry.is_dir()]
        
        decoded_dir_path = self._dataset_path / Path('decoded')
        decoded_dir_path.mkdir(exist_ok=True)
        
        barcode_path = decoded_dir_path / Path(tile_ids[self._tile_idx]+'_decoded_features.csv')
        self._barcodes_df.to_csv(barcode_path)
 
    
    def run_decoding(self):

        pass