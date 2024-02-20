import numpy as np
from pathlib import Path
import zarr
import gc
import cupy as cp
from cucim.skimage.filters import gaussian
from cuml.neighbors import NearestNeighbors
from typing import Union, Optional, Sequence
from numpy.typing import NDArray
from cupy.typing import ArrayLike
import pandas as pd
from cucim.skimage.measure import label, regionprops
from tqdm import tqdm
#from skimage.measure import label, regionprops
from numba import jit, prange
from multiprocessing import Pool, cpu_count
from cupyx.scipy.ndimage import minimum_filter, maximum_filter
from localize_psf.localize import filter_convolve, get_filter_kernel
from functools import partial
import dask.array as da
import dask
dask.config.set({'distributed.admin.low-level-log-length': 100})


class PixelDecoder():
    def __init__(self,
                 data_dir_path: Union[Path,str],
                 tile_idx: int = 0,
                 scale_factors: Optional[NDArray] = None,
                 background_factors: Optional[NDArray] = None):

        self._dataset_path = data_dir_path
        self._tile_idx = tile_idx
        
        self._load_experiment_parameters()
        self._load_codebook()
        self._decoding_matrix_no_errors = self._normalize_codebook(include_errors=False)
        self._decoding_matrix_1bit_errors = self._normalize_codebook(include_errors=True)
        self._decoding_matrix = self._decoding_matrix_no_errors.copy()
        self._barcode_count = self._decoding_matrix.shape[0]
        self._bit_count = self._decoding_matrix.shape[1]
        
        if scale_factors is not None:
            self._scale_factors = scale_factors
        else:
            self._scale_factors = None
            
        if background_factors is not None:
            self._background_factors = background_factors
        else:
            self._background_factors = None

        self._neighbors_l2 = NearestNeighbors(n_neighbors=1, 
                                           algorithm='brute',
                                           metric = 'minkowski',
                                           p=2)
        self._neighbors_l2.fit(self._decoding_matrix)
        self._neighbors_cosine = NearestNeighbors(n_neighbors=1, 
                                           algorithm='brute',
                                           metric = 'cosine',
                                           p=2)
        self._neighbors_cosine.fit(self._decoding_matrix)
        
        self._codebook_style = 1
        
        self._filter_type = None
        
    def _load_experiment_parameters(self):
        
        calibration_dir_path = self._dataset_path / Path("calibrations.zarr")
        self._calibration_zarr = zarr.open(calibration_dir_path,mode='r')
        
        # TO DO: fix in calibration.zarr creation
        self._na = 1.35
        self._ri = 1.4
    
    def _load_codebook(self):

        self._df_codebook = pd.DataFrame(self._calibration_zarr.attrs['codebook'])
        self._df_codebook.fillna(0, inplace=True)
        
            
        self._codebook_matrix = self._df_codebook.iloc[:, 1:17].to_numpy().astype(int)
        self._gene_ids = self._df_codebook.iloc[:, 0].tolist()
        
    def _load_filtered_images(self,
                              overwrite: bool = False):
        readout_dir_path = self._dataset_path / Path('readouts')
        tile_ids = [entry.name for entry in readout_dir_path.iterdir() if entry.is_dir()]
        tile_dir_path = readout_dir_path / Path(tile_ids[self._tile_idx])
        
        print('loading filtered data')
        filtered_images = []
        data_not_found = False
        for bit_id in tqdm(self._bit_ids,desc='bit',leave=False):
            tile_dir_path = readout_dir_path / Path(tile_ids[self._tile_idx])
            bit_dir_path = tile_dir_path / Path(bit_id)
            current_bit = zarr.open(bit_dir_path,mode='r')
            try:
                filtered_images.append(np.asarray(current_bit["filtered_data"], dtype=np.float32))
                self._filter_type = 'background'
            except:
                data_not_found = True
                self._filter_type = None
                
            del current_bit
                
        if data_not_found or overwrite==True:
            from numcodecs import blosc
            compressor = blosc.Blosc(cname='zstd', clevel=5, shuffle=blosc.Blosc.BITSHUFFLE)
            
            print('filtered data not found...generating')
            #self._perform_lp_filter()
            self._perform_DoG_filter()
            #self._perform_bkd_subtract()
            for bit_idx, bit_id in tqdm(enumerate(self._bit_ids),desc='bit'):
                tile_dir_path = readout_dir_path / Path(tile_ids[self._tile_idx])
                bit_dir_path = tile_dir_path / Path(bit_id)
                current_bit = zarr.open(bit_dir_path,mode='a')
                try:
                    filtered_data = current_bit.zeros('filtered_data',
                                                            shape=(self._image_data_bkd[bit_idx,:].shape[0],self._image_data_bkd[bit_idx,:].shape[1],self._image_data_bkd[bit_idx,:].shape[2]),
                                                            chunks=(1,self._image_data_bkd[bit_idx,:].shape[1],self._image_data_bkd[bit_idx,:].shape[2]),
                                                            compressor=compressor,
                                                            dtype=np.uint16)
                except Exception:
                    filtered_data = current_bit['filtered_data']
                
                filtered_data[:] = self._image_data_dog[bit_idx,:]
            
                del current_bit
        else:
            self._image_data_bkd = np.stack(filtered_images,axis=0)
            
            del filtered_images
            gc.collect()
    

    def _normalize_codebook(self, include_errors: bool = True):
        self._barcode_set = cp.asarray(self._codebook_matrix)
        magnitudes = cp.sqrt(cp.sum(self._barcode_set**2, axis=1))

        # Avoid division by zero by ensuring magnitudes are not zero
        magnitudes = cp.where(magnitudes == 0, 1, magnitudes)

        if not include_errors:
            # Normalize directly using broadcasting
            normalized_barcodes = self._barcode_set / magnitudes[:, None]
            return normalized_barcodes
        else:
            # Pre-compute the normalized barcodes
            normalized_barcodes = self._barcode_set / magnitudes[:, None]

            # Initialize an empty list to hold all barcodes with single errors
            barcodes_with_single_errors = [normalized_barcodes]

            # Generate single-bit errors
            for bit_index in range(self._barcode_set.shape[1]):
                flipped_barcodes = self._barcode_set.copy()
                flipped_barcodes[:, bit_index] = 1 - flipped_barcodes[:, bit_index]
                flipped_magnitudes = cp.sqrt(cp.sum(flipped_barcodes**2, axis=1))
                flipped_magnitudes = cp.where(flipped_magnitudes == 0, 1, flipped_magnitudes)
                normalized_flipped = flipped_barcodes / flipped_magnitudes[:, None]
                barcodes_with_single_errors.append(normalized_flipped)

            # Stack all barcodes (original normalized + with single errors)
            all_barcodes = cp.vstack(barcodes_with_single_errors)
            return all_barcodes
        
    def _load_bit_data(self):
        readout_dir_path = self._dataset_path / Path('readouts')
        tile_ids = [entry.name for entry in readout_dir_path.iterdir() if entry.is_dir()]
        tile_dir_path = readout_dir_path / Path(tile_ids[self._tile_idx])
        self._bit_ids = [entry.name for entry in tile_dir_path.iterdir() if entry.is_dir()]
        
        print('loading raw data')
        images = []
        self._em_wvl = []
        for bit_id in tqdm(self._bit_ids,desc='bit',leave=False):
            tile_dir_path = readout_dir_path / Path(tile_ids[self._tile_idx])
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
        
    def _perform_bkd_subtract(self,
                              size: float = (15,7,7)):
        
        if self._filter_type is None:        
            image_data_cp = cp.asarray(self._image_data,dtype=cp.float32)
        else:
            image_data_cp = cp.asarray(self._image_data_dog,dtype=cp.float32)

        print('background subtraction')
        for i in tqdm(range(image_data_cp.shape[0]),desc='bit'):
            image_data_cp[i, :, :, :] = image_data_cp[i, :, :, :] - maximum_filter(minimum_filter(image_data_cp[i, :, :, :], size=size),size)
        image_data_cp[image_data_cp<=0] = 0
        self._image_data_bkd = cp.asnumpy(image_data_cp).astype(np.float32)
                    
        self._filter = 'background'
                    
        del image_data_cp
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        
    def _perform_DoG_cartesian(self,
                            image: Union[NDArray,ArrayLike],
                            kernel_small: Sequence[float],
                            kernel_large: Sequence[float],
                            pixel_size: float,
                            scan_step:  float) -> NDArray:
        """
        Perform difference of gaussian filter on cartesian image chunk

        Parameters
        ----------
        image : array
            Image to filter
        kernel_small : Sequence[float]
            Small Gaussian kernel sigmas (z,y,x)
        kernel_large : Sequence[float]
            Large Gaussian kernel sigmas (z,y,x)
        pixel_size : float
            Effective camera pixel size
        scan_step : float
            Spacing between z-planes

        Returns
        -------
        filtered : array
            Filtered image
        """
        kernel_small = get_filter_kernel(kernel_small,
                                        [scan_step,
                                        pixel_size,
                                        pixel_size],
                                        sigma_cutoff = 2)
        kernel_large = get_filter_kernel(kernel_large,
                                        [scan_step,
                                        pixel_size,
                                        pixel_size],
                                        sigma_cutoff = 2)
        
        
        image_cp = cp.asarray(image,dtype=cp.float32)
        image_hp = filter_convolve(image_cp, kernel_small.astype(cp.float32))
        image_lp = filter_convolve(image_cp, kernel_large.astype(cp.float32))
        image_filtered = cp.asnumpy(image_hp - image_lp)
        image_filtered[image_filtered<0.0]=0.0
        

        del image_cp, image_lp, image_hp
        gc.collect()
        cp.clear_memo()
        cp._default_memory_pool.free_all_blocks()
        
        return image_filtered.astype(np.float32)

    def _perform_DoG_filter(self,
                            overlap_depth: Sequence[int]  = [0,128,128]):
        """
        Run difference of gaussian filter using Dask and GPU.

        Parameters
        ----------
        image : NDArray
            set of images to filter
        DoG_filter_params : dict
            DoG filtering parameters
        overlap_depth : Sequence[int]
            Size of overlap for dask map_overlap
        image_params : dict
            {'pixel_size','scan_step'}. Camera pixel size (µm), spacing between z-planes(µm).

        Returns
        -------
        fitered_image : NDArray
            DoG filtered image
        """
        image_da = da.from_array(self._image_data,
                                chunks=(1,
                                        self._image_data.shape[1],
                                        self._image_data.shape[2],
                                        self._image_data.shape[3]))

        self._image_data_dog = np.zeros_like(self._image_data,dtype=np.float32)
        
        print('DoG filter')
        for bit_idx in tqdm(range(self._image_data.shape[0]),desc='bit',leave=False):
            
            em_wvl = self._em_wvl[bit_idx]
            
            sigma_xy = 0.22 * em_wvl / self._na
            sigma_z = np.sqrt(6)/ np.pi * self._ri* em_wvl / self._na ** 2
            
            kernel_small = [0.707 * sigma_z,
                            0.707 * sigma_xy,
                            0.707 * sigma_xy]

            kernel_large = [1.414 * sigma_z,
                            1.414 * sigma_xy,
                            1.414 * sigma_xy]
        
            DoG_dask_func = partial(self._perform_DoG_cartesian,
                                    kernel_small=kernel_small,
                                    kernel_large=kernel_large,
                                    pixel_size=self._pixel_size,
                                    scan_step=self._axial_step)
            
            dask_dog_filter = da.map_overlap(DoG_dask_func,
                                            image_da[bit_idx,:],
                                            depth=overlap_depth,
                                            boundary='reflect',
                                            trim=True,
                                            meta=np.array((), dtype=np.float32))

            self._image_data_dog[bit_idx,:] = dask_dog_filter.compute(scheduler='single-threaded')
                
            del kernel_small, kernel_large, DoG_dask_func, dask_dog_filter
            cp.clear_memo()
            cp._default_memory_pool.free_all_blocks()
            gc.collect()
            
            self._filter_type = 'dog'
            
    @staticmethod            
    def find_global_min_max(image_stack):

        min_val = image_stack.min()
        max_val = image_stack.max()
        return min_val, max_val
    
    @staticmethod
    def calculate_pixel_histogram(image_stack, min_val, max_val):
        bit_count = image_stack.shape[0]
        pixel_histograms = []
        for i in range(bit_count):
            # Flatten the 3D image (z, y, x) to 1D for histogram calculation
            image_flat = cp.asarray(image_stack[i]).ravel()
            histogram, _ = cp.histogram(image_flat, bins=int(max_val)-int(min_val)+1, range=(int(min_val), int(max_val)))
            pixel_histograms.append(histogram)
            del image_flat
            cp.clear_memo()
            cp._default_memory_pool.free_all_blocks()
            gc.collect()
        return pixel_histograms
    
    
    def _calculate_initial_scale_factors(self):
        if self._filter_type == 'dog':
            min_val, max_val = self.find_global_min_max(self._image_data_dog)
            pixel_histograms = self.calculate_pixel_histogram(self._image_data_dog, min_val, max_val)
        else:
            min_val, max_val = self.find_global_min_max(self._image_data_bkd)
            pixel_histograms = self.calculate_pixel_histogram(self._image_data_bkd, min_val, max_val)
        bitCount = len(pixel_histograms)
        for i in range(bitCount):
            cumulative_histogram = cp.cumsum(pixel_histograms[i])
            cumulative_histogram = cumulative_histogram / cumulative_histogram[-1]
            self._scale_factors[i] = cp.asnumpy(cp.argmin(cp.abs(cumulative_histogram - 0.9)) + 2)
            cp.clear_memo()
            cp._default_memory_pool.free_all_blocks()
            gc.collect()
            
        del pixel_histograms, cumulative_histogram
        
    def _intialize_scale_factors(self):
        if self._scale_factors is None:
            self._scale_factors = np.zeros(self._bit_count)
            self._calculate_initial_scale_factors()
            self._overwrite = False

        if self._background_factors is None:
            self._background_factors = np.zeros(self._bit_count)
            self._overwrite = False

    def _decode_pixels(self,
                       codebook_style = 1,
                       l2_distance_threshold: float=1.5,
                       cosine_distance_threshold: float=.423,
                       magnitude_threshold: float=1.5):
        
        # Memory cleanup if overwrite is enabled
        if self._overwrite:
            del self._decoded_image, self._scaled_pixel_traces, self._l2_distances, self._cosine_distances
            gc.collect()
            
        if not(codebook_style == self._codebook_style):
            self._codebook_style = codebook_style
            del self._neighbors
            if self._codebook_style == 1:
                self._decoding_matrix = self._decoding_matrix_no_errors.copy()
            elif self._codebook_style == 2:
                self._decoding_matrix = self._decoding_matrix_1bit_errors.copy()
            self._neighbors = NearestNeighbors(n_neighbors=1, 
                                           algorithm='brute',
                                           metric = 'minkowski',
                                           p=2)    
            self._neighbors.fit(self._decoding_matrix)
            
        scale_factors = self._scale_factors.copy()
        background_factors = self._background_factors.copy()
        
        # Ensure scale_factors and background_factors are cupy arrays and have the correct shape
        scale_factors_cp = cp.asarray(scale_factors.reshape(-1, 1),dtype = cp.float32)  # Reshape for broadcasting
        background_factors_cp = cp.asarray(background_factors.reshape(-1, 1),dtype = cp.float32)  # Reshape for broadcasting

        original_shape = self._image_data.shape

        print('pixel decoding')
        for z_idx in tqdm(range(original_shape[1]),desc='z plane',leave=False):
            
            if self._filter_type == 'dog':
                zplane_shape = self._image_data_dog[:,z_idx,:].shape
                current_zplane_cp = cp.asarray(self._image_data_dog[:,z_idx,:],dtype=cp.float32)
            elif self._filter_type == 'background':
                zplane_shape = self._image_data_bkd[:,z_idx,:].shape
                current_zplane_cp = cp.asarray(self._image_data_bkd[:,z_idx,:],dtype=cp.float32)
            
            prod_dims_zplane_cp = cp.prod(cp.array(current_zplane_cp.shape[1:])).item()
            scaled_pixel_traces_zplane_cp = cp.reshape(current_zplane_cp, (current_zplane_cp.shape[0], prod_dims_zplane_cp))      

            scaled_pixel_traces_zplane_cp = (scaled_pixel_traces_zplane_cp - background_factors_cp) / scale_factors_cp
            scaled_pixel_traces_zplane_cp[scaled_pixel_traces_zplane_cp<0] = 0
            
            pixel_magnitudes_zplane_cp = cp.linalg.norm(scaled_pixel_traces_zplane_cp, axis=0, keepdims=True).astype(cp.float32)
            pixel_magnitudes_zplane_cp = cp.where(pixel_magnitudes_zplane_cp == 0., 1., pixel_magnitudes_zplane_cp)
            scaled_pixel_traces_zplane_cp= scaled_pixel_traces_zplane_cp / pixel_magnitudes_zplane_cp
            
            l2_distances_zplane_cp, l2_indexes_zplane_cp = self._neighbors_l2.kneighbors(cp.transpose(scaled_pixel_traces_zplane_cp),
                                                                                        return_distance=True,
                                                                                        two_pass_precision=True)
            
            cosine_distances_zplane_cp, cosine_indexes_zplane_cp = self._neighbors_cosine.kneighbors(cp.transpose(scaled_pixel_traces_zplane_cp),
                                                                                                    return_distance=True,
                                                                                                    two_pass_precision=True)
            
            decoded_zplane_flat_cp = cp.full((l2_indexes_zplane_cp.shape[0],), -1, dtype=cp.int16)

            mask_zplane_cp = (l2_distances_zplane_cp[:, 0] >= l2_distance_threshold) & (cosine_distances_zplane_cp[:, 0] < cosine_distance_threshold)
        
            decoded_zplane_flat_cp[mask_zplane_cp] = l2_indexes_zplane_cp[mask_zplane_cp, 0].astype(cp.int16)
            decoded_zplane_flat_cp[pixel_magnitudes_zplane_cp[0,:] <= magnitude_threshold] = -1
            decoded_image_zplane_cp = cp.reshape(decoded_zplane_flat_cp, zplane_shape[1:])

            pixel_magnitudes_zplane_cp = cp.reshape(pixel_magnitudes_zplane_cp[0,:], zplane_shape[1:])
            scaled_pixel_traces_zplane_cp = cp.reshape(scaled_pixel_traces_zplane_cp, zplane_shape)
            l2_distances_zplane_cp = np.reshape(l2_distances_zplane_cp, zplane_shape[1:])
            cosine_distances_zplane_cp = np.reshape(cosine_distances_zplane_cp, zplane_shape[1:])
            
            if z_idx == 0:
                self._decoded_image = np.zeros((original_shape[1],original_shape[2],original_shape[3]),dtype=np.int16)
                self._scaled_pixel_traces = np.zeros((original_shape),dtype=np.float32)
                self._l2_distances = np.zeros((original_shape[1],original_shape[2],original_shape[3]),dtype=np.float32)
                self._cosine_distances = np.zeros((original_shape[1],original_shape[2],original_shape[3]),dtype=np.float32)
                self._pixel_magnitudes = np.zeros((original_shape[1],original_shape[2],original_shape[3]),dtype=np.float32)
            
            self._decoded_image[z_idx,:] = cp.asnumpy(decoded_image_zplane_cp).astype(np.int16)
            self._scaled_pixel_traces[:,z_idx,:] = cp.asnumpy(scaled_pixel_traces_zplane_cp).astype(np.float32)
            self._l2_distances[z_idx,:] = cp.asnumpy(l2_distances_zplane_cp).astype(np.float32)
            self._cosine_distances[z_idx,:] = cp.asnumpy(cosine_distances_zplane_cp).astype(np.float32)
            self._pixel_magnitudes[z_idx,:] = cp.asnumpy(pixel_magnitudes_zplane_cp).astype(np.float32)


    def _extract_barcodes(self,
                          minimum_area: int = 4,
                          maximum_area: int = 40):
        
        scaled_pixel_traces_cp = cp.asarray(self._scaled_pixel_traces,dtype=cp.float32)
        l2_distances_cp = cp.asarray(self._l2_distances)
        cosine_distances_cp = cp.asarray(self._cosine_distances)
        
        accumulated_barcodes = []

        print('spot calling')
        column_names = ['barcode_id', 'gene_id', 'tile_idx', 'mean_intensity', 'max_intensity',
                'area', 'mean_dispersion', 'min_dispersion', 'mean_distance_l2', 'mean_distance_cosine', 
                'called_distance_l2', 'called_distance_cosine', 'mean_on_intensity', 'mean_off_intensity', 
                'z', 'y', 'x', 'global_z', 'global_y', 'global_x', 'cell_index']
        intensity_columns = [f'intensity_bit_{i+1}' for i in range(self._scaled_pixel_traces.shape[0])]
        for barcode_index in tqdm(range(self._barcode_count),desc='barcode',leave=True):
            labeled_image = label(cp.asarray(self._decoded_image,dtype=cp.int16)==barcode_index, 
                                connectivity=3)
            barcode_regions = [x for x in regionprops(labeled_image,intensity_image=cp.asarray(self._pixel_magnitudes,dtype=cp.float32)) if (x.area >= minimum_area and x.area < maximum_area)]
            
            del labeled_image
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
            
            if len(barcode_regions) == 0:
                df = pd.DataFrame(columns=column_names + intensity_columns)
                accumulated_barcodes.append(df)
                continue
    
            for br in tqdm(barcode_regions, desc='feature', leave=False):
                mean_intensity = br.mean_intensity.item()  # Assuming scalar values are on GPU
                max_intensity = br.max_intensity.item()
                area = br.area.item()
                centroid = cp.array([element.get() for element in br.weighted_centroid])  # Keep centroid on GPU for calculations

                all_coords = br.coords  # Keep all_coords on GPU
                disperson = cp.linalg.norm(all_coords - centroid, axis=1)
                mean_disperson = cp.mean(disperson).get()  # Perform mean on GPU, then transfer
                min_disperson = cp.min(disperson).get()  # Perform min on GPU, then transfer
                mean_distance_l2 = cp.round(cp.mean(l2_distances_cp[all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]], axis=0),2).get()
                mean_distance_cosine = cp.round(cp.mean(cosine_distances_cp[all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]], axis=0),2).get()

                intensities_mean = cp.round(cp.mean(scaled_pixel_traces_cp[:, all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]], axis=1),2).get()  # Transfer array from GPU to CPU
                df_intensities = pd.DataFrame([intensities_mean], columns=intensity_columns)
                barcode_sequence = cp.asnumpy(self._decoding_matrix[barcode_index,:])
                called_distance_l2 = np.round(np.sqrt(np.sum((intensities_mean-barcode_sequence)**2)),3)
                called_distance_cosine = np.round(1 - (np.dot(intensities_mean,barcode_sequence) / (np.linalg.norm(intensities_mean)*np.linalg.norm(barcode_sequence))),3)
                mean_on_intensity = np.round(np.mean(intensities_mean[barcode_sequence!=0]),3)
                mean_off_intensity = np.round(np.mean(intensities_mean[barcode_sequence==0]),3)

                df = pd.DataFrame(np.zeros((1, len(column_names))), columns=column_names)
                df['barcode_id'] = barcode_index + 1
                df['gene_id'] = self._gene_ids[barcode_index]
                df['tile_idx'] = self._tile_idx
                df.loc[:, ['mean_intensity', 'max_intensity', 'area']] = [np.round(mean_intensity,2), np.round(max_intensity,2), np.round(area,2)]
                df.loc[:, ['mean_dispersion', 'min_dispersion']] = [np.round(mean_disperson,2), np.round(min_disperson,2)]
                df.loc[:, ['mean_distance_l2']] = mean_distance_l2
                df.loc[:, ['mean_distance_cosine']] = mean_distance_cosine
                df.loc[:, ['called_distance_l2']] = called_distance_l2
                df.loc[:, ['called_distance_cosine']] = called_distance_cosine
                df.loc[:, ['mean_on_intensity']] = mean_on_intensity
                df.loc[:, ['mean_off_intensity']] = mean_off_intensity
                df.loc[:, ['z', 'y', 'x']] = np.round(centroid.get(),2) 
                df.loc[:, ['global_z', 'global_y', 'global_x']] = np.round(centroid.get(),2)
                df['cell_index'] = -1

                df_full = pd.concat([df, df_intensities], axis=1)

                accumulated_barcodes.append(df_full)
                
                del df_full

        self._barcodes_df = pd.concat(accumulated_barcodes, ignore_index=True)
        del accumulated_barcodes, scaled_pixel_traces_cp, l2_distances_cp, cosine_distances_cp
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        
    def _save_barcodes(self):
        
        readout_dir_path = self._dataset_path / Path('readouts')
        tile_ids = [entry.name for entry in readout_dir_path.iterdir() if entry.is_dir()]
        
        decoded_dir_path = self._dataset_path / Path('decoded')
        decoded_dir_path.mkdir(exist_ok=True)
        
        barcode_path = decoded_dir_path / Path(tile_ids[self._tile_idx]+'_decoded_features.csv')
        self._barcodes_df.to_csv(barcode_path)
        
    def _extract_refactors(self,
                           minimum_area: int = 4,
                           maximum_area: int = 40,
                           extract_backgrounds: Optional[bool] = False):
        
        if self._overwrite:
            del self._scale_factors, self._background_factors, self._barcodes_seen
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
        
        if extract_backgrounds:
            background_refactors = self._extract_backgrounds()
        else:
            background_refactors = cp.zeros(self._bit_count)

        sum_pixel_traces = cp.zeros((self._barcode_count, self._bit_count))
        barcodes_seen = cp.zeros(self._barcode_count)

        print('extract scaling factors')
        for b in tqdm(range(self._barcode_count),desc='barcode',leave=False):
            labeled_image = label(cp.asarray(self._decoded_image,dtype=cp.int16) == b,connectivity=3)
            barcode_regions = [x for x in regionprops(labeled_image) if (x.area >= minimum_area and x.area < maximum_area)]
            barcodes_seen[b] = len(barcode_regions)

            for br in barcode_regions:
                mean_pixel_traces = []
                for coord in br.coords:
                    z, y, x = map(int, coord)
                    pixel_trace = cp.asarray(self._scaled_pixel_traces[:, z,y,x],dtype=cp.float32) * cp.asarray(self._pixel_magnitudes[z,y,x],dtype=cp.float32)
                    mean_pixel_traces.append(pixel_trace)
                mean_pixel_trace = cp.mean(cp.stack(mean_pixel_traces),axis=0)  - background_refactors 
                norm_pixel_trace = mean_pixel_trace / cp.linalg.norm(mean_pixel_trace)
                sum_pixel_traces[b, :] += norm_pixel_trace / barcodes_seen[b]
                
                del mean_pixel_trace, norm_pixel_trace

        sum_pixel_traces[self._decoding_matrix == 0] = cp.nan
        on_bit_intensity = cp.nanmean(sum_pixel_traces, axis=0)
        refactors = on_bit_intensity / cp.mean(on_bit_intensity)

        self._scale_factors = cp.asnumpy(cp.round(refactors,3))
        self._background_factors = cp.asnumpy(background_refactors)
        self._barcodes_seen = cp.asnumpy(barcodes_seen)

        del refactors, background_refactors, barcodes_seen, sum_pixel_traces, on_bit_intensity
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    def _extract_backgrounds(self,
                            minimum_area: int = 6,
                            maximum_area: int = 60):
        
        sum_min_pixel_traces = cp.zeros((self._barcode_count, self._bit_count))
        barcodes_seen = cp.zeros(self._barcode_count)

        print('extract background factors')
        for b in tqdm(range(self._barcode_count),desc='barcode',leave=False):
            labeled_image = label(cp.asarray(self._decoded_image == b,dtype=cp.int16),connectivity=3)
            barcode_regions = [x for x in regionprops(labeled_image) if (x.area >= minimum_area and x.area < maximum_area)]
            barcodes_seen[b] = len(barcode_regions)

            for br in barcode_regions:
                # Initialize an empty list to store min pixel trace values for each coordinate
                min_pixel_traces = []
                for coord in br.coords:
                    z, y, x = map(int, coord)
                    pixel_trace = cp.asarray(self._scaled_pixel_traces[:, z, y, x], dtype=cp.float32) * cp.asarray(self._pixel_magnitudes[z, y, x], dtype=cp.float32)
                    min_pixel_traces.append(pixel_trace)
                # Convert list to CuPy array and calculate the minimum across all coordinates
                min_pixel_trace = cp.min(cp.stack(min_pixel_traces), axis=0)
                sum_min_pixel_traces[b, :] += min_pixel_trace

            off_pixel_traces = sum_min_pixel_traces.copy()
            off_pixel_traces[self._decoding_matrix > 0] = cp.nan
            off_bit_intensity = cp.nansum(off_pixel_traces, axis=0) / cp.sum((self._decoding_matrix == 0) * barcodes_seen[:, None], axis=0)

        del labeled_image, barcode_regions, off_pixel_traces
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

        return off_bit_intensity
    
    def run_decoding(self,
                     num_iterative_rounds: int = 10):
        
        self._load_bit_data()

        self._overwrite = False
        for i in range(num_iterative_rounds):
            self._decode_pixels()
            if i==0:
                self._overwrite = True
            self._extract_refactors()
        self._overwrite = False
        self._extract_barcodes(minimum_area=16)

        return self._barcodes_df
    
    def _extract_refactors_with_multiprocessing(self, minimum_area=8, maximum_area=30, extract_backgrounds=False):
    # Convert to NumPy arrays for multiprocessing and Numba processing
        if not extract_backgrounds:
            background_refactors_np = np.zeros(self._bit_count) 
        else: 
            np.asarray(self._extract_backgrounds_with_multiprocessing())

        # Extract features in parallel
        coords_list = extract_features_in_parallel(self._decoded_image, self._barcode_count, minimum_area, maximum_area)
        
        #filtered_coords_list = [coords for coords in coords_list if coords.size > 0]
        # Compute mean pixel traces using Numba
        sum_pixel_traces, barcodes_seen = compute_mean_pixel_traces(self._scaled_pixel_traces, 
                                                                    self._pixel_magnitudes, 
                                                                    coords_list, 
                                                                    self._barcode_count, 
                                                                    self._scaled_pixel_traces.shape[1], 
                                                                    background_refactors_np)

        # Calculate scaling factors
        sum_pixel_traces[self._decoding_matrix == 0] = np.nan
        on_bit_intensity = np.nanmean(sum_pixel_traces, axis=0)
        refactors = on_bit_intensity / np.mean(on_bit_intensity)

        self._scale_factors = np.round(refactors, 3)
        self._background_factors = background_refactors_np
        self._barcodes_seen = barcodes_seen
    
    
    # Integrated function that uses multiprocessing and Numba
    def _extract_backgrounds_with_multiprocessing(self, minimum_area=15, maximum_area=100):
                
        # Extract features in parallel
        coords_list = extract_features_in_parallel(self._decoded_image, self._barcode_count, minimum_area, maximum_area)

        # Compute min pixel traces using Numba
        sum_min_pixel_traces, barcodes_seen = compute_min_pixel_traces(self._scaled_pixel_trace, 
                                                                       self._pixel_magnitudes, 
                                                                       coords_list, 
                                                                       self._barcode_count, 
                                                                       self._scaled_pixel_traces.shape[1])

        # Calculate off_bit_intensity
        off_pixel_traces = sum_min_pixel_traces.copy()
        off_pixel_traces[self._decoding_matrix_np > 0] = np.nan
        off_bit_intensity = np.nansum(off_pixel_traces, axis=0) / np.sum((self._decoding_matrix_np == 0) * barcodes_seen[:, None], axis=0)

        return off_bit_intensity
    
    
# Function to be executed in parallel for each barcode
def process_barcode(args):
    decoded_image, barcode, minimum_area, maximum_area = args
    try:
        labeled_image = label(decoded_image == barcode, connectivity=decoded_image.ndim)
        barcode_regions = [x for x in regionprops(labeled_image) if minimum_area <= x.area < maximum_area]
        coords = [x.coords for x in barcode_regions]
        return barcode, coords
    except Exception as e:
        print(f"Error processing barcode {barcode}: {e}")
        return barcode, []

# Parallel feature extraction using multiprocessing
def extract_features_in_parallel(decoded_image, barcode_count, minimum_area, maximum_area):
    num_processes = min(cpu_count(), barcode_count)
    tasks = [(decoded_image, b, minimum_area, maximum_area) for b in range(barcode_count)]
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_barcode, tasks)
    
    # Sort results by barcode to ensure correct order
    results.sort(key=lambda x: x[0])
    # Extract just the coordinates list in the correct order
    coords_list = [coords for _, coords in results]
    return coords_list

# Numba-accelerated function for processing pixel traces
@jit(nopython=True, parallel=True)
def compute_min_pixel_traces(scaled_pixel_traces, pixel_magnitudes, coords_list, barcode_count, bit_count):
    sum_min_pixel_traces = np.full((barcode_count, bit_count), np.nan)
    barcodes_seen = np.zeros(barcode_count)

    for b in prange(barcode_count):
        if len(coords_list[b]) == 0:
            continue
        
        min_pixel_traces = []
        for coords in coords_list[b]:
            for coord in coords:
                pixel_trace = scaled_pixel_traces[:, coord[0], coord[1], coord[2]] * pixel_magnitudes[coord[0], coord[1], coord[2]]
                min_pixel_traces.append(pixel_trace)
        if min_pixel_traces:
            min_pixel_trace = np.min(np.stack(min_pixel_traces), axis=0)
            sum_min_pixel_traces[b, :] += min_pixel_trace
            barcodes_seen[b] = len(min_pixel_traces)

    return sum_min_pixel_traces, barcodes_seen

@jit(nopython=True, parallel=True)
def compute_mean_pixel_traces(scaled_pixel_traces, pixel_magnitudes, coords_list, barcode_count, bit_count, background_refactors):
    sum_pixel_traces = np.full((barcode_count, bit_count), np.nan)
    barcodes_seen = np.zeros(barcode_count)

    for b in prange(barcode_count):
        if coords_list[b].shape[0] == 0:  # Check if the list for the current barcode is empty
            continue  # Skip this barcode if there are no coordinates to process

        mean_pixel_traces = []
        for coords in coords_list[b]:
            for coord in coords:
                pixel_trace = scaled_pixel_traces[:, coord[0], coord[1], coord[2]] * pixel_magnitudes[coord[0], coord[1], coord[2]]
                mean_pixel_traces.append(pixel_trace)
        if mean_pixel_traces:
            mean_pixel_trace = np.mean(np.stack(mean_pixel_traces), axis=0) - background_refactors
            norm_pixel_trace = mean_pixel_trace / np.linalg.norm(mean_pixel_trace)
            sum_pixel_traces[b, :] += norm_pixel_trace
            barcodes_seen[b] = len(mean_pixel_traces)

    return sum_pixel_traces, barcodes_seen

    