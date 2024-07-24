from merfish3danalysis.utils._imageprocessing import deskew_shape_estimator, deskew, lab2cam, chunk_indices, downsample_deskewed_z
from ryomen import Slicer
from tqdm import tqdm
import numpy as np
from numpy.typing import ArrayLike
import gc

def pad_to_nearest_16_z_reflect(image):
    def next_multiple_of_16(x):
        return (x + 15) // 16 * 16
    
    z, y, x = image.shape
    
    new_z = next_multiple_of_16(z)
    pad_z = new_z - z
    
    # Distribute padding evenly on both sides
    pad_z_before = pad_z // 2
    pad_z_after = pad_z - pad_z_before
    
    # Padding configuration for numpy.pad
    pad_width = ((pad_z_before, pad_z_after), (0, 0), (0, 0))
    
    padded_image = np.pad(image, pad_width, mode='reflect')
    
    return padded_image, pad_z_before, pad_z_after

def remove_padding_z(image, pad_z_before, pad_z_after):
    return image[pad_z_before:-pad_z_after,:]

def chunked_orthogonal_deskew(oblique_image: ArrayLike,
                            psf_data: ArrayLike,
                            chunk_size: int = 15000,
                            overlap_size: int = 550,
                            scan_crop: int = 700,
                            camera_bkd: int = 100,
                            camera_cf: float = .24,
                            camera_qe: float = 0.9,
                            z_downsample_level = 2,
                            perform_decon: bool = True,
                            decon_iterations: int = 10,
                            decon_chunks: int = 256) -> ArrayLike:

    output_shape = deskew_shape_estimator(oblique_image.shape)
    output_shape[0] = output_shape[0]//z_downsample_level
    output_shape[1] = output_shape[1] - scan_crop
    deskewed_image = np.zeros(output_shape, dtype=np.uint16)

    if chunk_size < output_shape[1]:
        idxs = chunk_indices(output_shape[1],chunk_size)
    else:
        idxs = [(0,output_shape[1])]
        overlap_size = 0

    for idx in tqdm(idxs):
        
        if idx[0] > 0:
            tile_px_start = idx[0] - overlap_size
            crop_start = True
        else:
            tile_px_start = idx[0]
            crop_start = False
            
        if idx[1] < output_shape[1]:
            tile_px_end = idx[1] + overlap_size
            crop_end = True
        else:
            if overlap_size == 0:
                tile_px_end = idx[1]+scan_crop
                crop_end = False
            else:
                ile_px_end = idx[1]
                crop_end = False

        xp, yp, sp_start = lab2cam(oblique_image.shape[2],
                                tile_px_start,
                                0,
                                30. * np.pi/180.)
        
        xp, yp, sp_stop = lab2cam(oblique_image.shape[2],
                                tile_px_end,
                                0,
                                30. * np.pi/180.)
        scan_px_start = np.maximum(0,np.int64(np.ceil(sp_start * (.115/.4))))
        scan_px_stop =  np.minimum(oblique_image.shape[0],np.int64(np.ceil(sp_stop * (.115/.4))))
        
        raw_data = np.array(oblique_image[scan_px_start:scan_px_stop,:]).astype(np.float32)
        raw_data = (raw_data - camera_bkd)
        raw_data[raw_data<0.] = 0.0
        raw_data = ((raw_data * camera_cf) / camera_qe).astype(np.uint16)
        
        if perform_decon:
            from pycudadecon import decon
            
            data_padded, pad_z_before, pad_z_after = pad_to_nearest_16_z_reflect(raw_data)
            del raw_data
            gc.collect()
            data_decon_padded = np.zeros_like(data_padded)
            
            slices = Slicer(data_padded,
                crop_size=(decon_chunks,1600,1600),
                overlap=(64,64,64),
                batch_size=1,
                pad=True)

            for crop, source, destination in slices: 
                data_decon_padded[destination] = decon(
                        images = crop,
                        psf = psf_data,
                        dzpsf=0.400,
                        dxpsf=.115,
                        dzdata=.400,
                        dxdata=.115,
                        wavelength=520,
                        na=1.3,
                        nimm=1.4,
                        n_iters=decon_iterations,
                        cleanup_otf=True,
                        napodize=30,
                        skewed_decon=True)[source]
                
            data_decon = remove_padding_z(data_decon_padded,pad_z_before,pad_z_after)
            del data_padded, data_decon_padded
            gc.collect()

            
            
            # from clij2fft.richardson_lucy import richardson_lucy_nc, getlib
            # import cupy as cp
            # lib = getlib()
            # slices = Slicer(raw_data,
            #                 crop_size=(decon_chunks,raw_data.shape[1],raw_data.shape[2]),
            #                 overlap=(64,0,0),
            #                 batch_size=1,
            #                 pad=True)
            
            # for crop, source, destination in tqdm(slices,leave=False):             
            #     raw_data[destination] = richardson_lucy_nc(img=crop,
            #                                             psf=psf_data,
            #                                             numiterations=decon_iterations,
            #                                             regularizationfactor=1e-4,
            #                                             lib=lib)[source].astype(np.uint16)
            #     cp.clear_memo()
            #     cp._default_memory_pool.free_all_blocks()
        
        temp_deskew = deskew(data_decon).astype(np.uint16)
        
        if crop_start and crop_end:
            crop_deskew = temp_deskew[:,overlap_size:-overlap_size,:]
        elif crop_start:
            crop_deskew = temp_deskew[:,overlap_size:-1,:]
        elif crop_end:
            crop_deskew = temp_deskew[:,0:-overlap_size,:]
        else:
            crop_deskew = temp_deskew[:,0:-scan_crop,:]
            
        if crop_deskew.shape[1] > (chunk_size):
            diff = crop_deskew.shape[1] - (chunk_size)
            crop_deskew = crop_deskew[:,:-diff,:]
        elif crop_deskew.shape[1] < (chunk_size):
            diff = (chunk_size) - crop_deskew.shape[1]

            if crop_start and crop_end:
                crop_deskew = temp_deskew[:,overlap_size:-overlap_size+diff,:]
            elif crop_start:
                crop_deskew = temp_deskew[:,overlap_size-diff:-1,:]
                    
        if z_downsample_level > 1:
            deskewed_image[:,idx[0]:idx[1],:] = downsample_deskewed_z(crop_deskew,z_downsample_level)
        else:
            deskewed_image[:,idx[0]:idx[1],:] = crop_deskew
                
    del temp_deskew, oblique_image
    gc.collect()
    
    return deskewed_image