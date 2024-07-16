"""
Image processing functions for qi2lab LED widefield MERFISH data
"""

import numpy as np
import gc
from numpy.typing import ArrayLike
from numba import njit, prange
from typing import Sequence, Tuple

# GPU
CUPY_AVIALABLE = True
try:
    import cupy as cp # type: ignore
except ImportError:
    xp = np
    CUPY_AVIALABLE = False
    from scipy import ndimage # type: ignore
else:
    xp = cp
    from cupyx.scipy import ndimage # type: ignore
 
def replace_hot_pixels(noise_map: ArrayLike, 
                       data: ArrayLike, 
                       threshold: float = 375.) -> ArrayLike:
    """
    Replace all hot pixels with median values surrounding them.

    Parameters
    ----------
    noise_map: ArrayLike
        darkfield image collected at long exposure time to get hot pixels
    data: ArrayLike
        ND data [broadcast_dim,z,y,x]

    Returns
    -------
    data: ArrayLike
        hotpixel corrected data
    """

    data = xp.asarray(data, dtype=xp.float32)
    noise_map = xp.asarray(noise_map, dtype=xp.float32)

    # threshold darkfield_image to generate bad pixel matrix
    hot_pixels = xp.squeeze(xp.asarray(noise_map))
    hot_pixels[hot_pixels<=threshold] = 0
    hot_pixels[hot_pixels>threshold] = 1
    hot_pixels = hot_pixels.astype(xp.float32)
    inverted_hot_pixels = xp.ones_like(hot_pixels) - hot_pixels.copy()
    
    data = xp.asarray(data,dtype=xp.float32)
    for z_idx in range(data.shape[0]):
        median = ndimage.median_filter(data[z_idx,:,:],size=3)
        data[z_idx,:]=inverted_hot_pixels*data[z_idx,:] + hot_pixels*median
    
    data[data<0] = 0
    
    if CUPY_AVIALABLE:
        data = xp.asnumpy(data).astype(np.uint16)
        gc.collect()
        cp.clear_memo()
        cp._default_memory_pool.free_all_blocks()
    else:
        data = data.astype(np.uint16)
    
    return data
    
def correct_shading(darkfield_image: ArrayLike,
                    shading_image: ArrayLike,
                    data: ArrayLike) -> ArrayLike:
    """Perform illumination shading correction.
    
    I_corrected = (I_raw - I_dark) / (I_bright - I_dark).
    Here, we assume I_bright is not normalized or background corrected.

    Parameters
    ----------
    noise_map: ArrayLike
        darkfield image collected at long exposure time to get hot pixels
    darkfield_image: ArrayLike
        darkfield image collected at data's exposure time
    shading_image: ArrayLike
        illumination shading correction
    data: ArrayLike
        ND data [broadcast_dim,z,y,x]

    Returns
    -------
    data: ArrayLike
        shading corrected data
    """
    
    darkfield_image = xp.squeeze(xp.asarray(darkfield_image, dtype=xp.float32))
    shading_image = xp.squeeze(xp.asarray(shading_image, dtype=xp.float32))
    noise_map = xp.squeeze(xp.asarray(shading_image, dtype=xp.float32))
    data = xp.asarray(data, astype=xp.float32)

    shading_image = replace_hot_pixels(noise_map,(shading_image-darkfield_image))
    shading_image = xp.asarray(shading_image, dtype=xp.float32)
    shading_image = shading_image/xp.max(shading_image,axis=(0,1))

    data = replace_hot_pixels(noise_map,(data-darkfield_image))
    data = xp.asarray(data, dtype=xp.float32)

    for z_idx in range(data.shape[0]):
        data[z_idx,:] = data[z_idx,:] / shading_image
   
    if CUPY_AVIALABLE:
        data = xp.asnumpy(data).astype(np.uint16)
        gc.collect()
        cp.clear_memo()
        cp._default_memory_pool.free_all_blocks()
    else:
        data = data.astype(np.uint16)

    return data

@njit
def deskew_shape_estimator(input_shape: Sequence[int],
                           theta: float = 30.0,
                           distance: float = .4,
                           pixel_size: float = .115):
    """Generate shape of orthogonal interpolation output array.
    
    Parameters
    ----------
    shape: Sequence[int]
        shape of oblique array
    theta: float 
        angle relative to coverslip
    distance: float 
        step between image planes along coverslip
    pizel_size: float 
        in-plane camera pixel size in OPM coordinates

    Returns
    -------
    output_shape: Sequence[int]
        shape of deskewed array
    """
    
    # change step size from physical space (nm) to camera space (pixels)
    pixel_step = distance/pixel_size    # (pixels)

    # calculate the number of pixels scanned during stage scan 
    scan_end = input_shape[0] * pixel_step  # (pixels)

    # calculate properties for final image
    final_ny = np.int64(np.ceil(scan_end+input_shape[1]*np.cos(theta*np.pi/180))) # (pixels)
    final_nz = np.int64(np.ceil(input_shape[1]*np.sin(theta*np.pi/180)))          # (pixels)
    final_nx = np.int64(input_shape[2])
    
    return [final_nz, final_ny, final_nx]
    
@njit(parallel=True)
def deskew(data: ArrayLike,
           theta: float = 30.0,
           distance: float = .4,
           pixel_size: float = .115):
    """Numba accelerated orthogonal interpolation for oblique data.
    
    Parameters
    ----------
    data: ArrayLike
        image stack of uniformly spaced OPM planes
    theta: float 
        angle relative to coverslip
    distance: float 
        step between image planes along coverslip
    pizel_size: float 
        in-plane camera pixel size in OPM coordinates

    Returns
    -------
    output: ArrayLike
        image stack of deskewed OPM planes on uniform grid
    """

    # unwrap parameters 
    [num_images,ny,nx]=data.shape     # (pixels)

    # change step size from physical space (nm) to camera space (pixels)
    pixel_step = distance/pixel_size    # (pixels)

    # calculate the number of pixels scanned during stage scan 
    scan_end = num_images * pixel_step  # (pixels)

    # calculate properties for final image
    final_ny = np.int64(np.ceil(scan_end+ny*np.cos(theta*np.pi/180))) # (pixels)
    final_nz = np.int64(np.ceil(ny*np.sin(theta*np.pi/180)))          # (pixels)
    final_nx = np.int64(nx)                                           # (pixels)

    # create final image
    output = np.zeros((final_nz, final_ny, final_nx),dtype=np.float32)  # (time, pixels,pixels,pixels - data is float32)

    # precalculate trig functions for scan angle
    tantheta = np.float32(np.tan(theta * np.pi/180)) # (float32)
    sintheta = np.float32(np.sin(theta * np.pi/180)) # (float32)
    costheta = np.float32(np.cos(theta * np.pi/180)) # (float32)

    # perform orthogonal interpolation

    # loop through output z planes
    # defined as parallel loop in numba
    for z in prange(0,final_nz):
        # calculate range of output y pixels to populate
        y_range_min=np.minimum(0,np.int64(np.floor(np.float32(z)/tantheta)))
        y_range_max=np.maximum(final_ny,np.int64(np.ceil(scan_end+np.float32(z)/tantheta+1)))

        # loop through final y pixels
        # defined as parallel loop in numba
        for y in prange(y_range_min,y_range_max):

            # find the virtual tilted plane that intersects the interpolated plane 
            virtual_plane = y - z/tantheta

            # find raw data planes that surround the virtual plane
            plane_before = np.int64(np.floor(virtual_plane/pixel_step))
            plane_after = np.int64(plane_before+1)

            # continue if raw data planes are within the data range
            if ((plane_before>=0) and (plane_after<num_images)):
                
                # find distance of a point on the  interpolated plane to plane_before and plane_after
                l_before = virtual_plane - plane_before * pixel_step
                l_after = pixel_step - l_before
                
                # determine location of a point along the interpolated plane
                za = z/sintheta
                virtual_pos_before = za + l_before*costheta
                virtual_pos_after = za - l_after*costheta

                # determine nearest data points to interpoloated point in raw data
                pos_before = np.int64(np.floor(virtual_pos_before))
                pos_after = np.int64(np.floor(virtual_pos_after))

                # continue if within data bounds
                if ((pos_before>=0) and (pos_after >= 0) and (pos_before<ny-1) and (pos_after<ny-1)):
                    
                    # determine points surrounding interpolated point on the virtual plane 
                    dz_before = virtual_pos_before - pos_before
                    dz_after = virtual_pos_after - pos_after

                    # compute final image plane using orthogonal interpolation
                    output[z,y,:] = (l_before * dz_after * data[plane_after,pos_after+1,:] +
                                    l_before * (1-dz_after) * data[plane_after,pos_after,:] +
                                    l_after * dz_before * data[plane_before,pos_before+1,:] +
                                    l_after * (1-dz_before) * data[plane_before,pos_before,:]) /pixel_step


    # return output
    return output

def lab2cam(x: int, 
            y: int,
            z: int,
            theta: float = 30. * (np.pi/180.)) -> Tuple[int,int,int]:
    """Convert xyz coordinates to camera coordinates sytem, x', y', and stage position.
    
    Parameters
    ----------
    x: int
        coverslip x coordinate
    y: int
        coverslip y coordinate
    z: int
        coverslip z coordinate
    theta: float
        OPM angle in radians
        
        
    Returns
    -------
    xp: int
        xp coordinate
    yp: int
        yp coordinate
    stage_pos: int
        distance of leading edge of camera frame from the y-axis
    """
    xp = x
    stage_pos = y - z / np.tan(theta)
    yp = z / np.sin(theta)
    return xp, yp, stage_pos

def chunk_indices(length: int, 
                  chunk_size: int) -> Sequence[int]:
    """Calculate indices for evenly distributed chunks.
    
    Parameters
    ----------
    length: int
        axis array length
    chunk_size: int
        size of chunks
        
    Returns
    -------
    indices: Sequence[int,...]
        chunk indices
    """
    
    indices = []
    for i in range(0, length - chunk_size, chunk_size):
        indices.append((i, i + chunk_size))
    if length % chunk_size != 0:
        indices.append((length - chunk_size, length))
    return indices
            
@njit(parallel=True)
def downsample_deskewed_z(image: ArrayLike, 
                          level: int = 2) -> ArrayLike:
    """Numba acelerated z downsampling for 3D image.
    
    Parameters
    ----------
    image: ArrayLike
        3D image to be downsampled in "z" (first axis)
    level: int
        amount of downsampling
        
    Returns
    -------
    downsampled_image: ArrayLike
        3D downsampled image
    
    """
    new_length = image.shape[0] // level + (1 if image.shape[0] % level != 0 else 0)    
    downsampled_image = np.zeros((new_length,image.shape[1],image.shape[2]),dtype=np.uint16)
    
    for y in prange(image.shape[1]):
        for x in range(image.shape[2]):
            for z in range(new_length):
                sum_value = 0.0
                count = 0
                for j in range(level):
                    original_index = z * level + j
                    if original_index < image.shape[0]:
                        sum_value += image[original_index,y,x]
                        count += 1
                if count > 0:
                    downsampled_image[z,y,x] = np.uint16(sum_value / count)
            
    return downsampled_image