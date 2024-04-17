"""
Image processing functions for qi2lab LED widefield MERFISH data
"""

import numpy as np
import gc
from numpy.typing import NDArray

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
 
def replace_hot_pixels(noise_map: NDArray, data: NDArray) -> NDArray:
    """
    Replace all hot pixels with median values surrounding them.

    Parameters
    ----------
    noise_map: NDArray
        darkfield image collected at long exposure time to get hot pixels
    data: NDArray
        ND data [broadcast_dim,z,y,x]

    Returns
    -------
    data: NDArray
        hotpixel corrected data
    """

    data = xp.asarray(data, dtype=xp.float32)
    noise_map = xp.asarray(noise_map, dtype=xp.float32)

    # threshold darkfield_image to generate bad pixel matrix
    hot_pixels = xp.squeeze(xp.asarray(noise_map))
    hot_pixels[hot_pixels<=375] = 0
    hot_pixels[hot_pixels>375] = 1
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
    
def correct_shading(darkfield_image: NDArray,
                    shading_image: NDArray,
                    data: NDArray) -> NDArray:
    """
    Perform illumination shading correction, I_corrected = (I_raw - I_dark) / (I_bright - I_dark).
    Here, we assume I_bright is not normalized or background corrected.

    Parameters
    ----------
    noise_map: NDArray
        darkfield image collected at long exposure time to get hot pixels
    darkfield_image: NDArray
        darkfield image collected at data's exposure time
    shading_image: NDArray
        illumination shading correction
    data: NDArray
        ND data [broadcast_dim,z,y,x]

    Returns
    -------
    data: NDArray
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