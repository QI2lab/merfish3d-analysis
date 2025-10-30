"""
Image processing functions for qi2lab 3D MERFISH.

This module includes various utilities for image processing, such as
downsampling, padding, and chunked GPU-based deconvolution.

History:
---------
- **2024/12**: Refactored repo structure.
- **2024/07**: Added numba-accelerated downsampling, padding helper functions,
               and chunked GPU deconvolution.
"""

import numpy as np
import gc
from numpy.typing import ArrayLike
from numba import njit, prange
import builtins

def replace_hot_pixels(
    noise_map: ArrayLike, 
    data: ArrayLike, 
    threshold: float = 375.0
) -> ArrayLike:
    """Replace hot pixels with median values surrounding them.

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

    # GPU
    import cupy as cp  # type: ignore
    from cupyx.scipy import ndimage  # type: ignore

    data = cp.asarray(data, dtype=cp.float32)
    noise_map = cp.asarray(noise_map, dtype=cp.float32)

    # threshold darkfield_image to generate bad pixel matrix
    hot_pixels = cp.squeeze(cp.asarray(noise_map))
    hot_pixels[hot_pixels <= threshold] = 0
    hot_pixels[hot_pixels > threshold] = 1
    hot_pixels = hot_pixels.astype(cp.float32)
    inverted_hot_pixels = cp.ones_like(hot_pixels) - hot_pixels.copy()

    data = cp.asarray(data, dtype=cp.float32)
    for z_idx in range(data.shape[0]):
        median = ndimage.median_filter(data[z_idx, :, :], size=3)
        data[z_idx, :] = inverted_hot_pixels * data[z_idx, :] + hot_pixels * median

    data[data < 0] = 0

    data = cp.asnumpy(data).astype(np.uint16)
    gc.collect()
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    return data

def estimate_shading(
    images: list[ArrayLike]
) -> ArrayLike:
    """Estimate shading using stack of images and BaSiCPy.
    
    Parameters
    ----------
    images: ArrayLike
        4D image stack [p,z,y,x]
        
    Returns
    -------
    shading_image: ArrayLike
        estimated shading image
    """

    # GPU

    import cupy as cp  # type: ignore
    from cupyx.scipy import ndimage  # type: ignore
    from basicpy import BaSiC # type: ignore

    maxz_images = []
    for image in images:
        maxz_images.append(cp.squeeze(cp.max(image.result(),axis=0)))    

    maxz_images = cp.asnumpy(maxz_images).astype(np.uint16)
    gc.collect()
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    original_print = builtins.print
    builtins.print = no_op
    basic = BaSiC(get_darkfield=False)
    basic.autotune(maxz_images[:])
    basic.fit(maxz_images[:])
    builtins.print = original_print
    shading_correction = basic.flatfield.astype(np.float32) / np.max(basic.flatfield.astype(np.float32),axis=(0,1))
    
    del basic
    gc.collect()

    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    
    return shading_correction

def downsample_image_anisotropic(image: ArrayLike, level: tuple[int,int,int] = (2,6,6)) -> ArrayLike:
    """Numba accelerated anisotropic downsampling

    Parameters
    ----------
    image: ArrayLike
        3D image to be downsampled
    level: tuple[int,int,int], default=(2,6,6)
        anisotropic downsampling level

    Returns
    -------
    downsampled_image: ArrayLike
        downsampled 3D image
    """

    downsampled_image = downsample_axis(
        downsample_axis(downsample_axis(image, level[0], 0), level[1], 1), level[2], 2
    )

    return downsampled_image


@njit(parallel=True)
def downsample_axis(
    image: ArrayLike, 
    level: int = 2, 
    axis: int = 0
) -> ArrayLike:
    """Numba accelerated downsampling for 3D images along a specified axis.

    Parameters
    ----------
    image: ArrayLike
        3D image to be downsampled.
    level: int
        Amount of downsampling.
    axis: int
        Axis along which to downsample (0, 1, or 2).

    Returns
    -------
    downsampled_image: ArrayLike
        3D downsampled image.

    """
    if axis == 0:
        new_length = image.shape[0] // level + (1 if image.shape[0] % level != 0 else 0)
        downsampled_image = np.zeros(
            (new_length, image.shape[1], image.shape[2]), dtype=image.dtype
        )

        for y in prange(image.shape[1]):
            for x in range(image.shape[2]):
                for z in range(new_length):
                    sum_value = 0.0
                    count = 0
                    for j in range(level):
                        original_index = z * level + j
                        if original_index < image.shape[0]:
                            sum_value += image[original_index, y, x]
                            count += 1
                    if count > 0:
                        downsampled_image[z, y, x] = sum_value / count

    elif axis == 1:
        new_length = image.shape[1] // level + (1 if image.shape[1] % level != 0 else 0)
        downsampled_image = np.zeros(
            (image.shape[0], new_length, image.shape[2]), dtype=image.dtype
        )

        for z in prange(image.shape[0]):
            for x in range(image.shape[2]):
                for y in range(new_length):
                    sum_value = 0.0
                    count = 0
                    for j in range(level):
                        original_index = y * level + j
                        if original_index < image.shape[1]:
                            sum_value += image[z, original_index, x]
                            count += 1
                    if count > 0:
                        downsampled_image[z, y, x] = sum_value / count

    elif axis == 2:
        new_length = image.shape[2] // level + (1 if image.shape[2] % level != 0 else 0)
        downsampled_image = np.zeros(
            (image.shape[0], image.shape[1], new_length), dtype=image.dtype
        )

        for z in prange(image.shape[0]):
            for y in range(image.shape[1]):
                for x in range(new_length):
                    sum_value = 0.0
                    count = 0
                    for j in range(level):
                        original_index = x * level + j
                        if original_index < image.shape[2]:
                            sum_value += image[z, y, original_index]
                            count += 1
                    if count > 0:
                        downsampled_image[z, y, x] = sum_value / count

    return downsampled_image

def subtract_background_imagej(
    image: np.ndarray, 
    bkd_sub_radius=50, 
):
    """Subtract background from a 3D image using ImageJ's rolling ball algorithm.
    
    Parameters
    ----------
    image : np.ndarray
        3D numpy array representing the image (Z, Y, X).
    bkd_sub_radius : float, optional
        Radius for background subtraction. Default is 50.
    ij : imagej.ImageJ, optional
        An existing ImageJ instance. If None, a new instance will be created.
    Returns
    -------
    bkd_image : np.ndarray
        Background subtracted 3D image as a numpy array (Z, Y, X).
    """
    

    import imagej
    import io
    import contextlib
    import os
    import time

    stderr_buffer = io.StringIO()
    with contextlib.redirect_stderr(stderr_buffer):
        ij_success = False
        while not(ij_success):
            try:
                os.environ.setdefault("CLIJ_OPENCL_ALLOWED_DEVICE_TYPE", "CPU")
                ij = imagej.init()
                ij_success = True
            except:
                time.sleep(.5)
                ij_success = False

    imp_array = ij.py.to_imageplus(image)
    imp_array.setStack(imp_array.getStack().duplicate())
    imp_array.show()
    ij.IJ.run(imp_array, "Subtract Background...", f"rolling={int(bkd_sub_radius)} disable stack")
    imp_array.show()
    ij.py.sync_image(imp_array)
    bkd_output = ij.py.from_java(imp_array.duplicate())
    imp_array.close()
    bkd_image = np.swapaxes(
        bkd_output.data.transpose(2, 1, 0), 1, 2
    ).clip(0, 2**16 - 1).astype(np.uint16).copy()
    del image, imp_array, bkd_output
    gc.collect()

    ij.dispose()
    del ij
    gc.collect()

    return bkd_image

def no_op(*args, **kwargs):
    """Function to monkey patch print to suppress output.
    
    Parameters
    ----------
    args: Any
        positional arguments
    kwargs: Any
        keyword arguments
    """
    
    pass
