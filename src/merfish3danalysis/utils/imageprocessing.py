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
from typing import Sequence, Tuple
from pycudadecon import decon
from ryomen import Slicer
import builtins
from basicpy import BaSiC

# GPU
CUPY_AVIALABLE = True
try:
    import cupy as cp  # type: ignore
except ImportError:
    xp = np
    CUPY_AVIALABLE = False
    from scipy import ndimage  # type: ignore
else:
    xp = cp
    from cupyx.scipy import ndimage  # type: ignore


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

    data = xp.asarray(data, dtype=xp.float32)
    noise_map = xp.asarray(noise_map, dtype=xp.float32)

    # threshold darkfield_image to generate bad pixel matrix
    hot_pixels = xp.squeeze(xp.asarray(noise_map))
    hot_pixels[hot_pixels <= threshold] = 0
    hot_pixels[hot_pixels > threshold] = 1
    hot_pixels = hot_pixels.astype(xp.float32)
    inverted_hot_pixels = xp.ones_like(hot_pixels) - hot_pixels.copy()

    data = xp.asarray(data, dtype=xp.float32)
    for z_idx in range(data.shape[0]):
        median = ndimage.median_filter(data[z_idx, :, :], size=3)
        data[z_idx, :] = inverted_hot_pixels * data[z_idx, :] + hot_pixels * median

    data[data < 0] = 0

    if CUPY_AVIALABLE:
        data = xp.asnumpy(data).astype(np.uint16)
        gc.collect()
        cp.clear_memo()
        cp._default_memory_pool.free_all_blocks()
    else:
        data = data.astype(np.uint16)

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
    maxz_images = []
    for image in images:
        maxz_images.append(xp.squeeze(xp.max(image.result(),axis=0)))    

    if CUPY_AVIALABLE:
        maxz_images = xp.asnumpy(maxz_images).astype(np.uint16)
        gc.collect()
        cp.clear_memo()
        cp._default_memory_pool.free_all_blocks()
    else:
        maxz_images = maxz_images.astype(np.uint16)


    original_print = builtins.print
    builtins.print = no_op
    basic = BaSiC(get_darkfield=False)
    basic.autotune(maxz_images[:])
    basic.fit(maxz_images[:])
    builtins.print = original_print
    shading_correction = basic.flatfield.astype(np.float32) / np.max(basic.flatfield.astype(np.float32),axis=(0,1))
    
    del basic
    gc.collect()
    if CUPY_AVIALABLE:
        cp.clear_memo()
        cp._default_memory_pool.free_all_blocks()
    
    return shading_correction

def downsample_image_isotropic(image: ArrayLike, level: int = 2) -> ArrayLike:
    """Numba accelerated isotropic downsampling

    Parameters
    ----------
    image: ArrayLike
        3D image to be downsampled
    level: int
        isotropic downsampling level

    Returns
    -------
    downsampled_image: ArrayLike
        downsampled 3D image
    """

    downsampled_image = downsample_axis(
        downsample_axis(downsample_axis(image, level, 0), level, 1), level, 2
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


def next_multiple_of_32(x: int) -> int:
    """Calculate next multiple of 32 for the given integer.

    Parameters
    ----------
    x: int
        value to check.

    Returns
    -------
    next_32_x: int
        next multiple of 32 above x.
    """

    next_32_x = int(np.ceil((x + 31) / 32)) * 32

    return next_32_x


def pad_z(image: ArrayLike) -> Tuple[ArrayLike, int, int]:
    """Pad z-axis of 3D array by 32 (zyx order).

    Parameters
    ----------
    image: ArrayLike
        3D image to pad.


    Returns
    -------
    padded_image: ArrayLike
        padded 3D image
    pad_z_before: int
        amount of padding at beginning
    pad_z_after: int
        amount of padding at end
    """

    z, y, x = image.shape

    new_z = next_multiple_of_32(z)
    pad_z = new_z - z

    # Distribute padding evenly on both sides
    pad_z_before = pad_z // 2
    pad_z_after = pad_z - pad_z_before

    # Padding configuration for numpy.pad
    pad_width = ((pad_z_before, pad_z_after), (0, 0), (0, 0))

    padded_image = np.pad(image, pad_width, mode="reflect")

    return padded_image, pad_z_before, pad_z_after


def remove_padding_z(
    padded_image: ArrayLike, 
    pad_z_before: int, 
    pad_z_after: int
) -> ArrayLike:
    """Removing z-axis padding of 3D array (zyx order).

    Parameters
    ----------
    padded_image: ArrayLike
        padded 3D image
    pad_z_before: int
        amount of padding at beginning
    pad_z_after: int
        amount of padding at end


    Returns
    -------
    image: ArrayLike
        unpadded 3D image
    """

    image = padded_image[pad_z_before:-pad_z_after, :]

    return image


def chunked_cudadecon(
    image: ArrayLike,
    psf: ArrayLike,
    image_voxel_zyx_um: Sequence[float],
    psf_voxel_zyx_um: Sequence[float],
    wavelength_um: float,
    na: float,
    ri: float,
    n_iters: int = 10,
    background: float = 100.0,
) -> ArrayLike:
    """Chunked and padded GPU deconvolution using pycudadecon.

    Parameters
    ----------
    image: ArrayLike
        3D image to deconvolve
    psf: ArrayLike
        3D psf
    image_voxel_zyx_um: Sequence[float]
        image voxel spacing in microns
    psf_voxel_zyx_um: Sequence[float]
        psf voxel spacing in microns
    wavelength_um: float
        emission wavelength in microns
    na: float
        numerical aperture
    ri: float
        immersion media refractive index
    n_iters: int
        number of deconvolution iterations
    background: float
        background level to subtract

    Returns
    -------
    image_decon: ArrayLike
        deconvolved image

    """
    original_print = builtins.print

    image_padded, pad_z_before, pad_z_after = pad_z(image)

    image_decon_padded = np.zeros_like(image_padded)

    slices = Slicer(
        image_padded,
        crop_size=(image_padded.shape[0], 1600, 1600),
        overlap=(0, 32, 32),
        batch_size=1,
        pad=True,
    )

    for crop, source, destination in slices:
        builtins.print= no_op
        image_decon_padded[destination] = decon(
            images=crop,
            psf=psf,
            dzpsf=float(psf_voxel_zyx_um[0]),
            dxpsf=float(psf_voxel_zyx_um[1]),
            dzdata=float(image_voxel_zyx_um[0]),
            dxdata=float(image_voxel_zyx_um[1]),
            wavelength=int(wavelength_um * 1000),
            na=float(na),
            nimm=float(ri),
            n_iters=int(n_iters),
            cleanup_otf=True,
            napodize=15,
            background=float(background),
        )[source]
        builtins.print = original_print

    image_decon = remove_padding_z(image_decon_padded, pad_z_before, pad_z_after)

    del image_padded, image_decon_padded
    gc.collect()

    return image_decon

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