"""
Image processing functions for qi2lab 3D MERFISH.

This module provides hot-pixel correction, signal screening, and shading estimation.

"""

import gc
import io
from contextlib import redirect_stdout
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tensorstore import Future


def replace_hot_pixels(
    noise_map: ArrayLike, data: ArrayLike, threshold: float = 375.0
) -> ArrayLike:
    """Replace hot pixels with median values surrounding them.

    Parameters
    ----------
    noise_map: ArrayLike
        darkfield image collected at long exposure time to get hot pixels
    data: ArrayLike
        ND data [broadcast_dim,z,y,x]
    threshold: float, default = 375.0
        threshold to determine hot pixel

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


def image_has_signal(
    image: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    background_sigma: float = 5.0,
    minimum_pixels: int = 9,
) -> bool:
    """Screen YX or ZYX data for connected signal above a robust background.

    Median-filter the Z maximum projection over 3x3 pixels to suppress isolated
    hot pixels. Require a connected region of at least ``minimum_pixels`` above
    median + ``background_sigma`` times the MAD-based noise estimate (1.4826 MAD).
    An optional YX mask restricts both background estimation and accepted signal.
    This is an intensity-contrast screen, not a calibrated significance test.
    Empty images/masks return False; invalid dimensions or nonfinite data raise.

    Parameters
    ----------
    image : ArrayLike
        Image in YX or ZYX order.
    mask : ArrayLike or None, optional
        YX mask limiting the background and signal regions.
    background_sigma : float, optional
        Multiplier for the MAD-based background noise estimate.
    minimum_pixels : int, optional
        Minimum size of a connected foreground region.

    Returns
    -------
    bool
        Whether a connected region passes the signal threshold.

    Raises
    ------
    ValueError
        The image is not YX or ZYX, or its projection has nonfinite values.
    """
    from scipy.ndimage import label, median_filter

    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError("Signal screening requires a YX or ZYX image.")
    if image.size == 0:
        return False
    projection = image.max(axis=0) if image.ndim == 3 else image
    if not np.all(np.isfinite(projection)):
        raise ValueError("Signal screening requires finite image values.")
    projection = median_filter(projection.astype(np.float32), size=3)
    pixels = projection if mask is None else projection[np.asarray(mask, dtype=bool)]
    if pixels.size == 0:
        return False
    background = np.median(pixels)
    noise = 1.4826 * np.median(np.abs(pixels - background))
    foreground = projection > background + background_sigma * noise
    if mask is not None:
        foreground &= np.asarray(mask, dtype=bool)
    labels, _ = label(foreground)
    return bool(np.any(np.bincount(labels.ravel())[1:] >= minimum_pixels))


def estimate_shading(images: "Iterable[Future]") -> ArrayLike:
    """Estimate shading using stack of images and BaSiCPy.

    Parameters
    ----------
    images: Iterable[Future]
        Read futures yielding ZYX images. Pass a generator to load one stack at
        a time. Only YX maximum projections are retained for fitting.

    Returns
    -------
    shading_image: ArrayLike
        estimated shading image
    """
    # GPU

    import cupy as cp  # type: ignore
    from basicpy import BaSiC  # type: ignore

    maxz_images = []
    for image in images:
        maxz_images.append(cp.squeeze(cp.max(image.result(), axis=0)))
        del image

    maxz_images = cp.asnumpy(maxz_images).astype(np.uint16)
    gc.collect()
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    working_size = [max(1, size // 2) for size in maxz_images.shape[-2:]]
    basic = BaSiC(
        get_darkfield=False,
        sort_intensity=True,
        working_size=working_size,
    )
    with redirect_stdout(io.StringIO()):
        basic.autotune(maxz_images[:])
        basic.fit(maxz_images[:])
    shading_correction = basic.flatfield.astype(np.float32) / np.max(
        basic.flatfield.astype(np.float32), axis=(0, 1)
    )

    del basic
    gc.collect()

    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    return shading_correction
