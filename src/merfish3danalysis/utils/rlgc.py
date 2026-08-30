"""
Richardson-Lucy Gradient Consensus (RLGC) deconvolution (Manton-style core).

Original idea for Gradient Consensus deconvolution:
James Manton and Andrew York, https://zenodo.org/records/10278919

Reference RLGC loop based on James Manton's implementation:
https://colab.research.google.com/drive/1mfVNSCaYHz1g56g92xBkIoa8190XNJpJ
"""

import gc
import logging
import timeit
from collections.abc import Callable

import cupy as cp
import numpy as np
from cupy import ElementwiseKernel

# -----------------------------------------------------------------------------
# CUDA kernel: multiplicative RL step gated by consensus (reference-accurate)
# -----------------------------------------------------------------------------
filter_update = ElementwiseKernel(
    "float32 recon, float32 HTratio, float32 consensus_map",
    "float32 out",
    """
    bool skip = consensus_map < 0;
    out = skip ? recon : recon * HTratio
    """,
    "filter_update",
)

# -----------------------------------------------------------------------------
# FFT caches (performance)
# -----------------------------------------------------------------------------
_fft_cache_3d: dict[tuple[int, int, int, int], cp.ndarray] = {}


def clear_rlgc_caches(clear_memory_pool: bool = False) -> None:
    """
    Clear cached FFT resources used by RLGC helper functions.

    Parameters
    ----------
    clear_memory_pool : bool, default=False
        If True, synchronize the current CUDA stream and release CuPy device
        and pinned memory pools in addition to clearing local FFT buffers and
        CuPy FFT plans.

    Returns
    -------
    None
    """
    _fft_cache_3d.clear()
    try:
        cp.fft.config.get_plan_cache().clear()
    except Exception:
        pass
    try:
        import cupyx

        cupyx.scipy.fft.clear_plan_cache()
    except Exception:
        pass
    if clear_memory_pool:
        gc.collect()
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def next_gpu_fft_size(x: int) -> int:
    """
    Return the smallest FFT-friendly size >= ``x`` with prime factors in {2, 3}.

    Parameters
    ----------
    x : int
        Minimum desired length.

    Returns
    -------
    int
        Next 2-3-smooth length >= ``x``.
    """
    if x <= 1:
        return 1
    n = x
    while True:
        m = n
        while (m % 2) == 0:
            m //= 2
        while (m % 3) == 0:
            m //= 3
        if m == 1:
            return n
        n += 1


def _axis_linear_fft_padding(
    length: int,
    psf_support: int,
    *,
    halo_multiplier: int = 1,
) -> tuple[int, int]:
    """
    Calculate symmetric linear-convolution padding for one axis.

    Parameters
    ----------
    length : int
        Input image length along the axis.
    psf_support : int
        PSF support length along the same axis.
    halo_multiplier : int, default=1
        Multiplier applied to the PSF half-width when constructing the
        reflected boundary halo.

    Returns
    -------
    tuple[int, int]
        Padding before and after the axis. The padding includes the PSF halo
        and any extra samples needed to reach an FFT-friendly length.
    """
    halo = max((int(psf_support) // 2) * int(halo_multiplier), 0)
    length_with_halo = length + 2 * halo
    new_length = next_gpu_fft_size(length_with_halo)
    fft_extra = new_length - length_with_halo
    pad_before = halo + fft_extra // 2
    pad_after = halo + fft_extra - fft_extra // 2
    return pad_before, pad_after


def pad_for_linear_fft(
    image: np.ndarray,
    psf_shape: tuple[int, int, int],
    pad_yx: bool = True,
) -> tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    """
    Pad a 3D image for linear FFT convolution with ``ndimage(..., mode="reflect")`` edges.

    Z is always padded by the PSF support. Y/X are padded by the PSF support and
    expanded to FFT-friendly sizes only when ``pad_yx`` is True.

    Parameters
    ----------
    image : numpy.ndarray
        3D input image in Z, Y, X order.
    psf_shape : tuple[int, int, int]
        PSF shape in Z, Y, X order.
    pad_yx : bool, default=True
        If True, pad and FFT-expand Y/X. If False, only pad Z.

    Returns
    -------
    tuple[numpy.ndarray, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]
        Padded image and the per-axis padding widths needed to remove padding
        after deconvolution.
    """
    if image.ndim != 3:
        raise ValueError(f"Expected 3D input, got shape {image.shape!r}")

    pad_z = _axis_linear_fft_padding(image.shape[0], psf_shape[0])
    if pad_yx:
        pad_y = _axis_linear_fft_padding(image.shape[1], psf_shape[1])
        pad_x = _axis_linear_fft_padding(image.shape[2], psf_shape[2])
    else:
        pad_y = (0, 0)
        pad_x = (0, 0)

    pad_width = (pad_z, pad_y, pad_x)
    padded_image = np.pad(image, pad_width, mode="symmetric")
    return padded_image, pad_width


def remove_padding_zyx(
    padded_image: cp.ndarray | np.ndarray,
    pad_width: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> cp.ndarray | np.ndarray:
    """
    Remove per-axis padding added by :func:`pad_for_linear_fft`.

    Parameters
    ----------
    padded_image : cupy.ndarray or numpy.ndarray
        Padded image in Z, Y, X order.
    pad_width : tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
        Per-axis padding widths returned by :func:`pad_for_linear_fft`.

    Returns
    -------
    cupy.ndarray or numpy.ndarray
        View of the input with padding removed.
    """
    slices = []
    for axis, (pad_before, pad_after) in enumerate(pad_width):
        start = pad_before
        stop = padded_image.shape[axis] - pad_after if pad_after > 0 else None
        slices.append(slice(start, stop))
    return padded_image[tuple(slices)]


def _symmetric_padded_axis_indices(
    length: int,
    pad_before: int,
    pad_after: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return source indices for a symmetric extension of one padded axis.

    Parameters
    ----------
    length : int
        Full padded axis length.
    pad_before : int
        Number of samples padded before the observed region.
    pad_after : int
        Number of samples padded after the observed region.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Source indices for the left and right padded regions.
    """
    observed = np.arange(pad_before, length - pad_after, dtype=np.int64)
    padded = np.pad(observed, (pad_before, pad_after), mode="symmetric")
    return padded[:pad_before], padded[length - pad_after :]


def enforce_symmetric_boundary(
    image: cp.ndarray,
    pad_width: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> None:
    """
    Constrain padded samples to be a symmetric extension of observed samples.

    Parameters
    ----------
    image : cupy.ndarray
        Padded 3D image in Z, Y, X order. The array is modified in place.
    pad_width : tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
        Per-axis padding widths returned by :func:`pad_for_linear_fft`.

    Returns
    -------
    None
    """
    for axis, (pad_before, pad_after) in enumerate(pad_width):
        if pad_before == 0 and pad_after == 0:
            continue
        left_indices, right_indices = _symmetric_padded_axis_indices(
            image.shape[axis],
            pad_before,
            pad_after,
        )
        if pad_before > 0:
            destination = [slice(None)] * image.ndim
            destination[axis] = slice(0, pad_before)
            image[tuple(destination)] = cp.take(
                image,
                cp.asarray(left_indices),
                axis=axis,
            )
        if pad_after > 0:
            destination = [slice(None)] * image.ndim
            destination[axis] = slice(image.shape[axis] - pad_after, None)
            image[tuple(destination)] = cp.take(
                image,
                cp.asarray(right_indices),
                axis=axis,
            )


def pad_psf(
    psf_temp: cp.ndarray,
    image_shape: tuple[int, int, int],
    normalize: bool = True,
) -> cp.ndarray:
    """
    Pad and center a PSF to match the target image shape.

    Parameters
    ----------
    psf_temp : cupy.ndarray
        Original PSF (Z, Y, X).
    image_shape : tuple of int
        Target shape (Z, Y, X).
    normalize : bool, default=True
        If True, normalize the padded PSF to unit sum. Set False only for
        diagnostics that intentionally preserve the input PSF scale.

    Returns
    -------
    cupy.ndarray
        Padded, centered, nonnegative PSF.
    """
    if psf_temp.ndim == 2:
        psf_temp = cp.expand_dims(psf_temp, axis=0)

    psf = cp.zeros(image_shape, dtype=cp.float32)
    psf[: psf_temp.shape[0], : psf_temp.shape[1], : psf_temp.shape[2]] = psf_temp

    # Center the PSF
    for axis, axis_size in enumerate(psf.shape):
        psf = cp.roll(psf, int(axis_size / 2), axis=axis)
    for axis, axis_size in enumerate(psf_temp.shape):
        psf = cp.roll(psf, -int(axis_size / 2), axis=axis)

    psf = cp.fft.ifftshift(psf)
    if normalize:
        s = cp.sum(psf)
        psf = psf / (s if s != 0 else 1.0)
    return psf.astype(cp.float32)


def fft_conv(
    image: cp.ndarray, H: cp.ndarray, shape: tuple[int, int, int]
) -> cp.ndarray:
    """
    Linear convolution via FFT with cached buffers (no clipping).

    This computes ``irfftn(rfftn(image) * H, s=shape)`` with preallocated
    work buffers. No clipping is applied here-this matches the reference
    implementation used to compute predictions, ratios and consensus.

    Parameters
    ----------
    image : cupy.ndarray
        Input array in object space.
    H : cupy.ndarray
        Frequency-domain transfer function (RFFTN of PSF or its conjugate).
    shape : tuple of int
        Target inverse FFT shape (Z, Y, X).

    Returns
    -------
    cupy.ndarray
        Convolved array in object space (float32).
    """
    device_id = int(cp.cuda.Device().id)
    cache_key = (device_id, *shape)
    if cache_key not in _fft_cache_3d:
        z, y, x = shape
        freq_shape = (z, y, x // 2 + 1)
        _fft_cache_3d[cache_key] = cp.empty(freq_shape, dtype=cp.complex64)

    fft_buf = _fft_cache_3d[cache_key]
    fft_buf[...] = cp.fft.rfftn(image)
    fft_buf[...] *= H
    return cp.fft.irfftn(fft_buf, s=shape).astype(cp.float32, copy=False)


def _observed_region_mask(
    shape: tuple[int, int, int],
    pad_width: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> cp.ndarray:
    """
    Build a mask that is one in the original image and zero in padding.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Full padded image shape.
    pad_width : tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
        Per-axis padding widths returned by :func:`pad_for_linear_fft`.

    Returns
    -------
    cupy.ndarray
        Float32 mask with observed voxels equal to one.
    """
    mask = cp.zeros(shape, dtype=cp.float32)
    slices = []
    for axis, (pad_before, pad_after) in enumerate(pad_width):
        start = pad_before
        stop = shape[axis] - pad_after if pad_after > 0 else None
        slices.append(slice(start, stop))
    mask[tuple(slices)] = 1
    return mask


def kl_div(p: cp.ndarray, q: cp.ndarray, mask: cp.ndarray | None = None) -> float:
    """
    Compute Kullback-Leibler divergence between two distributions.

    Parameters
    ----------
    p : cupy.ndarray
        First distribution (nonnegative).
    q : cupy.ndarray
        Second distribution (nonnegative).
    mask : cupy.ndarray or None, default=None
        Optional mask selecting the observed image region. Values outside the
        mask are excluded before normalization.

    Returns
    -------
    float
        Sum over all elements of ``p * (log(p) - log(q))``, with NaNs set to 0.
    """
    eps = cp.float32(1e-4)
    if mask is not None:
        p = (p + eps) * mask
        q = (q + eps) * mask
    else:
        p = p + eps
        q = q + eps
    p = p / cp.sum(p)
    q = q / cp.sum(q)
    kldiv = p * (cp.log(p) - cp.log(q))
    kldiv[cp.isnan(kldiv)] = 0
    return float(cp.sum(kldiv))


def _child_log_prefix(base_prefix: str, suffix: str) -> str:
    """
    Append one structured suffix to an RLGC log prefix.

    Parameters
    ----------
    base_prefix : str
        Existing structured log prefix, or an empty string.
    suffix : str
        Suffix to append as a separate token.

    Returns
    -------
    str
        Combined log prefix. If ``base_prefix`` is empty, returns ``suffix``.
    """
    return suffix if not base_prefix else f"{base_prefix} {suffix}"


def _resolve_tiled_axis_geometry(
    requested_crop: int,
    image_size: int,
    psf_support: int,
    axis_name: str,
) -> tuple[int, int]:
    """
    Resolve retained crop size and discarded processing halo for one axis.

    Parameters
    ----------
    requested_crop : int
        Requested retained crop size along the axis.
    image_size : int
        Full image size along the axis.
    psf_support : int
        PSF support length along the axis.
    axis_name : str
        Name used in validation error messages.

    Returns
    -------
    tuple[int, int]
        Retained tile size and hidden processing halo for the axis.
    """
    if requested_crop <= 0:
        raise ValueError(f"{axis_name} must be greater than 0 for tiled 3D RLGC.")

    retained_size = min(int(requested_crop), int(image_size))
    if retained_size >= image_size:
        return retained_size, 0

    tile_pad = int(psf_support)
    return retained_size, tile_pad


def _axis_retained_bounds(retained_size: int, image_size: int) -> list[tuple[int, int]]:
    """
    Build non-overlapping retained tile bounds that exactly cover one axis.

    Parameters
    ----------
    retained_size : int
        Number of output samples retained from each tile along the axis.
    image_size : int
        Full image size along the axis.

    Returns
    -------
    list[tuple[int, int]]
        Inclusive-exclusive retained bounds covering ``[0, image_size)``.
    """
    if retained_size <= 0:
        raise ValueError("retained_size must be greater than 0.")
    bounds = []
    start = 0
    while start < image_size:
        stop = min(start + retained_size, image_size)
        bounds.append((start, stop))
        start = stop
    return bounds


def rlgc(
    image: np.ndarray,
    psf: np.ndarray,
    gpu_id: int = 0,
    safe_mode: bool = True,
    limit: float = 0.01,
    max_delta: float = 0.001,
    pad_yx: bool = True,
    rng_seed: int | None = 42,
    normalize_psf: bool = True,
    release_memory: bool = True,
    logger: logging.Logger | None = None,
    log_prefix: str = "",
) -> np.ndarray:
    """
    Richardson-Lucy Gradient Consensus deconvolution.

    The implementation follows the non-accelerated reference loop with
    split-KLD stopping and the consensus-gated multiplicative update.

    Parameters
    ----------
    image : numpy.ndarray
        2D or 3D image to deconvolve. 2D input is treated as a single-z stack.
    psf : numpy.ndarray
        2D or 3D point-spread function. The PSF is padded to the processing
        shape, centered using the reference RLGC convention, and transformed on
        the GPU to form the forward and adjoint OTFs.
    gpu_id : int, default=0
        CUDA device ID to use.
    safe_mode : bool, default=True
        If True, stop when either split KLD increases. If False, stop only when
        both split KLDs increase.
    limit : float, default=0.01
        Minimum fraction of pixels updated per iteration before early stopping.
    max_delta : float, default=0.001
        Stop when the largest relative update falls below this threshold.
    pad_yx : bool, default=True
        If True, pad Y/X by PSF support and expand to FFT-friendly sizes. Z is
        always padded by PSF support. Padding is removed before returning.
    rng_seed : int or None, default=42
        Seed for the per-iteration 50:50 data split. Set to None for
        nondeterministic splits.
    normalize_psf : bool, default=True
        If True, normalize the padded PSF to unit sum before deconvolution.
        Set False only for diagnostics that intentionally preserve PSF scale.
    release_memory : bool, default=True
        If True, release CuPy memory pools and FFT caches before returning. On
        exceptions, cleanup is always forced.
    logger : logging.Logger or None, default=None
        Optional logger for per-iteration diagnostics.
    log_prefix : str, default=""
        Structured prefix prepended to emitted log lines.

    Returns
    -------
    numpy.ndarray
        Deconvolved image as float32. A 2D input returns with singleton z
        removed by the caller when routed through :func:`chunked_rlgc`.
    """
    cp.cuda.Device(gpu_id).use()
    logging_enabled = logger is not None and logger.isEnabledFor(logging.INFO)
    log_tag = f"{log_prefix} " if log_prefix else ""
    solver_start_time = timeit.default_timer() if logging_enabled else None
    cleanup_memory_pool = bool(release_memory)

    try:
        rng = cp.random.default_rng(rng_seed)

        if psf.ndim == 2:
            psf = np.expand_dims(psf, axis=0)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)

        image_gpu_np, pad_width = pad_for_linear_fft(
            image=image,
            psf_shape=tuple(int(v) for v in psf.shape),
            pad_yx=pad_yx,
        )
        image_gpu = cp.asarray(image_gpu_np, dtype=cp.float32)
        del image_gpu_np
        psf_gpu = pad_psf(
            cp.asarray(psf, dtype=cp.float32), image_gpu.shape, normalize=normalize_psf
        )

        otf = cp.fft.rfftn(psf_gpu)
        otfT = cp.conjugate(otf)
        otfotfT = otf * otfT
        del psf_gpu

        observed_mask = _observed_region_mask(image_gpu.shape, pad_width)
        observed_image = image_gpu * observed_mask
        update_norm = fft_conv(observed_mask, otfT, image_gpu.shape)
        update_norm = cp.maximum(update_norm, cp.float32(1e-6))

        num_z = image_gpu.shape[0]
        num_y = image_gpu.shape[1]
        num_x = image_gpu.shape[2]
        num_pixels = int(cp.sum(observed_mask))
        num_iters = 0
        prev_kld1 = np.inf
        prev_kld2 = np.inf

        recon = cp.mean(observed_image[observed_mask > 0]) * cp.ones(
            (num_z, num_y, num_x), dtype=cp.float32
        )
        previous_recon = recon

        if logging_enabled:
            logger.info(
                "%ssolver_started image_shape=%s padded_shape=%s psf_shape=%s reference_core=non_accelerated safe_mode=%s pad_yx=%s",
                log_tag,
                tuple(int(v) for v in image.shape),
                tuple(int(v) for v in image_gpu.shape),
                tuple(int(v) for v in psf.shape),
                safe_mode,
                pad_yx,
            )

        while True:
            iter_start_time = timeit.default_timer() if logging_enabled else None

            split1 = rng.binomial(observed_image.astype("int64"), p=0.5).astype(
                cp.float32
            )
            split2 = observed_image - split1

            Hu = fft_conv(recon, otf, image_gpu.shape)

            kldim = kl_div(Hu, observed_image, observed_mask)
            kld1 = kl_div(Hu, split1, observed_mask)
            kld2 = kl_div(Hu, split2, observed_mask)

            if safe_mode:
                should_restore = (kld1 > prev_kld1) or (kld2 > prev_kld2)
            else:
                should_restore = (kld1 > prev_kld1) and (kld2 > prev_kld2)
            if should_restore:
                recon[...] = previous_recon
                if logging_enabled:
                    logger.info(
                        "%sstop=restore_previous_recon best_iteration=%d elapsed_s=%.2f safe_mode=%s kld_image=%.6f kld_split1=%.6f prev_kld_split1=%.6f kld_split2=%.6f prev_kld_split2=%.6f",
                        log_tag,
                        max(num_iters - 1, 0),
                        timeit.default_timer() - solver_start_time,
                        safe_mode,
                        float(kldim),
                        float(kld1),
                        float(prev_kld1),
                        float(kld2),
                        float(prev_kld2),
                    )
                break

            prev_kld1 = kld1
            prev_kld2 = kld2

            HTratio1 = (
                fft_conv(
                    observed_mask
                    * cp.divide(split1, 0.5 * (Hu + 1e-12), dtype=cp.float32),
                    otfT,
                    image_gpu.shape,
                )
                / update_norm
            )
            del split1
            HTratio2 = (
                fft_conv(
                    observed_mask
                    * cp.divide(split2, 0.5 * (Hu + 1e-12), dtype=cp.float32),
                    otfT,
                    image_gpu.shape,
                )
                / update_norm
            )
            del split2
            HTratio = HTratio1 + HTratio2
            del Hu

            consensus_map = fft_conv(
                (HTratio1 - 1) * (HTratio2 - 1), otfotfT, recon.shape
            )

            previous_recon = recon
            recon = filter_update(recon, HTratio, consensus_map)
            enforce_symmetric_boundary(recon, pad_width)

            num_updated = cp.sum((consensus_map >= 0) * observed_mask)
            observed_recon = recon * observed_mask
            observed_previous = previous_recon * observed_mask
            recon_max = cp.maximum(cp.max(observed_recon), cp.float32(1e-12))
            updated_fraction = float(num_updated) / float(num_pixels)
            min_HTratio = cp.min(HTratio)
            max_HTratio = cp.max(HTratio)
            max_relative_delta = float(
                cp.max(cp.abs(observed_recon - observed_previous) / recon_max)
            )
            del observed_recon, observed_previous
            del HTratio

            num_iters += 1
            if logging_enabled:
                logger.info(
                    "%siteration=%03d elapsed_s=%.2f kld_image=%.6f kld_split1=%.6f kld_split2=%.6f update_min=%.3f update_max=%.3f updated_fraction=%.5f max_relative_delta=%.5f",
                    log_tag,
                    num_iters,
                    timeit.default_timer() - iter_start_time,
                    float(kldim),
                    float(kld1),
                    float(kld2),
                    float(min_HTratio),
                    float(max_HTratio),
                    updated_fraction,
                    max_relative_delta,
                )

            del HTratio1, HTratio2, consensus_map

            if updated_fraction < limit:
                if logging_enabled:
                    logger.info(
                        "%sstop=limit iteration=%03d updated_fraction=%.5f limit=%.5f",
                        log_tag,
                        num_iters,
                        updated_fraction,
                        limit,
                    )
                break

            if max_relative_delta < max_delta:
                if logging_enabled:
                    logger.info(
                        "%sstop=max_delta iteration=%03d max_relative_delta=%.5f threshold=%.5f",
                        log_tag,
                        num_iters,
                        max_relative_delta,
                        max_delta,
                    )
                break

        recon = remove_padding_zyx(recon, pad_width)
        recon_cpu = cp.asnumpy(recon).astype(np.float32)

        if logging_enabled:
            logger.info(
                "%ssolver_completed iterations=%d elapsed_s=%.2f output_shape=%s",
                log_tag,
                num_iters,
                timeit.default_timer() - solver_start_time,
                tuple(int(v) for v in recon_cpu.shape),
            )
        return recon_cpu
    except Exception:
        cleanup_memory_pool = True
        raise
    finally:
        gc.collect()
        if cleanup_memory_pool:
            cp.cuda.Stream.null.synchronize()
            clear_rlgc_caches(clear_memory_pool=True)


def _is_gpu_memory_error(exc: BaseException) -> bool:
    """
    Identify CUDA allocation failures that should trigger chunk fallback.

    Parameters
    ----------
    exc : BaseException
        Exception raised during an RLGC attempt.

    Returns
    -------
    bool
        True if the exception appears to be a GPU or host memory allocation
        failure; otherwise False.
    """
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, cp.cuda.memory.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "oom" in message


def _chunked_rlgc_once(
    image: np.ndarray,
    psf: np.ndarray,
    gpu_id: int = 0,
    crop_yx: int = 2048,
    crop_z: int | None = None,
    safe_mode: bool = True,
    limit: float = 0.01,
    max_delta: float = 0.001,
    rng_seed: int | None = 42,
    normalize_psf: bool = True,
    verbose: int = 0,
    release_memory: bool = True,
    logger: logging.Logger | None = None,
    log_prefix: str = "",
) -> np.ndarray:
    """
    Run one chunked RLGC attempt without GPU-memory fallback retries.

    This function performs either full-frame RLGC or lateral tiling for the
    requested crop size. It keeps the full z extent for every tile unless
    ``crop_z`` is explicitly supplied by an internal caller.

    Parameters
    ----------
    image : numpy.ndarray
        2D or 3D image to deconvolve.
    psf : numpy.ndarray
        2D or 3D point-spread function.
    gpu_id : int, default=0
        CUDA device ID to use.
    crop_yx : int, default=2048
        Retained lateral tile size for Y and X.
    crop_z : int or None, default=None
        Internal axial tile size. Public callers should leave this as None.
    safe_mode : bool, default=True
        If True, stop when either split KLD increases. If False, stop only when
        both split KLDs increase.
    limit : float, default=0.01
        Minimum fraction of pixels updated per iteration before early stopping.
    max_delta : float, default=0.001
        Stop when the largest relative update falls below this threshold.
    rng_seed : int or None, default=42
        Seed for the per-iteration 50:50 data split.
    normalize_psf : bool, default=True
        If True, normalize the padded PSF to unit sum before deconvolution.
    verbose : int, default=0
        If at least 1, show a progress bar over lateral tiles.
    release_memory : bool, default=True
        If True, release CuPy memory pools and FFT caches before returning.
    logger : logging.Logger or None, default=None
        Optional logger for route and per-iteration diagnostics.
    log_prefix : str, default=""
        Structured prefix prepended to emitted log lines.

    Returns
    -------
    numpy.ndarray
        Deconvolved image as float32.
    """
    cp.cuda.Device(gpu_id).use()
    if crop_yx <= 0:
        raise ValueError("crop_yx must be greater than 0.")
    if crop_z is not None and crop_z <= 0:
        raise ValueError("crop_z must be greater than 0 when provided.")

    image_arr = np.asarray(image)
    original_ndim = image_arr.ndim
    if original_ndim == 2:
        image_work = image_arr[np.newaxis, ...]
    elif original_ndim == 3:
        image_work = image_arr
    else:
        raise ValueError(f"Expected a 2D or 3D image, got shape {image_arr.shape}")

    psf_arr = np.asarray(psf)
    if psf_arr.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D PSF, got shape {psf_arr.shape}")
    psf_shape = psf_arr.shape if psf_arr.ndim == 3 else (1, *psf_arr.shape)

    effective_crop_z = image_work.shape[0] if crop_z is None else crop_z

    # Full-frame path if tiling not needed
    if (
        effective_crop_z >= image_work.shape[0]
        and crop_yx >= image_work.shape[-2]
        and crop_yx >= image_work.shape[-1]
    ):
        if logger is not None and logger.isEnabledFor(logging.INFO):
            logger.info(
                "%spath=3d_fullframe image_shape=%s psf_shape=%s",
                f"{log_prefix} " if log_prefix else "",
                tuple(int(v) for v in image_arr.shape),
                tuple(int(v) for v in psf_shape),
            )
        output = rlgc(
            image_arr,
            psf,
            gpu_id,
            safe_mode=safe_mode,
            limit=limit,
            max_delta=max_delta,
            pad_yx=True,
            rng_seed=rng_seed,
            normalize_psf=normalize_psf,
            release_memory=False,
            logger=logger,
            log_prefix=_child_log_prefix(log_prefix, "path=3d_fullframe"),
        )
        if original_ndim == 2 and output.ndim == 3 and output.shape[0] == 1:
            output = np.squeeze(output, axis=0)

    # Tiled 3D deconvolution with discarded processing halos. The halo is wider
    # than a single convolution radius because RLGC is iterative, so boundary
    # influence can propagate farther than one PSF half-width.
    else:
        full_shape = image_work.shape
        if effective_crop_z >= full_shape[0]:
            retained_z = full_shape[0]
            tile_pad_z = 0
        else:
            tile_pad_z = int(psf_shape[0])
            if effective_crop_z <= 2 * tile_pad_z:
                raise ValueError(
                    "crop_z must be larger than twice the axial processing halo "
                    f"({2 * tile_pad_z} for PSF support {int(psf_shape[0])})."
                )
            retained_z = int(effective_crop_z) - 2 * tile_pad_z
        retained_y, tile_pad_y = _resolve_tiled_axis_geometry(
            crop_yx,
            full_shape[1],
            int(psf_shape[-2]),
            "crop_yx",
        )
        retained_x, tile_pad_x = _resolve_tiled_axis_geometry(
            crop_yx,
            full_shape[2],
            int(psf_shape[-1]),
            "crop_yx",
        )
        output = np.zeros_like(image_work, dtype=np.float32)

        retained_bounds_z = _axis_retained_bounds(retained_z, full_shape[0])
        retained_bounds_y = _axis_retained_bounds(retained_y, full_shape[1])
        retained_bounds_x = _axis_retained_bounds(retained_x, full_shape[2])
        tile_bounds = [
            (z_bounds, y_bounds, x_bounds)
            for x_bounds in retained_bounds_x
            for y_bounds in retained_bounds_y
            for z_bounds in retained_bounds_z
        ]
        num_tiles = len(tile_bounds)

        if logger is not None and logger.isEnabledFor(logging.INFO):
            logger.info(
                "%spath=3d_tiled image_shape=%s psf_shape=%s retained_shape=%s processing_halo=%s num_tiles=%d",
                f"{log_prefix} " if log_prefix else "",
                tuple(int(v) for v in image_work.shape),
                tuple(int(v) for v in psf_shape),
                (retained_z, retained_y, retained_x),
                (tile_pad_z, tile_pad_y, tile_pad_x),
                num_tiles,
            )

        if verbose >= 1:
            from rich.progress import track

            iterator = track(
                enumerate(tile_bounds),
                description="Chunks",
                total=num_tiles,
                transient=True,
            )
        else:
            iterator = enumerate(tile_bounds)

        for tile_idx, (
            (z_dest_start, z_dest_stop),
            (y_dest_start, y_dest_stop),
            (x_dest_start, x_dest_stop),
        ) in iterator:
            z_crop_start = max(z_dest_start - tile_pad_z, 0)
            z_crop_stop = min(z_dest_stop + tile_pad_z, full_shape[0])
            y_crop_start = max(y_dest_start - tile_pad_y, 0)
            y_crop_stop = min(y_dest_stop + tile_pad_y, full_shape[1])
            x_crop_start = max(x_dest_start - tile_pad_x, 0)
            x_crop_stop = min(x_dest_stop + tile_pad_x, full_shape[2])

            crop = image_work[
                z_crop_start:z_crop_stop,
                y_crop_start:y_crop_stop,
                x_crop_start:x_crop_stop,
            ]
            crop_array = rlgc(
                crop,
                psf,
                gpu_id,
                safe_mode=safe_mode,
                limit=limit,
                max_delta=max_delta,
                rng_seed=None if rng_seed is None else rng_seed + tile_idx,
                normalize_psf=normalize_psf,
                release_memory=False,
                logger=logger,
                log_prefix=_child_log_prefix(
                    log_prefix, f"path=3d_tiled tile={tile_idx:04d}"
                ),
            )

            z_source_start = z_dest_start - z_crop_start
            z_source_stop = z_source_start + (z_dest_stop - z_dest_start)
            y_source_start = y_dest_start - y_crop_start
            y_source_stop = y_source_start + (y_dest_stop - y_dest_start)
            x_source_start = x_dest_start - x_crop_start
            x_source_stop = x_source_start + (x_dest_stop - x_dest_start)
            crop_sub = crop_array[
                z_source_start:z_source_stop,
                y_source_start:y_source_stop,
                x_source_start:x_source_stop,
            ]
            output[
                z_dest_start:z_dest_stop,
                y_dest_start:y_dest_stop,
                x_dest_start:x_dest_stop,
            ] = crop_sub

        del crop_sub
        gc.collect()

        if original_ndim == 2:
            output = np.squeeze(output, axis=0)

    if release_memory:
        clear_rlgc_caches(clear_memory_pool=True)

    return output


def chunked_rlgc(
    image: np.ndarray,
    psf: np.ndarray,
    gpu_id: int = 0,
    crop_yx: int = 2048,
    crop_z: int | None = None,
    safe_mode: bool = True,
    limit: float = 0.01,
    max_delta: float = 0.001,
    rng_seed: int | None = 42,
    normalize_psf: bool = True,
    verbose: int = 0,
    release_memory: bool = True,
    logger: logging.Logger | None = None,
    log_prefix: str = "",
    on_successful_crop_yx: Callable[[int], None] | None = None,
) -> np.ndarray:
    """
    Chunked RLGC deconvolution with automatic lateral chunk fallback.

    The solver first attempts the requested ``crop_yx``. If CUDA memory
    allocation fails, it retries with lateral chunks reduced by 128 pixels at a
    time. Axial chunking is not used by this fallback; Z is always processed as
    a full stack. If provided, ``on_successful_crop_yx`` is called with the
    crop size that completed successfully.

    Parameters
    ----------
    image : numpy.ndarray
        2D or 3D image to deconvolve.
    psf : numpy.ndarray
        2D or 3D point-spread function.
    gpu_id : int, default=0
        CUDA device ID to use.
    crop_yx : int, default=2048
        Requested retained lateral tile size. Values larger than the image are
        treated as full-frame processing.
    crop_z : None, default=None
        Axial chunking is no longer supported; this must remain None.
    safe_mode : bool, default=True
        If True, stop when either split KLD increases. If False, stop only when
        both split KLDs increase.
    limit : float, default=0.01
        Minimum fraction of pixels updated per iteration before early stopping.
    max_delta : float, default=0.001
        Stop when the largest relative update falls below this threshold.
    rng_seed : int or None, default=42
        Seed for the per-iteration 50:50 data split. Tiled calls offset this
        seed by tile index.
    normalize_psf : bool, default=True
        If True, normalize the padded PSF to unit sum before deconvolution.
    verbose : int, default=0
        If at least 1, show a progress bar over lateral tiles.
    release_memory : bool, default=True
        If True, release CuPy memory pools and FFT caches before returning.
    logger : logging.Logger or None, default=None
        Optional logger for route, retry, and per-iteration diagnostics.
    log_prefix : str, default=""
        Structured prefix prepended to emitted log lines.
    on_successful_crop_yx : Callable[[int], None] or None, default=None
        Optional callback receiving the crop size that completed successfully.

    Returns
    -------
    numpy.ndarray
        Deconvolved image as float32.
    """
    if crop_z is not None:
        raise ValueError("RLGC no longer supports axial chunking; leave crop_z=None.")

    image_arr = np.asarray(image)
    if image_arr.ndim == 2:
        max_yx = max(image_arr.shape)
    elif image_arr.ndim == 3:
        max_yx = max(image_arr.shape[-2:])
    else:
        raise ValueError(f"Expected a 2D or 3D image, got shape {image_arr.shape}")

    fallback_step = 128
    min_crop_yx = max(int(np.asarray(psf).shape[-2]), int(np.asarray(psf).shape[-1]))
    attempted_crop_yx = min(int(crop_yx), int(max_yx))

    while True:
        try:
            if logger is not None and logger.isEnabledFor(logging.INFO):
                logger.info(
                    "%sattempt_rlgc_crop_yx=%d requested_crop_yx=%d",
                    f"{log_prefix} " if log_prefix else "",
                    attempted_crop_yx,
                    crop_yx,
                )
            output = _chunked_rlgc_once(
                image=image_arr,
                psf=psf,
                gpu_id=gpu_id,
                crop_yx=attempted_crop_yx,
                crop_z=None,
                safe_mode=safe_mode,
                limit=limit,
                max_delta=max_delta,
                rng_seed=rng_seed,
                normalize_psf=normalize_psf,
                verbose=verbose,
                release_memory=release_memory,
                logger=logger,
                log_prefix=log_prefix,
            )
            if on_successful_crop_yx is not None:
                on_successful_crop_yx(attempted_crop_yx)
            if logger is not None and logger.isEnabledFor(logging.INFO):
                logger.info(
                    "%ssuccessful_rlgc_crop_yx=%d requested_crop_yx=%d",
                    f"{log_prefix} " if log_prefix else "",
                    attempted_crop_yx,
                    crop_yx,
                )
            return output
        except Exception as exc:
            if not _is_gpu_memory_error(exc):
                raise

            clear_rlgc_caches(clear_memory_pool=True)
            next_crop_yx = attempted_crop_yx - fallback_step
            if next_crop_yx < min_crop_yx:
                raise RuntimeError(
                    "RLGC failed due to GPU memory constraints even at the "
                    f"minimum lateral crop size {attempted_crop_yx}."
                ) from exc

            if logger is not None and logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "%sretry_after_gpu_oom previous_crop_yx=%d next_crop_yx=%d",
                    f"{log_prefix} " if log_prefix else "",
                    attempted_crop_yx,
                    next_crop_yx,
                )
            attempted_crop_yx = next_crop_yx
