"""
Richardson-Lucy Gradient Consensus (RLGC) deconvolution (Manton-style core).

Original idea for Gradient Consensus deconvolution:
James Manton and Andrew York, https://zenodo.org/records/10278919

Reference RLGC loop based on James Manton's implementation:
https://colab.research.google.com/drive/1mfVNSCaYHz1g56g92xBkIoa8190XNJpJ

Biggs-Andrews acceleration:
Biggs & Andrews, 1997, https://doi.org/10.1364/AO.36.001766
"""

import gc
import logging
import timeit

import cupy as cp
import numpy as np
from cupy import ElementwiseKernel

# -----------------------------------------------------------------------------
# CUDA kernel: multiplicative RL step gated by consensus (reference-accurate)
# -----------------------------------------------------------------------------
filter_update_ba = ElementwiseKernel(
    "float32 recon, float32 HTratio, float32 consensus_map",
    "float32 out",
    """
    bool skip = consensus_map < 0;
    out = skip ? recon : recon * HTratio;
    out = out < 0 ? 0 : out
    """,
    "filter_update_ba",
)

# -----------------------------------------------------------------------------
# FFT caches (performance)
# -----------------------------------------------------------------------------
_fft_cache_3d: dict[tuple[int, int, int, int], cp.ndarray] = {}


def clear_rlgc_caches(clear_memory_pool: bool = False) -> None:
    """Clear cached FFT buffers used by RLGC helper functions."""

    _fft_cache_3d.clear()
    if clear_memory_pool:
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()


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


def _axis_linear_fft_padding(length: int, psf_support: int) -> tuple[int, int]:
    """Return PSF halo plus FFT-friendly padding for one axis."""

    halo = max(int(psf_support) // 2, 0)
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
    """Remove per-axis padding added by :func:`pad_for_linear_fft`."""

    slices = []
    for axis, (pad_before, pad_after) in enumerate(pad_width):
        start = pad_before
        stop = padded_image.shape[axis] - pad_after if pad_after > 0 else None
        slices.append(slice(start, stop))
    return padded_image[tuple(slices)]


def pad_psf(psf_temp: cp.ndarray, image_shape: tuple[int, int, int]) -> cp.ndarray:
    """
    Pad and center a PSF to match the target image shape; normalize to unit sum.

    Parameters
    ----------
    psf_temp : cupy.ndarray
        Original PSF (Z, Y, X).
    image_shape : tuple of int
        Target shape (Z, Y, X).

    Returns
    -------
    cupy.ndarray
        Padded, centered, nonnegative PSF normalized to unit sum.
    """
    psf = cp.zeros(image_shape, dtype=cp.float32)
    psf[: psf_temp.shape[0], : psf_temp.shape[1], : psf_temp.shape[2]] = psf_temp

    # Center the PSF
    for axis, axis_size in enumerate(psf.shape):
        psf = cp.roll(psf, int(axis_size / 2), axis=axis)
    for axis, axis_size in enumerate(psf_temp.shape):
        psf = cp.roll(psf, -int(axis_size / 2), axis=axis)

    psf = cp.fft.ifftshift(psf)
    s = cp.sum(psf)
    psf = psf / (s if s != 0 else 1.0)
    return psf.astype(cp.float32)


def fft_conv(
    image: cp.ndarray, H: cp.ndarray, shape: tuple[int, int, int]
) -> cp.ndarray:
    """
    Linear convolution via FFT with cached buffers (no clipping).

    This computes ``irfftn(rfftn(image) * H, s=shape)`` with preallocated
    work buffers. No clipping is applied here—this matches the reference
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


def _normalize_psf_to_2d(psf: np.ndarray) -> np.ndarray:
    """Convert a PSF to a normalized 2D kernel for 2D batched RLGC."""

    psf_arr = np.asarray(psf, dtype=np.float32)
    if psf_arr.ndim == 3:
        # 2D acquisition stores one effective PSF per z-plane. Use center plane.
        psf_arr = psf_arr[psf_arr.shape[0] // 2]
    if psf_arr.ndim != 2:
        raise ValueError(f"Expected a 2D or 3D PSF, got shape {psf_arr.shape}")

    psf_sum = float(np.sum(psf_arr))
    if psf_sum <= 0:
        raise ValueError("2D PSF must have positive total intensity.")
    return (psf_arr / psf_sum).astype(np.float32, copy=False)


def kl_div(p: cp.ndarray, q: cp.ndarray) -> float:
    """
    Compute Kullback-Leibler divergence between two distributions.

    Parameters
    ----------
    p : cupy.ndarray
        First distribution (nonnegative).
    q : cupy.ndarray
        Second distribution (nonnegative).

    Returns
    -------
    float
        Sum over all elements of ``p * (log(p) - log(q))``, with NaNs set to 0.
    """
    p = p + 1e-4
    q = q + 1e-4
    p = p / cp.sum(p)
    q = q / cp.sum(q)
    kldiv = p * (cp.log(p) - cp.log(q))
    kldiv[cp.isnan(kldiv)] = 0
    kldiv = cp.sum(kldiv)
    return kldiv


def _median_init_scalar(image: np.ndarray) -> float:
    """Return a scalar median-based initializer clamped to at least 1."""

    median_value = float(np.median(image))
    return 1.0 if median_value < 1.0 else median_value


def _median_init_planes(image: np.ndarray) -> np.ndarray:
    """Return per-plane median initializers clamped to at least 1."""

    plane_medians = np.asarray(np.median(image, axis=(-2, -1)), dtype=np.float32)
    return np.maximum(plane_medians, 1.0, dtype=np.float32)


def _child_log_prefix(base_prefix: str, suffix: str) -> str:
    """Append one structured suffix to an RLGC log prefix."""

    return suffix if not base_prefix else f"{base_prefix} {suffix}"


def _resolve_tiled_axis_geometry(
    requested_crop: int,
    image_size: int,
    psf_support: int,
    axis_name: str,
) -> tuple[int, int]:
    """Resolve retained crop size and discarded processing halo for one axis."""

    if requested_crop <= 0:
        raise ValueError(f"{axis_name} must be greater than 0 for tiled 3D RLGC.")

    retained_size = min(int(requested_crop), int(image_size))
    if retained_size >= image_size:
        return retained_size, 0

    tile_pad = int(psf_support)
    return retained_size, tile_pad


def _axis_retained_bounds(retained_size: int, image_size: int) -> list[tuple[int, int]]:
    """Return non-overlapping retained tile bounds that exactly cover one axis."""

    if retained_size <= 0:
        raise ValueError("retained_size must be greater than 0.")
    bounds = []
    start = 0
    while start < image_size:
        stop = min(start + retained_size, image_size)
        bounds.append((start, stop))
        start = stop
    return bounds


def rlgc_biggs_ba(
    image: np.ndarray,
    psf: np.ndarray,
    gpu_id: int = 0,
    safe_mode: bool = True,
    auto_delta_scale: float = 5.0,
    init_value: float | None = None,
    limit: float = 0.01,
    max_delta: float = 0.001,
    pad_yx: bool = True,
    release_memory: bool = True,
    logger: logging.Logger | None = None,
    log_prefix: str = "",
) -> np.ndarray:
    """
    Biggs-Andrews accelerated Richardson-Lucy Gradient Consensus.

    Parameters
    ----------
    image : numpy.ndarray
        3D image (Z, Y, X) to be deconvolved.
    psf : numpy.ndarray
        3D point-spread function. This PSF is padded and transformed on the GPU
        to form the forward and adjoint OTFs internally.
    gpu_id : int, default=0
        Which GPU to use.
    safe_mode : bool, default=True
        If True, stop when EITHER split KLD increases (play-it-safe).
        If False, stop only when BOTH split KLDs increase.
    auto_delta_scale : float, default=5.0
        Scale factor in the automatic small-update stop threshold
        ``auto_delta_scale / max(image)``. Smaller values make this stop
        criterion more permissive.
    init_value : float or None, default=None
        Constant initializer value for the reconstruction. If None, use the
        reference RLGC initializer ``mean(image)`` for each solver call.
    limit : float, default=0.01
        Minimum fraction of pixels that must be updated per iteration before
        early stopping is triggered.
    max_delta : float, default=0.001
        Maximum allowed relative update magnitude before early stopping is
        triggered.
    pad_yx : bool, default=True
        If True, pad Y/X by the PSF support and expand them to FFT-friendly
        sizes. Padding uses the same edge convention as
        ``ndimage(..., mode="reflect")``. If False, Y/X are left unpadded while
        Z is still padded by the PSF support.
    release_memory : bool, default=True
        If True, release GPU memory pools after each call. Set False when
        calling in tight loops to avoid allocator thrashing.
    logger : logging.Logger or None, default=None
        Optional logger for per-iteration RLGC diagnostics.
    log_prefix : str, default=""
        Structured prefix prepended to every emitted log line.

    Returns
    -------
    numpy.ndarray
        Deconvolved 3D image (float32).
    """

    cp.cuda.Device(gpu_id).use()
    if auto_delta_scale <= 0:
        raise ValueError("auto_delta_scale must be positive.")
    rng = cp.random.default_rng(42)
    logging_enabled = logger is not None and logger.isEnabledFor(logging.INFO)
    log_tag = f"{log_prefix} " if log_prefix else ""
    solver_start_time = timeit.default_timer() if logging_enabled else None

    # Ensure 3D inputs
    if psf.ndim == 2:
        psf = np.expand_dims(psf, axis=0)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)

    if logging_enabled:
        logger.info(
            "%ssolver_started image_shape=%s psf_shape=%s pad_yx=%s safe_mode=%s limit=%.4f max_delta=%.4f auto_delta_scale=%.3f",
            log_tag,
            tuple(int(v) for v in image.shape),
            tuple(int(v) for v in psf.shape),
            pad_yx,
            safe_mode,
            limit,
            max_delta,
            auto_delta_scale,
        )

    image_gpu_np, pad_width = pad_for_linear_fft(
        image=image,
        psf_shape=tuple(int(v) for v in psf.shape),
        pad_yx=pad_yx,
    )
    image_gpu = cp.asarray(image_gpu_np, dtype=cp.float32)
    del image_gpu_np

    # OTFs
    psf_gpu = pad_psf(cp.asarray(psf, dtype=cp.float32), image_gpu.shape)
    otf = cp.fft.rfftn(psf_gpu)
    otfT = cp.conjugate(otf)
    del psf_gpu
    cp.get_default_memory_pool().free_all_blocks()

    otfotfT = otf * otfT  # H^T H in frequency domain
    shape = image_gpu.shape
    z, y, x = shape
    num_pixels = z * y * x
    image_peak = cp.maximum(cp.max(image_gpu), cp.float32(1.0))

    resolved_init_value = (
        cp.mean(image_gpu) if init_value is None else cp.float32(init_value)
    )
    recon = cp.full((z, y, x), resolved_init_value, dtype=cp.float32)
    previous_recon = recon.copy()

    # Pre-allocations
    recon_next = cp.empty_like(recon)
    Hu = cp.empty_like(recon)

    # Biggs-Andrews state
    g1 = cp.zeros_like(recon)
    g2 = cp.zeros_like(recon)

    prev_kld1 = np.inf
    prev_kld2 = np.inf
    num_iters = 0

    while True:
        iter_start_time = timeit.default_timer() if logging_enabled else None

        # BA momentum: y = recon + alpha * (recon - previous_recon)
        if num_iters >= 1:
            numerator = cp.sum(g1 * g2)
            denominator = cp.sum(g2 * g2)
            alpha = numerator / denominator
            alpha = cp.clip(alpha, 0.0, 1.0)
            if cp.isnan(alpha):
                alpha = 0.0
            alpha = float(alpha)
        else:
            alpha = 0.0

        y_vec = recon + alpha * (recon - previous_recon)

        # Forward prediction
        Hu[...] = fft_conv(y_vec, otf, shape)

        # 50:50 split of the data (counts)
        split1 = rng.binomial(image_gpu.astype(cp.int64), p=0.5).astype(cp.float32)
        split2 = image_gpu - split1

        # KLDs & stopping
        kld1 = kl_div(Hu, split1)
        kld2 = kl_div(Hu, split2)

        if safe_mode:
            if (kld1 > prev_kld1) or (kld2 > prev_kld2):
                recon[...] = previous_recon
                if logging_enabled:
                    logger.info(
                        "%sstop=restore_previous_recon best_iteration=%d elapsed_s=%.2f safe_mode=%s kld_split1=%.6f prev_kld_split1=%.6f kld_split2=%.6f prev_kld_split2=%.6f",
                        log_tag,
                        max(num_iters - 1, 0),
                        timeit.default_timer() - solver_start_time,
                        safe_mode,
                        float(kld1),
                        float(prev_kld1),
                        float(kld2),
                        float(prev_kld2),
                    )
                break
        else:
            if (kld1 > prev_kld1) and (kld2 > prev_kld2):
                recon[...] = previous_recon
                if logging_enabled:
                    logger.info(
                        "%sstop=restore_previous_recon best_iteration=%d elapsed_s=%.2f safe_mode=%s kld_split1=%.6f prev_kld_split1=%.6f kld_split2=%.6f prev_kld_split2=%.6f",
                        log_tag,
                        max(num_iters - 1, 0),
                        timeit.default_timer() - solver_start_time,
                        safe_mode,
                        float(kld1),
                        float(prev_kld1),
                        float(kld2),
                        float(prev_kld2),
                    )
                break

        prev_kld1 = kld1
        prev_kld2 = kld2

        # BA bookkeeping: move recon forward to y
        previous_recon[...] = recon
        recon[...] = y_vec

        # RL ratios: H^T( split / (0.5*(Hu+eps)) )
        eps = 1e-12
        ratio_denom = 0.5 * (Hu + eps)
        HTratio1 = fft_conv(
            cp.divide(split1, ratio_denom, dtype=cp.float32), otfT, shape
        )
        HTratio2 = fft_conv(
            cp.divide(split2, ratio_denom, dtype=cp.float32), otfT, shape
        )
        HTratio = HTratio1 + HTratio2

        # Consensus: H^T H * ((HTratio1 - 1)*(HTratio2 - 1))
        consensus_map = fft_conv(
            (HTratio1 - 1.0) * (HTratio2 - 1.0), otfotfT, recon.shape
        )

        # Gated multiplicative update
        filter_update_ba(recon, HTratio, consensus_map, recon_next)
        recon_next = cp.nan_to_num(recon_next, nan=0.0, posinf=0.0, neginf=0.0)

        # BA vectors
        g2[...] = g1
        g1[...] = recon_next - y_vec

        recon[...] = recon_next

        num_updated = num_pixels - cp.sum(consensus_map < 0)
        recon_max = cp.maximum(cp.max(recon), eps)
        updated_fraction = float(num_updated) / float(num_pixels)
        max_relative_delta = float(cp.max(cp.abs(recon - previous_recon) / recon_max))
        auto_delta_threshold = float(auto_delta_scale / image_peak)

        num_iters += 1
        if logging_enabled:
            logger.info(
                "%siteration=%03d elapsed_s=%.2f kld_image=%.6f kld_split1=%.6f kld_split2=%.6f update_min=%.3f update_max=%.3f updated_fraction=%.5f max_relative_delta=%.5f",
                log_tag,
                num_iters,
                timeit.default_timer() - iter_start_time,
                float(kl_div(Hu, image_gpu)),
                float(kld1),
                float(kld2),
                float(cp.min(HTratio)),
                float(cp.max(HTratio)),
                updated_fraction,
                max_relative_delta,
            )

        # Cleanup per-iter temporaries
        del split1, split2, HTratio1, HTratio2, HTratio, consensus_map

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

        # (2) Hit max delta: updates have become very small
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

        # (3) Auto delta: updates small relative to overall image intensity
        if max_relative_delta < auto_delta_threshold:
            if logging_enabled:
                logger.info(
                    "%sstop=auto_delta iteration=%03d max_relative_delta=%.5f threshold=%.5f",
                    log_tag,
                    num_iters,
                    max_relative_delta,
                    auto_delta_threshold,
                )
            break

    # Enforce nonnegativity and unpad back to original
    recon = cp.maximum(recon, 0.0)
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

    # Cleanup
    del g1, g2, recon, previous_recon, Hu, otf, otfT, otfotfT, image_gpu
    gc.collect()
    if release_memory:
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    return recon_cpu


def rlgc_biggs_ba_2d_batched(
    image: np.ndarray,
    psf: np.ndarray,
    gpu_id: int = 0,
    safe_mode: bool = True,
    auto_delta_scale: float = 5.0,
    init_value: float | np.ndarray | None = None,
    limit: float = 0.01,
    max_delta: float = 0.001,
    pad_yx: bool = True,
    release_memory: bool = True,
    logger: logging.Logger | None = None,
    log_prefix: str = "",
) -> np.ndarray:
    """
    Quality-preserving 2D fast path over z-planes.

    This keeps the original per-plane RLGC behavior while reducing overhead by
    routing through one function call for a full z-stack and deferring GPU
    memory-pool release to outer scopes.
    """

    image_arr = np.asarray(image)

    if image_arr.ndim == 2:
        init_arr = None if init_value is None else np.asarray(init_value)
        plane_init_value = None
        if init_arr is not None:
            plane_init_value = (
                float(init_arr.ravel()[0]) if init_arr.ndim > 0 else float(init_arr)
            )
        return rlgc_biggs_ba(
            image=image_arr,
            psf=_normalize_psf_to_2d(psf),
            gpu_id=gpu_id,
            safe_mode=safe_mode,
            auto_delta_scale=auto_delta_scale,
            init_value=plane_init_value,
            limit=limit,
            max_delta=max_delta,
            pad_yx=pad_yx,
            release_memory=release_memory,
            logger=logger,
            log_prefix=log_prefix,
        )
    if image_arr.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D image, got shape {image_arr.shape}")

    psf_2d = _normalize_psf_to_2d(psf)
    if logger is not None and logger.isEnabledFor(logging.INFO):
        logger.info(
            "%sbatch_mode=2d planes=%d image_shape=%s psf_shape=%s",
            f"{log_prefix} " if log_prefix else "",
            int(image_arr.shape[0]),
            tuple(int(v) for v in image_arr.shape),
            tuple(int(v) for v in psf_2d.shape),
        )
    if init_value is None:
        plane_init_values = [None] * image_arr.shape[0]
    else:
        init_arr = np.asarray(init_value)
        if init_arr.ndim == 0:
            plane_init_values = np.full(
                image_arr.shape[0], float(init_arr), dtype=np.float32
            )
        else:
            plane_init_values = np.asarray(init_arr, dtype=np.float32).reshape(-1)
            if plane_init_values.size != image_arr.shape[0]:
                raise ValueError(
                    "Batched 2D RLGC expected one init value per z plane or a scalar; "
                    f"got shape {init_arr.shape} for {image_arr.shape[0]} planes."
                )
    output = np.zeros_like(image_arr, dtype=np.float32)
    for z_idx in range(image_arr.shape[0]):
        plane_init_value = plane_init_values[z_idx]
        output[z_idx] = rlgc_biggs_ba(
            image=image_arr[z_idx],
            psf=psf_2d,
            gpu_id=gpu_id,
            safe_mode=safe_mode,
            auto_delta_scale=auto_delta_scale,
            init_value=(None if plane_init_value is None else float(plane_init_value)),
            limit=limit,
            max_delta=max_delta,
            pad_yx=pad_yx,
            release_memory=False,
            logger=logger,
            log_prefix=_child_log_prefix(log_prefix, f"z={z_idx:04d}"),
        )

    if release_memory:
        clear_rlgc_caches(clear_memory_pool=True)
    return output


def chunked_rlgc(
    image: np.ndarray,
    psf: np.ndarray,
    gpu_id: int = 0,
    crop_yx: int = 1500,
    crop_z: int | None = None,
    safe_mode: bool = True,
    auto_delta_scale: float = 5.0,
    limit: float = 0.01,
    max_delta: float = 0.001,
    verbose: int = 0,
    release_memory: bool = True,
    logger: logging.Logger | None = None,
    log_prefix: str = "",
) -> np.ndarray:
    """
    Chunked RLGC deconvolution with hidden-halo stitching.

    Parameters
    ----------
    image : numpy.ndarray
        2D or 3D image to be deconvolved. A 2D PSF or single-z 3D PSF routes
        to the batched 2D path; a full 3D PSF routes to full-frame or tiled 3D.
    psf : numpy.ndarray
        Point-spread function (PSF) to use for deconvolution.
    gpu_id : int, default=0
        Which GPU to use.
    crop_yx : int, default=1500
        Retained tile size in Y and X. A discarded processing halo is added
        internally around each lateral tile before deconvolution.
    crop_z : int or None, default=None
        Total tile size in Z for chunked 3D deconvolution, including the
        discarded processing halo used by each z tile. If ``None``, the tiled
        3D path uses the full z-extent and only subdivides laterally. This is
        ignored for the 2D batched path.
    safe_mode : bool, default=True
        RLGC stopping: play-it-safe if True.
    auto_delta_scale : float, default=5.0
        Scale factor in the automatic small-update stop threshold
        ``auto_delta_scale / max(image)``. Smaller values make this stop
        criterion more permissive.
    limit : float, default=0.01
        Minimum fraction of pixels that must be updated per iteration before
        early stopping is triggered.
    max_delta : float, default=0.001
        Maximum allowed relative update magnitude before early stopping is
        triggered.
    verbose : int, default=0
        If ≥ 1, show a progress bar over subtiles.
    release_memory : bool, default=True
        If True, clear RLGC caches and release memory pools on exit.
    logger : logging.Logger or None, default=None
        Optional logger for RLGC route and per-iteration diagnostics.
    log_prefix : str, default=""
        Structured prefix prepended to every emitted log line.

    Returns
    -------
    numpy.ndarray
        Deconvolved image (float32).
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
    is_2d_psf = psf_arr.ndim == 2 or (psf_arr.ndim == 3 and psf_arr.shape[0] == 1)

    if original_ndim == 2 and not is_2d_psf:
        raise ValueError("2D RLGC requires a 2D PSF or a single-z 3D PSF.")

    if is_2d_psf:
        if logger is not None and logger.isEnabledFor(logging.INFO):
            logger.info(
                "%spath=2d_batched image_shape=%s psf_shape=%s",
                f"{log_prefix} " if log_prefix else "",
                tuple(int(v) for v in image_arr.shape),
                tuple(int(v) for v in np.asarray(psf).shape),
            )
        output = rlgc_biggs_ba_2d_batched(
            image_arr,
            psf,
            gpu_id,
            safe_mode=safe_mode,
            auto_delta_scale=auto_delta_scale,
            limit=limit,
            max_delta=max_delta,
            pad_yx=True,
            release_memory=False,
            logger=logger,
            log_prefix=_child_log_prefix(log_prefix, "path=2d_batched"),
        )
        if original_ndim == 2 and output.ndim == 3 and output.shape[0] == 1:
            output = np.squeeze(output, axis=0)
        if release_memory:
            clear_rlgc_caches(clear_memory_pool=True)
        return output

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
                tuple(int(v) for v in psf_arr.shape),
            )
        output = rlgc_biggs_ba(
            image_arr,
            psf,
            gpu_id,
            safe_mode=safe_mode,
            auto_delta_scale=auto_delta_scale,
            limit=limit,
            max_delta=max_delta,
            pad_yx=True,
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
            tile_pad_z = int(psf_arr.shape[0])
            if effective_crop_z <= 2 * tile_pad_z:
                raise ValueError(
                    "crop_z must be larger than twice the axial processing halo "
                    f"({2 * tile_pad_z} for PSF support {int(psf_arr.shape[0])})."
                )
            retained_z = int(effective_crop_z) - 2 * tile_pad_z
        retained_y, tile_pad_y = _resolve_tiled_axis_geometry(
            crop_yx,
            full_shape[1],
            int(psf_arr.shape[-2]),
            "crop_yx",
        )
        retained_x, tile_pad_x = _resolve_tiled_axis_geometry(
            crop_yx,
            full_shape[2],
            int(psf_arr.shape[-1]),
            "crop_yx",
        )
        init_value = float(np.mean(image_work))
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
                tuple(int(v) for v in psf_arr.shape),
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
            crop_array = rlgc_biggs_ba(
                crop,
                psf,
                gpu_id,
                safe_mode=safe_mode,
                auto_delta_scale=auto_delta_scale,
                init_value=init_value,
                limit=limit,
                max_delta=max_delta,
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
