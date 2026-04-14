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
import timeit

import cupy as cp
import numpy as np
from cupy import ElementwiseKernel
from ryomen import Slicer

DEBUG = False

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


def make_feather_weight(
    shape: tuple[int, int, int],
    feather_px: int = 64,
    taper_top: bool = True,
    taper_bottom: bool = True,
    taper_left: bool = True,
    taper_right: bool = True,
) -> np.ndarray:
    """
    Create a feathered weight mask using directional cosine tapers on Y/X only.

    Z is uniform. Feather taper width is explicitly specified in pixels. Each
    lateral edge can be tapered independently so outer image boundaries can stay
    untapered while interior tile overlaps are blended.

    Parameters
    ----------
    shape : tuple of int
        Crop shape as (z, y, x).
    feather_px : int
        Number of pixels to taper at each Y/X edge.
    taper_top, taper_bottom, taper_left, taper_right : bool, default=True
        Whether to apply the taper on the corresponding edge.

    Returns
    -------
    numpy.ndarray
        Feather mask of shape (z, y, x), values in [0, 1].
    """

    def cosine_taper(
        length: int,
        feather: int,
        taper_low: bool,
        taper_high: bool,
    ) -> np.ndarray:
        window = np.ones(length, dtype=np.float32)
        if feather > 0:
            feather = min(feather, length)
            ramp = 0.5 * (
                1 - np.cos(np.linspace(0, np.pi, feather + 2, dtype=np.float32)[1:-1])
            )
            if taper_low:
                window[:feather] = np.minimum(window[:feather], ramp)
            if taper_high:
                window[-feather:] = np.minimum(window[-feather:], ramp[::-1])
        return window

    y_win = cosine_taper(shape[1], feather_px, taper_top, taper_bottom)
    x_win = cosine_taper(shape[2], feather_px, taper_left, taper_right)
    weight2d = np.outer(y_win, x_win).astype(np.float32)
    weight = np.broadcast_to(weight2d[None, :, :], shape)
    return weight


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
    return psf_arr


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


def _resolve_blend_overlap_yx(
    psf_shape: tuple[int, ...],
    crop_yx: int,
) -> int:
    """
    Derive a lateral feather overlap from the PSF support.

    The retained overlap is set to the full lateral PSF support width minus the
    center pixel, which corresponds to twice the lateral PSF half-width for odd
    support sizes.
    """

    if crop_yx <= 1:
        raise ValueError("crop_yx must be greater than 1 for tiled 3D RLGC.")

    lateral_overlap = max(int(psf_shape[-2]) - 1, int(psf_shape[-1]) - 1, 1)
    if lateral_overlap >= crop_yx:
        raise ValueError(
            "crop_yx must be larger than the PSF-derived lateral overlap "
            f"({lateral_overlap} for PSF shape {psf_shape[-2:]!r})."
        )
    return lateral_overlap


def rlgc_biggs_ba(
    image: np.ndarray,
    psf: np.ndarray,
    gpu_id: int = 0,
    safe_mode: bool = True,
    auto_delta_scale: float = 5.0,
    init_value: float = 1,
    limit: float = 0.05,
    max_delta: float = 0.001,
    pad_yx: bool = True,
    release_memory: bool = True,
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
    init_value : float, default=1
        Constant initializer value for the reconstruction.
    limit : float, default=0.2
        Minimum fraction of pixels that must be updated per iteration before
        early stopping is triggered.
    max_delta : float, default=0.02
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

    Returns
    -------
    numpy.ndarray
        Deconvolved 3D image (float32).
    """

    cp.cuda.Device(gpu_id).use()
    if auto_delta_scale <= 0:
        raise ValueError("auto_delta_scale must be positive.")
    rng = cp.random.default_rng(42)

    # Ensure 3D inputs
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

    recon = cp.full((z, y, x), cp.float32(init_value), dtype=cp.float32)
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
    if DEBUG:
        start_time = timeit.default_timer()

    while True:
        if DEBUG:
            iter_start_time = timeit.default_timer()

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
        kldim = kl_div(Hu, image_gpu)
        kld1 = kl_div(Hu, split1)
        kld2 = kl_div(Hu, split2)

        if safe_mode:
            if (
                (kld1 > prev_kld1)
                or (kld2 > prev_kld2)
                or (kld1 < 1e-4)
                or (kld2 < 1e-4)
            ):
                recon[...] = previous_recon
                if DEBUG:
                    total_time = timeit.default_timer() - start_time
                    print(f"Optimum after {num_iters - 1} iters in {total_time:.1f} s.")
                break
        else:
            if (
                ((kld1 > prev_kld1) and (kld2 > prev_kld2))
                or (kld1 < 1e-4)
                or (kld2 < 1e-4)
            ):
                recon[...] = previous_recon
                if DEBUG:
                    total_time = timeit.default_timer() - start_time
                    print(f"Optimum after {num_iters - 1} iters in {total_time:.1f} s.")
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
        HTratio = 0.5 * (HTratio1 + HTratio2)

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
        max_relative_delta = cp.max(cp.abs(recon - previous_recon) / recon_max)

        num_iters += 1
        if DEBUG:
            calc_time = timeit.default_timer() - iter_start_time
            min_HTratio = cp.min(HTratio)
            max_HTratio = cp.max(HTratio)
            max_relative_delta = cp.max((recon - previous_recon) / cp.max(recon))
            print(
                f"Iteration {num_iters:03d} completed in {calc_time:.2f}s. "
                f"KLDs: {kldim:.6f} (image), {kld1:.6f} (split1), {kld2:.6f} (split2)."
                f"Update range : {min_HTratio:.3f} to {max_HTratio:.3f}."
                f"Largest relative delta = {max_relative_delta:.5f}."
            )
        # Cleanup per-iter temporaries
        del split1, split2, HTratio1, HTratio2, HTratio, consensus_map

        if num_updated / num_pixels < limit:
            if DEBUG:
                print("Hit limit")
            break

        # (2) Hit max delta: updates have become very small
        if max_relative_delta < max_delta:
            if DEBUG:
                print("Hit max delta")
            break

        # (3) Auto delta: updates small relative to overall image intensity
        if max_relative_delta < auto_delta_scale / image_peak:
            if DEBUG:
                print("Hit auto delta")
            break

    # Enforce nonnegativity and unpad back to original
    recon = cp.maximum(recon, 0.0)
    recon = remove_padding_zyx(recon, pad_width)
    recon_cpu = cp.asnumpy(recon).astype(np.float32)

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
    init_value: float | np.ndarray = 1,
    limit: float = 0.05,
    max_delta: float = 0.001,
    pad_yx: bool = True,
    release_memory: bool = True,
) -> np.ndarray:
    """
    Quality-preserving 2D fast path over z-planes.

    This keeps the original per-plane RLGC behavior while reducing overhead by
    routing through one function call for a full z-stack and deferring GPU
    memory-pool release to outer scopes.
    """

    image_arr = np.asarray(image)
    init_arr = np.asarray(init_value)

    if image_arr.ndim == 2:
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
        )
    if image_arr.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D image, got shape {image_arr.shape}")

    psf_2d = _normalize_psf_to_2d(psf)
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
        output[z_idx] = rlgc_biggs_ba(
            image=image_arr[z_idx],
            psf=psf_2d,
            gpu_id=gpu_id,
            safe_mode=safe_mode,
            auto_delta_scale=auto_delta_scale,
            init_value=float(plane_init_values[z_idx]),
            limit=limit,
            max_delta=max_delta,
            pad_yx=pad_yx,
            release_memory=False,
        )

    if release_memory:
        clear_rlgc_caches(clear_memory_pool=True)
    return output


def chunked_rlgc(
    image: np.ndarray,
    psf: np.ndarray,
    gpu_id: int = 0,
    crop_yx: int = 1500,
    safe_mode: bool = True,
    auto_delta_scale: float = 5.0,
    verbose: int = 0,
    release_memory: bool = True,
) -> np.ndarray:
    """
    Chunked RLGC deconvolution with feathered blending.

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
        Tile size in Y and X. The feathered retained overlap is derived
        internally from the lateral PSF support.
    safe_mode : bool, default=True
        RLGC stopping: play-it-safe if True.
    auto_delta_scale : float, default=5.0
        Scale factor in the automatic small-update stop threshold
        ``auto_delta_scale / max(image)``. Smaller values make this stop
        criterion more permissive.
    verbose : int, default=0
        If ≥ 1, show a progress bar over subtiles.
    release_memory : bool, default=True
        If True, clear RLGC caches and release memory pools on exit.

    Returns
    -------
    numpy.ndarray
        Deconvolved image (float32).
    """
    cp.cuda.Device(gpu_id).use()
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
        init_value = (
            _median_init_planes(image_arr)
            if image_arr.ndim == 3
            else _median_init_scalar(image_arr)
        )
        output = rlgc_biggs_ba_2d_batched(
            image_arr,
            psf,
            gpu_id,
            safe_mode=safe_mode,
            auto_delta_scale=auto_delta_scale,
            init_value=init_value,
            pad_yx=True,
            release_memory=False,
        )
        if original_ndim == 2 and output.ndim == 3 and output.shape[0] == 1:
            output = np.squeeze(output, axis=0)
        if release_memory:
            clear_rlgc_caches(clear_memory_pool=True)
        return output

    # Full-frame path if tiling not needed
    if crop_yx >= image_work.shape[-2] and crop_yx >= image_work.shape[-1]:
        output = rlgc_biggs_ba(
            image_arr,
            psf,
            gpu_id,
            safe_mode=safe_mode,
            auto_delta_scale=auto_delta_scale,
            init_value=_median_init_scalar(image_arr),
            pad_yx=True,
            release_memory=False,
        )
        if original_ndim == 2 and output.ndim == 3 and output.shape[0] == 1:
            output = np.squeeze(output, axis=0)

    # Tiled 3D deconvolution with hidden PSF halo and feathered blending
    else:
        blend_overlap_yx = _resolve_blend_overlap_yx(psf_arr.shape, crop_yx)

        expand_before = blend_overlap_yx // 2
        expand_after = blend_overlap_yx - expand_before
        psf_half_y = int(psf_arr.shape[-2]) // 2
        psf_half_x = int(psf_arr.shape[-1]) // 2
        tile_pad_y = psf_half_y + max(expand_before, expand_after)
        tile_pad_x = psf_half_x + max(expand_before, expand_after)
        init_value = _median_init_scalar(image_work)
        output_sum = np.zeros_like(image_work, dtype=np.float32)
        output_weight = np.zeros_like(image_work, dtype=np.float32)

        crop_size = (
            image_work.shape[0],
            crop_yx + 2 * tile_pad_y,
            crop_yx + 2 * tile_pad_x,
        )
        overlap = (0, tile_pad_y, tile_pad_x)
        slices = Slicer(image_work, crop_size=crop_size, overlap=overlap, pad=True)

        if verbose >= 1:
            from rich.progress import track

            iterator = track(
                enumerate(slices),
                description="Chunks",
                total=len(slices),
                transient=True,
            )
        else:
            iterator = enumerate(slices)

        for _, (crop, source, destination) in iterator:
            crop_array = rlgc_biggs_ba(
                crop,
                psf,
                gpu_id,
                safe_mode=safe_mode,
                auto_delta_scale=auto_delta_scale,
                init_value=init_value,
                release_memory=False,
            )

            def resolve_slice(s: slice, dim: int) -> tuple[int, int]:
                start = s.start if s.start is not None else 0
                stop = s.stop if s.stop is not None else dim
                if stop < 0:
                    stop = dim + stop
                return start, stop

            y_source_slice, x_source_slice = source[-2:]
            y_dest_slice, x_dest_slice = destination[-2:]

            y_source_start, y_source_stop = resolve_slice(y_source_slice, crop.shape[1])
            x_source_start, x_source_stop = resolve_slice(x_source_slice, crop.shape[2])
            y_dest_start, y_dest_stop = resolve_slice(
                y_dest_slice, image_work.shape[1]
            )
            x_dest_start, x_dest_stop = resolve_slice(
                x_dest_slice, image_work.shape[2]
            )

            y_expand_before = (
                min(expand_before, y_dest_start, y_source_start)
                if y_dest_start > 0
                else 0
            )
            y_expand_after = (
                min(
                    expand_after,
                    image_work.shape[1] - y_dest_stop,
                    crop.shape[1] - y_source_stop,
                )
                if y_dest_stop < image_work.shape[1]
                else 0
            )
            x_expand_before = (
                min(expand_before, x_dest_start, x_source_start)
                if x_dest_start > 0
                else 0
            )
            x_expand_after = (
                min(
                    expand_after,
                    image_work.shape[2] - x_dest_stop,
                    crop.shape[2] - x_source_stop,
                )
                if x_dest_stop < image_work.shape[2]
                else 0
            )

            y_source_start -= y_expand_before
            y_source_stop += y_expand_after
            x_source_start -= x_expand_before
            x_source_stop += x_expand_after
            y_dest_start -= y_expand_before
            y_dest_stop += y_expand_after
            x_dest_start -= x_expand_before
            x_dest_stop += x_expand_after

            crop_sub = crop_array[
                :,
                y_source_start:y_source_stop,
                x_source_start:x_source_stop,
            ]
            feather_weight = make_feather_weight(
                crop_sub.shape,
                feather_px=blend_overlap_yx,
                taper_top=y_dest_start > 0,
                taper_bottom=y_dest_stop < image_work.shape[1],
                taper_left=x_dest_start > 0,
                taper_right=x_dest_stop < image_work.shape[2],
            )

            weighted_sub = crop_sub * feather_weight
            weight_sub = feather_weight

            output_sum[:, y_dest_start:y_dest_stop, x_dest_start:x_dest_stop] += (
                weighted_sub
            )
            output_weight[:, y_dest_start:y_dest_stop, x_dest_start:x_dest_stop] += (
                weight_sub
            )

        del feather_weight, weighted_sub, weight_sub, crop_sub
        gc.collect()

        nonzero = output_weight > 0
        output = np.zeros_like(output_sum, dtype=output_sum.dtype)
        output[nonzero] = output_sum[nonzero] / output_weight[nonzero]

        if original_ndim == 2:
            output = np.squeeze(output, axis=0)

        del output_sum, output_weight, nonzero
        gc.collect()

    if release_memory:
        clear_rlgc_caches(clear_memory_pool=True)

    return output
