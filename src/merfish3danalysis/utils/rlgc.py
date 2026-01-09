"""
Richardson–Lucy Gradient Consensus (RLGC) deconvolution (Manton-style core).

Original idea for Gradient Consensus deconvolution:
James Manton and Andrew York, https://zenodo.org/records/10278919

Reference RLGC loop based on James Manton's implementation:
https://colab.research.google.com/drive/1mfVNSCaYHz1g56g92xBkIoa8190XNJpJ

Biggs–Andrews acceleration:
Biggs & Andrews, 1997, https://doi.org/10.1364/AO.36.001766
"""

import gc
import timeit
from typing import Tuple
import cupy as cp
import numpy as np
from cupy import ElementwiseKernel
from ryomen import Slicer

DEBUG = False

# -----------------------------------------------------------------------------
# CUDA kernel: multiplicative RL step gated by consensus (reference-accurate)
# -----------------------------------------------------------------------------
filter_update_ba = ElementwiseKernel(
    'float32 recon, float32 HTratio, float32 consensus_map',
    'float32 out',
    '''
    bool skip = consensus_map < 0;
    out = skip ? recon : recon * HTratio;
    out = out < 0 ? 0 : out
    ''',
    'filter_update_ba'
)

# -----------------------------------------------------------------------------
# FFT work-buffer cache (performance)
# -----------------------------------------------------------------------------
_fft_cache: dict[Tuple[int, int, int], Tuple[cp.ndarray, cp.ndarray]] = {}


def make_feather_weight(shape: tuple[int, int, int], feather_px: int = 64) -> np.ndarray:
    """
    Create a feathered weight mask using a cosine taper on Y/X only.

    Z is uniform. Feather taper width is explicitly specified in pixels.

    Parameters
    ----------
    shape : tuple of int
        Crop shape as (z, y, x).
    feather_px : int
        Number of pixels to taper at each Y/X edge.

    Returns
    -------
    numpy.ndarray
        Feather mask of shape (z, y, x), values in [0, 1].
    """
    def cosine_taper(length: int, feather: int) -> np.ndarray:
        window = np.ones(length, dtype=np.float32)
        if feather > 0:
            ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 2 * feather)))
            window[:feather] = ramp[:feather]
            window[-feather:] = ramp[feather:]
        return window

    y_win = cosine_taper(shape[1], feather_px)
    x_win = cosine_taper(shape[2], feather_px)
    weight2d = np.outer(y_win, x_win).astype(np.float32)
    weight = np.broadcast_to(weight2d[None, :, :], shape)
    return weight


def next_gpu_fft_size(x: int) -> int:
    """
    Return the smallest FFT-friendly size ≥ ``x`` with prime factors in {2, 3}.

    Parameters
    ----------
    x : int
        Minimum desired length.

    Returns
    -------
    int
        Next 2–3–smooth length ≥ ``x``.
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


def pad_z(image: np.ndarray,init_value:float = None) -> tuple[np.ndarray, int, int]:
    """
    Pad the Z axis of a 3D array to the next 2–3–smooth length (ZYX order).

    A constant background is subtracted before padding and values are
    clamped into [0, 65535] for numerical stability with uint16 inputs.

    Parameters
    ----------
    image : numpy.ndarray
        3D image to pad, shaped (z, y, x).

    Returns
    -------
    padded_image : numpy.ndarray
        Padded 3D image.
    pad_z_before : int
        Z padding at the start.
    pad_z_after : int
        Z padding at the end.
    """
    z, _, _ = image.shape
    new_z = next_gpu_fft_size(z)
    pad_amt = new_z - z
    pad_z_before = pad_amt // 2
    pad_z_after = pad_amt - pad_z_before
    pad_width = ((pad_z_before, pad_z_after), (0, 0), (0, 0))
    padded_image = np.pad(
        image,
        pad_width,
        mode="reflect"
    ).astype(np.uint16)
    return padded_image, pad_z_before, pad_z_after


def remove_padding_z(padded_image: np.ndarray, pad_z_before: int, pad_z_after: int) -> np.ndarray:
    """
    Remove Z padding added by :func:`pad_z`.

    Parameters
    ----------
    padded_image : numpy.ndarray
        Padded 3D image.
    pad_z_before : int
        Z padding at the start.
    pad_z_after : int
        Z padding at the end.

    Returns
    -------
    numpy.ndarray
        Unpadded 3D image.
    """
    if pad_z_before == 0 and pad_z_after == 0:
        return padded_image
    return padded_image[pad_z_before:-pad_z_after, :, :]


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
    psf[:psf_temp.shape[0], :psf_temp.shape[1], :psf_temp.shape[2]] = psf_temp

    # Center the PSF
    for axis, axis_size in enumerate(psf.shape):
        psf = cp.roll(psf, int(axis_size / 2), axis=axis)
    for axis, axis_size in enumerate(psf_temp.shape):
        psf = cp.roll(psf, -int(axis_size / 2), axis=axis)

    psf = cp.fft.ifftshift(psf)
    s = cp.sum(psf)
    psf = psf / (s if s != 0 else 1.0)
    return psf.astype(cp.float32)


def fft_conv(image: cp.ndarray, H: cp.ndarray, shape: tuple[int, int, int]) -> cp.ndarray:
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
    if shape not in _fft_cache:
        z, y, x = shape
        freq_shape = (z, y, x // 2 + 1)
        fft_buf = cp.empty(freq_shape, dtype=cp.complex64)
        ifft_buf = cp.empty(shape, dtype=cp.float32)
        _fft_cache[shape] = (fft_buf, ifft_buf)

    fft_buf, ifft_buf = _fft_cache[shape]
    fft_buf[...] = cp.fft.rfftn(image)
    fft_buf[...] *= H
    ifft_buf[...] = cp.fft.irfftn(fft_buf, s=shape)
    return ifft_buf


def kl_div(p: cp.ndarray, q: cp.ndarray) -> float:
    """
    Compute Kullback–Leibler divergence between two distributions.

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

def rlgc_biggs_ba(
    image: np.ndarray,
    psf: np.ndarray,
    gpu_id: int = 0,
    otf: cp.ndarray | None = None,
    otfT: cp.ndarray | None = None,
    safe_mode: bool = True,
    init_value: float = 1,
    limit: float = 0.2,
    max_delta: float = 0.02,
) -> np.ndarray:
    """
    Biggs–Andrews accelerated Richardson–Lucy Gradient Consensus.

    Parameters
    ----------
    image : numpy.ndarray
        3D image (Z, Y, X) to be deconvolved.
    psf : numpy.ndarray
        3D point-spread function. If ``otf`` and ``otfT`` are None, this PSF
        will be padded and transformed to form the OTF internally.
    gpu_id : int, default=0
        Which GPU to use.
    otf, otfT : cupy.ndarray, optional
        Precomputed OTF and its conjugate in RFFTN layout.
    safe_mode : bool, default=True
        If True, stop when EITHER split KLD increases (play-it-safe).
        If False, stop only when BOTH split KLDs increase.
    init_value : float, default=1
        Constant initializer value for the reconstruction.
    limit : float, default=0.01
        Minimum fraction of pixels that must be updated per iteration before
        early stopping is triggered.
    max_delta : float, default=0.01
        Maximum allowed relative update magnitude before early stopping is
        triggered.

    Returns
    -------
    numpy.ndarray
        Deconvolved 3D image (float32).
    """

    cp.cuda.Device(gpu_id).use()
    rng = cp.random.default_rng(42)

    # Ensure 3D inputs
    if psf.ndim == 2:
        psf = np.expand_dims(psf, axis=0)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)

    _, y0, x0 = image.shape  # (z, y, x)

    # Pad Y/X to FFT-friendly sizes
    new_y = next_gpu_fft_size(y0)
    new_x = next_gpu_fft_size(x0)
    pad_y_before = (new_y - y0) // 2
    pad_y_after = (new_y - y0) - pad_y_before
    pad_x_before = (new_x - x0) // 2
    pad_x_after = (new_x - x0) - pad_x_before

    image_padded = np.pad(
        image,
        pad_width=((0, 0), (pad_y_before, pad_y_after), (pad_x_before, pad_x_after)),
        mode="reflect",
    ).astype(np.uint16)

    # Z padding
    if image.ndim == 3:
        image_gpu_np, pad_z_before, pad_z_after = pad_z(image_padded)
        image_gpu = cp.asarray(image_gpu_np, dtype=cp.float32)
        
        del image_gpu_np
    else:
        image_gpu = cp.asarray(image_padded, dtype=cp.float32)[cp.newaxis, ...]
        pad_z_before = pad_z_after = 0

    # OTFs
    if (otf is None) or (otfT is None):
        # Pad PSF to image size first
        psf_gpu = pad_psf(cp.asarray(psf, dtype=cp.float32), image_gpu.shape)
        otf = cp.fft.rfftn(psf_gpu)
        otfT = cp.conjugate(otf)
        del psf_gpu
        cp.get_default_memory_pool().free_all_blocks()
    else:
        otfT = cp.conjugate(otf) if otfT is None else otfT

    otfotfT = otf * otfT  # H^T H in frequency domain
    shape = image_gpu.shape
    z, y, x = shape
    num_pixels = z * y * x

    recon = cp.full((z, y, x), cp.float32(init_value), dtype=cp.float32)
    previous_recon = recon.copy()

    # Pre-allocations
    recon_next = cp.empty_like(recon)
    Hu = cp.empty_like(recon)

    # Biggs–Andrews state
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

        # 50:50 split of the data (counts)
        split1 = rng.binomial(image_gpu.astype(cp.int64), p=0.5).astype(cp.float32)
        split2 = image_gpu - split1

        # BA momentum: y = recon + alpha * (recon - previous_recon)
        if num_iters >= 1:
            numerator = cp.sum(g1 * g2)
            denominator = cp.sum(g2 * g2)
            alpha = numerator / denominator
            alpha = cp.clip(alpha, 0.0, 1.0)
            if cp.isnan(alpha):
                alpha = 0.0
            alpha = float(alpha)
            alpha = 0.0
        else:
            alpha = 0.0

        y_vec = recon + alpha * (recon - previous_recon)

        # Forward prediction
        Hu[...] = fft_conv(y_vec, otf, shape)

        # KLDs & stopping
        kldim = kl_div(Hu, image_gpu)
        kld1 = kl_div(Hu, split1)
        kld2 = kl_div(Hu, split2)

        if safe_mode:
            if (kld1 > prev_kld1) or (kld2 > prev_kld2) or (kld1 < 1e-4) or (kld2 < 1e-4):
                recon[...] = previous_recon
                if DEBUG:
                    total_time = timeit.default_timer() - start_time
                    print(
                        f"Optimum after {num_iters - 1} iters in {total_time:.1f} s."
                    )
                break
        else:
            if ((kld1 > prev_kld1) and (kld2 > prev_kld2)) or (kld1 < 1e-4) or (kld2 < 1e-4):
                recon[...] = previous_recon
                if DEBUG:
                    total_time = timeit.default_timer() - start_time
                    print(
                        f"Optimum after {num_iters - 1} iters in {total_time:.1f} s."
                    )
                break

        prev_kld1 = kld1
        prev_kld2 = kld2

        # BA bookkeeping: move recon forward to y
        previous_recon[...] = recon
        recon[...] = y_vec

        # RL ratios: H^T( split / (0.5*(Hu+eps)) )
        eps = 1e-12
        HTratio1 = fft_conv(cp.divide(split1, 0.5 * (Hu + eps), dtype=cp.float32), otfT, shape)
        HTratio2 = fft_conv(cp.divide(split2, 0.5 * (Hu + eps), dtype=cp.float32), otfT, shape)
        HTratio = HTratio1 + HTratio2

        # Consensus: H^T H * ((HTratio1 - 1)*(HTratio2 - 1))
        consensus_map = fft_conv((HTratio1 - 1.0) * (HTratio2 - 1.0), otfotfT, recon.shape)

        # Gated multiplicative update
        filter_update_ba(recon, HTratio, consensus_map,recon_next)

        # BA vectors
        g2[...] = g1
        g1[...] = recon_next - y_vec

        recon[...] = recon_next

        num_updated = num_pixels - cp.sum(consensus_map < 0)
        max_relative_delta = cp.max((recon - previous_recon) / cp.max(recon))

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
        if max_relative_delta < 5.0 / cp.max(image_gpu):
            if DEBUG:
                print("Hit auto delta")
            break

    # Enforce nonnegativity and unpad back to original
    recon = cp.maximum(recon, 0.0)

    if image.ndim == 3:
        recon = remove_padding_z(recon, pad_z_before, pad_z_after)
    else:
        recon = cp.squeeze(recon)

    y_end = -pad_y_after if pad_y_after > 0 else None
    x_end = -pad_x_after if pad_x_after > 0 else None
    recon = recon[:, pad_y_before:y_end, pad_x_before:x_end]

    recon_cpu = cp.asnumpy(recon).astype(np.float32)

    # Cleanup
    del g1, g2, recon, previous_recon, Hu, otf, otfT, otfotfT, image_gpu
    gc.collect()
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return recon_cpu

def chunked_rlgc(
    image: np.ndarray,
    psf: np.ndarray,
    gpu_id: int = 0,
    crop_yx: int = 1500,
    overlap_yx: int = 128,
    safe_mode: bool = True,
    verbose: int = 0
) -> np.ndarray:
    """
    Chunked RLGC deconvolution with feathered blending.

    Parameters
    ----------
    image : numpy.ndarray
        3D image to be deconvolved.
    psf : numpy.ndarray
        Point-spread function (PSF) to use for deconvolution.
    gpu_id : int, default=0
        Which GPU to use.
    crop_yx : int, default=1500
        Tile size in Y and X.
    overlap_yx : int, default=128
        Overlap width in pixels between tiles (for feathering).
    safe_mode : bool, default=True
        RLGC stopping: play-it-safe if True.
    verbose : int, default=0
        If ≥ 1, show a progress bar over subtiles.

    Returns
    -------
    numpy.ndarray
        Deconvolved image (float32).
    """
    cp.cuda.Device(gpu_id).use()
    # Avoid cuFFT plan accumulation across tiles
    cp.fft._cache.PlanCache(memsize=0)

    # Full-frame path if tiling not needed
    if crop_yx >= image.shape[-2] and crop_yx >= image.shape[-1]:
        output = rlgc_biggs_ba(image, psf, gpu_id, safe_mode=safe_mode, init_value=float(np.median(image)))

    # Tiled deconvolution with feathered blending
    else:
        init_value = np.mean(image)
        output_sum = np.zeros_like(image, dtype=np.float32)
        output_weight = np.zeros_like(image, dtype=np.float32)

        crop_size = (image.shape[0], crop_yx, crop_yx)
        overlap = (0, overlap_yx, overlap_yx)
        slices = Slicer(image, crop_size=crop_size, overlap=overlap, pad=True)

        if verbose >= 1:
            from rich.progress import track
            iterator = track(enumerate(slices), description="Chunks", total=len(slices), transient=True)
        else:
            iterator = enumerate(slices)

        for _, (crop, source, destination) in iterator:
            crop_array = rlgc_biggs_ba(
                crop,
                psf,
                gpu_id,
                safe_mode=safe_mode,
                init_value=init_value
            )

            # Resolve subtile edge status to decide feathering
            _, y_slice, x_slice = source[1:]

            def resolve_slice(s: slice, dim: int) -> tuple[int, int]:
                start = s.start if s.start is not None else 0
                stop = s.stop if s.stop is not None else dim
                if stop < 0:
                    stop = dim + stop
                return start, stop

            y_start, y_stop = resolve_slice(y_slice, crop.shape[1])
            x_start, x_stop = resolve_slice(x_slice, crop.shape[2])
            is_y_edge = (y_start == 0) or (y_stop == crop.shape[1])
            is_x_edge = (x_start == 0) or (x_stop == crop.shape[2])

            if is_y_edge or is_x_edge:
                feather_weight = np.ones_like(crop_array, dtype=np.float32)
            else:
                feather_weight = make_feather_weight(crop.shape, feather_px=overlap_yx)

            weighted_crop = crop_array * feather_weight
            weighted_sub = weighted_crop[source]
            weight_sub = feather_weight[source]

            output_sum[destination] += weighted_sub
            output_weight[destination] += weight_sub

        del feather_weight, weighted_crop, weighted_sub, weight_sub
        gc.collect()

        nonzero = output_weight > 0
        output = np.zeros_like(output_sum, dtype=output_sum.dtype)
        output[nonzero] = output_sum[nonzero] / output_weight[nonzero]

        del output_sum, output_weight, nonzero
        gc.collect()

    # Clear caches and pools
    _fft_cache.clear()
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()

    return output