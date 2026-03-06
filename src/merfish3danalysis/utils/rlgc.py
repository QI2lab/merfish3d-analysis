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
import hashlib
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
# FFT / OTF caches (performance)
# -----------------------------------------------------------------------------
_fft_cache_3d: dict[tuple[int, int, int], tuple[cp.ndarray, cp.ndarray]] = {}
_fft_cache_2d_batched: dict[tuple[int, int, int], tuple[cp.ndarray, cp.ndarray]] = {}
_otf_cache_2d: dict[tuple[int, int, str], tuple[cp.ndarray, cp.ndarray]] = {}


def clear_rlgc_caches(clear_memory_pool: bool = False) -> None:
    """Clear cached FFT/OTF buffers used by RLGC helper functions."""

    _fft_cache_3d.clear()
    _fft_cache_2d_batched.clear()
    _otf_cache_2d.clear()
    if clear_memory_pool:
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()


def make_feather_weight(
    shape: tuple[int, int, int], feather_px: int = 64
) -> np.ndarray:
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
        Next 2-3-smooth length ≥ ``x``.
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


def pad_z(image: np.ndarray, psf_z: int = 1) -> tuple[np.ndarray, int, int]:
    """
    Pad the Z axis of a 3D array using PSF-aware halo plus FFT-friendly padding.

    The halo radius is ``psf_z // 2`` on each side so circular FFT convolution
    does not wrap into the interior volume. Additional symmetric padding is then
    applied to reach the next 2-3-smooth FFT length.

    Parameters
    ----------
    image : numpy.ndarray
        3D image to pad, shaped (z, y, x).
    psf_z : int, default=1
        PSF support size along Z.

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
    z_halo = max(int(psf_z) // 2, 0)

    z_with_halo = z + 2 * z_halo
    new_z = next_gpu_fft_size(z_with_halo)
    fft_extra = new_z - z_with_halo

    pad_z_before = z_halo + fft_extra // 2
    pad_z_after = z_halo + fft_extra - fft_extra // 2
    pad_width = ((pad_z_before, pad_z_after), (0, 0), (0, 0))
    padded_image = np.pad(image, pad_width, mode="reflect").astype(np.uint16)
    return padded_image, pad_z_before, pad_z_after


def remove_padding_z(
    padded_image: np.ndarray, pad_z_before: int, pad_z_after: int
) -> np.ndarray:
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
    if shape not in _fft_cache_3d:
        z, y, x = shape
        freq_shape = (z, y, x // 2 + 1)
        fft_buf = cp.empty(freq_shape, dtype=cp.complex64)
        ifft_buf = cp.empty(shape, dtype=cp.float32)
        _fft_cache_3d[shape] = (fft_buf, ifft_buf)

    fft_buf, ifft_buf = _fft_cache_3d[shape]
    fft_buf[...] = cp.fft.rfftn(image)
    fft_buf[...] *= H
    ifft_buf[...] = cp.fft.irfftn(fft_buf, s=shape)
    return ifft_buf


def _normalize_psf_to_2d(psf: np.ndarray) -> np.ndarray:
    """Convert a PSF to a normalized 2D kernel for 2D batched RLGC."""

    psf_arr = np.asarray(psf, dtype=np.float32)
    if psf_arr.ndim == 3:
        # 2D acquisition stores one effective PSF per z-plane. Use center plane.
        psf_arr = psf_arr[psf_arr.shape[0] // 2]
    if psf_arr.ndim != 2:
        raise ValueError(f"Expected a 2D or 3D PSF, got shape {psf_arr.shape}")
    return psf_arr


def _pad_psf_2d(psf_2d: cp.ndarray, image_shape: tuple[int, int]) -> cp.ndarray:
    """Pad and center a 2D PSF to target YX shape; normalize to unit sum."""

    y, x = image_shape
    padded = cp.zeros((y, x), dtype=cp.float32)
    padded[: psf_2d.shape[0], : psf_2d.shape[1]] = psf_2d

    for axis, axis_size in enumerate(padded.shape):
        padded = cp.roll(padded, int(axis_size / 2), axis=axis)
    for axis, axis_size in enumerate(psf_2d.shape):
        padded = cp.roll(padded, -int(axis_size / 2), axis=axis)

    padded = cp.fft.ifftshift(padded)
    s = cp.sum(padded)
    padded = padded / (s if s != 0 else 1.0)
    return padded.astype(cp.float32)


def _psf_cache_key(
    psf_2d: np.ndarray, image_shape: tuple[int, int]
) -> tuple[int, int, str]:
    """Build a stable cache key for padded 2D PSF OTFs."""

    y, x = image_shape
    digest = hashlib.sha1(psf_2d.tobytes()).hexdigest()
    return y, x, digest


def _get_otf_pair_2d(
    psf: np.ndarray,
    image_shape: tuple[int, int],
    use_cache: bool = True,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Return (OTF, conj(OTF)) for a 2D PSF and target image shape."""

    psf_2d = _normalize_psf_to_2d(psf)
    key = _psf_cache_key(psf_2d, image_shape)
    if use_cache and key in _otf_cache_2d:
        return _otf_cache_2d[key]

    psf_gpu = _pad_psf_2d(cp.asarray(psf_2d, dtype=cp.float32), image_shape)
    otf = cp.fft.rfftn(psf_gpu)
    otfT = cp.conjugate(otf)
    del psf_gpu
    pair = (otf, otfT)
    if use_cache:
        _otf_cache_2d[key] = pair
    return pair


def fft_conv_batched_2d(
    image: cp.ndarray,
    H: cp.ndarray,
    shape: tuple[int, int, int],
) -> cp.ndarray:
    """Apply FFT convolution independently to each z-plane in a batch."""

    if shape not in _fft_cache_2d_batched:
        z, y, x = shape
        freq_shape = (z, y, x // 2 + 1)
        fft_buf = cp.empty(freq_shape, dtype=cp.complex64)
        ifft_buf = cp.empty(shape, dtype=cp.float32)
        _fft_cache_2d_batched[shape] = (fft_buf, ifft_buf)

    fft_buf, ifft_buf = _fft_cache_2d_batched[shape]
    fft_buf[...] = cp.fft.rfftn(image, axes=(-2, -1))
    fft_buf[...] *= H
    ifft_buf[...] = cp.fft.irfftn(fft_buf, s=shape[-2:], axes=(-2, -1))
    return ifft_buf


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


def rlgc_biggs_ba(
    image: np.ndarray,
    psf: np.ndarray,
    gpu_id: int = 0,
    otf: cp.ndarray | None = None,
    otfT: cp.ndarray | None = None,
    safe_mode: bool = True,
    init_value: float = 1,
    limit: float = 0.01,
    max_delta: float = 0.01,
    release_memory: bool = True,
) -> np.ndarray:
    """
    Biggs-Andrews accelerated Richardson-Lucy Gradient Consensus.

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
    release_memory : bool, default=True
        If True, release GPU memory pools after each call. Set False when
        calling in tight loops to avoid allocator thrashing.

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

    psf_z = int(psf.shape[0])

    # Z padding
    if image.ndim == 3:
        image_gpu_np, pad_z_before, pad_z_after = pad_z(image_padded, psf_z=psf_z)
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
        HTratio1 = fft_conv(
            cp.divide(split1, 0.5 * (Hu + eps), dtype=cp.float32), otfT, shape
        )
        HTratio2 = fft_conv(
            cp.divide(split2, 0.5 * (Hu + eps), dtype=cp.float32), otfT, shape
        )
        HTratio = HTratio1 + HTratio2

        # Consensus: H^T H * ((HTratio1 - 1)*(HTratio2 - 1))
        consensus_map = fft_conv(
            (HTratio1 - 1.0) * (HTratio2 - 1.0), otfotfT, recon.shape
        )

        # Gated multiplicative update
        filter_update_ba(recon, HTratio, consensus_map, recon_next)

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
    if release_memory:
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    return recon_cpu


def rlgc_biggs_ba_2d_batched(
    image: np.ndarray,
    psf: np.ndarray,
    gpu_id: int = 0,
    otf: cp.ndarray | None = None,
    otfT: cp.ndarray | None = None,
    safe_mode: bool = True,
    init_value: float = 1,
    limit: float = 0.01,
    max_delta: float = 0.01,
    cache_otf: bool = True,
    release_memory: bool = True,
) -> np.ndarray:
    """Quality-preserving 2D fast path over z-planes.

    This keeps the original per-plane RLGC behavior while reducing overhead by:
    1) routing through one function call for a full z-stack,
    2) keeping FFT/OTF caches warm across z-planes and calls, and
    3) deferring GPU memory-pool release to outer scopes.
    """

    image_arr = np.asarray(image)
    if image_arr.ndim == 2:
        return rlgc_biggs_ba(
            image=image_arr,
            psf=_normalize_psf_to_2d(psf),
            gpu_id=gpu_id,
            safe_mode=safe_mode,
            init_value=init_value,
            limit=limit,
            max_delta=max_delta,
            release_memory=release_memory,
        )
    if image_arr.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D image, got shape {image_arr.shape}")

    psf_2d = _normalize_psf_to_2d(psf)
    output = np.zeros_like(image_arr, dtype=np.float32)
    for z_idx in range(image_arr.shape[0]):
        output[z_idx] = rlgc_biggs_ba(
            image=image_arr[z_idx],
            psf=psf_2d,
            gpu_id=gpu_id,
            safe_mode=safe_mode,
            init_value=init_value,
            limit=limit,
            max_delta=max_delta,
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
    overlap_yx: int = 128,
    safe_mode: bool = True,
    verbose: int = 0,
    use_batched_2d: bool | None = None,
    cache_otf: bool = True,
    release_memory: bool = True,
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
    use_batched_2d : bool | None, default=None
        If True, run batched 2D RLGC over z-planes. If None, auto-detect 2D
        input based on image and PSF dimensionality.
    cache_otf : bool, default=True
        Reuse cached 2D OTF pairs across calls.
    release_memory : bool, default=True
        If True, clear RLGC caches and release memory pools on exit.

    Returns
    -------
    numpy.ndarray
        Deconvolved image (float32).
    """
    cp.cuda.Device(gpu_id).use()

    if use_batched_2d is None:
        psf_arr = np.asarray(psf)
        use_batched_2d = image.ndim == 3 and (
            psf_arr.ndim == 2 or (psf_arr.ndim == 3 and psf_arr.shape[0] == 1)
        )

    # Full-frame path if tiling not needed
    if crop_yx >= image.shape[-2] and crop_yx >= image.shape[-1]:
        if use_batched_2d:
            output = rlgc_biggs_ba_2d_batched(
                image,
                psf,
                gpu_id,
                safe_mode=safe_mode,
                init_value=float(np.median(image)),
                cache_otf=cache_otf,
                release_memory=False,
            )
        else:
            output = rlgc_biggs_ba(
                image,
                psf,
                gpu_id,
                safe_mode=safe_mode,
                init_value=float(np.median(image)),
                release_memory=False,
            )

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

            iterator = track(
                enumerate(slices),
                description="Chunks",
                total=len(slices),
                transient=True,
            )
        else:
            iterator = enumerate(slices)

        for _, (crop, source, destination) in iterator:
            if use_batched_2d:
                crop_array = rlgc_biggs_ba_2d_batched(
                    crop,
                    psf,
                    gpu_id,
                    safe_mode=safe_mode,
                    init_value=init_value,
                    cache_otf=cache_otf,
                    release_memory=False,
                )
            else:
                crop_array = rlgc_biggs_ba(
                    crop,
                    psf,
                    gpu_id,
                    safe_mode=safe_mode,
                    init_value=init_value,
                    release_memory=False,
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

    if release_memory:
        clear_rlgc_caches(clear_memory_pool=True)

    return output
