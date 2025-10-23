"""Richardson-Lucy Gradient Consensus deconvolution.

Original idea for gradient consensus deconvolution from: James Maton and Andrew York, https://zenodo.org/records/10278919

RLGC code based on James Manton's implementation, https://colab.research.google.com/drive/1mfVNSCaYHz1g56g92xBkIoa8190XNJpJ

Biggs-Andrews acceleration based on their 1997 paper, https://doi.org/10.1364/AO.36.001766
"""

import cupy as cp
import numpy as np
from cupy import ElementwiseKernel
from ryomen import Slicer
import timeit
import gc

DEBUG = False

# -----------------------------------------------------------------------------
# High-speed fused CUDA kernel: additive RL step with GC gate + nonnegativity
# -----------------------------------------------------------------------------
gc_add_update_kernel = ElementwiseKernel(
    # inputs
    'float32 recon, float32 gradient, float32 H_T_ones, float32 local_dot',
    # output
    'float32 recon_next',
    r'''
    // Scaled-gradient RL step (additive), gated by Gradient Consensus:
    // step_size = recon / max(H_T_ones, 1e-6)
    // if local_dot <= 0 --> step_size = 0
    // recon_next = max(recon + gradient * step_size, 0)
    const float denom = (H_T_ones > 1e-6f) ? H_T_ones : 1e-6f;
    const float step  = (local_dot > 0.0f) ? (recon / denom) : 0.0f;
    const float val   = recon + gradient * step;
    recon_next = (val > 0.0f) ? val : 0.0f;
    ''',
    'gc_add_update_kernel'
)

# -----------------------------------------------------------------------------
# Work-buffer caches for FFTs (kept for performance)
# -----------------------------------------------------------------------------
_fft_cache: dict[tuple[int, int, int], tuple[cp.ndarray, cp.ndarray]] = {}
_H_T_cache: dict[tuple[int, int, int], cp.ndarray] = {}


def make_feather_weight(shape: tuple[int, int, int], feather_px: int = 64):
    """
    Create a feathered weight mask using a cosine taper on Y/X axes only.
    Z is uniform. Feather taper width is explicitly specified in pixels.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Shape of the crop (z, y, x)
    feather_px : int
        Number of pixels to taper at each edge of Y and X

    Returns
    -------
    weight : np.ndarray
        Feather mask of shape (z, y, x), values in [0, 1]
    """
    def cosine_taper(length, feather):
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
    Pick the smallest FFT-friendly size ≥ x whose only prime factors are 2 or 3.

    Parameters
    ----------
    x : int
        Minimum desired length.

    Returns
    -------
    int
        The next length ≥ x that is 2–3–smooth.
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


def pad_z(image: np.ndarray, bkd: int = 0) -> tuple[np.ndarray, int, int]:
    """Pad z-axis of 3D array by the next 2–3–smooth length (zyx order).

    Parameters
    ----------
    image: np.ndarray
        3D image to pad.
    bkd: int
        Background value to subtract before padding.

    Returns
    -------
    padded_image: np.ndarray
        Padded 3D image.
    pad_z_before: int
        Amount of padding at the start of the z-axis.
    pad_z_after: int
        Amount of padding at the end of the z-axis.
    """
    z, y, x = image.shape
    new_z = next_gpu_fft_size(z)
    pad_z_amt = new_z - z
    pad_z_before = pad_z_amt // 2
    pad_z_after = pad_z_amt - pad_z_before
    pad_width = ((pad_z_before, pad_z_after), (0, 0), (0, 0))
    padded_image = np.pad(
        (image.astype(np.float32) - float(bkd)).clip(0, None),
        pad_width, mode="reflect"
    )
    return padded_image, pad_z_before, pad_z_after


def remove_padding_z(
    padded_image: np.ndarray,
    pad_z_before: int,
    pad_z_after: int
) -> np.ndarray:
    """Remove z-axis padding added by pad_z."""
    if pad_z_before == 0 and pad_z_after == 0:
        return padded_image
    return padded_image[pad_z_before:-pad_z_after, :, :]


def pad_psf(psf_temp: cp.ndarray, image_shape: tuple[int, int, int]) -> cp.ndarray:
    """Pad and center a PSF to match the target image shape; normalize to unit sum."""
    psf = cp.zeros(image_shape, dtype=cp.float32)
    psf[:psf_temp.shape[0], :psf_temp.shape[1], :psf_temp.shape[2]] = psf_temp

    # center
    for axis, axis_size in enumerate(psf.shape):
        psf = cp.roll(psf, int(axis_size / 2), axis=axis)
    for axis, axis_size in enumerate(psf_temp.shape):
        psf = cp.roll(psf, -int(axis_size / 2), axis=axis)

    psf = cp.fft.ifftshift(psf)
    psf = cp.maximum(psf, 0.0)
    s = cp.sum(psf)
    psf = psf / (s if s != 0 else 1.0)
    return psf.astype(cp.float32)


def fft_conv(image: cp.ndarray, H: cp.ndarray, shape: tuple[int, int, int]) -> cp.ndarray:
    """Perform convolution via FFT with cached buffers:
       irfftn( rfftn(image) * H ). Enforce nonnegativity only.
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
    cp.maximum(ifft_buf, 0.0, out=ifft_buf)  # no upper clamp
    return ifft_buf


def kl_div(p: cp.ndarray, q: cp.ndarray) -> float:
    """Compute Kullback–Leibler divergence with stabilizer and normalization."""
    p = p + 1e-4
    q = q + 1e-4
    p = p / cp.sum(p)
    q = q / cp.sum(q)
    kldiv = p * (cp.log(p) - cp.log(q))
    kldiv[cp.isnan(kldiv)] = 0
    return float(cp.sum(kldiv))


def _get_H_T_ones(otf: cp.ndarray, otfT: cp.ndarray, shape: tuple[int, int, int]) -> cp.ndarray:
    """Compute H_T(ones) with cache."""
    if shape in _H_T_cache:
        return _H_T_cache[shape]
    ones = cp.ones(shape, dtype=cp.float32)
    tmp = fft_conv(fft_conv(ones, otf, shape), otfT, shape)
    _H_T_cache[shape] = tmp
    return tmp


def rlgc_biggs(
    image: np.ndarray,
    psf: np.ndarray,
    bkd: int = 0,
    gpu_id: int = 0,
    otf: cp.ndarray | None = None,
    otfT: cp.ndarray | None = None,
    eager_mode: bool = False,
    image_mean: float | None = None  # unused now, kept for API compat
) -> np.ndarray:
    """
    Biggs–Andrews accelerated RLGC deconvolution (reference-accurate core).

    Parameters
    ----------
    image: np.ndarray
        3D image (zyx) to be deconvolved.
    psf: np.ndarray
        3D point-spread function. If `otf` and `otfT` are None, this PSF
        will be padded and transformed to form the OTF internally.
    bkd: int, default = 0
        Constant background to subtract before deconvolution (applied by pad_z).
    gpu_id: int, default = 0
        Which GPU to use.
    otf, otfT: cp.ndarray, optional
        Precomputed OTF and its conjugate in rfftn layout.
    eager_mode: bool, default = False
        If True, stop when EITHER split KLD increases (play-it-safe).
        If False, stop only when BOTH split KLDs increase.
    image_mean: float, optional
        If provided, use as the constant initializer for recon.

    Returns
    -------
    output: np.ndarray
        Deconvolved 3D image (float32)
    """
    cp.cuda.Device(gpu_id).use()
    rng = cp.random.default_rng(42)

    # Ensure 3D
    if psf.ndim == 2:
        psf = np.expand_dims(psf, axis=0)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)

    _, y0, x0 = image.shape  # (z, y, x)

    # YX padding to FFT-friendly sizes
    new_y = next_gpu_fft_size(y0)
    new_x = next_gpu_fft_size(x0)
    pad_y_before = (new_y - y0) // 2
    pad_y_after = (new_y - y0) - pad_y_before
    pad_x_before = (new_x - x0) // 2
    pad_x_after = (new_x - x0) - pad_x_before

    image_padded = np.pad(
        image,
        pad_width=((0, 0),
                   (pad_y_before, pad_y_after),
                   (pad_x_before, pad_x_after)),
        mode="reflect"
    )

    # Z padding (and background subtract inside pad_z)
    if image.ndim == 3:
        image_gpu_np, pad_z_before, pad_z_after = pad_z(image_padded, bkd)
        image_gpu = cp.asarray(image_gpu_np, dtype=cp.float32)
        del image_gpu_np
    else:
        image_gpu = cp.asarray(image_padded, dtype=cp.float32)[cp.newaxis, ...]
        pad_z_before = pad_z_after = 0

    # OTFs
    if (otf is None) or (otfT is None):
        psf_gpu = pad_psf(cp.asarray(psf, dtype=cp.float32), image_gpu.shape)
        otf = cp.fft.rfftn(psf_gpu)
        otfT = cp.conjugate(otf)
        del psf_gpu
        cp.get_default_memory_pool().free_all_blocks()

    otfotfT = otf * otfT  # H^T H in frequency domain
    shape = image_gpu.shape
    z, y, x = shape

    # ---- Ones initialization (item 5) ----
    recon = cp.ones((z, y, x), dtype=cp.float32)

    previous_recon = cp.empty_like(recon)
    previous_recon[...] = recon

    # Pre-allocations
    Hu = cp.empty_like(recon)
    recon_next = cp.empty_like(recon)

    # Biggs-Andrews state (kept)
    g1 = cp.zeros_like(recon)
    g2 = cp.zeros_like(recon)

    # Precompute H_T(1) for scaled step size
    H_T_ones = _get_H_T_ones(otf, otfT, shape)

    prev_kld1 = np.inf
    prev_kld2 = np.inf
    num_iters = 0
    start_time = timeit.default_timer()
    eps = 1e-12

    while True:
        iter_start_time = timeit.default_timer()

        # 50:50 split of the data (counts)
        split1 = rng.binomial(image_gpu.astype(cp.int64), p=0.5).astype(cp.float32)
        split2 = image_gpu - split1

        # --- Biggs–Andrews momentum (kept) ---
        if num_iters >= 1:
            numerator = cp.sum(g1 * g2)
            denominator = cp.sum(g2 * g2) + 1e-12
            alpha = numerator / denominator
            alpha = cp.clip(alpha, 0.0, 1.0)
            alpha = float(alpha)
        else:
            alpha = 0.0

        y_vec = recon + alpha * (recon - previous_recon)

        # Forward prediction at y_vec
        Hu[...] = fft_conv(y_vec, otf, shape)

        # KLDs & stopping (kept)
        kld1 = kl_div(Hu, split1)
        kld2 = kl_div(Hu, split2)

        if eager_mode:
            if (kld1 > prev_kld1) or (kld2 > prev_kld2) or (kld1 < 1e-4) or (kld2 < 1e-4):
                recon[...] = previous_recon
                if DEBUG:
                    total_time = timeit.default_timer() - start_time
                    print(
                        f"Optimum result obtained after {num_iters - 1} iterations "
                        f"in {total_time:.1f} seconds."
                    )
                break
        else:
            if ((kld1 > prev_kld1) and (kld2 > prev_kld2)) or (kld1 < 1e-4) or (kld2 < 1e-4):
                recon[...] = previous_recon
                if DEBUG:
                    total_time = timeit.default_timer() - start_time
                    print(
                        f"Optimum result obtained after {num_iters - 1} iterations "
                        f"in {total_time:.1f} seconds."
                    )
                break

        prev_kld1 = kld1
        prev_kld2 = kld2

        # Advance baseline to y_vec (BA bookkeeping)
        previous_recon[...] = recon
        recon[...] = y_vec

        # ---- Additive scaled-gradient update with reference GC gate (item 4) ----
        Hu_safe = Hu + eps

        # Heads gradient (centered at 0.5)
        heads_grad = fft_conv(split1 / Hu_safe - 0.5, otfT, shape)

        # Full-data gradient: H^T(d/Hu - 1)
        full_ratio = fft_conv(image_gpu / Hu_safe, otfT, shape)  # H^T(d/Hu)
        gradient = full_ratio - H_T_ones

        # Tails gradient via reuse (gradient - heads_grad)
        tails_grad = gradient - heads_grad

        # Consensus gate: (H^T H) * (heads_grad * tails_grad)
        local_dot = fft_conv(heads_grad * tails_grad, otfotfT, shape)

        # Fused kernel does: recon_next = max(recon + gradient * step, 0)
        # where step = recon / max(H_T_ones, 1e-6) and zeroed if local_dot <= 0
        gc_add_update_kernel(
            recon,
            gradient,
            H_T_ones,
            local_dot,
            recon_next
        )

        # Biggs–Andrews vectors
        g2[...] = g1
        g1[...] = recon_next - y_vec

        # Swap to recon
        recon[...] = recon_next

        # Cleanup per-iter temporaries
        del split1, split2, Hu_safe, heads_grad, tails_grad, full_ratio, gradient, local_dot

        num_iters += 1
        if DEBUG:
            calc_time = timeit.default_timer() - iter_start_time
            print(
                f"Iteration {num_iters:03d} completed in {calc_time:.3f}s. "
                f"KLDs: {kld1:.4f} (split1), {kld2:.4f} (split2)."
            )

    # Nonnegativity only (no upper clamp)
    recon = cp.maximum(recon, 0.0).astype(cp.float32)

    # Unpad back to original
    if image.ndim == 3:
        recon = remove_padding_z(recon, pad_z_before, pad_z_after)
    else:
        recon = cp.squeeze(recon)

    y_end = -pad_y_after if pad_y_after > 0 else None
    x_end = -pad_x_after if pad_x_after > 0 else None
    recon = recon[:, pad_y_before:y_end, pad_x_before:x_end]

    recon_cpu = cp.asnumpy(recon).astype(np.float32)

    # Aggressive cleanup
    del recon_next, g1, g2, recon, previous_recon, Hu, otf, otfT, otfotfT, image_gpu
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
    overlap_yx: int = 32,
    eager_mode: bool = False,
    bkd: bool = False,
    ij=None,
    bkd_sub_radius: int = 200,
    verbose: int = 0
) -> np.ndarray:
    """Chunked RLGC deconvolution with feathered blending.

    Parameters
    ----------
    image: np.ndarray
        3D image to be deconvolved.
    psf: np.ndarray
        Point spread function (PSF) to use for deconvolution.
    gpu_id: int, default = 0
        Which GPU to use.
    crop_yx: int, default = 1500
        Size of the tile in Y and X.
    overlap_yx: int, default = 32
        Overlap width in pixels between tiles (for feathering).
    eager_mode: bool, default = False
        RLGC stopping: play-it-safe if True.
    bkd: bool, default = False
        Subtract background using ImageJ rolling-ball before deconvolution.
    ij: imagej instance, optional
        If provided and bkd=True, use this ImageJ instance.
    bkd_sub_radius: int, default = 200
        Rolling-ball radius for background subtraction.
    verbose: int, default = 0
        If >=1, show a tqdm over tiles.

    Returns
    -------
    output: np.ndarray
        Deconvolved image (uint16).
    """
    cp.cuda.Device(gpu_id).use()
    # Disable CuPy's FFT plan cache to avoid cross-tile bloat:
    cp.fft._cache.PlanCache(memsize=0)

    # Optional ImageJ background subtraction
    if bkd:
        if ij is None:
            import imagej
            ij = imagej.init()
            delete_ij = True
        else:
            delete_ij = False

        imp_array = ij.py.to_imageplus(image)
        imp_array.setStack(imp_array.getStack().duplicate())
        imp_array.show()
        ij.IJ.run(imp_array, "Subtract Background...", f"rolling={int(bkd_sub_radius)} disable stack")
        imp_array.show()
        ij.py.sync_image(imp_array)
        bkd_output = ij.py.from_java(imp_array.duplicate())
        imp_array.close()
        bkd_image = np.swapaxes(bkd_output.data.transpose(2, 1, 0), 1, 2).clip(0, None).astype(np.float32).copy()
        del image, imp_array, bkd_output
        gc.collect()
        if delete_ij:
            ij.dispose()
            del ij
            gc.collect()
    else:
        bkd_image = image.astype(np.float32, copy=True)
        del image
        gc.collect()

    # Full-frame path if tiling not needed
    if crop_yx >= bkd_image.shape[1] and crop_yx >= bkd_image.shape[2]:
        output = rlgc_biggs(bkd_image, psf, 0, gpu_id, eager_mode=eager_mode)
        output = np.maximum(output, 0.0).astype(np.float32)  # nonnegative

    # Tiled deconvolution with feathered blending
    else:
        output_sum = np.zeros_like(bkd_image, dtype=np.float32)
        output_weight = np.zeros_like(bkd_image, dtype=np.float32)

        crop_size = (bkd_image.shape[0], crop_yx, crop_yx)
        overlap = (0, overlap_yx, overlap_yx)
        slices = Slicer(bkd_image, crop_size=crop_size, overlap=overlap, pad=True)

        if verbose >= 1:
            from tqdm import tqdm
            iterator = enumerate(tqdm(slices, desc="Chunks"))
        else:
            iterator = enumerate(slices)

        for i, (crop, source, destination) in iterator:
            crop_array = rlgc_biggs(crop, psf, 0, gpu_id, eager_mode=eager_mode)
            crop_array = np.maximum(crop_array, 0.0, dtype=np.float32)

            # Resolve tile edge status to decide feathering
            z_slice, y_slice, x_slice = source[1:]

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

        # Blend
        del feather_weight, weighted_crop, weighted_sub, weight_sub
        gc.collect()

        nonzero = output_weight > 0
        output = np.zeros_like(output_sum, dtype=output_sum.dtype)
        output[nonzero] = output_sum[nonzero] / output_weight[nonzero]
        output = np.maximum(output, 0.0, dtype=np.float32)  # nonnegative

        del output_sum, output_weight, nonzero
        gc.collect()

    # Clear caches and pools
    _fft_cache.clear()
    _H_T_cache.clear()
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()

    return output
