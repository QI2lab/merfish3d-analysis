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

gc_update_kernel = ElementwiseKernel(
    'float32 recon, float32 gradient, float32 consensus_map, float32 H_T_ones',
    'float32 recon_next',
    '''
    float denom = (H_T_ones > 1e-6f) ? H_T_ones : 1e-6f;
    float step  = recon / denom;
    float upd   = recon + gradient * step;
    if (consensus_map < 0) {
        upd = recon;
    }
    if (upd < 0) {
        upd = 0;
    }
    recon_next = upd;
    ''',
    'gc_update_kernel'
)

_fft_cache: dict[tuple[int,int,int], tuple[cp.ndarray, cp.ndarray]] = {}
_H_T_cache: dict[tuple[int,int,int], cp.ndarray] = {}

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
            window[:feather] = ramp[:feather]          # ramp up
            window[-feather:] = ramp[feather:]         # ramp down
        return window

    y_win = cosine_taper(shape[1], feather_px)
    x_win = cosine_taper(shape[2], feather_px)
    weight2d = np.outer(y_win, x_win)
    weight = np.broadcast_to(weight2d[None, :, :], shape)
    return weight

def next_gpu_fft_size(x: int) -> int:
    """
    Pick the smallest FFT‑friendly size ≥ x whose only prime factors are 2 or 3.
    GPUs (cuFFT) are super‑fast on radix‑2 and radix‑3, so this trades a few extra
    pixels (vs. a pure power‑of‑two) for peak throughput.

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

    # scan upward until we hit a 2–3–smooth number
    n = x
    while True:
        m = n
        # pull out all factors of 2
        while (m % 2) == 0:
            m //= 2
        # pull out all factors of 3
        while (m % 3) == 0:
            m //= 3
        if m == 1:
            return n
        n += 1

def pad_z(image: np.ndarray, bkd: int = 0) -> tuple[np.ndarray, int, int]:
    """Pad z-axis of 3D array by the next multiple of 32 (zyx order).

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
        Amount of padding at the start of the y-axis.
    pad_z_after: int
        Amount of padding at the end of the y-axis.
    """
    z, y, x = image.shape
    new_z = next_gpu_fft_size(z)
    pad_z = new_z - z
    pad_z_before = pad_z // 2
    pad_z_after = pad_z - pad_z_before
    pad_width = ((pad_z_before, pad_z_after),(0, 0), (0, 0))
    padded_image = np.pad(
        (image.astype(np.float32) - float(bkd)).clip(0, 2**16 - 1),
        pad_width, mode="reflect"
    )
    return padded_image, pad_z_before, pad_z_after

def remove_padding_z(
    padded_image: np.ndarray,
    pad_z_before: int,
    pad_z_after: int
) -> cp.ndarray:
    """Remove y-axis padding added by pad_z.

    Parameters
    ----------
    padded_image: np.ndarray
        Padded 3D image.
    pad_z_before: int
        Amount of padding at the start of the z-axis.
    pad_z_after: int
        Amount of padding at the end of the z-axis.

    Returns
    -------
    image: np.ndarray
        Unpadded 3D image.
    """
    if pad_z_before == 0 and pad_z_after == 0:
        return padded_image
    else:
        image = padded_image[pad_z_before:-pad_z_after, :, :]
        return image

def pad_psf(psf_temp: cp.ndarray, image_shape: tuple[int, int, int]) -> cp.ndarray:
    """Pad and center a PSF to match the target image shape.

    Parameters
    ----------
    psf_temp: cp.ndarray
        Original PSF array.
    image_shape: tuple[int, int, int]
        Desired shape for the padded PSF.

    Returns
    -------
    psf: cp.ndarray
        Padded and centered PSF, normalized to unit sum.
    """
    psf = cp.zeros(image_shape, dtype=cp.float32)
    psf[:psf_temp.shape[0], :psf_temp.shape[1], :psf_temp.shape[2]] = psf_temp
    for axis, axis_size in enumerate(psf.shape):
        psf = cp.roll(psf, int(axis_size / 2), axis=axis)
    for axis, axis_size in enumerate(psf_temp.shape):
        psf = cp.roll(psf, -int(axis_size / 2), axis=axis)
    psf = cp.fft.ifftshift(psf)
    psf = psf / cp.sum(psf)
    psf = cp.maximum(psf, 0.0)  
    return psf.astype(cp.float32)

def fft_conv(image: cp.ndarray, OTF: cp.ndarray, shape) -> cp.ndarray:
    """Perform convolution via FFT: irfftn(rfftn(image) * OTF).

    Parameters
    ----------
    image: cp.ndarray
        Input image in object space.
    OTF: cp.ndarray
        Frequency-domain transfer function (rfftn of PSF).
    shape: tuple[int, int, int]
        Target shape for the inverse FFT.

    Returns
    -------
    result: cp.ndarray
        Convolved image in object space.
    """
    if shape not in _fft_cache:
        z, y, x = shape
        freq_shape = (z, y, x // 2 + 1)
        fft_buf = cp.empty(freq_shape, dtype=cp.complex64)
        ifft_buf = cp.empty(shape, dtype=cp.float32)
        _fft_cache[shape] = (fft_buf, ifft_buf)
    fft_buf, ifft_buf = _fft_cache[shape]
    fft_buf[...] = cp.fft.rfftn(image)
    fft_buf[...] *= OTF
    ifft_buf[...] = cp.fft.irfftn(fft_buf, s=shape)
    cp.clip(ifft_buf, 0.0, 2**16-1, out=ifft_buf)
    return ifft_buf

def kl_div(p: cp.ndarray, q: cp.ndarray) -> float:
    """Compute Kullback–Leibler divergence between two distributions.

    Parameters
    ----------
    p: cp.ndarray
        First distribution (nonnegative).
    q: cp.ndarray
        Second distribution (nonnegative).

    Returns
    -------
    kldiv: float
        Sum over all elements of p * (log(p) - log(q)), with NaNs set to zero.
    """
    p = p + 1e-4
    q = q + 1e-4
    p = p / cp.sum(p)
    q = q / cp.sum(q)
    kldiv = p * (cp.log(p) - cp.log(q))
    kldiv[cp.isnan(kldiv)] = 0
    kldiv = cp.sum(kldiv)
    return float(kldiv)

def _get_H_T_ones(otf: cp.ndarray, otfT: cp.ndarray, shape: tuple[int,int,int]) -> cp.ndarray:
    if shape in _H_T_cache:
        return _H_T_cache[shape]
    ones = cp.ones(shape, dtype=cp.float32)
    freq_shape = (shape[0], shape[1], shape[2] // 2 + 1)
    fft_buf = cp.empty(freq_shape, dtype=cp.complex64)
    ifft_buf = cp.empty(shape, dtype=cp.float32)
    fft_buf[...] = cp.fft.rfftn(ones)
    fft_buf[...] *= otf
    ifft_buf[...] = cp.fft.irfftn(fft_buf, s=shape)
    fft_buf[...] = cp.fft.rfftn(ifft_buf)
    fft_buf[...] *= otfT
    ifft_buf[...] = cp.fft.irfftn(fft_buf, s=shape)
    _H_T_cache[shape] = ifft_buf
    return _H_T_cache[shape]

def rlgc_biggs(
    image: np.ndarray,
    psf: np.ndarray,
    bkd: int = 0,
    gpu_id: int = 0,
    otf: cp.ndarray = None,
    otfT: cp.ndarray = None,
    eager_mode: bool = False
) -> np.ndarray:
    """
    Andrew–Biggs accelerated RLGC deconvolution with Gradient Consensus.

    This routine performs Richardson–Lucy + Gradient Consensus (GC)
    deconvolution on a 3D image stack, using a Biggs–Andrews momentum
    term and an early stopping criterion based on Kullback–Leibler divergence.

    Parameters
    ----------
    image: np.ndarray
        3D image (zyx) to be deconvolved.
    psf: np.ndarray
        3D point-spread function. If `otf` and `otfT` are None, this PSF
        will be padded and transformed to form the OTF internally.
    bkd: int, default = 0
        Constant background to subtract before deconvolution (default=0).
    gpu_id: int, default = 0
        Which GPU to use
    otf: cp.ndarray, optional
        Precomputed rfftn of the padded PSF. If provided, `psf` is ignored.
    otfT: cp.ndarray, optional
        Conjugate of `otf`. Must match shape of `otf`.
    eager_mode: bool, default = False
        Use stricter iteration cutoff, potentially leading to over-fitting.

    Returns
    -------
    output: np.ndarray
        Deconvolved 3D image, clipped to [0, 2^16-1] and cast to uint16.
    """
    cp.cuda.Device(gpu_id).use()
    rng = cp.random.default_rng(42)
    
    if psf.ndim == 2:
        psf = np.expand_dims(psf,axis=0)
    if image.ndim == 2:
        image = np.expand_dims(image,axis=0)
    _, y0, x0 = image.shape  # image is (z,y,x)

    new_y = next_gpu_fft_size(y0)  # no +1 here
    new_x = next_gpu_fft_size(x0)

    pad_y = new_y - y0
    pad_x = new_x - x0

    pad_y_before = pad_y // 2
    pad_y_after  = pad_y - pad_y_before

    pad_x_before = pad_x // 2
    pad_x_after  = pad_x - pad_x_before

    image_padded = np.pad(
        image,
        pad_width=((0, 0),
                (pad_y_before, pad_y_after),
                (pad_x_before, pad_x_after)),
        mode="reflect"
    )

    if image.ndim == 3:
        image_gpu, pad_z_before, pad_z_after = pad_z(cp.asarray(image_padded, dtype=cp.float32))
    else:
        image_gpu = cp.asarray(image_padded, dtype=cp.float32)
        image_gpu = image_gpu[cp.newaxis, ...]

    otf = None
    otfT = None

    if isinstance(psf, np.ndarray) and otf is None and otfT is None:
        psf_gpu = pad_psf(cp.asarray(psf, dtype=cp.float32), image_gpu.shape)
        otf = cp.fft.rfftn(psf_gpu)
        otfT = cp.conjugate(otf)
        del psf_gpu
        cp.get_default_memory_pool().free_all_blocks()
    otfotfT = cp.real(otf * otfT).astype(cp.float32)
    shape = image_gpu.shape
    z, y, x = shape
    recon = image_gpu.copy().astype(cp.float32)
    previous_recon = cp.empty_like(recon)
    previous_recon[...] = recon
    recon_next = cp.empty_like(recon)
    split1 = cp.empty_like(recon)
    split2 = cp.empty_like(recon)
    Hu = cp.empty_like(recon)
    Hu_safe = cp.empty_like(recon)
    HTratio1 = cp.empty_like(recon)
    HTratio2 = cp.empty_like(recon)
    HTratio = cp.empty_like(recon)
    consensus_map = cp.empty_like(recon)
    g1 = cp.zeros_like(recon)
    g2 = cp.zeros_like(recon)
    H_T_ones = _get_H_T_ones(otf, otfT, shape)
    prev_kld1 = np.inf
    prev_kld2 = np.inf
    num_iters = 0
    start_time = timeit.default_timer()
    while True:
        iter_start_time = timeit.default_timer()
        split1[...] = rng.binomial(image_gpu.astype(cp.int64), p=0.5).astype(cp.float32)
        cp.subtract(image_gpu, split1, out=split2)
        if num_iters >= 2:
            numerator = cp.sum(g1 * g2)
            denominator = cp.sum(g2 * g2) + 1e-12
            alpha = numerator / denominator
            alpha = cp.clip(alpha, 0.0, 1.0)
            alpha = float(alpha)
        else:
            alpha = 0.0
        temp = recon - previous_recon
        cp.multiply(temp, alpha, out=temp)
        cp.add(recon, temp, out=recon_next)
        recon, recon_next = recon_next, recon
        previous_recon[...] = recon
        Hu[...] = fft_conv(recon, otf, shape)
        kld1 = kl_div(Hu, split1)
        kld2 = kl_div(Hu, split2)

        if not(eager_mode):
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
        cp.add(Hu, 1e-12, out=Hu_safe)
        cp.divide(split1, Hu_safe, out=split1)
        cp.subtract(split1, 0.5, out=split1)
        HTratio1[...] = fft_conv(split1, otfT, shape)
        cp.divide(split2, Hu_safe, out=split2)
        cp.subtract(split2, 0.5, out=split2)
        HTratio2[...] = fft_conv(split2, otfT, shape)
        cp.add(HTratio1, HTratio2, out=HTratio)
        cp.multiply(HTratio1, HTratio2, out=split1)
        consensus_map[...] = fft_conv(split1, otfotfT, shape)
        gc_update_kernel(
            recon,
            HTratio,
            consensus_map,
            H_T_ones,
            recon_next
        )
        temp_g2 = recon_next - recon
        g2[...] = g1
        g1[...] = temp_g2
        recon[...] = recon_next
        num_iters += 1
        if DEBUG:
            calc_time = timeit.default_timer() - iter_start_time
            print(
                f"Iteration {num_iters:03d} completed in {calc_time:.3f}s. "
                f"KLDs: {kld1:.4f} (split1), {kld2:.4f} (split2)."
            )
    recon = cp.clip(recon, 0, 2**16 - 1).astype(cp.float32)
    if image.ndim == 3:
        recon = remove_padding_z(recon,pad_z_before,pad_z_after)
    else:
        recon = cp.squeeze(recon)

    y_end = -pad_y_after if pad_y_after > 0 else None
    x_end = -pad_x_after if pad_x_after > 0 else None
    recon = recon[:, pad_y_before:y_end, pad_x_before:x_end]

    recon_cpu = cp.asnumpy(recon).astype(np.float32)
    del recon_next, g1, g2, H_T_ones, recon, previous_recon, split1, split2
    if num_iters >= 2:
        del numerator, denominator, alpha, temp, temp_g2
    del Hu, Hu_safe, HTratio1, HTratio2, HTratio, consensus_map, otf, otfT, otfotfT, image_gpu
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
    ij = None,
    bkd_sub_radius: int = 200,
    verbose: int = 0
) -> np.ndarray:
    """Chunked RLGC deconvolution.
    
    Parameters
    ----------
    image: np.ndarray
        3D image to be deconvolved.
    psf: np.ndarray
        point spread function (PSF) to use for deconvolution.
    gpu_id: int, default = 0
        which GPU to use
    crop_yx: int, default = 1500
        size of the chunk to process at a time.
    overlap_yx: int, default = 32
        size of the overlap between chunks.
    eager_mode: bool, default = False
        Use stricter iteration cutoff, potentially leading to over-fitting.
    bkd: bool, default = False
        subtract background using imagej
    ij: imagej instance, default = None
        if provided, use this imagej instance for background subtraction.
    bkd_sub_radius: int, default = 200
        rolling ball radius for imagej background subtraction.
        
    Returns
    -------
    output: np.ndarray
        deconvolved image.
    """
    

    cp.cuda.Device(gpu_id).use()
    cp.fft._cache.PlanCache(memsize=0)

    # if requested, subtract background before deconvolution. This can help with registration across rounds.
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
        ij.IJ.run(imp_array,"Subtract Background...", f"rolling={int(bkd_sub_radius)} disable stack")
        imp_array.show()
        ij.py.sync_image(imp_array)
        bkd_output = ij.py.from_java(imp_array.duplicate())
        imp_array.close()
        bkd_image = np.swapaxes(bkd_output.data.transpose(2,1,0),1,2).clip(0,2**16-1).astype(np.uint16).copy()
        del image, imp_array, bkd_output
        gc.collect()
        if delete_ij:
            ij.dispose()
            del ij
            gc.collect()
    else:
        bkd_image = image.copy()
        del image
        gc.collect()

    # Check if deconvolution is tiled. If not, do full deconvolution in one go
    if crop_yx >= bkd_image.shape[1] and crop_yx >= bkd_image.shape[2]:
        output = rlgc_biggs(bkd_image, psf, 0, gpu_id, eager_mode=eager_mode)
        output = output.clip(0,2**16-1).astype(np.uint16)

    # Tiled deconvolution with feathered weighting across tiles
    else:
        output_sum   = np.zeros_like(bkd_image, dtype=np.float32)
        output_weight = np.zeros_like(bkd_image, dtype=np.float32)
        crop_size = (bkd_image.shape[0], crop_yx, crop_yx)
        overlap = (0, overlap_yx, overlap_yx)
        slices = Slicer(bkd_image, crop_size=crop_size, overlap=overlap, pad = True)

        if verbose >= 1:
            from tqdm import tqdm
            iterator = enumerate(tqdm(slices, desc="Chunks"))
        else:
            iterator = enumerate(slices)
        
        for i, (crop, source, destination) in iterator:
            crop_array = rlgc_biggs(crop, psf, bkd, gpu_id, eager_mode)
             # --- Parse slices ---
            # Get source slices
            z_slice, y_slice, x_slice = source[1:]

            # Convert negative stops to absolute positions
            def resolve_slice(s: slice, dim: int) -> tuple[int, int]:
                start = s.start if s.start is not None else 0
                stop = s.stop if s.stop is not None else dim
                if stop < 0:
                    stop = dim + stop
                return start, stop

            y_start, y_stop = resolve_slice(y_slice, crop.shape[1])
            x_start, x_stop = resolve_slice(x_slice, crop.shape[2])

            # Determine edge tiles correctly
            is_y_edge = y_start == 0 or y_stop == crop.shape[1]
            is_x_edge = x_start == 0 or x_stop == crop.shape[2]

            
            if is_y_edge or is_x_edge:
                feather_weight = np.ones_like(crop_array, dtype=np.float32)
            else:
                feather_weight = make_feather_weight(crop.shape, feather_px=overlap_yx)

            # --- Apply weight to valid region only ---
            weighted_crop = crop_array * feather_weight
            weighted_sub = weighted_crop[source]
            weight_sub = feather_weight[source]

            output_sum[destination] += weighted_sub
            output_weight[destination] += weight_sub

        del feather_weight, weighted_crop, weighted_sub, weight_sub
        gc.collect()
        nonzero = output_weight > 0
        output  = np.zeros_like(output_sum, dtype=output_sum.dtype)
        output[nonzero] = output_sum[nonzero] / output_weight[nonzero]
        output = output.clip(0,2**16-1).astype(np.uint16)
        del output_sum, output_weight, nonzero
        gc.collect()

    _fft_cache.clear()
    _H_T_cache.clear()
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()

    return output

    

    