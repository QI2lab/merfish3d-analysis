"""Richardson-Lucy Gradient Consensus deconvolution.

Original idea for gradient consensus deconvolution from: James Maton and Andrew York, https://zenodo.org/records/10278919

RLGC code based on James Manton's implementation, https://colab.research.google.com/drive/1mfVNSCaYHz1g56g92xBkIoa8190XNJpJ

Biggs-Andrews acceleration based on their 1997 paper, https://doi.org/10.1364/AO.36.001766
"""

import cupy as cp
import numpy as np
from cupy import ElementwiseKernel
from ryomen import Slicer
from tqdm import tqdm
import timeit
import gc

DEBUG = False

gc_update_kernel = ElementwiseKernel(
    'float32 recon, float32 gradient, float32 consensus_map, float32 H_T_ones',
    'float32 recon_next',
    '''
    float step = recon / H_T_ones;
    float upd  = recon + gradient * step;
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

def next_multiple_of_32(x: int) -> int:
    """Determine the next multiple of 64 greater than or equal to x.
    
    Parameters
    ----------
    x: int
        The input integer to round up to the next multiple of 32.
        
    Returns
    -------
    next__x: int
        The next multiple of 32 that is greater than or equal to x.
    """
    next_32_x = int(np.ceil((x + 15) / 15)) * 16
    return next_32_x

def pad_z(image: np.ndarray, bkd: int) -> tuple[np.ndarray, int, int]:
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
    new_z = next_multiple_of_32(z)
    pad_z = new_z - z
    pad_z_before = pad_z // 2
    pad_z_after = pad_z - pad_z_before
    pad_width = ((pad_z_before, pad_z_after),(0, 0), (0, 0))
    padded_image = np.pad(
        (image.astype(np.float32) - float(bkd)).clip(0, 2**16 - 1).astype(np.uint16),
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
    image = padded_image[pad_z_before:-pad_z_after, :, :]
    return image

def pad_psf_new(psf_temp: cp.ndarray, image_shape: tuple[int, int, int]) -> cp.ndarray:
    """Pad and center a PSF to match the target image shape.

    This will crop or zero‑pad psf_temp so that its center voxel
    ends up at the center of an array of shape `image_shape`.  After
    embedding, the result is fft‑shifted (center->corner), normalized,
    and clipped for numeric stability.

    Parameters
    ----------
    psf_temp : cp.ndarray
        Original PSF volume.
    image_shape : tuple of int (3,)
        Desired output shape (z, y, x).

    Returns
    -------
    cp.ndarray
        Padded, centered, fft‑shifted, and unit‑sum PSF of shape `image_shape`.
    """
    # prepare output
    psf = cp.zeros(image_shape, dtype=psf_temp.dtype)

    in_shape = psf_temp.shape
    # centers
    out_ctr = [dim // 2 for dim in image_shape]
    in_ctr  = [dim // 2 for dim in in_shape]

    # build slices for each axis
    slices_in  = []
    slices_out = []
    for ax in range(3):
        N = image_shape[ax]
        M = in_shape[ax]
        # how much to shift input so its center lands at output center
        shift = out_ctr[ax] - in_ctr[ax]

        # input slice bounds
        i0 = max(0, -shift)
        i1 = min(M, N - shift if shift >= 0 else N)

        # output slice bounds
        o0 = max(0, shift)
        o1 = o0 + (i1 - i0)

        slices_in.append(slice(i0, i1))
        slices_out.append(slice(o0, o1))

    # copy the overlapping region
    psf[slices_out[0], slices_out[1], slices_out[2]] = \
        psf_temp[slices_in[0], slices_in[1], slices_in[2]]

    # move center→corner for FFT convolution
    psf = cp.fft.ifftshift(psf)

    # normalize to unit sum
    total = psf.sum()
    if total != 0:
        psf = psf / total

    # clip for stability
    return cp.clip(psf, a_min=1e-12, a_max=2**16-1).astype(cp.float32)


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
    return cp.clip(psf,a_min=1e-12, a_max=2**16-1).astype(cp.float32)

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
    return cp.clip(ifft_buf,a_min=1e-12, a_max=2**16-1)

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
    return cp.clip(ifft_buf,a_min=1e-12, a_max=2**16-1).astype(cp.float32)

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
    if image.ndim == 3:
        image_gpu =cp.asarray(image, dtype=cp.float32)
    else:
        image_gpu = cp.asarray(image, dtype=cp.float32)
        image_gpu = image_gpu[cp.newaxis, ...]
    if isinstance(psf, np.ndarray) and otf is None and otfT is None:
        psf_gpu = pad_psf_new(cp.asarray(psf, dtype=cp.float32), image_gpu.shape)
        otf = cp.fft.rfftn(psf_gpu)
        otfT = cp.conjugate(otf)
        del psf_gpu
        cp.get_default_memory_pool().free_all_blocks()
    otfotfT = cp.clip(cp.real(otf * otfT).astype(cp.float32),a_min=1e-12, a_max=2**16-1).astype(cp.float32)
    shape = image_gpu.shape
    z, y, x = shape
    recon = cp.full(shape, cp.mean(image_gpu), dtype=cp.float32)
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
    recon = cp.clip(recon, 0, 2**16 - 1).astype(cp.uint16)
    if not(image.ndim == 3):
        recon = cp.squeeze(recon)

    recon_cpu = cp.asnumpy(recon).astype(np.uint16)
    del recon_next, g1, g2, H_T_ones, recon, temp_g2, previous_recon, split1, split2
    del numerator, denominator, alpha, temp
    del Hu, Hu_safe, HTratio1, HTratio2, HTratio, consensus_map, otf, otfT, otfotfT, image_gpu
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return recon_cpu

def chunked_rlgc(
    image: np.ndarray, 
    psf: np.ndarray,
    gpu_id: int = 0,
    crop_z: int = 36,
    overlap_z: int = 12,
    bkd: int = 0,
    eager_mode: bool = False
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
    crop_size: int, default = 512
        size of the chunk to process at a time.
    overlap_size: int, default = 32
        size of the overlap between chunks.
    bkd: int, default = 0
        background value to subtract from the image.
    eager_mode: bool, default = False
        Use stricter iteration cutoff, potentially leading to over-fitting.
        
    Returns
    -------
    output: np.ndarray
        deconvolved image.
    """
    
    cp.cuda.Device(gpu_id).use()
    cp.fft._cache.PlanCache(memsize=0)
    if image.ndim == 3:
        image_padded, pad_z_before, pad_z_after = pad_z(
            image, bkd
        )
    image_padded = np.pad(image_padded,pad_width=((0,0),(128,128),(128,128)),mode="symmetric")

    output = np.zeros_like(image_padded)
    crop_size = (crop_z, image_padded.shape[-2], image_padded.shape[-1])
    overlap = (overlap_z, 0, 0)
    slices = Slicer(image_padded, crop_size=crop_size, overlap=overlap, pad = True)
    #for crop, source, destination in tqdm(slices,desc="decon chunk:",leave=False):
    for crop, source, destination in slices:
        crop_array = rlgc_biggs(crop, psf, bkd, gpu_id, eager_mode=eager_mode)
        output[destination] = crop_array[source]
    if image.ndim == 3:
        output = remove_padding_z(output,pad_z_before,pad_z_after)
    output = output[:, 128:-128, 128:-128]
    _fft_cache.clear()
    _H_T_cache.clear()
    cp.get_default_memory_pool().free_all_blocks()
    return output