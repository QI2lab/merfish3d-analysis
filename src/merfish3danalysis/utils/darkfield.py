import cupy as cp
from cupyx.scipy.special import j1
from cupyx.scipy.ndimage import minimum_filter, gaussian_filter, convolve
import numpy as np
from typing import Any

def window_sum_filter(image2d: cp.ndarray, r: int) -> cp.ndarray:
    """
    Compute local windowed sum via two-pass cumulative sum.

    Parameters
    ----------
    image2d : cp.ndarray
        2D input array.
    r : int
        Radius of the window (half-size), window size = 2*r + 1.

    Returns
    -------
    cp.ndarray
        Summed image of same shape as input.
    """
    # size = 2 * r + 1
    # kernel = cp.ones((size, size), dtype=image2d.dtype)
    # return convolve(image2d, kernel, mode='reflect')
    m, n = image2d.shape
    im_cum = cp.cumsum(image2d, axis=0)
    sum_img = cp.empty_like(image2d)
    sum_img[:r+1, :] = im_cum[r:2*r+1, :]
    sum_img[r+1:m-r, :] = im_cum[2*r+1:m, :] - im_cum[:m-2*r-1, :]
    sum_img[m-r:m, :] = (
        im_cum[m-1, :][None, :].repeat(r, axis=0)
        - im_cum[m-2*r-1:m-r-1, :]
    )
    im_cum2 = cp.cumsum(sum_img, axis=1)
    sum_img[:, :r+1] = im_cum2[:, r:2*r+1]
    sum_img[:, r+1:n-r] = (
        im_cum2[:, 2*r+1:n] - im_cum2[:, :n-2*r-1]
    )
    sum_img[:, n-r:n] = (
        im_cum2[:, n-1][:, None].repeat(r, axis=1)
        - im_cum2[:, n-2*r-1:n-r-1]
    )
    return sum_img


def lpgauss(h: int, w: int, sigma: float) -> cp.ndarray:
    """
    Create a 2D Gaussian low-pass filter in Fourier domain.

    Parameters
    ----------
    h : int
        Height of filter.
    w : int
        Width of filter.
    sigma : float
        Standard deviation of Gaussian.

    Returns
    -------
    cp.ndarray
        Low-pass filter kernel.
    """
    x = cp.arange(-w // 2, w - w // 2)
    y = cp.arange(-h // 2, h - h // 2)
    X, Y = cp.meshgrid(x, y)
    temp = -(X**2 + Y**2) / (sigma**2)
    return cp.fft.ifftshift(cp.exp(temp))


def hpgauss(h: int, w: int, sigma: float) -> cp.ndarray:
    """
    Create a 2D Gaussian high-pass filter.

    Parameters
    ----------
    h : int
        Height of filter.
    w : int
        Width of filter.
    sigma : float
        Standard deviation of Gaussian.

    Returns
    -------
    cp.ndarray
        High-pass filter kernel.
    """
    return 1 - lpgauss(h, w, sigma)


def psf_generator(
    lam: float,
    pixel_size: float,
    na: float,
    w: int,
    factor: float
) -> cp.ndarray:
    """
    Generate a pupil-based PSF matching MATLAB's generator.

    Parameters
    ----------
    lam : float
        Emission wavelength.
    pixel_size : float
        Pixel size.
    na : float
        Numerical aperture.
    w : int
        PSF grid size.
    factor : float
        Scaling factor for PSF support.

    Returns
    -------
    cp.ndarray
        PSF kernel of shape (w, w).
    """
    coords = cp.linspace(0, w - 1, w)
    X, Y = cp.meshgrid(coords, coords)
    scale = 2 * cp.pi * na / lam * pixel_size * factor
    eps = cp.finfo(cp.float32).eps
    R = cp.sqrt(
        cp.minimum(X, cp.abs(X - w))**2
        + cp.minimum(Y, cp.abs(Y - w))**2
    )
    psf = cp.abs(2 * j1(scale * R + eps) / (scale * R + eps))**2
    psf /= psf.sum()
    return cp.fft.fftshift(psf)


def separate_hi_lo(
    image2d: cp.ndarray,
    params: dict[str, Any],
    deg: float,
    divide: float
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Separate high and low frequency components of a 2D slice.

    Parameters
    ----------
    image2d : cp.ndarray
        Input 2D image.
    params : dict
        Dictionary with keys 'emwavelength', 'NA', 'pixelsize', 'factor'.
    deg : float
        Degree for high-pass scaling.
    divide : float
        Divide factor for low-pass.

    Returns
    -------
    hi, lo, lp, el : cp.ndarray
        High-frequency, low-frequency, low-pass kernel, and envelope.
    """
    nx, ny = image2d.shape
    res = 0.5 * params['emwavelength'] / params['NA'] / params['factor']
    k_m = ny / (res / params['pixelsize'])
    kc = int(cp.floor(k_m * 0.2).item())
    sigma_lp = kc * 2 / 2.355
    lp = lpgauss(nx, ny, sigma_lp * 2 * divide)
    hp = hpgauss(nx, ny, sigma_lp * 2 * divide)
    elp = lpgauss(nx, ny, sigma_lp / deg)
    fft_img = cp.fft.fft2(image2d)
    hi = cp.real(cp.fft.ifft2(fft_img * hp))
    lo = cp.real(cp.fft.ifft2(fft_img * lp))
    el = cp.real(cp.fft.ifft2(fft_img * elp))
    return hi, lo, lp, el


def confirm_block(
    params: dict[str, Any],
    lp: cp.ndarray
) -> int:
    """
    Compute PSF block size from low-frequency support.

    Parameters
    ----------
    params : dict
        Dictionary with PSF parameters, including 'Nx'.
    lp : cp.ndarray
        Low-pass filter kernel.

    Returns
    -------
    int
        Block radius where PSF drops below threshold.
    """
    psf = psf_generator(
        params['emwavelength'],
        params['pixelsize'],
        params['NA'],
        params['Nx'],
        params['factor'],
    )
    psf_lo = cp.abs(
        cp.fft.ifft2(cp.fft.fftshift(cp.fft.fft2(psf)))
        * cp.fft.fftshift(lp)
    )
    psf_lo /= psf_lo.max()
    center = params['Nx'] // 2
    for i in range(center, params['Nx']):
        if psf_lo[i, center] < 0.01:
            return i - center
    return params['Nx'] - center


def dehaze_fast2(
    image2d: cp.ndarray,
    omega: float,
    win_size: int,
    el: cp.ndarray,
    dep: float,
    thres: float
) -> cp.ndarray:
    """
    Fast dehazing for a 2D slice.

    Parameters
    ----------
    image2d : cp.ndarray
        Input low-frequency 2D image.
    omega : float
        Weight for transmission estimation.
    win_size : int
        Window size for dark channel.
    el : cp.ndarray
        Low-frequency envelope.
    dep : float
        Depth weight.
    thres : float
        Threshold for dark mask.

    Returns
    -------
    cp.ndarray
        Dehazed 2D image.
    """
    h, w = image2d.shape
    win_size = min(win_size, h, w)
    # Ensure odd window
    if win_size % 2 == 0:
        win_size = max(1, win_size - 1)

    mask = (image2d < thres).astype(cp.float32)
    dc1 = get_dark_channel(image2d * mask, win_size)
    min_atm = get_atmosphere(image2d * mask, dc1)
    dc2 = get_dark_channel(image2d, win_size)
    max_atm = get_atmosphere(image2d, dc2)

    el_c = el - el.min()
    el_max = el_c.max() if el_c.max() != 0 else cp.finfo(cp.float32).eps
    rep_atm = el_c / el_max * (max_atm - min_atm) + min_atm
    rep_atm *= dep

    trans_est = get_transmission_estimate(rep_atm, image2d, omega, win_size)
    refined = guided_filter(image2d, trans_est, 15, 0.001)
    return get_radiance(rep_atm, image2d, refined)


def get_dark_channel(
    image2d: cp.ndarray,
    win_size: int
) -> cp.ndarray:
    """
    Compute dark channel via local minimum with Inf padding.

    Parameters
    ----------
    image2d : cp.ndarray
        Input 2D image.
    win_size : int
        Window size for local minimum.

    Returns
    -------
    cp.ndarray
        Dark channel image.
    """
    return minimum_filter(
        image2d, size=win_size, mode='reflect'
    )


def get_atmosphere(
    image2d: cp.ndarray,
    dark_channel: cp.ndarray
) -> float:
    """
    Estimate atmospheric light from dark channel.

    Parameters
    ----------
    image2d : cp.ndarray
        Input 2D image.
    dark_channel : cp.ndarray
        Dark channel of the input image.

    Returns
    -------
    float
        Estimated atmospheric light.
    """
    m, n = image2d.shape
    n_search = int(cp.floor(m * n * 0.01).item())
    idx = cp.argsort(dark_channel.ravel())[::-1][:n_search]
    return float(image2d.ravel()[idx].sum() / n_search)


def get_transmission_estimate(
    rep_atm: float,
    image2d: cp.ndarray,
    omega: float,
    win_size: int
) -> cp.ndarray:
    """
    Estimate transmission based on dark channel.

    Parameters
    ----------
    rep_atm : float
        Recovered atmospheric light.
    image2d : cp.ndarray
        Input 2D image.
    omega : float
        Weight parameter.
    win_size : int
        Window size for dark channel.

    Returns
    -------
    cp.ndarray
        Transmission estimate.
    """
    return 1 - omega * get_dark_channel(image2d / rep_atm, win_size)


def guided_filter(
    guide: cp.ndarray,
    target: cp.ndarray,
    r: int,
    eps: float
) -> cp.ndarray:
    """
    Perform guided filtering on 2D image.

    Parameters
    ----------
    guide : cp.ndarray
        Guidance image.
    target : cp.ndarray
        Target image to be filtered.
    r : int
        Radius of window.
    eps : float
        Regularization term.

    Returns
    -------
    cp.ndarray
        Filtered image.
    """
    h, w = guide.shape
    # Clamp radius so window fits inside the image
    max_r = min((h - 1) // 2, (w - 1) // 2)
    r = min(r, max_r)

    denom = window_sum_filter(cp.ones((h, w), dtype=guide.dtype), r)
    mean_g = window_sum_filter(guide, r) / denom
    mean_t = window_sum_filter(target, r) / denom
    corr_gg = window_sum_filter(guide * guide, r) / denom
    corr_gt = window_sum_filter(guide * target, r) / denom

    var_g = corr_gg - mean_g**2
    cov_gt = corr_gt - mean_g * mean_t

    a = cov_gt / (var_g + eps)
    b = mean_t - a * mean_g

    mean_a = window_sum_filter(a, r) / denom
    mean_b = window_sum_filter(b, r) / denom

    return mean_a * guide + mean_b


def get_radiance(
    rep_atm: float,
    image2d: cp.ndarray,
    transmission: cp.ndarray
) -> cp.ndarray:
    """
    Recover scene radiance from transmission.

    Parameters
    ----------
    rep_atm : float
        Recovered atmospheric light.
    image2d : cp.ndarray
        Input 2D image.
    transmission : cp.ndarray
        Transmission map.

    Returns
    -------
    cp.ndarray
        Radiance image.
    """
    t = cp.maximum(transmission, 0.1)
    return (image2d - rep_atm) / t + rep_atm


def dark_sectioning(
    input_image: np.ndarray,
    emwavelength: float,
    na: float,
    pixel_size: float,
    factor: float
) -> np.ndarray:
    """
    Perform 3D dark-sectioning dehazing on an image stack.

    Parameters
    ----------
    input_image : np.ndarray
        3D input image in (z, y, x) order.
    emwavelength : float
        Emission wavelength.
    na : float
        Numerical aperture.
    pixel_size : float
        Pixel size.
    factor : float
        Factor for PSF support.

    Returns
    -------
    cp.ndarray
        Dehazed 3D image in (z, y, x) order on GPU.
    """

    if input_image.ndim == 2:
        input_image = input_image[None, ...]

    image0 = cp.asarray(input_image, dtype=cp.float32).transpose(2, 1, 0)
    mn, mx = image0.min(), image0.max()
    image0 = 255 * (image0 - mn) / (mx - mn)

    nx0, ny0, nz = image0.shape
    # Make square by padding smaller dimension
    if ny0 < nx0:
        image0 = cp.pad(
            image0,
            ((0, 0), (0, nx0 - ny0), (0, 0)),
            mode='constant',
            constant_values=0
        )
    elif ny0 > nx0:
        image0 = cp.pad(
            image0,
            ((0, ny0 - nx0), (0, 0), (0, 0)),
            mode='constant',
            constant_values=0
        )
    nx, ny, nz = image0.shape

    pad_size = 40
    padx = nx // pad_size + 1
    pady = ny // pad_size + 1
    pad_mode = 'symmetric'
    denoise = False

    # Initial padding for convolution
    image = cp.zeros((nx + 2*padx, ny + 2*pady, nz), dtype=cp.float32)
    for j in range(nz):
        image[:, :, j] = cp.pad(
            image0[:, :, j],
            ((padx, padx), (pady, pady)),
            mode=pad_mode
        )

    params: dict[str, Any] = {
        'Nx': image.shape[0],
        'Ny': image.shape[1],
        'NA': na,
        'emwavelength': emwavelength,
        'pixelsize': pixel_size,
        'factor': factor
    }
    background = False
    thres = 50
    divide = 0.5

    if background:
        maxtime = 2
        deg_mat = [6, 3, 1.2]
        dep_mat = [3, 2, 2]
        hl_mat = [1, 1, 1]
    else:
        maxtime = 1
        deg_mat = [10]
        dep_mat = [.7]
        hl_mat = [2]

    result_stack = cp.zeros((nx, ny, nz), dtype=cp.float32)
    for t in range(maxtime):
        for j in range(nz):
            hi, lo, lp, el = separate_hi_lo(
                image[:, :, j], params, deg_mat[t], divide
            )
            block = confirm_block(params, lp)
            lo_proc = dehaze_fast2(lo, 0.95, block, el, dep_mat[t], thres)
            res = lo_proc / hl_mat[t] + hi
            result_stack[:, :, j] = res[
                padx:padx+nx, pady:pady+ny
            ]
        for j in range(nz):
            image[:, :, j] = cp.pad(
                result_stack[:, :, j],
                ((padx, padx), (pady, pady)),
                mode=pad_mode
            )

    if denoise:
        result_final = cp.zeros_like(result_stack)
        for j in range(nz):
            tmp = cp.pad(
                result_stack[:, :, j],
                ((padx, padx), (pady, pady)),
                mode=pad_mode
            )
            tmp1 = gaussian_filter(tmp, sigma=1, mode='reflect')
            result_final[:, :, j] = tmp1[padx:padx+nx, pady:pady+ny]
    else:
        result_final = result_stack

    # Crop back to original
    result_final = result_final[:nx0, :ny0, :]
    result_final = result_final / result_final.max() * 65535
    result_final[result_final < 0] = 0

    # Move result back to CPU
    output_cpu = cp.asnumpy(result_final.transpose(2, 1, 0)).astype(np.uint16)


    del result_final
    del result_stack
    del image
    del image0
    del el, lo, hi, lp

    # Free any cached FFT and memory-pool blocks
    try:
        cp.fft.clear_plan_cache()      # drop cuFFT plans
    except AttributeError:
        # older CuPy: no public API, but you already set PlanCache(memsize=0) earlier
        pass

    cp.get_default_memory_pool().free_all_blocks()

    # Now return the CPU array
    return output_cpu