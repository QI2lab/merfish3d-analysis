"""Small multiview-stitcher adapters for MERFISH registration."""

import gc
import timeit
from collections.abc import Sequence
from typing import Any

import numpy as np

LOCAL_ROUND_TRANSFORM_KEY = "local_round_registered"


def _diag(message: str, *, enabled: bool) -> None:
    """
    Print one multiview registration diagnostic message when enabled.

    Parameters
    ----------
    message : str
        Diagnostic message body.
    enabled : bool
        If True, print the diagnostic message.

    Returns
    -------
    None
        The message is printed only when diagnostics are enabled.
    """
    if enabled:
        print(f"[multiview-registration] {message}", flush=True)


def _clear_cupy_memory(cp: Any) -> None:
    """
    Release cached CuPy allocations and FFT plans after a registration stage.

    Parameters
    ----------
    cp : Any
        Imported ``cupy`` module.

    Returns
    -------
    None
        CuPy memory pools and FFT plan cache are cleared in-place.
    """
    cp.cuda.Stream.null.synchronize()
    try:
        cp.fft.config.get_plan_cache().clear()
    except Exception:
        pass
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()


def _max_z_projection_gpu(image: np.ndarray, cp: Any) -> Any:
    """
    Compute a maximum Z projection without retaining the full GPU volume.

    Parameters
    ----------
    image : numpy.ndarray
        Input image in Z, Y, X order.
    cp : Any
        Imported ``cupy`` module.

    Returns
    -------
    cupy.ndarray
        Maximum projection over Z as a GPU array.
    """
    image_gpu = cp.asarray(image, dtype=cp.float32)
    projection = cp.max(image_gpu, axis=0)
    del image_gpu
    _clear_cupy_memory(cp)
    return projection


def _overlap_slices_after_translation(
    shape: Sequence[int],
    translation_px: Sequence[float],
) -> tuple[slice, ...] | None:
    """
    Return output slices whose translated coordinates stay inside the input.

    Parameters
    ----------
    shape : Sequence[int]
        Image shape.
    translation_px : Sequence[float]
        Translation used by ``cupyx.scipy.ndimage.affine_transform``. An output
        coordinate ``p`` samples input coordinate ``p + translation_px``.

    Returns
    -------
    tuple[slice, ...] or None
        Valid overlap slices, or None if the translation leaves no overlap.
    """
    slices = []
    for axis_size, axis_translation_px in zip(shape, translation_px, strict=True):
        start = int(np.ceil(max(0.0, -float(axis_translation_px))))
        stop = int(
            np.floor(min(float(axis_size), float(axis_size) - axis_translation_px))
        )
        if stop <= start:
            return None
        slices.append(slice(start, stop))
    return tuple(slices)


def _zyx_dict(values: Sequence[float]) -> dict[str, float]:
    """
    Convert a ZYX vector into the axis dictionary expected by multiview-stitcher.

    Parameters
    ----------
    values : Sequence[float]
        Three values in Z, Y, X order.

    Returns
    -------
    dict[str, float]
        Mapping with ``"z"``, ``"y"``, and ``"x"`` keys.
    """
    values = np.asarray(values, dtype=np.float32).tolist()
    return {"z": float(values[0]), "y": float(values[1]), "x": float(values[2])}


def sim_from_array(
    image: np.ndarray,
    *,
    spacing_zyx_um: Sequence[float],
    origin_zyx_um: Sequence[float] = (0.0, 0.0, 0.0),
) -> Any:
    """
    Create a multiview-stitcher SpatialImage from a ZYX array.

    Parameters
    ----------
    image : numpy.ndarray
        Image data in Z, Y, X axis order.
    spacing_zyx_um : Sequence[float]
        Physical voxel spacing in microns in Z, Y, X order.
    origin_zyx_um : Sequence[float], default=(0.0, 0.0, 0.0)
        Physical origin in microns in Z, Y, X order.

    Returns
    -------
    Any
        SpatialImage object created by ``multiview_stitcher``.
    """
    from multiview_stitcher import spatial_image_utils as si_utils

    return si_utils.get_sim_from_array(
        np.asarray(image),
        dims=("z", "y", "x"),
        scale=_zyx_dict(spacing_zyx_um),
        translation=_zyx_dict(origin_zyx_um),
    )


def msim_from_array(
    image: np.ndarray,
    *,
    spacing_zyx_um: Sequence[float],
    origin_zyx_um: Sequence[float] = (0.0, 0.0, 0.0),
    transform_zyx_um: np.ndarray | None = None,
    transform_key: str = LOCAL_ROUND_TRANSFORM_KEY,
) -> Any:
    """
    Create a MultiscaleSpatialImage with one affine transform.

    Parameters
    ----------
    image : numpy.ndarray
        Image data in Z, Y, X axis order.
    spacing_zyx_um : Sequence[float]
        Physical voxel spacing in microns in Z, Y, X order.
    origin_zyx_um : Sequence[float], default=(0.0, 0.0, 0.0)
        Physical origin in microns in Z, Y, X order.
    transform_zyx_um : numpy.ndarray or None, default=None
        Homogeneous 4x4 affine transform in physical Z, Y, X coordinates. If
        None, an identity transform is stored.
    transform_key : str, default=LOCAL_ROUND_TRANSFORM_KEY
        Name under which the transform is stored in the multiscale image.

    Returns
    -------
    Any
        MultiscaleSpatialImage object with the requested transform metadata.
    """
    from multiview_stitcher import msi_utils

    sim = sim_from_array(
        image, spacing_zyx_um=spacing_zyx_um, origin_zyx_um=origin_zyx_um
    )
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
    transform = (
        np.eye(4, dtype=np.float32) if transform_zyx_um is None else transform_zyx_um
    )
    msi_utils.set_affine_transform(
        msim, np.asarray(transform)[None, ...], transform_key
    )
    return msim


def register_pair_to_fixed(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    spacing_zyx_um: Sequence[float],
    diagnostics: bool = False,
) -> np.ndarray:
    """
    Register a moving image to a fixed image with staged GPU phase correlation.

    The input arrays are interpreted as Z, Y, X images with physical spacing in
    microns. The registration first estimates lateral translation from maximum
    Z projections, warps the moving volume by that lateral estimate, then runs
    phase correlation on the full volume to estimate the residual translation.
    The returned affine maps fixed/reference physical coordinates to
    moving-image physical coordinates, matching the convention expected by
    :func:`warp_array_to_reference_gpu`.

    Parameters
    ----------
    fixed : numpy.ndarray
        Reference image in Z, Y, X order.
    moving : numpy.ndarray
        Image to align to ``fixed``, in Z, Y, X order.
    spacing_zyx_um : Sequence[float]
        Physical voxel spacing in microns in Z, Y, X order.
    diagnostics : bool, default=False
        If True, print detailed timing diagnostics.

    Returns
    -------
    numpy.ndarray
        Homogeneous 4x4 affine transform in physical Z, Y, X coordinates. The
        transform maps coordinates in the fixed reference space to coordinates
        sampled from the moving image, matching the convention expected by
        :func:`warp_array_to_reference_gpu`.
    """
    import cupy as cp
    from cucim.skimage.registration import phase_cross_correlation

    _diag(
        "register_pair_to_fixed_start "
        f"fixed_shape={tuple(int(v) for v in fixed.shape)} "
        f"moving_shape={tuple(int(v) for v in moving.shape)} "
        f"spacing_zyx_um={tuple(float(v) for v in spacing_zyx_um)}",
        enabled=diagnostics,
    )
    if fixed.shape != moving.shape or fixed.ndim != 3:
        raise ValueError(
            "register_pair_to_fixed expects fixed and moving 3D arrays with "
            f"matching shapes, got {fixed.shape!r} and {moving.shape!r}."
        )

    start_time = timeit.default_timer()
    spacing = np.asarray(spacing_zyx_um, dtype=np.float32)
    fixed_projection = _max_z_projection_gpu(fixed, cp)
    moving_projection = _max_z_projection_gpu(moving, cp)
    xy_push_shift_px = phase_cross_correlation(
        fixed_projection,
        moving_projection,
        upsample_factor=10,
        disambiguate=True,
    )[0]
    xy_pull_shift_px = -cp.asnumpy(xy_push_shift_px).astype(np.float32)
    del fixed_projection, moving_projection, xy_push_shift_px
    _clear_cupy_memory(cp)

    xy_transform = np.eye(4, dtype=np.float32)
    xy_transform[1, 3] = float(xy_pull_shift_px[0]) * float(spacing[1])
    xy_transform[2, 3] = float(xy_pull_shift_px[1]) * float(spacing[2])
    moving_xy_registered = warp_array_to_reference_gpu(
        moving,
        transform_zyx_um=xy_transform,
        spacing_zyx_um=spacing,
        reference_shape=fixed.shape,
        order=1,
        diagnostics=diagnostics,
    )

    fixed_gpu = cp.asarray(fixed, dtype=cp.float32)
    moving_xy_registered_gpu = cp.asarray(moving_xy_registered, dtype=cp.float32)
    overlap_slices = _overlap_slices_after_translation(
        fixed.shape,
        (0.0, float(xy_pull_shift_px[0]), float(xy_pull_shift_px[1])),
    )
    if overlap_slices is None:
        residual_push_shift_px = np.zeros(3, dtype=np.float32)
    else:
        residual_push_shift_px = phase_cross_correlation(
            fixed_gpu[overlap_slices],
            moving_xy_registered_gpu[overlap_slices],
            upsample_factor=10,
            disambiguate=True,
        )[0]
        residual_push_shift_px = cp.asnumpy(residual_push_shift_px).astype(np.float32)
    residual_pull_shift_px = -residual_push_shift_px.astype(np.float32, copy=False)
    del fixed_gpu, moving_xy_registered_gpu, moving_xy_registered
    total_shift_px = residual_pull_shift_px.copy()
    total_shift_px[1] += xy_pull_shift_px[0]
    total_shift_px[2] += xy_pull_shift_px[1]

    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = total_shift_px * spacing
    _diag(
        "register_pair_to_fixed_done "
        f"xy_pull_shift_px=(0.000, {float(xy_pull_shift_px[0]):.3f}, {float(xy_pull_shift_px[1]):.3f}) "
        f"residual_pull_shift_px={tuple(float(v) for v in residual_pull_shift_px)} "
        f"total_pull_shift_px={tuple(float(v) for v in total_shift_px)} "
        f"elapsed_s={timeit.default_timer() - start_time:.2f}",
        enabled=diagnostics,
    )
    _clear_cupy_memory(cp)
    return transform


def _score_axial_shift_gpu(
    fixed: Any,
    moving: Any,
    z_shift_px: float,
) -> float:
    """
    Score one axial shift on the GPU over the full valid overlap.

    Parameters
    ----------
    fixed : cupy.ndarray
        Fixed image in Z, Y, X order.
    moving : cupy.ndarray
        Moving image in Z, Y, X order after lateral alignment.
    z_shift_px : float
        Candidate fixed-to-moving z shift in pixels.

    Returns
    -------
    float
        Pearson score after applying the candidate z shift. Only out-of-bounds
        z samples are excluded; all valid Y/X pixels are used.
    """
    import cupy as cp

    if fixed.shape != moving.shape or fixed.ndim != 3:
        return float("-inf")

    num_z = fixed.shape[0]
    z_coords = cp.arange(num_z, dtype=cp.float32) + cp.float32(z_shift_px)
    valid = (z_coords >= 0) & (z_coords <= (num_z - 1))
    if int(cp.sum(valid).item()) < 8:
        return float("-inf")

    fixed_valid = fixed[valid].astype(cp.float32, copy=False)
    z_valid = z_coords[valid]
    z0 = cp.floor(z_valid).astype(cp.int64)
    z1 = cp.minimum(z0 + 1, num_z - 1)
    alpha = (z_valid - z0.astype(cp.float32)).astype(cp.float32)
    moving0 = moving[z0].astype(cp.float32, copy=False)
    moving1 = moving[z1].astype(cp.float32, copy=False)
    moving_interp = (1.0 - alpha)[:, np.newaxis, np.newaxis] * moving0 + alpha[
        :, np.newaxis, np.newaxis
    ] * moving1

    a = fixed_valid.ravel()
    b = moving_interp.ravel()
    a = a - cp.mean(a)
    b = b - cp.mean(b)
    denom = cp.linalg.norm(a) * cp.linalg.norm(b)
    if float(denom) == 0.0:
        return float("-inf")
    return float(cp.dot(a, b) / denom)


def _best_axial_score(
    fixed: Any,
    moving: Any,
    candidates: np.ndarray,
) -> tuple[float, float]:
    """
    Return the z shift and GPU score with maximal full-volume correlation.

    Parameters
    ----------
    fixed : cupy.ndarray
        Fixed image in Z, Y, X order.
    moving : cupy.ndarray
        Moving image in Z, Y, X order after lateral alignment.
    candidates : numpy.ndarray
        Candidate z shifts in pixels.

    Returns
    -------
    tuple[float, float]
        Best z shift in pixels and its Pearson score.
    """
    scores = [
        _score_axial_shift_gpu(fixed, moving, float(candidate))
        for candidate in candidates
    ]
    best_index = int(np.nanargmax(scores))
    return float(candidates[best_index]), float(scores[best_index])


def _refine_axial_translation(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    transform_zyx_um: np.ndarray,
    spacing_zyx_um: Sequence[float],
    radius_px: int,
    step_px: float,
    diagnostics: bool = False,
) -> np.ndarray:
    """
    Refine only the axial component of a phase-correlation transform.

    Parameters
    ----------
    fixed : numpy.ndarray
        Reference image in Z, Y, X order.
    moving : numpy.ndarray
        Moving image in Z, Y, X order.
    transform_zyx_um : numpy.ndarray
        Transform mapping fixed coordinates to moving coordinates.
    spacing_zyx_um : Sequence[float]
        Voxel spacing in microns in Z, Y, X order.
    radius_px : int
        Search radius centered on the phase-derived z shift.
    step_px : float
        Fine search step in pixels.
    diagnostics : bool, default=False
        If True, print detailed timing diagnostics.

    Returns
    -------
    numpy.ndarray
        Transform with the same lateral components and rescored axial
        translation.
    """
    if fixed.shape != moving.shape or fixed.ndim != 3:
        return np.asarray(transform_zyx_um, dtype=np.float32)

    spacing = np.asarray(spacing_zyx_um, dtype=np.float32)
    transform = np.asarray(transform_zyx_um, dtype=np.float32).copy()
    phase_translation_px = transform[:3, 3] / spacing
    center_z_px = round(float(phase_translation_px[0]))

    lateral_transform = transform.copy()
    lateral_transform[0, 3] = 0.0
    lateral_warped = warp_array_to_reference_gpu(
        moving,
        transform_zyx_um=lateral_transform,
        spacing_zyx_um=spacing,
        reference_shape=fixed.shape,
        order=1,
        diagnostics=diagnostics,
    )

    coarse_candidates = np.arange(
        center_z_px - int(radius_px),
        center_z_px + int(radius_px) + 1,
        1.0,
        dtype=np.float32,
    )
    import cupy as cp

    fixed_float = cp.asarray(fixed, dtype=cp.float32)
    moving_float = cp.asarray(lateral_warped, dtype=cp.float32)
    best_integer_z_px, _coarse_score = _best_axial_score(
        fixed_float,
        moving_float,
        coarse_candidates,
    )

    step_px = max(float(step_px), 0.01)
    fine_candidates = np.arange(
        best_integer_z_px - 1.0,
        best_integer_z_px + 1.0 + (0.5 * step_px),
        step_px,
        dtype=np.float32,
    )
    best_z_px, best_score = _best_axial_score(
        fixed_float,
        moving_float,
        fine_candidates,
    )

    refined = transform.copy()
    refined[0, 3] = float(best_z_px) * float(spacing[0])
    _diag(
        "axial_refinement "
        f"phase_z_px={float(phase_translation_px[0]):.3f} "
        f"coarse_z_px={best_integer_z_px:.3f} "
        f"refined_z_px={best_z_px:.3f} "
        f"step_px={step_px:.3f} "
        f"score={best_score:.6f}",
        enabled=diagnostics,
    )
    return refined.astype(np.float32, copy=False)


def _cupy_rankdata_average(values: Any) -> Any:
    """
    Return average ranks for a one-dimensional CuPy array.

    Parameters
    ----------
    values : cupy.ndarray
        One-dimensional values to rank.

    Returns
    -------
    cupy.ndarray
        Average ranks with the same shape as ``values``.
    """
    import cupy as cp

    values = cp.ravel(values)
    if values.size == 0:
        return cp.asarray([], dtype=cp.float32)

    order = cp.argsort(values, kind="stable")
    sorted_values = values[order]
    group_start = cp.empty(values.size, dtype=cp.bool_)
    group_start[0] = True
    group_start[1:] = sorted_values[1:] != sorted_values[:-1]
    group_id = cp.cumsum(group_start, dtype=cp.int64) - 1
    counts = cp.bincount(group_id)
    starts = cp.cumsum(cp.concatenate((cp.asarray([0], dtype=cp.int64), counts[:-1])))
    average_ranks = starts.astype(cp.float32) + (counts.astype(cp.float32) - 1) / 2
    sorted_ranks = average_ranks[group_id]
    ranks = cp.empty(values.size, dtype=cp.float32)
    ranks[order] = sorted_ranks
    return ranks


def _cupy_spearman_correlation(values0: Any, values1: Any) -> float:
    """
    Compute Spearman correlation for two CuPy arrays.

    Parameters
    ----------
    values0 : cupy.ndarray
        First vector.
    values1 : cupy.ndarray
        Second vector.

    Returns
    -------
    float
        Spearman rank correlation, or NaN for empty or constant inputs.
    """
    import cupy as cp

    values0 = cp.ravel(values0)
    values1 = cp.ravel(values1)
    if values0.size == 0 or values1.size == 0:
        return float("nan")

    ranks0 = _cupy_rankdata_average(values0)
    ranks1 = _cupy_rankdata_average(values1)
    ranks0 = ranks0 - cp.mean(ranks0)
    ranks1 = ranks1 - cp.mean(ranks1)
    denom = cp.sqrt(cp.sum(ranks0 * ranks0) * cp.sum(ranks1 * ranks1))
    if float(denom) == 0.0:
        return float("nan")
    return float(cp.sum(ranks0 * ranks1) / denom)


def cucim_phase_correlation_registration(
    fixed_data: Any,
    moving_data: Any,
    disambiguate_region_mode: str | None = None,
    **skimage_phase_corr_kwargs: Any,
) -> dict[str, Any]:
    """
    GPU phase-correlation pairwise registration for multiview-stitcher.

    This function follows the pixel-space pairwise registration contract used by
    ``multiview_stitcher.registration.register``. It mirrors
    ``multiview_stitcher.registration.phase_correlation_registration`` while
    using CuPy, CuCIM, and CuPy ndimage operations for the image-sized work.

    Parameters
    ----------
    fixed_data : Any
        Fixed image data passed by multiview-stitcher. The ``.data`` attribute
        is interpreted as a float array in pixel coordinates.
    moving_data : Any
        Moving image data passed by multiview-stitcher. The ``.data`` attribute
        is interpreted as a float array in pixel coordinates.
    disambiguate_region_mode : str or None, default=None
        Manual shift-disambiguation region mode. If None, matches
        multiview-stitcher's default behavior: ``"intersection"`` when either
        input has NaNs and ``"union"`` otherwise.
    **skimage_phase_corr_kwargs : Any
        Additional keyword arguments forwarded to
        ``cucim.skimage.registration.phase_cross_correlation``.

    Returns
    -------
    dict[str, Any]
        Dictionary with ``"affine_matrix"`` and ``"quality"`` entries, matching
        multiview-stitcher's pairwise registration API.
    """
    import cupy as cp
    from cucim.skimage.exposure import rescale_intensity
    from cucim.skimage.metrics import structural_similarity
    from cucim.skimage.registration import phase_cross_correlation
    from cupyx.scipy import ndimage
    from multiview_stitcher import param_utils

    im0 = cp.asarray(fixed_data.data)
    im1 = cp.asarray(moving_data.data)
    ndim = im0.ndim

    im0, im1 = (
        rescale_intensity(
            im,
            in_range=(float(cp.nanmin(im)), float(cp.nanmax(im))),
            out_range=(0, 1),
        )
        for im in (im0, im1)
    )

    im0nm = cp.isnan(im0)
    im1nm = cp.isnan(im1)
    has_nan = bool(cp.any(im0nm).item() or cp.any(im1nm).item())

    if disambiguate_region_mode is None:
        disambiguate_region_mode = "intersection" if has_nan else "union"

    valid_pixels1 = cp.sum(~im1nm)

    if has_nan:
        im0nn = cp.nan_to_num(im0)
        im1nn = cp.nan_to_num(im1)
    else:
        im0nn = im0
        im1nn = im1

    if "upsample_factor" not in skimage_phase_corr_kwargs:
        skimage_phase_corr_kwargs["upsample_factor"] = 10 if ndim == 2 else 2

    shift_candidates = []
    for normalization in ("phase", None):
        shift_candidates.append(
            phase_cross_correlation(
                im0nn,
                im1nn,
                disambiguate=False,
                normalization=normalization,
                **skimage_phase_corr_kwargs,
            )[0]
        )

    if has_nan:
        shift_candidates.append(
            phase_cross_correlation(
                im0,
                im1,
                reference_mask=im0nm,
                moving_mask=im1nm,
                disambiguate=False,
                **skimage_phase_corr_kwargs,
            )[0]
        )

    max_shift_per_dim = max(max(im0.shape), max(im1.shape))
    data_max = cp.nanmax(cp.stack((cp.nanmax(im0), cp.nanmax(im1))))
    data_min = cp.nanmin(cp.stack((cp.nanmin(im0), cp.nanmin(im1))))
    data_range = float(data_max - data_min)
    im1_min = float(cp.nanmin(im1))

    disambiguate_metric_vals = []
    quality_metric_vals = []

    t_candidates = []
    for shift_candidate in shift_candidates:
        shift_candidate = cp.asnumpy(shift_candidate)
        for s in np.ndindex(
            tuple(1 if shift_candidate[d] == 0 else 4 for d in range(ndim))
        ):
            t_candidate = []
            for d in range(ndim):
                if s[d] == 0:
                    t_candidate.append(shift_candidate[d])
                elif s[d] == 1:
                    t_candidate.append(-shift_candidate[d])
                elif s[d] == 2:
                    t_candidate.append(-(shift_candidate[d] - im1.shape[d]))
                elif s[d] == 3:
                    t_candidate.append(-shift_candidate[d] - im1.shape[d])
            if np.max(np.abs(t_candidate)) < max_shift_per_dim:
                t_candidates.append(t_candidate)

    if not t_candidates:
        return {"affine_matrix": np.eye(ndim + 1, dtype=np.float32), "quality": 1.0}

    def get_bb_from_nanmask(mask: Any) -> list[list[int]]:
        """
        Return per-axis valid bounds from a boolean mask.

        Parameters
        ----------
        mask : Any
            Boolean CuPy mask.

        Returns
        -------
        list[list[int]]
            Inclusive lower and upper bounds for each axis.
        """
        bbs = []
        for idim in range(mask.ndim):
            axes = list(range(mask.ndim))
            axes.remove(idim)
            valids = cp.where(cp.max(mask, axis=tuple(axes)))[0]
            bbs.append([int(cp.min(valids).item()), int(cp.max(valids).item())])
        return bbs

    im0_bb = get_bb_from_nanmask(~im0nm)

    for t_ in t_candidates:
        im1t = ndimage.affine_transform(
            im1,
            cp.asarray(param_utils.affine_from_translation(list(t_))),
            order=1,
            mode="constant",
            cval=cp.nan,
        )
        mask = (~cp.isnan(im1t)) * (~im0nm)

        if bool(cp.all(~mask).item()) or float(cp.sum(mask) / valid_pixels1) < 0.1:
            disambiguate_metric_val = -1
            quality_metric_val = -1
        else:
            im1t_bb = get_bb_from_nanmask(~cp.isnan(im1t))

            if disambiguate_region_mode == "union":
                mask_slices = tuple(
                    slice(
                        min(im0_bb[idim][0], im1t_bb[idim][0]),
                        max(im0_bb[idim][1], im1t_bb[idim][1]) + 1,
                    )
                    for idim in range(ndim)
                )
            elif disambiguate_region_mode == "intersection":
                mask_slices = tuple(
                    slice(
                        max(im0_bb[idim][0], im1t_bb[idim][0]),
                        min(im0_bb[idim][1], im1t_bb[idim][1]) + 1,
                    )
                    for idim in range(ndim)
                )
            else:
                raise ValueError(
                    "disambiguate_region_mode must be 'union', 'intersection', or None."
                )

            if float(cp.nanmax(im1t[mask_slices])) <= im1_min:
                disambiguate_metric_val = -1
                quality_metric_val = -1
                continue

            min_shape = min(im0[mask_slices].shape)
            ssim_win_size = min(7, min_shape - ((min_shape - 1) % 2))
            if ssim_win_size < 3 or float(cp.max(im1t[mask_slices])) <= im1_min:
                disambiguate_metric_val = -1
            else:
                disambiguate_metric_val = float(
                    structural_similarity(
                        cp.nan_to_num(im0[mask_slices]),
                        cp.nan_to_num(im1t[mask_slices]),
                        data_range=data_range,
                        win_size=ssim_win_size,
                    )
                )
            quality_metric_val = _cupy_spearman_correlation(im0[mask], im1t[mask] - 1)

        disambiguate_metric_vals.append(disambiguate_metric_val)
        quality_metric_vals.append(quality_metric_val)

    argmax_index = int(np.nanargmax(disambiguate_metric_vals))
    t = t_candidates[argmax_index]

    return {
        "affine_matrix": param_utils.affine_from_translation(t),
        "quality": quality_metric_vals[argmax_index],
    }


def warp_array_to_reference_gpu(
    image: np.ndarray,
    *,
    transform_zyx_um: np.ndarray,
    spacing_zyx_um: Sequence[float],
    reference_shape: Sequence[int],
    reference_origin_zyx_um: Sequence[float] = (0.0, 0.0, 0.0),
    mode: str = "constant",
    cval: float = 0.0,
    order: int = 1,
    gpu_id: int = 0,
    z_batch_size: int = 4,
    diagnostics: bool = False,
) -> np.ndarray:
    """
    Warp an image into a reference ZYX grid using CuPy affine interpolation.

    The physical transform convention matches the local registration adapter:
    the 4x4 matrix maps output/reference physical coordinates to input/moving
    physical coordinates. The matrix is converted to the pixel-coordinate
    convention expected by ``cupyx.scipy.ndimage.affine_transform``.

    Parameters
    ----------
    image : numpy.ndarray
        Moving image in Z, Y, X order.
    transform_zyx_um : numpy.ndarray
        Homogeneous 4x4 affine transform in physical Z, Y, X coordinates. The
        transform maps output/reference coordinates to input/moving
        coordinates.
    spacing_zyx_um : Sequence[float]
        Physical voxel spacing in microns in Z, Y, X order for both input and
        output grids.
    reference_shape : Sequence[int]
        Output grid shape in Z, Y, X order.
    reference_origin_zyx_um : Sequence[float], default=(0.0, 0.0, 0.0)
        Physical output origin in microns in Z, Y, X order. The moving image is
        assumed to use the same origin convention as the reference grid.
    mode : str, default="constant"
        Boundary mode passed to ``cupyx.scipy.ndimage.affine_transform``.
    cval : float, default=0.0
        Constant fill value used when ``mode="constant"``. This matches the
        old SimpleITK registration path, which filled samples outside the
        moving image with background.
    order : int, default=1
        Interpolation order passed to ``cupyx.scipy.ndimage.affine_transform``.
    gpu_id : int, default=0
        CUDA device ID to use.
    z_batch_size : int, default=4
        Number of output z planes to process per GPU batch. Keeping this small
        avoids allocating full-volume coordinate grids for large tiles.
    diagnostics : bool, default=False
        If True, print detailed timing diagnostics.

    Returns
    -------
    numpy.ndarray
        Warped image sampled on the reference grid.
    """
    import cupy as cp
    from cupyx.scipy import ndimage

    cp.cuda.Device(gpu_id).use()

    spacing = np.asarray(spacing_zyx_um, dtype=np.float32)
    origin = np.asarray(reference_origin_zyx_um, dtype=np.float32)
    transform = np.asarray(transform_zyx_um, dtype=np.float32)
    linear_um = transform[:3, :3]
    translation_um = transform[:3, 3]

    matrix_px = (linear_um * spacing[np.newaxis, :]) / spacing[:, np.newaxis]
    offset_px = (linear_um @ origin + translation_um - origin) / spacing

    _diag(
        "warp_array_to_reference_gpu_start "
        f"image_shape={tuple(int(v) for v in image.shape)} "
        f"reference_shape={tuple(int(v) for v in reference_shape)} "
        f"spacing_zyx_um={tuple(float(v) for v in spacing_zyx_um)} "
        f"mode={mode} "
        f"cval={float(cval)} "
        f"order={order} "
        f"gpu_id={gpu_id}",
        enabled=diagnostics,
    )
    start_time = timeit.default_timer()
    image_gpu = cp.asarray(image)
    warped_gpu = ndimage.affine_transform(
        image_gpu,
        matrix=cp.asarray(matrix_px),
        offset=cp.asarray(offset_px),
        output_shape=tuple(int(v) for v in reference_shape),
        order=order,
        mode=mode,
        cval=float(cval),
    )
    warped = cp.asnumpy(warped_gpu)
    del image_gpu, warped_gpu
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    _diag(
        "warp_array_to_reference_gpu_done "
        f"elapsed_s={timeit.default_timer() - start_time:.2f}",
        enabled=diagnostics,
    )
    return np.asarray(warped)


def warp_array_to_reference_with_affine_and_sofima_flow_gpu(
    image: np.ndarray,
    *,
    transform_zyx_um: np.ndarray,
    spacing_zyx_um: Sequence[float],
    reference_shape: Sequence[int],
    sofima_flow_field_xyz_px: np.ndarray,
    flow_field_stride_zyx_px: Sequence[float],
    flow_field_box_start_xyz_px: Sequence[float],
    reference_origin_zyx_um: Sequence[float] = (0.0, 0.0, 0.0),
    mode: str = "constant",
    cval: float = 0.0,
    order: int = 1,
    gpu_id: int = 0,
    z_batch_size: int = 4,
    diagnostics: bool = False,
) -> np.ndarray:
    """
    Warp an image with a stored affine transform and SOFIMA flow field.

    The image is sampled exactly once. The SOFIMA flow field is interpolated in
    reference pixel space, composed with the stored affine transform, and the
    original moving image is sampled at the composed source coordinates.

    Deformable-field convention
    ---------------------------
    ``sofima_flow_field_xyz_px`` has channel-first shape ``(3, z, y, x)``.
    Channels are ordered ``X, Y, Z`` and spatial axes are ordered ``Z, Y, X``.
    Each vector is a relative displacement in reference pixels from a
    reference-grid coordinate toward the affine-initialized moving image. The
    first map sample is located at ``flow_field_box_start_xyz_px`` in ``X, Y,
    Z`` pixel coordinates. SOFIMA estimates patch-centered vectors, so fields
    produced by :func:`estimate_sofima_flow_field_xyz_px` use half the patch
    size as this origin. The map stride is stored separately in ``Z, Y, X``
    order.

    Parameters
    ----------
    image : numpy.ndarray
        Moving image in native Z, Y, X order.
    transform_zyx_um : numpy.ndarray
        Homogeneous 4x4 physical transform mapping reference Z, Y, X
        coordinates to moving native Z, Y, X coordinates.
    spacing_zyx_um : Sequence[float]
        Voxel spacing in microns in Z, Y, X order.
    reference_shape : Sequence[int]
        Output shape in Z, Y, X order.
    sofima_flow_field_xyz_px : numpy.ndarray
        Relative SOFIMA flow field with channels X, Y, Z and spatial axes Z, Y,
        X. It maps reference pixels toward affine-initialized moving pixels.
    flow_field_stride_zyx_px : Sequence[float]
        Flow-field sampling stride in reference pixels in Z, Y, X order.
    flow_field_box_start_xyz_px : Sequence[float]
        Reference pixel coordinate of the first flow sample in X, Y, Z order.
    reference_origin_zyx_um : Sequence[float], default=(0.0, 0.0, 0.0)
        Physical origin for the reference and moving local grids.
    mode : str, default="constant"
        Boundary mode for flow-field interpolation and image sampling.
    cval : float, default=0.0
        Constant fill value used when sampling outside the flow field or moving
        image.
    order : int, default=1
        Interpolation order for the final image sampling.
    gpu_id : int, default=0
        CUDA device ID to use.
    z_batch_size : int, default=4
        Number of output z planes to process per GPU batch.
    diagnostics : bool, default=False
        If True, print detailed timing diagnostics.

    Returns
    -------
    numpy.ndarray
        Warped image on the reference grid.
    """
    import cupy as cp
    from cupyx.scipy import ndimage

    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {image.shape!r}.")
    if len(reference_shape) != 3:
        raise ValueError("reference_shape must have three ZYX elements.")

    cp.cuda.Device(gpu_id).use()

    ref_shape = tuple(int(v) for v in reference_shape)
    spacing = cp.asarray(spacing_zyx_um, dtype=cp.float32)
    origin = cp.asarray(reference_origin_zyx_um, dtype=cp.float32)
    transform = cp.asarray(transform_zyx_um, dtype=cp.float32)
    flow_field = cp.asarray(sofima_flow_field_xyz_px, dtype=cp.float32)
    if flow_field.ndim != 4:
        raise ValueError("sofima_flow_field_xyz_px must have channel plus ZYX axes.")
    if flow_field.shape[0] != 3 and flow_field.shape[-1] == 3:
        flow_field = cp.moveaxis(flow_field, -1, 0)
    if flow_field.shape[0] != 3:
        raise ValueError("SOFIMA flow field must have three XYZ channels.")

    stride_zyx = cp.asarray(flow_field_stride_zyx_px, dtype=cp.float32)
    box_start_xyz = cp.asarray(flow_field_box_start_xyz_px, dtype=cp.float32)
    box_start_zyx = box_start_xyz[[2, 1, 0]]

    _diag(
        "warp_array_to_reference_with_affine_and_sofima_flow_gpu_start "
        f"image_shape={tuple(int(v) for v in image.shape)} "
        f"reference_shape={ref_shape} "
        f"flow_field_shape={tuple(int(v) for v in flow_field.shape)} "
        f"mode={mode} "
        f"cval={float(cval)} "
        f"order={order} "
        f"gpu_id={gpu_id} "
        f"z_batch_size={int(z_batch_size)}",
        enabled=diagnostics,
    )
    start_time = timeit.default_timer()

    image_gpu = cp.asarray(image)
    warped = np.empty(ref_shape, dtype=np.asarray(image).dtype)
    z_batch_size = max(1, int(z_batch_size))
    y_indices = cp.arange(ref_shape[1], dtype=cp.float32)
    x_indices = cp.arange(ref_shape[2], dtype=cp.float32)
    grid_y, grid_x = cp.meshgrid(y_indices, x_indices, indexing="ij")

    for z_start in range(0, ref_shape[0], z_batch_size):
        z_stop = min(z_start + z_batch_size, ref_shape[0])
        z_indices = cp.arange(z_start, z_stop, dtype=cp.float32)
        grid_z = cp.broadcast_to(
            z_indices[:, cp.newaxis, cp.newaxis],
            (z_stop - z_start, ref_shape[1], ref_shape[2]),
        )
        batch_grid_y = cp.broadcast_to(
            grid_y[cp.newaxis, :, :],
            (z_stop - z_start, ref_shape[1], ref_shape[2]),
        )
        batch_grid_x = cp.broadcast_to(
            grid_x[cp.newaxis, :, :],
            (z_stop - z_start, ref_shape[1], ref_shape[2]),
        )
        flow_coords = cp.stack(
            [
                (grid_z - box_start_zyx[0]) / stride_zyx[0],
                (batch_grid_y - box_start_zyx[1]) / stride_zyx[1],
                (batch_grid_x - box_start_zyx[2]) / stride_zyx[2],
            ],
            axis=0,
        )

        affine_initialized_xyz = []
        for channel_index, identity_channel in enumerate(
            (batch_grid_x, batch_grid_y, grid_z)
        ):
            flow_component = ndimage.map_coordinates(
                flow_field[channel_index],
                flow_coords,
                order=1,
                mode=mode,
                cval=float(cval),
            )
            affine_initialized_xyz.append(identity_channel + flow_component)

        physical_z = affine_initialized_xyz[2] * spacing[0] + origin[0]
        physical_y = affine_initialized_xyz[1] * spacing[1] + origin[1]
        physical_x = affine_initialized_xyz[0] * spacing[2] + origin[2]

        moving_z = (
            transform[0, 0] * physical_z
            + transform[0, 1] * physical_y
            + transform[0, 2] * physical_x
            + transform[0, 3]
        )
        moving_y = (
            transform[1, 0] * physical_z
            + transform[1, 1] * physical_y
            + transform[1, 2] * physical_x
            + transform[1, 3]
        )
        moving_x = (
            transform[2, 0] * physical_z
            + transform[2, 1] * physical_y
            + transform[2, 2] * physical_x
            + transform[2, 3]
        )
        source_coords = cp.stack(
            [
                (moving_z - origin[0]) / spacing[0],
                (moving_y - origin[1]) / spacing[1],
                (moving_x - origin[2]) / spacing[2],
            ],
            axis=0,
        )
        warped_batch = ndimage.map_coordinates(
            image_gpu,
            source_coords,
            order=order,
            mode=mode,
            cval=float(cval),
        )
        warped[z_start:z_stop] = cp.asnumpy(warped_batch)
        del (
            grid_z,
            flow_coords,
            affine_initialized_xyz,
            physical_z,
            physical_y,
            physical_x,
            moving_z,
            moving_y,
            moving_x,
            source_coords,
            warped_batch,
        )

    del (
        image_gpu,
        flow_field,
        grid_y,
        grid_x,
    )
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    _diag(
        "warp_array_to_reference_with_affine_and_sofima_flow_gpu_done "
        f"elapsed_s={timeit.default_timer() - start_time:.2f}",
        enabled=diagnostics,
    )
    return np.asarray(warped)


def transform_points_to_reference(
    points_zyx: np.ndarray,
    *,
    transform_zyx_um: np.ndarray,
    spacing_zyx_um: Sequence[float],
    origin_zyx_um: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Map native moving-image ZYX pixel points into the reference grid.

    Parameters
    ----------
    points_zyx : numpy.ndarray
        Point coordinates in the moving image pixel grid, with columns in Z, Y,
        X order.
    transform_zyx_um : numpy.ndarray
        Homogeneous 4x4 affine transform in physical Z, Y, X coordinates. This
        is the same transform used by :func:`warp_array_to_reference_gpu`.
    spacing_zyx_um : Sequence[float]
        Physical voxel spacing in microns in Z, Y, X order.
    origin_zyx_um : Sequence[float], default=(0.0, 0.0, 0.0)
        Physical origin in microns in Z, Y, X order.

    Returns
    -------
    numpy.ndarray
        Point coordinates in the reference pixel grid, with columns in Z, Y, X
        order.
    """
    points = np.asarray(points_zyx, dtype=np.float32)
    if points.size == 0:
        return points.reshape((-1, 3))
    spacing = np.asarray(spacing_zyx_um, dtype=np.float32)
    origin = np.asarray(origin_zyx_um, dtype=np.float32)
    physical = points * spacing + origin
    homogeneous = np.concatenate(
        [physical, np.ones((physical.shape[0], 1), dtype=np.float32)], axis=1
    )
    reference_physical = homogeneous @ np.linalg.inv(transform_zyx_um).T
    return (reference_physical[:, :3] - origin) / spacing
