"""Small multiview-stitcher adapters for MERFISH registration."""

import os
import timeit
from collections.abc import Sequence
from typing import Any

import numpy as np

LOCAL_ROUND_TRANSFORM_KEY = "local_round_registered"


def _diagnostics_enabled() -> bool:
    """
    Return whether registration timing diagnostics should be printed.

    Returns
    -------
    bool
        True when ``MERFISH3D_REGISTRATION_DIAGNOSTICS`` is set to a truthy
        value.
    """

    return os.environ.get("MERFISH3D_REGISTRATION_DIAGNOSTICS", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _diag(message: str) -> None:
    """
    Print one multiview registration diagnostic message when enabled.

    Parameters
    ----------
    message : str
        Diagnostic message body.

    Returns
    -------
    None
        The message is printed only when diagnostics are enabled.
    """

    if _diagnostics_enabled():
        print(f"[multiview-registration] {message}", flush=True)


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
    registration_binning: dict[str, int] | None = None,
) -> np.ndarray:
    """
    Register a moving image to a fixed image with GPU phase correlation.

    The input arrays are interpreted as Z, Y, X images with physical spacing in
    microns. Registration is delegated to ``multiview-stitcher`` using a
    CuPy/CuCIM pairwise phase-correlation function. The returned affine maps
    fixed/reference physical coordinates to moving-image physical coordinates,
    matching the convention expected by :func:`warp_array_to_reference_gpu`.

    Parameters
    ----------
    fixed : numpy.ndarray
        Reference image in Z, Y, X order.
    moving : numpy.ndarray
        Image to align to ``fixed``, in Z, Y, X order.
    spacing_zyx_um : Sequence[float]
        Physical voxel spacing in microns in Z, Y, X order.
    registration_binning : dict[str, int] or None, default=None
        Spatial binning passed to ``multiview-stitcher`` before pairwise phase
        registration. If None, all spatial axes are binned by 2.

    Returns
    -------
    numpy.ndarray
        Homogeneous 4x4 affine transform in physical Z, Y, X coordinates. The
        transform maps coordinates in the fixed reference space to coordinates
        sampled from the moving image, matching the convention expected by
        :func:`warp_array_to_reference_gpu`.
    """

    from dask import config as dask_config
    from multiview_stitcher import registration

    if registration_binning is None:
        registration_binning = {"z": 2, "y": 2, "x": 2}

    _diag(
        "register_pair_to_fixed_start "
        f"fixed_shape={tuple(int(v) for v in fixed.shape)} "
        f"moving_shape={tuple(int(v) for v in moving.shape)} "
        f"spacing_zyx_um={tuple(float(v) for v in spacing_zyx_um)} "
        f"registration_binning={registration_binning}"
    )
    fixed_msim = msim_from_array(
        fixed,
        spacing_zyx_um=spacing_zyx_um,
        transform_key=LOCAL_ROUND_TRANSFORM_KEY,
    )
    moving_msim = msim_from_array(
        moving,
        spacing_zyx_um=spacing_zyx_um,
        transform_key=LOCAL_ROUND_TRANSFORM_KEY,
    )
    start_time = timeit.default_timer()
    with dask_config.set(scheduler="single-threaded"):
        transforms = registration.register(
            [fixed_msim, moving_msim],
            reg_channel_index=0,
            transform_key=LOCAL_ROUND_TRANSFORM_KEY,
            new_transform_key=LOCAL_ROUND_TRANSFORM_KEY,
            pairwise_reg_func=cucim_phase_correlation_registration,
            registration_binning=registration_binning,
            groupwise_resolution_kwargs={
                "reference_view": 0,
                "transform": "translation",
            },
        )
    _diag(
        f"register_pair_to_fixed_done elapsed_s={timeit.default_timer() - start_time:.2f}"
    )
    return np.asarray(transforms[1].values[0], dtype=np.float32)


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
    mode: str = "nearest",
    order: int = 1,
    gpu_id: int = 0,
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
    mode : str, default="nearest"
        Boundary mode passed to ``cupyx.scipy.ndimage.affine_transform``.
    order : int, default=1
        Interpolation order passed to ``cupyx.scipy.ndimage.affine_transform``.
    gpu_id : int, default=0
        CUDA device ID to use.

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
        f"order={order} "
        f"gpu_id={gpu_id}"
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
    )
    warped = cp.asnumpy(warped_gpu)
    del image_gpu, warped_gpu
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    _diag(
        "warp_array_to_reference_gpu_done "
        f"elapsed_s={timeit.default_timer() - start_time:.2f}"
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
