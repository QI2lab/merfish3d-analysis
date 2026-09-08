"""Small multiview-stitcher adapters for MERFISH registration."""

import gc
import timeit
from collections.abc import Sequence
from typing import Any

import numpy as np


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


def _maximum_overlap_phase_shift_px(
    shift_px: Sequence[float],
    shape: Sequence[int],
) -> np.ndarray:
    """
    Select periodic phase-shift aliases with maximum image overlap.

    FFT phase correlation identifies shifts modulo each axis length. Real-space
    disambiguation can therefore report a shift close to ``+/-axis_size`` that
    overlaps only a small part of the images even when the equivalent
    near-zero shift is physically supported. For equal-shaped local-
    registration images, the representative in
    ``[-axis_size / 2, axis_size / 2)`` retains at least half of every field of
    view.

    Parameters
    ----------
    shift_px : Sequence[float]
        Phase-correlation shift in pixels for each axis.
    shape : Sequence[int]
        Phase-correlation image shape in the same axis order.

    Returns
    -------
    numpy.ndarray
        Equivalent periodic shifts with maximum overlap on every axis.
    """
    shift = np.asarray(shift_px, dtype=np.float64)
    period = np.asarray(shape, dtype=np.float64)
    if shift.ndim != 1 or period.shape != shift.shape:
        raise ValueError("shift_px and shape must be one-dimensional and equal length.")
    if np.any(period <= 0):
        raise ValueError("All image dimensions must be positive.")
    return np.remainder(shift + period / 2.0, period) - period / 2.0


def _translation_overlap_slices(
    shape: Sequence[int],
    pull_shift_px: Sequence[float],
) -> tuple[tuple[slice, ...], tuple[slice, ...]] | None:
    """
    Return fixed and moving slices for an integerized pull translation.

    Parameters
    ----------
    shape : Sequence[int]
        Equal fixed and moving image shape.
    pull_shift_px : Sequence[float]
        Translation for which output coordinate ``p`` samples moving
        coordinate ``p + pull_shift_px``.

    Returns
    -------
    tuple[tuple[slice, ...], tuple[slice, ...]] or None
        Matching fixed and moving overlap slices, or ``None`` when there is no
        overlap.
    """
    fixed_slices = []
    moving_slices = []
    for axis_size, axis_shift_px in zip(shape, pull_shift_px, strict=True):
        integer_shift = int(np.rint(float(axis_shift_px)))
        fixed_start = max(0, -integer_shift)
        fixed_stop = min(int(axis_size), int(axis_size) - integer_shift)
        if fixed_stop <= fixed_start:
            return None
        fixed_slices.append(slice(fixed_start, fixed_stop))
        moving_slices.append(
            slice(fixed_start + integer_shift, fixed_stop + integer_shift)
        )
    return tuple(fixed_slices), tuple(moving_slices)


def _overlap_weighted_translation_score(
    fixed: Any,
    moving: Any,
    *,
    pull_shift_px: Sequence[float],
    array_module: Any,
) -> float:
    """
    Score a pull translation using correlation and retained image overlap.

    Parameters
    ----------
    fixed : Any
        Fixed image array.
    moving : Any
        Moving image array with the same shape as ``fixed``.
    pull_shift_px : Sequence[float]
        Candidate fixed-to-moving pull translation in pixels.
    array_module : Any
        NumPy-compatible module implementing array arithmetic. Runtime GPU
        registration supplies CuPy; CPU tests supply NumPy.

    Returns
    -------
    float
        Pearson correlation multiplied by the fraction of fixed-image pixels
        retained in the overlap. Invalid or constant overlaps score negative
        infinity.
    """
    overlap = _translation_overlap_slices(fixed.shape, pull_shift_px)
    if overlap is None:
        return float("-inf")
    fixed_slices, moving_slices = overlap
    fixed_values = fixed[fixed_slices].astype(array_module.float32, copy=False)
    moving_values = moving[moving_slices].astype(array_module.float32, copy=False)
    fixed_centered = fixed_values - array_module.mean(fixed_values)
    moving_centered = moving_values - array_module.mean(moving_values)
    denominator = array_module.sqrt(
        array_module.sum(fixed_centered * fixed_centered)
        * array_module.sum(moving_centered * moving_centered)
    )
    denominator_value = float(denominator)
    if not np.isfinite(denominator_value) or denominator_value <= 0:
        return float("-inf")
    correlation = float(
        array_module.sum(fixed_centered * moving_centered) / denominator
    )
    if not np.isfinite(correlation):
        return float("-inf")
    overlap_fraction = float(fixed_values.size) / float(fixed.size)
    return correlation * overlap_fraction


def _select_phase_correlation_pull_shift_px(
    fixed: Any,
    moving: Any,
    *,
    phase_cross_correlation: Any,
    array_module: Any,
    to_numpy: Any,
    diagnostics: bool = False,
) -> np.ndarray:
    """
    Select a reliable phase-correlation translation candidate.

    Phase normalization can amplify decorrelated high-frequency noise in
    fiducial images and produce a strong but physically unsupported peak near
    half an image period. This function evaluates phase-normalized,
    unnormalized, and identity candidates in real space. It selects the
    candidate with the best overlap-weighted Pearson correlation after mapping
    every periodic shift to its maximum-overlap representative.

    Parameters
    ----------
    fixed : Any
        Fixed image array.
    moving : Any
        Moving image array with the same shape as ``fixed``.
    phase_cross_correlation : Any
        NumPy-compatible phase-correlation callable.
    array_module : Any
        NumPy-compatible array module used for real-space scoring.
    to_numpy : Any
        Callable converting a phase-correlation result to a NumPy array.
    diagnostics : bool, default=False
        If True, print candidate shifts and scores.

    Returns
    -------
    numpy.ndarray
        Selected fixed-to-moving pull translation in pixels.
    """
    if fixed.shape != moving.shape:
        raise ValueError(
            "Phase-correlation candidate images must have matching shapes, got "
            f"{fixed.shape!r} and {moving.shape!r}."
        )

    candidates: list[tuple[str, np.ndarray]] = [
        ("identity", np.zeros(fixed.ndim, dtype=np.float32))
    ]
    for normalization in ("phase", None):
        push_shift_px = phase_cross_correlation(
            fixed,
            moving,
            upsample_factor=10,
            disambiguate=False,
            normalization=normalization,
        )[0]
        pull_shift_px = _maximum_overlap_phase_shift_px(
            -np.asarray(to_numpy(push_shift_px), dtype=np.float64),
            fixed.shape,
        ).astype(np.float32)
        if not any(
            np.allclose(pull_shift_px, existing_shift)
            for _label, existing_shift in candidates
        ):
            label = "phase" if normalization == "phase" else "unnormalized"
            candidates.append((label, pull_shift_px))

    scores = [
        _overlap_weighted_translation_score(
            fixed,
            moving,
            pull_shift_px=pull_shift_px,
            array_module=array_module,
        )
        for _label, pull_shift_px in candidates
    ]
    selected_index = int(np.argmax(scores))
    if diagnostics:
        details = ", ".join(
            f"{label}:pull_px={tuple(float(v) for v in shift)}:score={score:.6f}"
            for (label, shift), score in zip(candidates, scores, strict=True)
        )
        _diag(
            f"phase_candidates {details} selected={candidates[selected_index][0]}",
            enabled=True,
        )
    return candidates[selected_index][1].copy()


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
    At both stages, phase-normalized, unnormalized, and identity candidates are
    scored in real space so decorrelated noise cannot promote a large
    half-period displacement. The returned affine maps fixed/reference
    physical coordinates to moving-image physical coordinates, matching the
    convention expected by :func:`warp_array_to_reference_gpu`.

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
    xy_pull_shift_px = _select_phase_correlation_pull_shift_px(
        fixed_projection,
        moving_projection,
        phase_cross_correlation=phase_cross_correlation,
        array_module=cp,
        to_numpy=cp.asnumpy,
        diagnostics=diagnostics,
    )
    del fixed_projection, moving_projection
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

    overlap_slices = _overlap_slices_after_translation(
        fixed.shape,
        (0.0, float(xy_pull_shift_px[0]), float(xy_pull_shift_px[1])),
    )
    if overlap_slices is None:
        residual_pull_shift_px = np.zeros(3, dtype=np.float32)
    else:
        fixed_overlap = cp.asarray(fixed[overlap_slices], dtype=cp.float32)
        moving_overlap = cp.asarray(
            moving_xy_registered[overlap_slices],
            dtype=cp.float32,
        )
        residual_pull_shift_px = _select_phase_correlation_pull_shift_px(
            fixed_overlap,
            moving_overlap,
            phase_cross_correlation=phase_cross_correlation,
            array_module=cp,
            to_numpy=cp.asnumpy,
            diagnostics=diagnostics,
        )
        del fixed_overlap, moving_overlap
    del moving_xy_registered
    total_shift_px = residual_pull_shift_px.copy()
    total_shift_px[1] += xy_pull_shift_px[0]
    total_shift_px[2] += xy_pull_shift_px[1]
    total_shift_px = _maximum_overlap_phase_shift_px(
        total_shift_px,
        fixed.shape,
    ).astype(np.float32)

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
