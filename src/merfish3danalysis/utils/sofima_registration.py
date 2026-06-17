"""SOFIMA flow-field estimation utilities."""

from typing import Any

import numpy as np

SOFIMA_RESIDUAL_ITERATIONS = 2
SOFIMA_PATCH_SIZE_ZYX = (10, 32, 32)
SOFIMA_PEAK_MIN_DISTANCE = 2
SOFIMA_PEAK_RADIUS = 8
SOFIMA_BATCH_SIZE = 32
SOFIMA_MAX_MASKED = 0.75
SOFIMA_MIN_PEAK_RATIO = 1.2
SOFIMA_MIN_PEAK_SHARPNESS = 1.2
SOFIMA_MAX_MAGNITUDE = 30.0
SOFIMA_MAX_DEVIATION = 5.0
SOFIMA_SUBPIXEL_OFFSETS = (-0.5, 0.0, 0.5)
SOFIMA_SUBPIXEL_BATCH_SIZE = 32
SOFIMA_MESH_DT = 0.001
SOFIMA_MESH_GAMMA = 0.0
SOFIMA_MESH_K0 = 1.0
SOFIMA_MESH_K = 0.01
SOFIMA_MESH_NUM_ITERS = 1000
SOFIMA_MESH_MAX_ITERS = 20000
SOFIMA_MESH_STOP_V_MAX = 0.001
SOFIMA_MESH_DT_MAX = 100.0
SOFIMA_MESH_START_CAP = 0.1
SOFIMA_MESH_FINAL_CAP = 10.0


def _resolve_patch_and_step(
    shape_zyx: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    Resolve SOFIMA patch size and stride for one fixed/moving volume pair.

    Parameters
    ----------
    shape_zyx : tuple[int, int, int]
        Shape of the fixed and moving volumes in Z, Y, X order.
    Returns
    -------
    tuple[tuple[int, int, int], tuple[int, int, int]]
        Patch size and step in Z, Y, X order, clipped to the image shape.
    """

    default_patch_size = tuple(
        max(4, min(axis_size, patch_size))
        for axis_size, patch_size in zip(
            shape_zyx,
            SOFIMA_PATCH_SIZE_ZYX,
            strict=False,
        )
    )
    patch_size = tuple(int(v) for v in default_patch_size)
    step = tuple(max(1, size // 2) for size in patch_size)
    return patch_size, step


def _compose_flow_fields_same_grid(
    base_flow_xyz: np.ndarray,
    residual_flow_xyz: np.ndarray,
    *,
    stride_zyx: tuple[float, float, float],
    box_start_xyz: tuple[float, float, float],
) -> np.ndarray:
    """
    Compose two SOFIMA flow fields sampled on the same reference grid.

    Parameters
    ----------
    base_flow_xyz : numpy.ndarray
        Existing flow field in ``(3, z, y, x)`` order. It maps reference-grid
        coordinates into the original affine-initialized moving image.
    residual_flow_xyz : numpy.ndarray
        Residual flow field in ``(3, z, y, x)`` order. It maps reference-grid
        coordinates into the image already warped by ``base_flow_xyz``.
    stride_zyx : tuple[float, float, float]
        Flow-grid spacing in reference-image pixels.
    box_start_xyz : tuple[float, float, float]
        Reference pixel coordinate of the first flow sample in X, Y, Z order.

    Returns
    -------
    numpy.ndarray
        Composed flow field mapping reference-grid coordinates directly into
        the original affine-initialized moving image.
    """

    import jax
    import jax.numpy as jnp
    from jax.scipy.ndimage import map_coordinates

    base_flow = jnp.asarray(base_flow_xyz, dtype=jnp.float32)
    residual_flow = jnp.asarray(residual_flow_xyz, dtype=jnp.float32)
    z_grid, y_grid, x_grid = jnp.indices(base_flow.shape[1:], dtype=jnp.float32)
    shifted_zyx = (
        box_start_xyz[2] + z_grid * stride_zyx[0] + residual_flow[2],
        box_start_xyz[1] + y_grid * stride_zyx[1] + residual_flow[1],
        box_start_xyz[0] + x_grid * stride_zyx[2] + residual_flow[0],
    )
    coords = jnp.stack(
        [
            (shifted_zyx[0] - box_start_xyz[2]) / stride_zyx[0],
            (shifted_zyx[1] - box_start_xyz[1]) / stride_zyx[1],
            (shifted_zyx[2] - box_start_xyz[0]) / stride_zyx[2],
        ],
        axis=0,
    )
    sampled_base = jnp.stack(
        [
            map_coordinates(
                base_flow[channel_index],
                coords,
                order=1,
                mode="nearest",
            )
            for channel_index in range(3)
        ],
        axis=0,
    )
    composed = residual_flow + sampled_base
    return np.asarray(jax.device_get(composed), dtype=np.float32)


def _refine_flow_vectors_subpixel(
    flow_xyz: np.ndarray,
    pre_image_zyx: np.ndarray,
    post_image_zyx: np.ndarray,
    *,
    patch_size_zyx: tuple[int, int, int],
    step_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, int]:
    """
    Refine valid SOFIMA integer vectors by local fractional patch matching.

    SOFIMA's local cross-correlation reports integer-pixel peak locations. This
    function keeps SOFIMA's accepted vectors and only searches a small
    fractional neighborhood around each one. Candidate shifts are scored by
    normalized sum-of-squared differences between the post patch and a
    subpixel-sampled pre patch.

    Parameters
    ----------
    flow_xyz : numpy.ndarray
        Cleaned SOFIMA flow field in ``(3, z, y, x)`` order.
    pre_image_zyx : numpy.ndarray
        Moving image in Z, Y, X order. This is SOFIMA's ``pre_image``.
    post_image_zyx : numpy.ndarray
        Fixed image in Z, Y, X order. This is SOFIMA's ``post_image``.
    patch_size_zyx : tuple[int, int, int]
        Patch size used for SOFIMA flow estimation.
    step_zyx : tuple[int, int, int]
        Flow-grid step used for SOFIMA flow estimation.
    Returns
    -------
    tuple[numpy.ndarray, int]
        Refined flow field and the number of vectors that were refined.
    """

    import itertools

    import jax
    import jax.numpy as jnp
    from jax.scipy.ndimage import map_coordinates

    offsets = np.asarray(SOFIMA_SUBPIXEL_OFFSETS, dtype=np.float32)

    valid_indices = np.argwhere(np.isfinite(flow_xyz[0]))
    if valid_indices.size == 0:
        return flow_xyz.copy(), 0

    patch_grid_zyx = jnp.asarray(np.indices(patch_size_zyx, dtype=np.float32))
    image_shape = np.asarray(pre_image_zyx.shape, dtype=np.float32)
    patch_size_array = np.asarray(patch_size_zyx, dtype=np.float32)
    step_array = np.asarray(step_zyx, dtype=np.float32)
    vector_xyz = np.stack(
        [
            flow_xyz[0][tuple(valid_indices.T)],
            flow_xyz[1][tuple(valid_indices.T)],
            flow_xyz[2][tuple(valid_indices.T)],
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    vector_zyx = vector_xyz[:, [2, 1, 0]]
    post_starts_zyx = valid_indices.astype(np.float32) * step_array
    base_pre_starts_zyx = post_starts_zyx + vector_zyx
    interior_mask = np.all(base_pre_starts_zyx >= 1.0, axis=1) & np.all(
        base_pre_starts_zyx + patch_size_array <= image_shape - 2.0,
        axis=1,
    )
    if not np.any(interior_mask):
        return flow_xyz.copy(), 0

    valid_indices = valid_indices[interior_mask]
    vector_xyz = vector_xyz[interior_mask]
    offset_grid_zyx = np.asarray(
        list(itertools.product(offsets, offsets, offsets)),
        dtype=np.float32,
    )
    batch_size = SOFIMA_SUBPIXEL_BATCH_SIZE
    pre_image = jnp.asarray(pre_image_zyx, dtype=jnp.float32)
    post_image = jnp.asarray(post_image_zyx, dtype=jnp.float32)
    candidate_offsets = jnp.asarray(offset_grid_zyx, dtype=jnp.float32)
    step = jnp.asarray(step_zyx, dtype=jnp.float32)

    @jax.jit
    def _refine_batch(
        batch_indices_zyx: jnp.ndarray,
        batch_vectors_xyz: jnp.ndarray,
    ) -> jnp.ndarray:
        post_starts = batch_indices_zyx.astype(jnp.float32) * step
        batch_vectors_zyx = batch_vectors_xyz[:, [2, 1, 0]]
        base_pre_starts = post_starts + batch_vectors_zyx

        post_coords = (
            patch_grid_zyx[:, jnp.newaxis, ...]
            + post_starts.T[:, :, jnp.newaxis, jnp.newaxis, jnp.newaxis]
        )
        post_patch = map_coordinates(
            post_image,
            post_coords,
            order=1,
            mode="constant",
            cval=0.0,
        )
        post_patch = post_patch - jnp.mean(post_patch, axis=(1, 2, 3), keepdims=True)
        post_patch = post_patch / jnp.maximum(
            jnp.sqrt(jnp.mean(post_patch**2, axis=(1, 2, 3), keepdims=True)),
            jnp.float32(1e-6),
        )

        pre_starts = (
            base_pre_starts[:, jnp.newaxis, :] + candidate_offsets[jnp.newaxis, :, :]
        )
        pre_coords = (
            patch_grid_zyx[:, jnp.newaxis, jnp.newaxis, ...]
            + jnp.moveaxis(pre_starts, -1, 0)[
                :, :, :, jnp.newaxis, jnp.newaxis, jnp.newaxis
            ]
        )
        pre_patch = map_coordinates(
            pre_image,
            pre_coords,
            order=1,
            mode="constant",
            cval=0.0,
        )
        pre_patch = pre_patch - jnp.mean(
            pre_patch,
            axis=(2, 3, 4),
            keepdims=True,
        )
        pre_patch = pre_patch / jnp.maximum(
            jnp.sqrt(jnp.mean(pre_patch**2, axis=(2, 3, 4), keepdims=True)),
            jnp.float32(1e-6),
        )
        scores = jnp.mean(
            (pre_patch - post_patch[:, jnp.newaxis, ...]) ** 2,
            axis=(2, 3, 4),
        )
        best_indices = jnp.argmin(scores, axis=1)
        best_offsets_zyx = candidate_offsets[best_indices]
        return batch_vectors_xyz + best_offsets_zyx[:, [2, 1, 0]]

    refined = flow_xyz.copy()
    refined_vectors = 0
    for start in range(0, valid_indices.shape[0], batch_size):
        stop = min(start + batch_size, valid_indices.shape[0])
        batch_indices = valid_indices[start:stop]
        batch_vectors = vector_xyz[start:stop]
        refined_batch = np.asarray(
            jax.device_get(
                _refine_batch(
                    jnp.asarray(batch_indices, dtype=jnp.int32),
                    jnp.asarray(batch_vectors, dtype=jnp.float32),
                )
            ),
            dtype=np.float32,
        )
        refined[0][tuple(batch_indices.T)] = refined_batch[:, 0]
        refined[1][tuple(batch_indices.T)] = refined_batch[:, 1]
        refined[2][tuple(batch_indices.T)] = refined_batch[:, 2]
        refined_vectors += batch_indices.shape[0]

    return refined.astype(np.float32, copy=False), int(refined_vectors)


def _relax_flow_field(
    cleaned_flow_xyz: np.ndarray,
    initial_flow_xyz: np.ndarray,
    step_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Relax a sparse SOFIMA flow field with SOFIMA's elastic mesh solver.

    SOFIMA's patch cross-correlation returns integer local flow vectors. The
    elastic mesh converts those measurements into the smooth float-valued
    coordinate map expected by downstream warping. Valid measured vectors pull
    mesh nodes through zero-length springs; invalid nodes are governed by the
    internal 3D mesh elasticity.

    Parameters
    ----------
    cleaned_flow_xyz : numpy.ndarray
        Cleaned SOFIMA flow field in ``(3, z, y, x)`` order. Invalid vectors
        are encoded as NaN.
    initial_flow_xyz : numpy.ndarray
        Initial dense flow field in ``(3, z, y, x)`` order.
    step_zyx : tuple[int, int, int]
        Flow-grid spacing in image pixels in Z, Y, X order.
    Returns
    -------
    tuple[numpy.ndarray, dict[str, Any]]
        Relaxed float-valued flow field and metadata for the mesh solve.
    """

    import jax
    import jax.numpy as jnp
    from sofima import mesh

    stride_xyz = (float(step_zyx[2]), float(step_zyx[1]), float(step_zyx[0]))
    mesh_config = mesh.IntegrationConfig(
        dt=SOFIMA_MESH_DT,
        gamma=SOFIMA_MESH_GAMMA,
        k0=SOFIMA_MESH_K0,
        k=SOFIMA_MESH_K,
        stride=stride_xyz,
        num_iters=SOFIMA_MESH_NUM_ITERS,
        max_iters=SOFIMA_MESH_MAX_ITERS,
        stop_v_max=SOFIMA_MESH_STOP_V_MAX,
        dt_max=SOFIMA_MESH_DT_MAX,
        prefer_orig_order=False,
        start_cap=SOFIMA_MESH_START_CAP,
        final_cap=SOFIMA_MESH_FINAL_CAP,
        remove_drift=False,
    )
    relaxed_flow, kinetic_energy, iterations = mesh.relax_mesh(
        jnp.asarray(initial_flow_xyz, dtype=jnp.float32),
        jnp.asarray(cleaned_flow_xyz, dtype=jnp.float32),
        mesh_config,
        mesh_force=mesh.elastic_mesh_3d,
    )
    relaxed_flow = np.asarray(jax.device_get(relaxed_flow), dtype=np.float32)
    metadata = {
        "mesh_iterations": int(iterations),
        "mesh_final_kinetic_energy": (
            float(kinetic_energy[-1]) if len(kinetic_energy) > 0 else 0.0
        ),
        "mesh_relaxation": True,
    }
    return relaxed_flow, metadata


def _estimate_sofima_flow_field_xyz_px_impl(
    fixed_zyx: np.ndarray,
    moving_affine_initialized_zyx: np.ndarray,
    *,
    single_residual_pass: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Estimate a SOFIMA flow field after affine initialization.

    The returned field follows the package deformable-registration convention:

    - Array shape is ``(3, z, y, x)``.
    - Channel order is ``X, Y, Z`` because this is SOFIMA's flow-component
      order.
    - Spatial map axes are ``Z, Y, X`` to match image arrays.
    - Values are relative displacements in reference-image pixels. Adding the
      interpolated field to a reference-grid coordinate gives the coordinate in
      the affine-initialized moving image.
    - ``map_box_start_xyz_px`` is the reference-grid coordinate of the first
      flow sample in ``X, Y, Z`` order. SOFIMA estimates patch-centered
      displacements, so this origin is half the patch size, not the image
      corner.

    Parameters
    ----------
    fixed_zyx : numpy.ndarray
        Reference round001 fiducial image in Z, Y, X order.
    moving_affine_initialized_zyx : numpy.ndarray
        Moving fiducial image already rendered into the reference grid by the
        stored affine transform.
    single_residual_pass : bool, default=False
        Internal recursion control. Public callers always use the fixed
        production path with two residual passes.

    Returns
    -------
    tuple[numpy.ndarray, dict[str, Any]]
        Relative SOFIMA flow field with XYZ channels and metadata describing
        the map spacing/origin. The metadata is sufficient to save the field
        as OME-Zarr and later reproduce the same warp from the reloaded field.

    Notes
    -----
    SOFIMA's masked cross-correlation estimator computes integer local flow
    vectors from ``post_image`` to ``pre_image``. Passing the affine-initialized
    moving image as ``pre_image`` and round001 as ``post_image`` gives the
    residual displacement from reference pixels into affine-initialized moving
    pixels. The local measurements are then relaxed with SOFIMA's 3D elastic
    mesh solver to produce the smooth float-valued field expected by
    :func:`merfish3danalysis.utils.multiview_registration.warp_array_to_reference_with_affine_and_sofima_flow_gpu`.
    """

    residual_iterations = 1 if single_residual_pass else SOFIMA_RESIDUAL_ITERATIONS
    if residual_iterations > 1:
        from merfish3danalysis.utils.multiview_registration import (
            warp_array_to_reference_with_affine_and_sofima_flow_gpu,
        )

        total_flow, total_metadata = _estimate_sofima_flow_field_xyz_px_impl(
            fixed_zyx,
            moving_affine_initialized_zyx,
            single_residual_pass=True,
        )
        corrected = warp_array_to_reference_with_affine_and_sofima_flow_gpu(
            moving_affine_initialized_zyx,
            transform_zyx_um=np.eye(4, dtype=np.float32),
            spacing_zyx_um=(1.0, 1.0, 1.0),
            reference_shape=fixed_zyx.shape,
            sofima_flow_field_xyz_px=total_flow,
            flow_field_stride_zyx_px=total_metadata["map_stride_zyx_px"],
            flow_field_box_start_xyz_px=total_metadata["map_box_start_xyz_px"],
            mode="nearest",
        ).astype(np.float32, copy=False)
        completed_iterations = 1
        for _iteration in range(1, residual_iterations):
            residual_flow, residual_metadata = _estimate_sofima_flow_field_xyz_px_impl(
                fixed_zyx,
                corrected,
                single_residual_pass=True,
            )
            if residual_metadata["status"] != "ok":
                break
            total_flow = _compose_flow_fields_same_grid(
                total_flow,
                residual_flow,
                stride_zyx=tuple(float(v) for v in total_metadata["map_stride_zyx_px"]),
                box_start_xyz=tuple(
                    float(v) for v in total_metadata["map_box_start_xyz_px"]
                ),
            )
            total_metadata["valid_flow_vectors"] = int(
                total_metadata["valid_flow_vectors"]
            ) + int(residual_metadata["valid_flow_vectors"])
            total_metadata["mesh_iterations"] = int(
                total_metadata["mesh_iterations"]
            ) + int(residual_metadata["mesh_iterations"])
            completed_iterations += 1
            if _iteration + 1 < residual_iterations:
                corrected = warp_array_to_reference_with_affine_and_sofima_flow_gpu(
                    moving_affine_initialized_zyx,
                    transform_zyx_um=np.eye(4, dtype=np.float32),
                    spacing_zyx_um=(1.0, 1.0, 1.0),
                    reference_shape=fixed_zyx.shape,
                    sofima_flow_field_xyz_px=total_flow,
                    flow_field_stride_zyx_px=total_metadata["map_stride_zyx_px"],
                    flow_field_box_start_xyz_px=total_metadata["map_box_start_xyz_px"],
                    mode="nearest",
                ).astype(np.float32, copy=False)
        total_metadata["residual_iterations"] = completed_iterations
        return total_flow.astype(np.float32, copy=False), total_metadata

    from sofima import flow_field, flow_utils, map_utils

    if fixed_zyx.shape != moving_affine_initialized_zyx.shape:
        raise ValueError(
            "fixed_zyx and moving_affine_initialized_zyx must have matching "
            f"shapes, got {fixed_zyx.shape!r} and "
            f"{moving_affine_initialized_zyx.shape!r}."
        )

    shape_zyx = tuple(int(v) for v in fixed_zyx.shape)
    patch_size, step = _resolve_patch_and_step(shape_zyx)

    calculator = flow_field.JAXMaskedXCorrWithStatsCalculator(
        mean=None,
        peak_min_distance=SOFIMA_PEAK_MIN_DISTANCE,
        peak_radius=SOFIMA_PEAK_RADIUS,
    )
    flow = calculator.flow_field(
        moving_affine_initialized_zyx.astype(np.float32, copy=False),
        fixed_zyx.astype(np.float32, copy=False),
        patch_size=patch_size,
        step=step,
        batch_size=SOFIMA_BATCH_SIZE,
        max_masked=SOFIMA_MAX_MASKED,
    )
    cleaned_flow = flow_utils.clean_flow(
        flow,
        min_peak_ratio=SOFIMA_MIN_PEAK_RATIO,
        min_peak_sharpness=SOFIMA_MIN_PEAK_SHARPNESS,
        max_magnitude=SOFIMA_MAX_MAGNITUDE,
        max_deviation=SOFIMA_MAX_DEVIATION,
        dim=3,
    )
    cleaned_flow, subpixel_refined_vectors = _refine_flow_vectors_subpixel(
        cleaned_flow,
        moving_affine_initialized_zyx.astype(np.float32, copy=False),
        fixed_zyx.astype(np.float32, copy=False),
        patch_size_zyx=patch_size,
        step_zyx=step,
    )
    valid_flow_mask = np.isfinite(cleaned_flow[0])
    valid_flow_vectors = int(np.sum(valid_flow_mask))
    if valid_flow_vectors == 0:
        sofima_flow_field = np.zeros_like(cleaned_flow, dtype=np.float32)
        flow_status = "identity_fallback_no_valid_vectors"
        relaxation_metadata = {
            "mesh_relaxation": False,
            "mesh_iterations": 0,
            "mesh_final_kinetic_energy": 0.0,
        }
    else:
        initial_flow_field = map_utils.fill_missing(
            cleaned_flow,
            extrapolate=True,
            invalid_to_zero=False,
            interpolate_first=True,
        )
        sofima_flow_field, relaxation_metadata = _relax_flow_field(
            cleaned_flow,
            initial_flow_field,
            step,
        )
        flow_status = "ok"
    map_stride_zyx_px = [float(v) for v in step]

    metadata = {
        "status": flow_status,
        "valid_flow_vectors": valid_flow_vectors,
        "subpixel_refined_vectors": subpixel_refined_vectors,
        "residual_iterations": 1,
        "map_stride_zyx_px": map_stride_zyx_px,
        "map_box_start_xyz_px": [
            float(patch_size[2]) / 2.0,
            float(patch_size[1]) / 2.0,
            float(patch_size[0]) / 2.0,
        ],
        "map_box_size_xyz_px": [
            float((sofima_flow_field.shape[3] - 1) * map_stride_zyx_px[2] + 1),
            float((sofima_flow_field.shape[2] - 1) * map_stride_zyx_px[1] + 1),
            float((sofima_flow_field.shape[1] - 1) * map_stride_zyx_px[0] + 1),
        ],
    }
    metadata.update(relaxation_metadata)
    return sofima_flow_field.astype(np.float32, copy=False), metadata


def estimate_sofima_flow_field_xyz_px(
    fixed_zyx: np.ndarray,
    moving_affine_initialized_zyx: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Estimate the production SOFIMA flow field after affine initialization.

    This public entry point intentionally exposes no tuning parameters. If
    deformable registration is enabled, the package always runs the same
    production path: two residual SOFIMA passes, accepted-vector subpixel
    refinement, and SOFIMA 3D elastic mesh relaxation.

    Parameters
    ----------
    fixed_zyx : numpy.ndarray
        Reference round001 fiducial image in Z, Y, X order.
    moving_affine_initialized_zyx : numpy.ndarray
        Moving fiducial image already rendered into the reference grid by the
        stored affine transform.

    Returns
    -------
    tuple[numpy.ndarray, dict[str, Any]]
        Relative SOFIMA flow field with XYZ channels and metadata describing
        the map spacing/origin.
    """

    return _estimate_sofima_flow_field_xyz_px_impl(
        fixed_zyx,
        moving_affine_initialized_zyx,
        single_residual_pass=False,
    )
