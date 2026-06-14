"""SOFIMA flow-field estimation utilities."""

from typing import Any

import numpy as np


def _resolve_patch_and_step(
    shape_zyx: tuple[int, int, int],
    config: dict[str, Any],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    Resolve SOFIMA patch size and stride for one fixed/moving volume pair.

    Parameters
    ----------
    shape_zyx : tuple[int, int, int]
        Shape of the fixed and moving volumes in Z, Y, X order.
    config : dict[str, Any]
        Optional runtime configuration. Recognized keys are
        ``patch_size_zyx`` and ``step_zyx``.

    Returns
    -------
    tuple[tuple[int, int, int], tuple[int, int, int]]
        Patch size and step in Z, Y, X order, clipped to the image shape.
    """

    default_patch_size = tuple(
        max(4, min(axis_size, patch_size))
        for axis_size, patch_size in zip(shape_zyx, (32, 64, 64), strict=False)
    )
    patch_size = tuple(int(v) for v in config.get("patch_size_zyx", default_patch_size))
    patch_size = tuple(
        max(2, min(size, axis_size))
        for size, axis_size in zip(patch_size, shape_zyx, strict=False)
    )

    default_step = tuple(max(1, size // 2) for size in patch_size)
    step = tuple(int(v) for v in config.get("step_zyx", default_step))
    step = tuple(
        max(1, min(size, axis_size))
        for size, axis_size in zip(step, shape_zyx, strict=False)
    )
    return patch_size, step


def estimate_sofima_flow_field_xyz_px(
    fixed_zyx: np.ndarray,
    moving_affine_initialized_zyx: np.ndarray,
    *,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Estimate a SOFIMA flow field after affine initialization.

    Parameters
    ----------
    fixed_zyx : numpy.ndarray
        Reference round001 fiducial image in Z, Y, X order.
    moving_affine_initialized_zyx : numpy.ndarray
        Moving fiducial image already rendered into the reference grid by the
        stored affine transform.
    config : dict[str, Any] or None, default=None
        SOFIMA configuration. The current implementation imports JAX/SOFIMA
        inside this function so the caller can run it in a short-lived process.

    Returns
    -------
    tuple[numpy.ndarray, dict[str, Any]]
        Relative SOFIMA flow field with XYZ channels and metadata describing
        the map spacing/origin.

    Notes
    -----
    This function estimates only the residual deformation between one fixed
    volume and one affine-initialized moving volume. It intentionally does not
    use SOFIMA's stitching mesh integration helpers, which are designed for
    multi-tile neighbor graphs rather than a single fixed/moving image pair.

    SOFIMA's masked cross-correlation estimator computes a flow field from
    ``post_image`` to ``pre_image``. Passing the affine-initialized moving image
    as ``pre_image`` and round001 as ``post_image`` gives the residual
    displacement from reference pixels into affine-initialized moving pixels.
    """

    config = {} if config is None else dict(config)
    from sofima import flow_field, flow_utils, map_utils

    if fixed_zyx.shape != moving_affine_initialized_zyx.shape:
        raise ValueError(
            "fixed_zyx and moving_affine_initialized_zyx must have matching "
            f"shapes, got {fixed_zyx.shape!r} and "
            f"{moving_affine_initialized_zyx.shape!r}."
        )

    shape_zyx = tuple(int(v) for v in fixed_zyx.shape)
    patch_size, step = _resolve_patch_and_step(shape_zyx, config)

    calculator = flow_field.JAXMaskedXCorrWithStatsCalculator(
        mean=config.get("mean"),
        peak_min_distance=int(config.get("peak_min_distance", 2)),
        peak_radius=int(config.get("peak_radius", 5)),
    )
    flow = calculator.flow_field(
        moving_affine_initialized_zyx.astype(np.float32, copy=False),
        fixed_zyx.astype(np.float32, copy=False),
        patch_size=patch_size,
        step=step,
        batch_size=int(config.get("batch_size", 32)),
        max_masked=float(config.get("max_masked", 0.75)),
    )
    cleaned_flow = flow_utils.clean_flow(
        flow,
        min_peak_ratio=float(config.get("min_peak_ratio", 1.4)),
        min_peak_sharpness=float(config.get("min_peak_sharpness", 1.4)),
        max_magnitude=float(config.get("max_magnitude", 0.0)),
        max_deviation=float(config.get("max_deviation", 5.0)),
        dim=3,
    )
    if not np.any(np.isfinite(cleaned_flow[0])):
        raise RuntimeError("SOFIMA did not produce any valid flow vectors.")
    sofima_flow_field = map_utils.fill_missing(
        cleaned_flow,
        extrapolate=True,
        invalid_to_zero=False,
        interpolate_first=True,
    )
    map_stride_zyx_px = [float(v) for v in step]

    metadata = {
        "map_stride_zyx_px": map_stride_zyx_px,
        "map_box_start_xyz_px": [0.0, 0.0, 0.0],
        "map_box_size_xyz_px": [
            float((sofima_flow_field.shape[3] - 1) * map_stride_zyx_px[2] + 1),
            float((sofima_flow_field.shape[2] - 1) * map_stride_zyx_px[1] + 1),
            float((sofima_flow_field.shape[1] - 1) * map_stride_zyx_px[0] + 1),
        ],
    }
    return sofima_flow_field.astype(np.float32, copy=False), metadata
