"""Small multiview-stitcher adapters for MERFISH registration."""

from __future__ import annotations

import os
import timeit
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

LOCAL_ROUND_TRANSFORM_KEY = "local_round_registered"
STAGE_TRANSFORM_KEY = "stage_metadata"


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
    transform_key: str = STAGE_TRANSFORM_KEY,
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
    transform_key : str, default=STAGE_TRANSFORM_KEY
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
    transform_types: Sequence[str] = ("translation", "rigid", "affine"),
    number_of_resolutions: int = 3,
    number_of_iterations: int = 500,
    metric: str = "AdvancedMattesMutualInformation",
) -> np.ndarray:
    """
    Register a moving image to a fixed image with ITK-Elastix.

    The input arrays are interpreted as Z, Y, X images with physical spacing in
    microns. Registration is delegated to ``multiview-stitcher`` using
    ``registration_ITKElastix``. Elastix handles its own multiresolution
    pyramid; this function does not manually downsample the arrays.

    Parameters
    ----------
    fixed : numpy.ndarray
        Reference image in Z, Y, X order.
    moving : numpy.ndarray
        Image to align to ``fixed``, in Z, Y, X order.
    spacing_zyx_um : Sequence[float]
        Physical voxel spacing in microns in Z, Y, X order.
    transform_types : Sequence[str], default=("translation", "rigid", "affine")
        Ordered elastix transform stages to run.
    number_of_resolutions : int, default=3
        Number of elastix pyramid levels.
    number_of_iterations : int, default=500
        Maximum elastix iterations per registration stage.
    metric : str, default="AdvancedMattesMutualInformation"
        Elastix similarity metric.

    Returns
    -------
    numpy.ndarray
        Homogeneous 4x4 affine transform in physical Z, Y, X coordinates. The
        transform maps coordinates in the fixed reference space to coordinates
        sampled from the moving image, matching the convention expected by
        :func:`warp_array_to_reference`.
    """

    from dask import config as dask_config
    from multiview_stitcher import registration

    _diag(
        "register_pair_to_fixed_start "
        f"fixed_shape={tuple(int(v) for v in fixed.shape)} "
        f"moving_shape={tuple(int(v) for v in moving.shape)} "
        f"spacing_zyx_um={tuple(float(v) for v in spacing_zyx_um)} "
        f"transform_types={tuple(transform_types)} "
        f"number_of_resolutions={number_of_resolutions} "
        f"number_of_iterations={number_of_iterations} "
        f"metric={metric}"
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
            pairwise_reg_func=registration.registration_ITKElastix,
            pairwise_reg_func_kwargs={
                "transform_types": list(transform_types),
                "number_of_resolutions": number_of_resolutions,
                "number_of_iterations": number_of_iterations,
                "metric": metric,
            },
            groupwise_resolution_kwargs={
                "reference_view": 0,
                "transform": "affine",
            },
        )
    _diag(
        f"register_pair_to_fixed_done elapsed_s={timeit.default_timer() - start_time:.2f}"
    )
    return np.asarray(transforms[1].values[0], dtype=np.float32)


def warp_array_to_reference(
    image: np.ndarray,
    *,
    transform_zyx_um: np.ndarray,
    spacing_zyx_um: Sequence[float],
    reference_shape: Sequence[int],
    reference_origin_zyx_um: Sequence[float] = (0.0, 0.0, 0.0),
    order: int = 1,
) -> np.ndarray:
    """
    Warp an image into a reference ZYX grid using a physical affine.

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
        Physical output origin in microns in Z, Y, X order.
    order : int, default=1
        Interpolation order passed to multiview-stitcher.

    Returns
    -------
    numpy.ndarray
        Warped image sampled on the reference grid.
    """

    from multiview_stitcher import spatial_image_utils as si_utils
    from multiview_stitcher import transformation

    _diag(
        "warp_array_to_reference_start "
        f"image_shape={tuple(int(v) for v in image.shape)} "
        f"reference_shape={tuple(int(v) for v in reference_shape)} "
        f"spacing_zyx_um={tuple(float(v) for v in spacing_zyx_um)} "
        f"order={order}"
    )
    sim = sim_from_array(
        image,
        spacing_zyx_um=spacing_zyx_um,
        origin_zyx_um=reference_origin_zyx_um,
    )
    field = si_utils.get_sim_field(sim)
    start_time = timeit.default_timer()
    warped = transformation.transform_sim(
        field,
        p=np.asarray(transform_zyx_um, dtype=np.float32),
        output_stack_properties={
            "spacing": _zyx_dict(spacing_zyx_um),
            "origin": _zyx_dict(reference_origin_zyx_um),
            "shape": {
                "z": int(reference_shape[0]),
                "y": int(reference_shape[1]),
                "x": int(reference_shape[2]),
            },
        },
        order=order,
    )
    data = warped.data
    if hasattr(data, "compute"):
        data = data.compute()
    _diag(
        f"warp_array_to_reference_done elapsed_s={timeit.default_timer() - start_time:.2f}"
    )
    return np.asarray(data)


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
        is the same transform used by :func:`warp_array_to_reference`.
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


def get_batch_processing_options(
    misc_utils: Any,
    n_batch: int,
    n_jobs: int,
) -> dict[str, Any]:
    """
    Build multiview-stitcher batch options for direct Zarr fusion.

    Parameters
    ----------
    misc_utils : Any
        ``multiview_stitcher.misc_utils`` module.
    n_batch : int
        Number of fusion batches to schedule.
    n_jobs : int
        Number of batches to process in parallel.

    Returns
    -------
    dict[str, Any]
        Batch options accepted by ``fusion.fuse``.
    """

    return {
        "batch_func": misc_utils.process_batch_using_joblib,
        "n_batch": int(n_batch),
        "batch_func_kwargs": {"n_jobs": int(n_jobs)},
    }


def get_scale0_sim_from_fusion_result(fused: Any, msi_utils: Any) -> Any:
    """
    Return a scale0 SpatialImage from a fusion result.

    Parameters
    ----------
    fused : Any
        SpatialImage or MultiscaleSpatialImage returned by
        ``multiview_stitcher.fusion.fuse``.
    msi_utils : Any
        ``multiview_stitcher.msi_utils`` module.

    Returns
    -------
    Any
        SpatialImage at the highest written resolution.
    """

    if hasattr(fused, "data"):
        return fused
    return msi_utils.get_sim_from_msim(fused, scale="scale0")


def get_gpu_fusion_backend_kwargs(fuse_func: Callable) -> dict[str, Any]:
    """
    Return CuPy backend keyword arguments for multiview-stitcher fusion.

    Parameters
    ----------
    fuse_func : Callable
        ``multiview_stitcher.fusion.fuse`` function.

    Returns
    -------
    dict[str, Any]
        Extra keyword arguments requesting CuPy-backed fusion.
    """

    import inspect

    fuse_parameters = inspect.signature(fuse_func).parameters
    if "backend" not in fuse_parameters:
        raise RuntimeError(
            "GPU fusion requires multiview-stitcher with fusion.fuse(..., "
            "backend='cupy'), expected in version 0.1.56 or newer."
        )

    try:
        import cupy  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "GPU fusion requires a CUDA-compatible CuPy package."
        ) from exc

    return {"backend": "cupy", "output_on_backend": False}
