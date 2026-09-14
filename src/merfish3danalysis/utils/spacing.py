"""Common precision for physical pixel and voxel sizes, in microns."""

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

SPACING_DECIMALS = 3


def round_pixel_size_um(value: float) -> float:
    """Round a physical pixel size to three decimal places."""
    return float(np.round(float(value), SPACING_DECIMALS))


def round_spacing_um(values: "ArrayLike") -> np.ndarray:
    """Round physical spacing in float64, suitable for calculations and JSON.

    Round after converting to float64 so serializing float32 input does not
    reintroduce digits such as 0.107999995. GPU callers may cast the rounded
    result to float32 for computation, but metadata should use this result's
    ``tolist()`` directly.
    """
    return np.round(np.asarray(values, dtype=np.float64), SPACING_DECIMALS)


def round_ome_spatial_scales(attributes: dict[str, Any]) -> None:
    """Round spatial scale metadata in place for every OME-Zarr pyramid level."""
    ome = attributes.get("ome", attributes)
    for multiscale in ome.get("multiscales", []):
        axes = multiscale.get("axes", [])
        spatial_axes = [
            index
            for index, axis in enumerate(axes)
            if (
                isinstance(axis, dict)
                and (axis.get("type") == "space" or axis.get("name") in ("z", "y", "x"))
            )
            or (isinstance(axis, str) and axis in ("z", "y", "x"))
        ]
        for dataset in multiscale.get("datasets", []):
            for transform in dataset.get("coordinateTransformations", []):
                if transform.get("type") == "scale":
                    scale = list(transform["scale"])
                    for index in spatial_axes:
                        scale[index] = round_pixel_size_um(scale[index])
                    transform["scale"] = scale
