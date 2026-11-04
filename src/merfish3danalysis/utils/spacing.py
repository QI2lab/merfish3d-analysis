"""Common precision for physical pixel and voxel sizes, in microns."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

SPACING_DECIMALS = 3


def round_pixel_size_um(value: float) -> float:
    """Round a physical pixel size to three decimal places.

    Parameters
    ----------
    value : float
        Pixel size in microns.

    Returns
    -------
    float
        Rounded pixel size in microns.
    """
    return float(np.round(float(value), SPACING_DECIMALS))


def round_spacing_um(values: "ArrayLike") -> np.ndarray:
    """Round physical spacing in float64, suitable for calculations and JSON.

    Round after converting to float64 so serializing float32 input does not
    reintroduce digits such as 0.107999995. GPU callers may cast the rounded
    result to float32 for computation, but metadata should use this result's
    ``tolist()`` directly.

    Parameters
    ----------
    values : ArrayLike
        Physical spacings in microns.

    Returns
    -------
    numpy.ndarray
        Rounded spacings as float64 values.
    """
    return np.round(np.asarray(values, dtype=np.float64), SPACING_DECIMALS)
