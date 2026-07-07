"""NDV display decoration helpers."""

from contextlib import suppress
from typing import Any

import numpy as np


def apply_lut_channel_labels(array_viewer: Any, labels: list[str]) -> int:
    """
    Apply human-readable labels and stable overlay colors to ndv LUT views.

    Parameters
    ----------
    array_viewer : Any
        array_viewer for this viewer operation.
    labels : list[str]
        labels for this viewer operation.

    Returns
    -------
    int
        Computed viewer result.
    """

    controllers = getattr(array_viewer, "_lut_controllers", {})
    applied = 0
    for key, controller in controllers.items():
        if not isinstance(key, int) or key < 0 or key >= len(labels):
            continue
        if ":fiducial " in labels[key] or "polyDT" in labels[key]:
            lut_model = getattr(controller, "lut_model", None)
            if lut_model is not None:
                lut_model.cmap = "gray"
        if "Cellpose outlines" in labels[key]:
            lut_model = getattr(controller, "lut_model", None)
            if lut_model is not None:
                lut_model.cmap = "white"
        if "codewords" in labels[key] or "Proseg transcripts" in labels[key]:
            lut_model = getattr(controller, "lut_model", None)
            if lut_model is not None:
                lut_model.cmap = "turbo"
                with suppress(Exception):
                    channel_data = np.asarray(array_viewer.data[key])
                    finite_values = channel_data[np.isfinite(channel_data)]
                    if finite_values.size > 0:
                        max_value = float(np.nanmax(finite_values))
                        lut_model.clims = (
                            (1.0, max_value) if max_value > 1.0 else (0.0, 1.0)
                        )
        for lut_view in getattr(controller, "lut_views", []):
            set_channel_name = getattr(lut_view, "set_channel_name", None)
            if set_channel_name is None:
                continue
            set_channel_name(labels[key])
            applied += 1
    return applied
