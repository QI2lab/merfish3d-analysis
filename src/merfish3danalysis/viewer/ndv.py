"""NDV display decoration helpers."""

from contextlib import suppress
from typing import Any


def channel_cmap_for_label(label: str) -> str | None:
    """
    Return the display colormap for a viewer channel label.

    Parameters
    ----------
    label : str
        Channel label.

    Returns
    -------
    str or None
        Colormap name, or ``None`` when the default should be used.
    """
    if ":fiducial " in label or "polyDT" in label or label == "global empty canvas":
        return "gray"
    if "Cellpose mask" in label:
        return "gray"
    return None


def apply_lut_channel_labels(array_viewer: Any, labels: list[str]) -> int:
    """
    Apply human-readable labels and stable overlay colors to ndv LUT views.

    Parameters
    ----------
    array_viewer : Any
        NDV array viewer.
    labels : list[str]
        Channel labels in channel order.

    Returns
    -------
    int
        Number of LUT views updated.
    """
    controllers = getattr(array_viewer, "_lut_controllers", {})
    applied = 0
    for key, controller in controllers.items():
        if not isinstance(key, int) or key < 0 or key >= len(labels):
            continue
        lut_model = getattr(controller, "lut_model", None)
        cmap = channel_cmap_for_label(labels[key])
        if lut_model is not None and cmap is not None:
            with suppress(Exception):
                lut_model.cmap = cmap
        for lut_view in getattr(controller, "lut_views", []):
            set_channel_name = getattr(lut_view, "set_channel_name", None)
            if set_channel_name is None:
                continue
            set_channel_name(labels[key])
            applied += 1
    return applied


def ndv_array_view(array_viewer: Any) -> Any | None:
    """
    Return NDV's array-view object when available.

    Parameters
    ----------
    array_viewer : Any
        NDV array viewer.

    Returns
    -------
    Any or None
        Private NDV array-view object.
    """
    return getattr(array_viewer, "_view", None)


def ndv_canvas_parts(array_viewer: Any) -> tuple[Any | None, Any | None, Any | None]:
    """
    Return NDV canvas controller, VisPy view, and VisPy canvas.

    Parameters
    ----------
    array_viewer : Any
        NDV array viewer.

    Returns
    -------
    tuple[Any or None, Any or None, Any or None]
        Canvas controller, scene view, and canvas.
    """
    canvas_controller = getattr(array_viewer, "_canvas", None)
    return (
        canvas_controller,
        getattr(canvas_controller, "_view", None),
        getattr(canvas_controller, "_canvas", None),
    )


def hide_ndv_volume_button(array_viewer: Any) -> None:
    """
    Hide NDV's built-in volume-rendering button when present.

    Parameters
    ----------
    array_viewer : Any
        NDV array viewer.
    """
    array_view = ndv_array_view(array_viewer)
    qwidget = getattr(array_view, "_qwidget", None)
    ndims_button = getattr(qwidget, "ndims_btn", None)
    if ndims_button is None:
        return
    ndims_button.setChecked(False)
    ndims_button.setVisible(False)
    ndims_button.setEnabled(False)


def ndv_current_index(array_viewer: Any) -> dict[str, Any]:
    """
    Return NDV's current index mapping when available.

    Parameters
    ----------
    array_viewer : Any
        NDV array viewer.

    Returns
    -------
    dict[str, Any]
        Current dimension index mapping.
    """
    array_view = ndv_array_view(array_viewer)
    if array_view is None:
        return {}
    with suppress(Exception):
        return dict(array_view.current_index())
    return {}


def ndv_current_index_signal(array_viewer: Any) -> Any | None:
    """
    Return NDV's current-index-changed signal when available.

    Parameters
    ----------
    array_viewer : Any
        NDV array viewer.

    Returns
    -------
    Any or None
        Current-index-changed Qt signal.
    """
    array_view = ndv_array_view(array_viewer)
    return getattr(array_view, "currentIndexChanged", None)
