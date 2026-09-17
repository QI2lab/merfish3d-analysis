"""NDV display decoration helpers."""

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
    controllers = array_viewer._lut_controllers
    applied = 0
    for key, controller in controllers.items():
        if not isinstance(key, int) or key < 0 or key >= len(labels):
            continue
        cmap = channel_cmap_for_label(labels[key])
        if cmap is not None:
            controller.lut_model.cmap = cmap
        for lut_view in controller.lut_views:
            lut_view.set_channel_name(labels[key])
            applied += 1
    return applied


def hide_ndv_volume_button(array_viewer: Any) -> None:
    """
    Hide NDV's built-in volume-rendering button.

    Parameters
    ----------
    array_viewer : Any
        NDV array viewer.
    """
    ndims_button = array_viewer._view._qwidget.ndims_btn
    ndims_button.setChecked(False)
    ndims_button.setVisible(False)
    ndims_button.setEnabled(False)
