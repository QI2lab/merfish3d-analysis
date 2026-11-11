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


def apply_lut_channel_labels(array_viewer: Any, labels: list[str]) -> None:
    """
    Apply human-readable labels and stable overlay colors to ndv LUT views.

    Parameters
    ----------
    array_viewer : Any
        NDV array viewer.
    labels : list[str]
        Channel labels in channel order.

    """
    from ndv.models import LUTModel

    for index, label in enumerate(labels):
        model = array_viewer.display_model.luts.setdefault(index, LUTModel())
        model.name = label
        cmap = channel_cmap_for_label(label)
        if cmap is not None:
            model.cmap = cmap
