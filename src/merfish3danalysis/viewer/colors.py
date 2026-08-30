"""Shared viewer color helpers."""

from matplotlib import colormaps


def transcript_color_hex(value: int, value_count: int) -> str:
    """
    Return the transcript display color for one selected transcript value.

    Parameters
    ----------
    value : int
        Positive transcript value.
    value_count : int
        Number of selected transcript values.

    Returns
    -------
    str
        Hex RGB color string.
    """
    if value_count <= 1:
        normalized_value = 1.0
    else:
        normalized_value = (value - 1) / (value_count - 1)
    rgb = colormaps["turbo"](normalized_value)[:3]
    return "#" + "".join(f"{round(channel * 255):02x}" for channel in rgb)
