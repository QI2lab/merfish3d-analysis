"""NDV display of decoder image volumes and optional point comparisons."""

from collections.abc import Mapping, Sequence

import dask.array as da
import numpy as np
import xarray as xr


def diagnostic_channel_data(
    images: Mapping[str, np.ndarray], spacing_zyx: Sequence[float]
) -> xr.DataArray:
    """Build a lazy CZYX stack with spatial coordinates for diagnostic images.

    Leading image dimensions are flattened into labeled channels. All images
    must share the same trailing ZYX shape. Spacing is in microns, or unity
    when the caller wants pixel coordinates.
    """
    spacing = np.asarray(spacing_zyx, dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing) & (spacing > 0)):
        raise ValueError("Diagnostic spacing must contain three positive values.")
    channels = []
    labels = []
    shape = None
    for name, image in images.items():
        array = da.asarray(image)
        if array.ndim < 2:
            raise ValueError("Diagnostic images must have at least two dimensions.")
        if array.ndim == 2:
            array = array[None]
        if shape is not None and array.shape[-3:] != shape:
            raise ValueError("Diagnostic images must share the same ZYX shape.")
        shape = array.shape[-3:]
        array = array.reshape((-1, *shape))
        channels.append(array)
        labels.extend(
            [name]
            if array.shape[0] == 1
            else [f"{name} {i}" for i in range(array.shape[0])]
        )
    if not channels:
        raise ValueError("Provide at least one diagnostic image.")
    return xr.DataArray(
        da.concatenate(channels),
        dims=("c", "z", "y", "x"),
        coords={
            "c": labels,
            "z": np.arange(shape[0]) * spacing[0],
            "y": np.arange(shape[1]) * spacing[1],
            "x": np.arange(shape[2]) * spacing[2],
        },
    )


def show_diagnostic_images(
    images: Mapping[str, np.ndarray],
    spacing_zyx: Sequence[float],
    *,
    points: Sequence[tuple[str, np.ndarray, str, str]] = (),
    units: str = "um",
) -> None:
    """Display decoder channels with optional ZYX pixel-coordinate points.

    Each point layer supplies a name, coordinates, color, and VisPy marker
    symbol. Points follow the selected Z plane. Images retain independent
    channel visibility and contrast controls in NDV.
    """
    import os

    import ndv
    from qtpy import QtWidgets
    from vispy import scene

    from merfish3danalysis.viewer.ndv import (
        apply_lut_channel_labels,
        hide_ndv_volume_button,
    )

    os.environ["NDV_GUI_FRONTEND"] = "qt"
    os.environ["NDV_CANVAS_BACKEND"] = "vispy"
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    data = diagnostic_channel_data(images, spacing_zyx)
    viewer = ndv.ArrayViewer(
        data, channel_axis="c", channel_mode="composite", visible_axes=("y", "x")
    )
    apply_lut_channel_labels(viewer, list(data.coords["c"].values))
    hide_ndv_volume_button(viewer)
    widget = viewer.widget()
    widget.setWindowTitle(f"Decoder diagnostics ({units})")
    widget.resize(1100, 850)
    widget.show()
    view = viewer._canvas._view
    canvas = viewer._canvas._canvas
    if points and view is None:
        raise RuntimeError("NDV VisPy canvas is unavailable for diagnostic points.")
    visuals = [scene.visuals.Markers(parent=view.scene) for _ in points]

    def update_points(*_args: object) -> None:
        """Draw points from the currently selected Z plane."""
        z_index = int(viewer.display_model.current_index.get("z", 0))
        for visual, (_name, coords, color, symbol) in zip(
            visuals, points, strict=False
        ):
            coords = np.asarray(coords, dtype=float).reshape((-1, 3))
            selected = coords[np.rint(coords[:, 0]).astype(int) == z_index]
            positions = selected[:, [2, 1]] * np.asarray(spacing_zyx)[[2, 1]]
            visual.set_data(
                pos=positions.astype(np.float32),
                size=5,
                face_color=color,
                edge_width=0,
                symbol=symbol,
            )
        if canvas is not None:
            canvas.update()

    signal = viewer.display_model.current_index.value_changed
    if signal is not None:
        signal.connect(update_points)
    update_points()
    try:
        app.exec()
    finally:
        if signal is not None:
            signal.disconnect(update_points)
        for visual in visuals:
            visual.parent = None
        widget.close()
