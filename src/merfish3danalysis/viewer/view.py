"""NDV and VisPy view layer for the datastore viewer."""

from contextlib import suppress
from typing import Any

import ndv
import numpy as np
from qtpy import QtCore

from merfish3danalysis.viewer.models import (
    SparseOverlayPayload,
    stack_with_micron_coords,
)
from merfish3danalysis.viewer.ndv import (
    apply_lut_channel_labels,
    channel_cmap_for_label,
    hide_ndv_volume_button,
    ndv_canvas_parts,
    ndv_current_index,
    ndv_current_index_signal,
)
from merfish3danalysis.viewer.sparse import SparseVispyOverlay


class _ScaleBar:
    """Own the VisPy scale-bar visuals and canvas callbacks."""

    def __init__(self, view: Any, canvas: Any, x_spacing_um: float) -> None:
        """
        Create scale-bar visuals on one VisPy view.

        Parameters
        ----------
        view : Any
            VisPy scene view.
        canvas : Any
            VisPy canvas.
        x_spacing_um : float
            X pixel spacing in microns.
        """
        from vispy import scene

        self.view = view
        self.canvas = canvas
        self.x_spacing_um = x_spacing_um
        self.line = scene.visuals.Line(
            pos=np.zeros((2, 2), dtype=np.float32),
            color="white",
            width=5,
            parent=view.scene,
        )
        self.text = scene.visuals.Text(
            "",
            color="white",
            font_size=12,
            anchor_x="center",
            anchor_y="bottom",
            parent=view.scene,
        )
        self.line.order = 200
        self.text.order = 201
        canvas.events.draw.connect(self.update)
        canvas.events.resize.connect(self.update)
        self.update()
        canvas.update()

    def close(self) -> None:
        """Disconnect callbacks and remove visuals."""
        with suppress(Exception):
            self.canvas.events.draw.disconnect(self.update)
        with suppress(Exception):
            self.canvas.events.resize.disconnect(self.update)
        self.line.parent = None
        self.text.parent = None

    def update(self, *_args: Any) -> None:
        """
        Update scale-bar length and position for the current camera.

        Parameters
        ----------
        *_args : Any
            Optional VisPy event arguments.
        """
        camera = getattr(self.view, "camera", None)
        rect = getattr(camera, "rect", None)
        if rect is None or rect.width <= 0 or rect.height <= 0:
            return
        target_um = rect.width * self.x_spacing_um / 5.0
        length_um = _nice_scale_bar_length_um(target_um)
        length_px = length_um / self.x_spacing_um
        margin_x = rect.width / 20.0
        margin_y = rect.height / 20.0
        x1 = rect.right - margin_x - length_px
        x2 = rect.right - margin_x
        y = rect.top - margin_y
        self.line.set_data(pos=np.asarray([[x1, y], [x2, y]], dtype=np.float32))
        self.text.text = f"{length_um:g} um"
        self.text.pos = ((x1 + x2) / 2.0, y - margin_y / 2.0)


class DatastoreNdvView(QtCore.QObject):
    """Own NDV windows, LUTs, scale bar, and sparse VisPy overlays."""

    def __init__(self) -> None:
        """Initialize an empty NDV view."""
        super().__init__()
        self.array_viewer: Any | None = None
        self.viewer_windows: list[Any] = []
        self.channel_labels: list[str] = []
        self._scale_bar: _ScaleBar | None = None
        self._sparse_overlay = SparseVispyOverlay()
        self._sparse_payload = SparseOverlayPayload()
        self._ndv_index_signal: Any | None = None
        self._use_3d = False
        self._project_z = False
        self._sparse_refresh_timer = QtCore.QTimer(self)
        self._sparse_refresh_timer.setSingleShot(True)
        self._sparse_refresh_timer.setInterval(15)
        self._sparse_refresh_timer.timeout.connect(self._apply_sparse_overlay_update)

    def show_stack(
        self,
        stack: Any,
        *,
        spacing_zyx_um: Any,
        origin_zyx_um: Any | None,
    ) -> None:
        """
        Display an image stack in a fresh NDV window.

        Parameters
        ----------
        stack : Any
            Image channel stack.
        spacing_zyx_um : Any
            Z, Y, X voxel spacing in microns.
        origin_zyx_um : Any or None
            Optional global origin in microns.
        """
        self.close()
        data = stack_with_micron_coords(stack, spacing_zyx_um, origin_zyx_um)
        self.channel_labels = list(stack.labels)
        self._reset_array_viewer(data, self.channel_labels)
        apply_lut_channel_labels(self.array_viewer, self.channel_labels)
        self._enable_scale_bar(spacing_zyx_um)

    def set_sparse_payload(self, payload: SparseOverlayPayload | None) -> None:
        """
        Install sparse geometry on the current NDV canvas.

        Parameters
        ----------
        payload : SparseOverlayPayload or None
            Sparse overlay payload.
        """
        self._sparse_payload = SparseOverlayPayload() if payload is None else payload
        self._sparse_overlay.set_payload(self._sparse_payload)
        self.refresh_sparse_overlay()

    def set_sparse_mode(self, *, use_3d: bool, project_z: bool) -> None:
        """
        Set sparse overlay display mode.

        Parameters
        ----------
        use_3d : bool
            Whether sparse geometry is shown in 3D.
        project_z : bool
            Whether sparse geometry is max-projected.
        """
        self._use_3d = use_3d
        self._project_z = project_z
        self.refresh_sparse_overlay()

    def refresh_sparse_overlay(self) -> None:
        """Schedule sparse VisPy geometry redraw for the current Z plane or mode."""
        self._sparse_refresh_timer.start()

    def _apply_sparse_overlay_update(self) -> None:
        """Redraw sparse VisPy geometry for the latest Z plane and mode."""
        self._sparse_overlay.update(
            z_index=self._current_z_index(),
            use_3d=self._use_3d,
            project_z=self._project_z,
        )

    def close(self) -> None:
        """Close all NDV windows and release visuals."""
        self._sparse_refresh_timer.stop()
        for widget in list(self.viewer_windows):
            widget.removeEventFilter(self)
            widget.close()
        self.viewer_windows.clear()
        self._disconnect_ndv_index_signal()
        self._disconnect_scale_bar()
        self._sparse_overlay.clear()
        self.array_viewer = None

    def eventFilter(self, watched: Any, event: Any) -> bool:
        """
        Track NDV viewer windows without controlling app lifetime.

        Parameters
        ----------
        watched : Any
            Watched Qt object.
        event : Any
            Qt event.

        Returns
        -------
        bool
            Whether the event was handled.
        """
        close_event = getattr(QtCore.QEvent, "Type", QtCore.QEvent).Close
        if event.type() == close_event and watched in self.viewer_windows:
            self.viewer_windows.remove(watched)
            current_widget = (
                None if self.array_viewer is None else self.array_viewer.widget()
            )
            if watched is current_widget or not self.viewer_windows:
                self._disconnect_ndv_index_signal()
                self._disconnect_scale_bar()
                self._sparse_overlay.clear()
                self.array_viewer = None
            return False
        return super().eventFilter(watched, event)

    def _reset_array_viewer(self, data: Any, labels: list[str]) -> None:
        """
        Replace NDV's viewer to avoid stale axis/channel state.

        Parameters
        ----------
        data : Any
            NDV-compatible image data.
        labels : list[str]
            Channel labels.
        """
        z_size = (
            int(data.sizes["z_um"]) if hasattr(data, "sizes") else int(data.shape[1])
        )
        display_model = self._display_model(data, labels, z_size)
        self.array_viewer = ndv.ArrayViewer(
            data,
            display_model=display_model,
            **(
                {}
                if display_model is not None
                else {
                    "channel_axis": "c",
                    "channel_mode": "composite",
                    "visible_axes": ("y_um", "x_um"),
                    "current_index": {"z_um": z_size // 2},
                }
            ),
        )
        widget = self.array_viewer.widget()
        widget.setWindowTitle("qi2lab NDV view")
        widget.resize(1100, 850)
        widget.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        widget.setParent(None)
        delete_on_close = getattr(
            QtCore.Qt, "WidgetAttribute", QtCore.Qt
        ).WA_DeleteOnClose
        widget.setAttribute(delete_on_close, True)
        widget.installEventFilter(self)
        widget.show()
        self.viewer_windows.append(widget)
        self._sparse_overlay.attach(self.array_viewer)
        self._hide_ndv_3d_button()
        self._connect_ndv_index_signal()

    def _display_model(self, data: Any, labels: list[str], z_size: int) -> Any | None:
        """
        Return an NDV display model for lazy global channel data.

        Parameters
        ----------
        data : Any
            NDV-compatible image data.
        labels : list[str]
            Channel labels.
        z_size : int
            Number of Z planes.

        Returns
        -------
        Any or None
            NDV display model, or ``None`` for NDV defaults.
        """
        if not hasattr(data, "channel_value_range"):
            return None
        from ndv.models import ArrayDisplayModel, LUTModel

        dtype = np.dtype(data.dtype)
        if np.issubdtype(dtype, np.integer):
            dtype_info = np.iinfo(dtype)
            image_clims = (float(dtype_info.min), float(dtype_info.max))
        else:
            image_clims = (0.0, 1.0)
        luts = {0: LUTModel(cmap="gray", clims=image_clims)}
        for channel_index in range(1, int(data.shape[0])):
            value_range = data.channel_value_range(channel_index)
            if value_range is None:
                continue
            cmap = channel_cmap_for_label(labels[channel_index])
            luts[channel_index] = LUTModel(
                cmap="gray" if cmap is None else cmap,
                clims=value_range,
            )
        return ArrayDisplayModel(
            channel_axis="c",
            channel_mode="composite",
            visible_axes=("y_um", "x_um"),
            current_index={"z_um": z_size // 2},
            luts=luts,
        )

    def _hide_ndv_3d_button(self) -> None:
        """Hide NDV's volume-rendering 3D toggle."""
        hide_ndv_volume_button(self.array_viewer)

    def _connect_ndv_index_signal(self) -> None:
        """Refresh sparse geometry when NDV's Z slider changes."""
        self._disconnect_ndv_index_signal()
        signal = ndv_current_index_signal(self.array_viewer)
        if signal is None:
            return
        signal.connect(self.refresh_sparse_overlay)
        self._ndv_index_signal = signal

    def _disconnect_ndv_index_signal(self) -> None:
        """Disconnect the current NDV Z-slider callback."""
        if self._ndv_index_signal is None:
            return
        with suppress(Exception):
            self._ndv_index_signal.disconnect(self.refresh_sparse_overlay)
        self._ndv_index_signal = None

    def _current_z_index(self) -> int:
        """Return the current NDV Z index."""
        if self.array_viewer is None:
            return 0
        current_index = ndv_current_index(self.array_viewer)
        value = current_index.get("z_um")
        if isinstance(value, slice):
            value = value.start
        if value is None:
            data = getattr(self.array_viewer, "data", None)
            z_size = int(data.shape[1]) if data is not None else 1
            return z_size // 2
        return int(value)

    def _disconnect_scale_bar(self) -> None:
        """Remove existing scale-bar visuals and callbacks."""
        if self._scale_bar is not None:
            self._scale_bar.close()
        self._scale_bar = None

    def _enable_scale_bar(self, spacing_zyx_um: Any) -> None:
        """
        Add a physically accurate scale bar to the NDV canvas.

        Parameters
        ----------
        spacing_zyx_um : Any
            Z, Y, X voxel spacing in microns.
        """
        self._disconnect_scale_bar()
        if self.array_viewer is None:
            return
        canvas_controller, view, canvas = ndv_canvas_parts(self.array_viewer)
        if canvas_controller is None or view is None or canvas is None:
            return

        self._scale_bar = _ScaleBar(view, canvas, _x_spacing(spacing_zyx_um))


def _x_spacing(spacing_zyx_um: Any) -> float:
    """
    Return positive X spacing.

    Parameters
    ----------
    spacing_zyx_um : Any
        Z, Y, X voxel spacing in microns.

    Returns
    -------
    float
        Positive X spacing in microns.
    """
    spacing = np.asarray(spacing_zyx_um, dtype=float)
    x_spacing_um = float(spacing[2]) if spacing.size >= 3 else 1.0
    if not np.isfinite(x_spacing_um) or x_spacing_um <= 0:
        return 1.0
    return x_spacing_um


def _nice_scale_bar_length_um(target_um: float) -> float:
    """
    Return a readable scale-bar length no larger than the target.

    Parameters
    ----------
    target_um : float
        Target scale-bar length in microns.

    Returns
    -------
    float
        Rounded scale-bar length in microns.
    """
    if target_um <= 0 or not np.isfinite(target_um):
        return 1.0
    exponent = np.floor(np.log10(target_um))
    base = 10.0**exponent
    for multiplier in (5.0, 2.0, 1.0):
        length_um = multiplier * base
        if length_um <= target_um:
            return float(length_um)
    return float(base)
