"""Qt controller and public launcher for the datastore viewer."""

from collections.abc import Callable
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from merfish3danalysis.viewer.datastore import (
    codebook_gene_bits,
    component_summary,
    normalize_datastore_path,
    open_datastore,
    unavailable_data_message,
)
from merfish3danalysis.viewer.models import (
    ChannelStack,
    ProsegRun,
    ViewerBuildResult,
    WarpChainOptions,
    stack_with_micron_coords,
)
from merfish3danalysis.viewer.ndv import apply_lut_channel_labels
from merfish3danalysis.viewer.overlays import (
    append_overlay_channel,
    cell_outline_overlay_for_tile,
    codeword_color_hex,
    discover_proseg_runs,
    global_cell_outline_overlay,
    load_global_image_channels,
    rasterize_cell_outlines,
    rasterize_global_cell_outlines,
    rasterize_global_decoded_spots,
    rasterize_global_proseg_transcripts,
    rasterize_local_decoded_spots,
    rasterize_local_proseg_transcripts,
)
from merfish3danalysis.viewer.warping import load_local_warped_image_channels


class Qi2labViewer:
    """View-only ndv/PyQt GUI for qi2lab datastores."""

    def __init__(self, initial_path: Path | None = None) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        initial_path : Path | None
            initial_path for this viewer operation.
        """
        self.initial_path = initial_path

    def run(self) -> None:
        """
        Launch the viewer.

        Returns
        -------
        None
            Computed viewer result.
        """

        run_viewer(self.initial_path)


def run_viewer(initial_path: Path | None = None) -> None:
    """
    Launch the view-only ndv/PyQt datastore viewer.

    Parameters
    ----------
    initial_path : Path | None
        initial_path for this viewer operation.

    Returns
    -------
    None
        Computed viewer result.
    """

    try:
        import ndv
        from qtpy import QtCore, QtGui, QtWidgets
    except ImportError as exc:
        raise RuntimeError(
            "The qi2lab viewer requires GUI dependencies. Run "
            "`uv sync` to install ndv and Qt support."
        ) from exc

    if hasattr(ndv, "set_gui_backend"):
        ndv.set_gui_backend("qt")
    if hasattr(ndv, "set_canvas_backend"):
        ndv.set_canvas_backend("vispy")

    class DisplayWorkerSignals(QtCore.QObject):
        """Signals emitted by a background display worker."""

        finished = QtCore.Signal(object)
        failed = QtCore.Signal(str)

    class DisplayWorker(QtCore.QRunnable):
        """Run display-stack preparation off the Qt UI thread."""

        def __init__(self, build_display: Callable[[], ViewerBuildResult]) -> None:
            """
            Initialize the object.

            Parameters
            ----------
            build_display : Callable[[], ViewerBuildResult]
                Callable that prepares the viewer stack.
            """

            super().__init__()
            self.build_display = build_display
            self.signals = DisplayWorkerSignals()

        def run(self) -> None:
            """
            Build display data and emit the result.

            Returns
            -------
            None
                Computed viewer result.
            """

            try:
                self.signals.finished.emit(self.build_display())
            except ValueError as exc:
                self.signals.failed.emit(unavailable_data_message(exc))
            except Exception as exc:
                self.signals.failed.emit(str(exc))

    class DatastoreViewerWindow(QtWidgets.QMainWindow):
        """Small view-only Qt wrapper around ndv.ArrayViewer."""

        def __init__(self, path: Path | None = None) -> None:
            """
            Initialize the object.

            Parameters
            ----------
            path : Path | None
                path for this viewer operation.
            """
            super().__init__()
            self.setWindowTitle("qi2lab datastore viewer")
            self.datastore: Any | None = None
            self.datastore_path: Path | None = None
            self.gene_to_bits: dict[str, list[str]] = {}
            self.proseg_runs: list[ProsegRun] = []
            self.channel_labels: list[str] = []
            self.array_viewer: Any | None = None
            self.viewer_windows: list[Any] = []
            self._control_rows: dict[Any, Any] = {}
            self._scale_bar_draw_callback: Callable[..., None] | None = None
            self._scale_bar_visuals: tuple[Any, Any] | None = None
            self._display_worker: DisplayWorker | None = None
            self._display_refresh_enabled = False
            self._display_refresh_pending = False
            self._transcript_refresh_pending = False
            self._transcript_refresh_context: dict[str, Any] | None = None
            self._global_decoded_spots_loaded = False
            self._global_decoded_spots: Any | None = None
            self._proseg_transcripts: dict[str, Any | None] = {}
            self._build_ui()
            if path is not None:
                self.load_path(path)

        def _build_ui(self) -> None:
            """
            Build the compact controller UI.

            Returns
            -------
            None
                No return value.
            """
            central = QtWidgets.QWidget()
            root_layout = QtWidgets.QHBoxLayout(central)
            root_layout.setContentsMargins(4, 4, 4, 4)
            root_layout.setSpacing(4)
            control_panel = QtWidgets.QWidget()
            control_panel.setObjectName("viewerControlPanel")
            control_panel.setMaximumWidth(360)
            self.control_panel = control_panel
            control_layout = QtWidgets.QVBoxLayout(control_panel)
            control_layout.setContentsMargins(6, 4, 6, 4)
            control_layout.setSpacing(3)

            def add_control_row(label: Any, widget: Any) -> None:
                row = QtWidgets.QWidget()
                row_layout = QtWidgets.QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(4)
                label.setMinimumWidth(92)
                row_layout.addWidget(label, stretch=0)
                row_layout.addWidget(widget, stretch=1)
                self._control_rows[label] = row
                self._control_rows[widget] = row
                control_layout.addWidget(row)

            self.path_label = QtWidgets.QLabel("No datastore selected")
            self.path_label.setWordWrap(True)
            open_button = QtWidgets.QPushButton("Open datastore")
            open_button.clicked.connect(self.open_directory)
            control_layout.addWidget(self.path_label)
            control_layout.addWidget(open_button)

            self.component_label = QtWidgets.QLabel("Components: none")
            self.component_label.setWordWrap(True)
            control_layout.addWidget(self.component_label)
            selection_mode = (
                getattr(
                    QtWidgets.QAbstractItemView,
                    "SelectionMode",
                    QtWidgets.QAbstractItemView,
                )
            ).NoSelection

            self.view_mode_label = QtWidgets.QLabel("View mode")
            self.view_mode_combo = QtWidgets.QComboBox()
            self.view_mode_combo.addItems(
                [
                    "Select view type...",
                    "Local native",
                    "Local warped",
                    "Global fused",
                ]
            )
            self.view_mode_combo.currentTextChanged.connect(self._on_view_mode_changed)
            add_control_row(self.view_mode_label, self.view_mode_combo)

            self.warp_preset_label = QtWidgets.QLabel("Warp preset")
            self.warp_preset_combo = QtWidgets.QComboBox()
            self.warp_preset_combo.addItems(
                [
                    "Decode-equivalent",
                    "Native",
                    "Affine only",
                    "Affine + chromatic",
                    "Full",
                ]
            )
            self.warp_preset_combo.currentTextChanged.connect(self._apply_warp_preset)
            add_control_row(self.warp_preset_label, self.warp_preset_combo)
            self.chromatic_checkbox = QtWidgets.QCheckBox("Chromatic affine")
            self.stage_affine_checkbox = QtWidgets.QCheckBox("Stage / round affine")
            self.sofima_checkbox = QtWidgets.QCheckBox("SOFIMA residual")
            self.chromatic_checkbox.setChecked(True)
            self.stage_affine_checkbox.setChecked(True)
            self.sofima_checkbox.setChecked(True)
            control_layout.addWidget(self.chromatic_checkbox)
            control_layout.addWidget(self.stage_affine_checkbox)
            control_layout.addWidget(self.sofima_checkbox)

            self.gpu_label = QtWidgets.QLabel("GPU")
            self.gpu_spinbox = QtWidgets.QSpinBox()
            self.gpu_spinbox.setMinimum(0)
            self.gpu_spinbox.setMaximum(16)
            add_control_row(self.gpu_label, self.gpu_spinbox)

            self.tile_label = QtWidgets.QLabel("Tile")
            self.tile_combo = QtWidgets.QComboBox()
            add_control_row(self.tile_label, self.tile_combo)

            self.fiducial_rounds_label = QtWidgets.QLabel("Fiducial rounds")
            control_layout.addWidget(self.fiducial_rounds_label)
            self.fiducial_round_list = QtWidgets.QListWidget()
            self.fiducial_round_list.setSelectionMode(selection_mode)
            self.fiducial_round_list.setMaximumHeight(86)
            control_layout.addWidget(self.fiducial_round_list)

            self.fiducial_sources_label = QtWidgets.QLabel("Fiducial sources")
            control_layout.addWidget(self.fiducial_sources_label)
            self.fiducial_corrected = QtWidgets.QCheckBox("corrected")
            self.fiducial_registered = QtWidgets.QCheckBox("registered/decon")
            self.fiducial_registered.setChecked(True)
            control_layout.addWidget(self.fiducial_corrected)
            control_layout.addWidget(self.fiducial_registered)

            self.readout_sources_label = QtWidgets.QLabel("Readout bit sources")
            control_layout.addWidget(self.readout_sources_label)
            self.bit_corrected = QtWidgets.QCheckBox("corrected")
            self.bit_registered = QtWidgets.QCheckBox("registered/decon")
            self.bit_feature = QtWidgets.QCheckBox("feature predictor")
            self.bit_feature.setChecked(True)
            control_layout.addWidget(self.bit_corrected)
            control_layout.addWidget(self.bit_registered)
            control_layout.addWidget(self.bit_feature)

            self.bits_label = QtWidgets.QLabel("Bits")
            control_layout.addWidget(self.bits_label)
            self.bit_list = QtWidgets.QListWidget()
            self.bit_list.setSelectionMode(selection_mode)
            self.bit_list.setMaximumHeight(110)
            control_layout.addWidget(self.bit_list)

            self.decoded_checkbox = QtWidgets.QCheckBox("datastore codewords")
            self.decoded_checkbox.toggled.connect(self._on_decoded_overlay_toggled)
            self.gene_list = QtWidgets.QListWidget()
            self.gene_list.setSelectionMode(selection_mode)
            self.gene_list.itemChanged.connect(self._on_gene_checked)
            self.cells_checkbox = QtWidgets.QCheckBox("Cellpose outlines")
            self.cells_checkbox.toggled.connect(self._request_display_refresh)
            self.global_segmentation_checkbox = QtWidgets.QCheckBox("Cellpose mask")
            self.global_segmentation_checkbox.toggled.connect(
                self._request_display_refresh
            )
            self.proseg_run_combo = QtWidgets.QComboBox()
            self.proseg_run_combo.currentTextChanged.connect(
                self._request_display_refresh
            )
            self.proseg_transcripts_checkbox = QtWidgets.QCheckBox("Proseg transcripts")
            self.proseg_transcripts_checkbox.toggled.connect(
                self._on_proseg_transcripts_toggled
            )
            self.proseg_polygons_checkbox = QtWidgets.QCheckBox(
                "Proseg refined polygons"
            )
            self.proseg_polygons_checkbox.toggled.connect(self._request_display_refresh)
            control_layout.addWidget(self.decoded_checkbox)
            self.proseg_run_label = QtWidgets.QLabel("Proseg run")
            add_control_row(self.proseg_run_label, self.proseg_run_combo)
            control_layout.addWidget(self.proseg_transcripts_checkbox)
            self.rna_label = QtWidgets.QLabel("Codeword filter")
            control_layout.addWidget(self.rna_label)
            self.codeword_button_panel = QtWidgets.QWidget()
            codeword_button_layout = QtWidgets.QHBoxLayout(self.codeword_button_panel)
            codeword_button_layout.setContentsMargins(0, 0, 0, 0)
            codeword_button_layout.setSpacing(3)
            self.select_all_codewords_button = QtWidgets.QPushButton("Select all")
            self.deselect_all_codewords_button = QtWidgets.QPushButton("Deselect all")
            self.apply_codewords_button = QtWidgets.QPushButton("Apply")
            self.select_all_codewords_button.clicked.connect(self._select_all_codewords)
            self.deselect_all_codewords_button.clicked.connect(
                self._deselect_all_codewords
            )
            self.apply_codewords_button.clicked.connect(self._apply_codeword_changes)
            self.apply_codewords_button.setEnabled(False)
            codeword_button_layout.addWidget(self.select_all_codewords_button)
            codeword_button_layout.addWidget(self.deselect_all_codewords_button)
            codeword_button_layout.addWidget(self.apply_codewords_button)
            control_layout.addWidget(self.codeword_button_panel)
            self.marker_radius_label = QtWidgets.QLabel("Marker radius")
            self.marker_radius_spinbox = QtWidgets.QSpinBox()
            self.marker_radius_spinbox.setMinimum(0)
            self.marker_radius_spinbox.setMaximum(20)
            self.marker_radius_spinbox.setValue(10)
            self.marker_radius_spinbox.valueChanged.connect(
                self._mark_codeword_changes_pending
            )
            add_control_row(self.marker_radius_label, self.marker_radius_spinbox)
            self.gene_list.setMaximumHeight(170)
            control_layout.addWidget(self.gene_list)
            control_layout.addWidget(self.cells_checkbox)
            control_layout.addWidget(self.global_segmentation_checkbox)
            control_layout.addWidget(self.proseg_polygons_checkbox)

            self.display_button = QtWidgets.QPushButton("Display")
            self.display_button.clicked.connect(self.display_selection)
            control_layout.addWidget(self.display_button)

            self.status_label = QtWidgets.QLabel("")
            self.status_label.setWordWrap(True)
            self.status_label.setMaximumHeight(58)
            control_layout.addWidget(self.status_label)
            self.progress_bar = QtWidgets.QProgressBar()
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setVisible(False)
            control_layout.addWidget(self.progress_bar)
            control_layout.addStretch()

            root_layout.addWidget(control_panel)
            self.setCentralWidget(central)
            self._update_view_options()

        def _on_view_mode_changed(self, _mode: str) -> None:
            """
            Rebuild visible option controls for the selected view mode.

            Parameters
            ----------
            _mode : str
                Selected view mode.

            Returns
            -------
            None
                Computed viewer result.
            """

            self._populate_selected_options()
            self._update_view_options()
            self.adjustSize()

        def _update_view_options(self) -> None:
            """
            Update visible controls for the selected view mode.

            Returns
            -------
            None
                Computed viewer result.
            """

            mode = self.view_mode_combo.currentText()
            local_mode = mode in {"Local native", "Local warped"}
            warped_mode = mode == "Local warped"
            global_mode = mode == "Global fused"
            mode_selected = local_mode or global_mode
            has_proseg = bool(self.proseg_runs)

            def set_option_state(
                widget: Any, visible: bool, enabled: bool | None = None
            ) -> None:
                target = self._control_rows.get(widget, widget)
                target.setVisible(visible)
                if enabled is not None:
                    target.setEnabled(enabled)
                    widget.setEnabled(enabled)

            self.control_panel.setStyleSheet(
                {
                    "Local native": (
                        "QWidget#viewerControlPanel { border-left: 3px solid #5b8def; }"
                    ),
                    "Local warped": (
                        "QWidget#viewerControlPanel { border-left: 3px solid #2aa876; }"
                    ),
                    "Global fused": (
                        "QWidget#viewerControlPanel { border-left: 3px solid #b56bd6; }"
                    ),
                }.get(mode, "QWidget#viewerControlPanel { border-left: none; }")
            )

            warp_widgets = [
                self.warp_preset_label,
                self.warp_preset_combo,
                self.chromatic_checkbox,
                self.stage_affine_checkbox,
                self.sofima_checkbox,
                self.gpu_label,
                self.gpu_spinbox,
            ]
            local_widgets = [
                self.tile_label,
                self.tile_combo,
                self.fiducial_rounds_label,
                self.fiducial_round_list,
                self.fiducial_sources_label,
                self.fiducial_corrected,
                self.fiducial_registered,
                self.readout_sources_label,
                self.bit_corrected,
                self.bit_registered,
                self.bit_feature,
                self.bits_label,
                self.bit_list,
            ]
            proseg_widgets = [
                self.proseg_run_label,
                self.proseg_run_combo,
                self.proseg_transcripts_checkbox,
                self.proseg_polygons_checkbox,
            ]

            for widget in warp_widgets:
                set_option_state(widget, warped_mode, warped_mode)
            for widget in local_widgets:
                set_option_state(widget, local_mode, local_mode)
            for widget in proseg_widgets:
                set_option_state(widget, mode_selected and has_proseg)

            transcript_source_selected = (
                self.decoded_checkbox.isChecked()
                or self.proseg_transcripts_checkbox.isChecked()
            )
            set_option_state(self.global_segmentation_checkbox, global_mode)
            set_option_state(self.decoded_checkbox, mode_selected)
            set_option_state(
                self.rna_label, mode_selected and transcript_source_selected
            )
            set_option_state(
                self.codeword_button_panel, mode_selected and transcript_source_selected
            )
            set_option_state(
                self.marker_radius_label, mode_selected and transcript_source_selected
            )
            set_option_state(
                self.marker_radius_spinbox, mode_selected and transcript_source_selected
            )
            set_option_state(
                self.gene_list, mode_selected and transcript_source_selected
            )
            set_option_state(self.cells_checkbox, mode_selected)
            set_option_state(self.display_button, mode_selected)
            self.display_button.setEnabled(mode_selected)
            self.adjustSize()

        def _request_display_refresh(self, *_args: Any) -> None:
            """
            Refresh the current NDV display after an option change.

            Parameters
            ----------
            *_args : Any
                Optional Qt signal arguments.

            Returns
            -------
            None
                Computed viewer result.
            """

            if (
                not self._display_refresh_enabled
                or self._display_refresh_pending
                or self.progress_bar.isVisible()
            ):
                return
            self._display_refresh_pending = True
            QtCore.QTimer.singleShot(250, self._refresh_current_display)

        def _request_transcript_refresh(self, *_args: Any) -> None:
            """
            Refresh only transcript overlays after codeword display changes.

            Parameters
            ----------
            *_args : Any
                Optional Qt signal arguments.

            Returns
            -------
            None
                Computed viewer result.
            """

            if (
                not self._display_refresh_enabled
                or self._transcript_refresh_pending
                or self.progress_bar.isVisible()
            ):
                return
            self._transcript_refresh_pending = True
            QtCore.QTimer.singleShot(0, self._refresh_transcript_overlay)

        def _refresh_transcript_overlay(self) -> None:
            """
            Redraw transcript overlays without reloading image channels.

            Returns
            -------
            None
                Computed viewer result.
            """

            self._transcript_refresh_pending = False
            if (
                not self._display_refresh_enabled
                or self.progress_bar.isVisible()
                or self._transcript_refresh_context is None
            ):
                return
            try:
                self._start_progress(0, "Updating codeword overlay...")
                stack = self._stack_with_current_transcript_overlay()
                if not self._replace_current_transcript_channel(stack):
                    self._update_current_stack(
                        stack,
                        spacing_zyx_um=self._transcript_refresh_context[
                            "spacing_zyx_um"
                        ],
                        origin_zyx_um=self._transcript_refresh_context.get(
                            "origin_zyx_um"
                        ),
                    )
                self._finish_progress("Displayed: " + ", ".join(stack.labels))
            except Exception as exc:
                self._finish_progress(str(exc))

        def _refresh_current_display(self) -> None:
            """
            Redraw the current NDV view after a control change.

            Returns
            -------
            None
                Computed viewer result.
            """

            self._display_refresh_pending = False
            if not self._display_refresh_enabled or self.progress_bar.isVisible():
                return
            self.display_selection()

        def _reset_array_viewer(self, data: Any) -> None:
            """
            Replace ndv's viewer to avoid stale axis/channel state.

            Parameters
            ----------
            data : Any
                data for this viewer operation.

            Returns
            -------
            None
                Computed viewer result.
            """

            self.array_viewer = ndv.ArrayViewer(
                data,
                channel_axis="c",
                channel_mode="composite",
                visible_axes=("y_um", "x_um"),
                current_index={"z_um": int(data.sizes["z_um"]) // 2},
            )
            widget = self.array_viewer.widget()
            widget.setWindowTitle("qi2lab NDV view")
            widget.resize(1100, 850)
            delete_on_close = getattr(
                QtCore.Qt, "WidgetAttribute", QtCore.Qt
            ).WA_DeleteOnClose
            widget.setAttribute(delete_on_close, True)
            widget.installEventFilter(self)
            widget.show()
            self.viewer_windows.append(widget)

        def _nice_scale_bar_length_um(self, target_um: float) -> float:
            """
            Return a readable scale-bar length no larger than the target.

            Parameters
            ----------
            target_um : float
                Target physical length in microns.

            Returns
            -------
            float
                Rounded physical scale-bar length in microns.
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

        def _disconnect_scale_bar(self) -> None:
            """
            Remove any existing scale-bar visuals and callbacks.

            Returns
            -------
            None
                Computed viewer result.
            """

            if (
                self.array_viewer is not None
                and self._scale_bar_draw_callback is not None
            ):
                canvas_controller = getattr(self.array_viewer, "_canvas", None)
                canvas = getattr(canvas_controller, "_canvas", None)
                if canvas is not None:
                    with suppress(Exception):
                        canvas.events.draw.disconnect(self._scale_bar_draw_callback)
                    with suppress(Exception):
                        canvas.events.resize.disconnect(self._scale_bar_draw_callback)
            if self._scale_bar_visuals is not None:
                for visual in self._scale_bar_visuals:
                    visual.parent = None
            self._scale_bar_draw_callback = None
            self._scale_bar_visuals = None

        def _enable_scale_bar(self, spacing_zyx_um: Any) -> None:
            """
            Add a physically accurate scale bar to the NDV canvas.

            Parameters
            ----------
            spacing_zyx_um : Any
                Display voxel spacing in Z, Y, X microns.

            Returns
            -------
            None
                Computed viewer result.
            """

            self._disconnect_scale_bar()
            if self.array_viewer is None:
                return
            canvas_controller = getattr(self.array_viewer, "_canvas", None)
            view = getattr(canvas_controller, "_view", None)
            canvas = getattr(canvas_controller, "_canvas", None)
            if canvas_controller is None or view is None or canvas is None:
                return

            from vispy import scene

            line = scene.visuals.Line(
                pos=np.zeros((2, 2), dtype=np.float32),
                color="white",
                width=5,
                parent=view.scene,
            )
            text = scene.visuals.Text(
                "",
                color="white",
                font_size=12,
                anchor_x="center",
                anchor_y="bottom",
                parent=view.scene,
            )
            line.order = 200
            text.order = 201
            spacing = np.asarray(spacing_zyx_um, dtype=float)
            x_spacing_um = float(spacing[2])

            def update_scale_bar(*_args: Any) -> None:
                camera = getattr(view, "camera", None)
                rect = getattr(camera, "rect", None)
                if rect is None or rect.width <= 0 or rect.height <= 0:
                    return
                target_um = rect.width * x_spacing_um / 5.0
                length_um = self._nice_scale_bar_length_um(target_um)
                length_px = length_um / x_spacing_um
                margin_x = rect.width / 20.0
                margin_y = rect.height / 20.0
                x1 = rect.right - margin_x - length_px
                x2 = rect.right - margin_x
                y = rect.top - margin_y
                line.set_data(pos=np.asarray([[x1, y], [x2, y]], dtype=np.float32))
                text.text = f"{length_um:g} um"
                text.pos = ((x1 + x2) / 2.0, y - margin_y / 2.0)

            self._scale_bar_visuals = (line, text)
            self._scale_bar_draw_callback = update_scale_bar
            canvas.events.draw.connect(update_scale_bar)
            canvas.events.resize.connect(update_scale_bar)
            update_scale_bar()
            canvas.update()

        def eventFilter(self, watched: Any, event: Any) -> bool:
            """
            Track NDV viewer windows without letting them control app lifetime.

            Parameters
            ----------
            watched : Any
                Qt object receiving the event.
            event : Any
                Qt event.

            Returns
            -------
            bool
                Whether the event was consumed.
            """

            close_event = getattr(QtCore.QEvent, "Type", QtCore.QEvent).Close
            if event.type() == close_event and watched in self.viewer_windows:
                self.viewer_windows.remove(watched)
                current_widget = (
                    None if self.array_viewer is None else self.array_viewer.widget()
                )
                if watched is current_widget or not self.viewer_windows:
                    self._disconnect_scale_bar()
                    self.array_viewer = None
                return False
            return super().eventFilter(watched, event)

        def closeEvent(self, event: Any) -> None:
            """
            Close NDV viewer windows when the controller window closes.

            Parameters
            ----------
            event : Any
                Main window close event.

            Returns
            -------
            None
                Computed viewer result.
            """

            for widget in list(self.viewer_windows):
                widget.removeEventFilter(self)
                widget.close()
            self.viewer_windows.clear()
            self._disconnect_scale_bar()
            self.array_viewer = None
            super().closeEvent(event)
            qt_app = QtWidgets.QApplication.instance()
            if qt_app is not None:
                QtCore.QTimer.singleShot(0, qt_app.quit)

        def _update_current_stack(
            self,
            stack: ChannelStack,
            spacing_zyx_um: Any,
            origin_zyx_um: Any | None = None,
        ) -> None:
            """
            Update the current NDV viewer data without opening a new window.

            Parameters
            ----------
            stack : ChannelStack
                Stack to display.
            spacing_zyx_um : Any
                Voxel spacing in Z, Y, X microns.
            origin_zyx_um : Any | None
                Optional origin in Z, Y, X microns.

            Returns
            -------
            None
                Computed viewer result.
            """

            if self.array_viewer is None:
                self._show_stack(stack, spacing_zyx_um, origin_zyx_um=origin_zyx_um)
                return
            self.channel_labels = stack.labels
            self.array_viewer.data = stack_with_micron_coords(
                stack,
                spacing_zyx_um,
                origin_zyx_um=origin_zyx_um,
            )
            self._apply_lut_names(self.channel_labels)
            self._enable_scale_bar(spacing_zyx_um)

        def _replace_current_transcript_channel(self, stack: ChannelStack) -> bool:
            """
            Replace only the current transcript overlay channel when possible.

            Parameters
            ----------
            stack : ChannelStack
                New stack containing the same image channels and a refreshed
                transcript overlay channel.

            Returns
            -------
            bool
                True when the existing NDV data was updated in place.
            """

            if self.array_viewer is None:
                return False
            current_data = getattr(self.array_viewer, "data", None)
            if current_data is None or len(stack.labels) == 0:
                return False
            if stack.labels[:-1] != self.channel_labels[:-1]:
                return False
            if tuple(getattr(current_data, "shape", ())) != stack.data.shape:
                return False
            try:
                current_data.data[-1, :, :, :] = stack.data[-1, :, :, :]
                self.array_viewer.data = current_data
            except Exception:
                return False
            self.channel_labels = stack.labels
            self._apply_lut_names(self.channel_labels)
            return True

        def _show_stack(
            self,
            stack: ChannelStack,
            spacing_zyx_um: Any,
            origin_zyx_um: Any | None = None,
        ) -> None:
            """
            Show a channel stack in a fresh NDV viewer.

            Parameters
            ----------
            stack : ChannelStack
                Stack to display.
            spacing_zyx_um : Any
                Voxel spacing in Z, Y, X microns.
            origin_zyx_um : Any | None
                Optional origin in Z, Y, X microns.

            Returns
            -------
            None
                Computed viewer result.
            """

            self.channel_labels = stack.labels
            self._reset_array_viewer(
                stack_with_micron_coords(
                    stack,
                    spacing_zyx_um,
                    origin_zyx_um=origin_zyx_um,
                )
            )
            self._enable_scale_bar(spacing_zyx_um)
            self._apply_lut_names(self.channel_labels)
            QtCore.QTimer.singleShot(50, self._apply_lut_names_callback(stack.labels))
            QtCore.QTimer.singleShot(250, self._apply_lut_names_callback(stack.labels))

        def _cached_global_decoded_spots(self) -> Any | None:
            """
            Return cached global decoded spots loaded through the datastore.

            Returns
            -------
            Any | None
                Global decoded spots, or None when unavailable.
            """

            if not self._global_decoded_spots_loaded:
                self._global_decoded_spots = (
                    self.datastore.load_global_filtered_decoded_spots()
                )
                self._global_decoded_spots_loaded = True
            return self._global_decoded_spots

        def _cached_proseg_transcripts(self, run_name: str) -> Any | None:
            """
            Return cached Proseg transcripts loaded through the datastore.

            Parameters
            ----------
            run_name : str
                Proseg run name.

            Returns
            -------
            Any | None
                Proseg transcript table, or None when unavailable.
            """

            if run_name not in self._proseg_transcripts:
                self._proseg_transcripts[run_name] = (
                    self.datastore.load_proseg_transcripts_3d(run_name=run_name)
                )
            return self._proseg_transcripts[run_name]

        def _stack_with_current_transcript_overlay(self) -> ChannelStack:
            """
            Return the current base stack plus selected transcript overlay.

            Returns
            -------
            ChannelStack
                Current display stack.
            """

            if self._transcript_refresh_context is None:
                raise RuntimeError("No displayed stack is available to refresh.")

            context = self._transcript_refresh_context
            stack = context["base_stack"]
            shape_zyx = stack.data.shape[1:]
            selected_genes = self._selected_decoded_genes()
            radius = self.marker_radius_spinbox.value()
            mode = context["mode"]

            if self.decoded_checkbox.isChecked():
                if mode == "global":
                    decoded_spots = self._cached_global_decoded_spots()
                    overlay = (
                        None
                        if decoded_spots is None
                        else rasterize_global_decoded_spots(
                            decoded_spots,
                            shape_zyx=shape_zyx,
                            origin_zyx_um=context["origin_zyx_um"],
                            spacing_zyx_um=context["spacing_zyx_um"],
                            genes=selected_genes,
                            radius=radius,
                        )
                    )
                    return append_overlay_channel(
                        stack,
                        overlay,
                        "global datastore codewords",
                    )

                decoded_spots = self._cached_global_decoded_spots()
                tile_transform = context.get("tile_transform")
                overlay = (
                    None
                    if decoded_spots is None or tile_transform is None
                    else rasterize_local_decoded_spots(
                        decoded_spots,
                        shape_zyx=shape_zyx,
                        affine_zyx_um=tile_transform[0],
                        origin_zyx_um=tile_transform[1],
                        spacing_zyx_um=tile_transform[2],
                        genes=selected_genes,
                        radius=radius,
                    )
                )
                return append_overlay_channel(stack, overlay, "datastore codewords")

            if self.proseg_transcripts_checkbox.isChecked():
                proseg_run = self._selected_proseg_run()
                if proseg_run is None:
                    return stack
                transcripts = self._cached_proseg_transcripts(proseg_run.name)
                if mode == "global":
                    overlay = rasterize_global_proseg_transcripts(
                        transcripts,
                        shape_zyx=shape_zyx,
                        origin_zyx_um=context["origin_zyx_um"],
                        spacing_zyx_um=context["spacing_zyx_um"],
                        genes=selected_genes,
                        radius=radius,
                    )
                    return append_overlay_channel(
                        stack,
                        overlay,
                        f"global Proseg transcripts: {proseg_run.name}",
                    )

                tile_transform = context.get("proseg_tile_transform")
                if tile_transform is None:
                    return append_overlay_channel(
                        stack,
                        None,
                        f"Proseg transcripts: {proseg_run.name}",
                    )
                affine, origin, spacing = tile_transform
                overlay = rasterize_local_proseg_transcripts(
                    transcripts,
                    shape_zyx=shape_zyx,
                    affine_zyx_um=affine,
                    origin_zyx_um=origin,
                    spacing_zyx_um=spacing,
                    genes=selected_genes,
                    radius=radius,
                )
                return append_overlay_channel(
                    stack,
                    overlay,
                    f"Proseg transcripts: {proseg_run.name}",
                )

            return stack

        def open_directory(self) -> None:
            """
            Open directory.

            Returns
            -------
            None
                Computed viewer result.
            """
            selected = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Select experiment root or qi2labdatastore",
                str(Path.home()),
            )
            if selected:
                self.load_path(Path(selected))

        def load_path(self, path: Path) -> None:
            """
            Load path.

            Parameters
            ----------
            path : Path
                path for this viewer operation.

            Returns
            -------
            None
                Computed viewer result.
            """
            try:
                datastore_path = normalize_datastore_path(path)
                self.datastore = open_datastore(datastore_path)
                self.datastore_path = datastore_path
            except Exception as exc:
                self.status_label.setText(str(exc))
                return

            self.path_label.setText(str(datastore_path))
            self._display_refresh_enabled = False
            self._display_refresh_pending = False
            self._transcript_refresh_pending = False
            self._transcript_refresh_context = None
            self._global_decoded_spots_loaded = False
            self._global_decoded_spots = None
            self._proseg_transcripts.clear()
            self._populate_controls()
            self.status_label.setText("Datastore loaded. Select a view type.")

        def _clear_option_controls(self) -> None:
            """
            Clear mode-specific option controls.

            Returns
            -------
            None
                Computed viewer result.
            """

            self.tile_combo.clear()
            self.bit_list.clear()
            self.fiducial_round_list.clear()
            self.gene_list.setUpdatesEnabled(False)
            self.gene_list.blockSignals(True)
            self.gene_list.clear()
            self.gene_list.blockSignals(False)
            self.gene_list.setUpdatesEnabled(True)
            self.apply_codewords_button.setEnabled(False)
            self.proseg_run_combo.clear()
            self.gene_to_bits = {}

        def _populate_controls(self) -> None:
            """
            Refresh datastore-level availability without populating view options.

            Returns
            -------
            None
                Computed viewer result.
            """
            if self.datastore is None:
                return

            self.proseg_runs = discover_proseg_runs(self.datastore)
            state = component_summary(self.datastore)
            enabled = [name for name, value in state.items() if value]
            self.component_label.setText(
                "Components: " + (", ".join(enabled) if enabled else "none")
            )
            self.decoded_checkbox.setEnabled(True)
            self.cells_checkbox.setEnabled(True)
            self.global_segmentation_checkbox.setEnabled(True)
            has_proseg = bool(self.proseg_runs)
            self.proseg_run_combo.setEnabled(has_proseg)
            self.proseg_transcripts_checkbox.setEnabled(has_proseg)
            self.proseg_polygons_checkbox.setEnabled(has_proseg)
            self._clear_option_controls()
            self._populate_selected_options()
            self._update_view_options()

        def _populate_selected_options(self) -> None:
            """
            Populate only the controls needed by the selected view mode.

            Returns
            -------
            None
                Computed viewer result.
            """

            self._clear_option_controls()
            if self.datastore is None:
                return

            mode = self.view_mode_combo.currentText()
            local_mode = mode in {"Local native", "Local warped"}
            global_mode = mode == "Global fused"
            if not (local_mode or global_mode):
                return

            item_flag = getattr(QtCore.Qt, "ItemFlag", QtCore.Qt).ItemIsUserCheckable
            checked_state = getattr(QtCore.Qt, "CheckState", QtCore.Qt).Checked
            unchecked_state = getattr(QtCore.Qt, "CheckState", QtCore.Qt).Unchecked

            if local_mode:
                for tile_id in self.datastore.tile_ids or []:
                    self.tile_combo.addItem(str(tile_id))

                for idx, round_id in enumerate(self.datastore.round_ids or []):
                    item = QtWidgets.QListWidgetItem(str(round_id))
                    item.setFlags(item.flags() | item_flag)
                    item.setCheckState(checked_state if idx == 0 else unchecked_state)
                    self.fiducial_round_list.addItem(item)

                for idx, bit_id in enumerate(self.datastore.bit_ids or []):
                    item = QtWidgets.QListWidgetItem(str(bit_id))
                    item.setFlags(item.flags() | item_flag)
                    item.setCheckState(checked_state if idx == 0 else unchecked_state)
                    self.bit_list.addItem(item)

            self.gene_to_bits = codebook_gene_bits(self.datastore)
            self.gene_list.setUpdatesEnabled(False)
            self.gene_list.blockSignals(True)
            for gene_id in sorted(self.gene_to_bits):
                item = QtWidgets.QListWidgetItem(gene_id)
                item.setFlags(item.flags() | item_flag)
                item.setData(
                    getattr(QtCore.Qt, "ItemDataRole", QtCore.Qt).UserRole, gene_id
                )
                item.setCheckState(checked_state)
                self.gene_list.addItem(item)
            self._update_codeword_color_key()
            self.gene_list.blockSignals(False)
            self.gene_list.setUpdatesEnabled(True)
            self.apply_codewords_button.setEnabled(False)

            for run in self.proseg_runs:
                self.proseg_run_combo.addItem(run.name)

        def _on_decoded_overlay_toggled(self, checked: bool) -> None:
            """
            Keep datastore and Proseg transcript overlays mutually exclusive.

            Parameters
            ----------
            checked : bool
                Whether datastore transcript overlays were selected.

            Returns
            -------
            None
                Computed viewer result.
            """

            if checked and self.proseg_transcripts_checkbox.isChecked():
                self.proseg_transcripts_checkbox.blockSignals(True)
                self.proseg_transcripts_checkbox.setChecked(False)
                self.proseg_transcripts_checkbox.blockSignals(False)
            self._update_view_options()
            self._request_transcript_refresh()

        def _on_proseg_transcripts_toggled(self, checked: bool) -> None:
            """
            Keep Proseg and datastore transcript overlays mutually exclusive.

            Parameters
            ----------
            checked : bool
                Whether Proseg transcript overlays were selected.

            Returns
            -------
            None
                Computed viewer result.
            """

            if checked and self.decoded_checkbox.isChecked():
                self.decoded_checkbox.blockSignals(True)
                self.decoded_checkbox.setChecked(False)
                self.decoded_checkbox.blockSignals(False)
            self._update_view_options()
            self._request_transcript_refresh()

        def _apply_warp_preset(self, preset: str) -> None:
            """
            Apply a named local warp-chain preset to the checkboxes.

            Parameters
            ----------
            preset : str
                Preset label from the UI.

            Returns
            -------
            None
                Computed viewer result.
            """

            if preset in {"Decode-equivalent", "Full"}:
                chromatic, stage_affine, sofima = True, True, True
            elif preset == "Native":
                chromatic, stage_affine, sofima = False, False, False
            elif preset == "Affine only":
                chromatic, stage_affine, sofima = False, True, False
            elif preset == "Affine + chromatic":
                chromatic, stage_affine, sofima = True, True, False
            else:
                return
            self.chromatic_checkbox.setChecked(chromatic)
            self.stage_affine_checkbox.setChecked(stage_affine)
            self.sofima_checkbox.setChecked(sofima)

        def _warp_options(self) -> WarpChainOptions:
            """
            Return selected local warp-chain options.

            Returns
            -------
            WarpChainOptions
                Selected transform components.
            """

            if self.view_mode_combo.currentText() == "Local native":
                return WarpChainOptions(
                    chromatic=False,
                    stage_affine=False,
                    sofima=False,
                )
            return WarpChainOptions(
                chromatic=self.chromatic_checkbox.isChecked(),
                stage_affine=self.stage_affine_checkbox.isChecked(),
                sofima=self.sofima_checkbox.isChecked(),
            )

        def _selected_fiducial_rounds(self) -> list[str]:
            """
            Return checked fiducial rounds.

            Returns
            -------
            list[str]
                Selected fiducial round identifiers.
            """

            checked: list[str] = []
            checked_state = getattr(QtCore.Qt, "CheckState", QtCore.Qt).Checked
            for row in range(self.fiducial_round_list.count()):
                item = self.fiducial_round_list.item(row)
                if item.checkState() == checked_state:
                    checked.append(item.text())
            return checked

        def _selected_proseg_run(self) -> ProsegRun | None:
            """
            Return selected Proseg run.

            Returns
            -------
            ProsegRun | None
                Selected run or None when unavailable.
            """

            name = self.proseg_run_combo.currentText()
            for run in self.proseg_runs:
                if run.name == name:
                    return run
            return None

        def _mark_codeword_changes_pending(self, *_args: Any) -> None:
            """
            Mark codeword display settings as changed but not yet applied.

            Parameters
            ----------
            *_args : Any
                Optional Qt signal arguments.

            Returns
            -------
            None
                Computed viewer result.
            """

            self.apply_codewords_button.setEnabled(True)
            self.status_label.setText("Codeword display changes are pending.")

        def _codeword_is_user_checkable(self, item: Any) -> bool:
            """
            Return whether a list item represents a checkable codeword.

            Parameters
            ----------
            item : Any
                Codeword list item.

            Returns
            -------
            bool
                True when the item is a codeword row.
            """

            item_flag = getattr(QtCore.Qt, "ItemFlag", QtCore.Qt).ItemIsUserCheckable
            return bool(item.flags() & item_flag)

        def _checked_codeword_genes(self) -> list[str]:
            """
            Return checked codeword genes in current list order.

            Returns
            -------
            list[str]
                Checked codeword gene ids.
            """

            checked_state = getattr(QtCore.Qt, "CheckState", QtCore.Qt).Checked
            selected: list[str] = []
            for row in range(self.gene_list.count()):
                item = self.gene_list.item(row)
                if (
                    self._codeword_is_user_checkable(item)
                    and item.checkState() == checked_state
                ):
                    selected.append(self._codeword_item_gene_id(item))
            return selected

        def _rebuild_codeword_list(self, selected_genes: list[str]) -> None:
            """
            Reorder codeword rows with selected genes above unselected genes.

            Parameters
            ----------
            selected_genes : list[str]
                Selected codeword gene ids in display order.

            Returns
            -------
            None
                Computed viewer result.
            """

            user_role = getattr(QtCore.Qt, "ItemDataRole", QtCore.Qt).UserRole
            item_flag = getattr(QtCore.Qt, "ItemFlag", QtCore.Qt).ItemIsUserCheckable
            checked_state = getattr(QtCore.Qt, "CheckState", QtCore.Qt).Checked
            unchecked_state = getattr(QtCore.Qt, "CheckState", QtCore.Qt).Unchecked
            selected_set = set(selected_genes)
            ordered_selected = [
                gene_id
                for gene_id in dict.fromkeys(selected_genes)
                if gene_id in self.gene_to_bits
            ]
            ordered_unselected = [
                gene_id
                for gene_id in sorted(self.gene_to_bits)
                if gene_id not in selected_set
            ]

            self.gene_list.setUpdatesEnabled(False)
            self.gene_list.blockSignals(True)
            self.gene_list.clear()
            for gene_id in ordered_selected:
                item = QtWidgets.QListWidgetItem(gene_id)
                item.setFlags(item.flags() | item_flag)
                item.setData(user_role, gene_id)
                item.setCheckState(checked_state)
                self.gene_list.addItem(item)
            if ordered_selected and ordered_unselected:
                separator = QtWidgets.QListWidgetItem(
                    "──────── Unselected codewords ────────"
                )
                separator.setFlags(
                    getattr(QtCore.Qt, "ItemFlag", QtCore.Qt).NoItemFlags
                )
                self.gene_list.addItem(separator)
            for gene_id in ordered_unselected:
                item = QtWidgets.QListWidgetItem(gene_id)
                item.setFlags(item.flags() | item_flag)
                item.setData(user_role, gene_id)
                item.setCheckState(unchecked_state)
                self.gene_list.addItem(item)
            self.gene_list.blockSignals(False)
            self.gene_list.setUpdatesEnabled(True)
            self._update_codeword_color_key()

        def _sync_bits_to_selected_codewords(self, selected_genes: list[str]) -> None:
            """
            Check readout bits used by the selected codewords.

            Parameters
            ----------
            selected_genes : list[str]
                Selected codeword gene ids.

            Returns
            -------
            None
                Computed viewer result.
            """

            checked_state = getattr(QtCore.Qt, "CheckState", QtCore.Qt).Checked
            unchecked_state = getattr(QtCore.Qt, "CheckState", QtCore.Qt).Unchecked
            bit_set: set[str] = set()
            if selected_genes:
                bit_set = set().union(
                    *(set(self.gene_to_bits[gene_id]) for gene_id in selected_genes)
                )
            self.bit_list.blockSignals(True)
            for row in range(self.bit_list.count()):
                item = self.bit_list.item(row)
                item.setCheckState(
                    checked_state if item.text() in bit_set else unchecked_state
                )
            self.bit_list.blockSignals(False)

        def _on_gene_checked(self, item: Any) -> None:
            """
            On gene checked.

            Parameters
            ----------
            item : Any
                Changed codeword list item.

            Returns
            -------
            None
                Computed viewer result.
            """

            if not self._codeword_is_user_checkable(item):
                return
            selected_genes = self._checked_codeword_genes()
            checked_state = getattr(QtCore.Qt, "CheckState", QtCore.Qt).Checked
            if item.checkState() == checked_state:
                changed_gene = self._codeword_item_gene_id(item)
                selected_genes = [
                    changed_gene,
                    *[gene_id for gene_id in selected_genes if gene_id != changed_gene],
                ]
            self._rebuild_codeword_list(selected_genes)
            self._sync_bits_to_selected_codewords(selected_genes)
            if selected_genes and not self.proseg_transcripts_checkbox.isChecked():
                self.decoded_checkbox.blockSignals(True)
                self.decoded_checkbox.setChecked(True)
                self.decoded_checkbox.blockSignals(False)
                self._update_view_options()
            self._mark_codeword_changes_pending()

        def _apply_codeword_changes(self) -> None:
            """
            Apply pending codeword display changes to the current NDV viewer.

            Returns
            -------
            None
                Computed viewer result.
            """

            self.apply_codewords_button.setEnabled(False)
            self._request_transcript_refresh()

        def _set_all_codewords(self, checked: bool) -> None:
            """
            Set every codeword filter checkbox to the same state.

            Parameters
            ----------
            checked : bool
                Whether all codewords should be selected.

            Returns
            -------
            None
                Computed viewer result.
            """

            checked_state = getattr(QtCore.Qt, "CheckState", QtCore.Qt).Checked
            unchecked_state = getattr(QtCore.Qt, "CheckState", QtCore.Qt).Unchecked
            self.gene_list.blockSignals(True)
            for row in range(self.gene_list.count()):
                item = self.gene_list.item(row)
                if self._codeword_is_user_checkable(item):
                    item.setCheckState(checked_state if checked else unchecked_state)
            self.gene_list.blockSignals(False)
            selected_genes = self._checked_codeword_genes()
            self._rebuild_codeword_list(selected_genes)
            self._sync_bits_to_selected_codewords(selected_genes)
            if selected_genes and not self.proseg_transcripts_checkbox.isChecked():
                self.decoded_checkbox.blockSignals(True)
                self.decoded_checkbox.setChecked(True)
                self.decoded_checkbox.blockSignals(False)
                self._update_view_options()
            self._mark_codeword_changes_pending()

        def _select_all_codewords(self) -> None:
            """
            Select all codewords in the filter table.

            Returns
            -------
            None
                Computed viewer result.
            """

            self._set_all_codewords(True)

        def _deselect_all_codewords(self) -> None:
            """
            Deselect all codewords in the filter table.

            Returns
            -------
            None
                Computed viewer result.
            """

            self._set_all_codewords(False)

        def _checked_bits(self) -> list[str]:
            """
            Checked bits.

            Returns
            -------
            list[str]
                Computed viewer result.
            """
            checked: list[str] = []
            checked_state = getattr(QtCore.Qt, "CheckState", QtCore.Qt).Checked
            for row in range(self.bit_list.count()):
                item = self.bit_list.item(row)
                if item.checkState() == checked_state:
                    checked.append(item.text())
            return checked

        def _fiducial_sources(self) -> list[str]:
            """
            Fiducial sources.

            Returns
            -------
            list[str]
                Computed viewer result.
            """
            sources: list[str] = []
            if self.fiducial_corrected.isChecked():
                sources.append("corrected")
            if self.fiducial_registered.isChecked():
                sources.append("registered")
            return sources

        def _bit_sources(self) -> list[str]:
            """
            Bit sources.

            Returns
            -------
            list[str]
                Computed viewer result.
            """
            sources: list[str] = []
            if self.bit_corrected.isChecked():
                sources.append("corrected")
            if self.bit_registered.isChecked():
                sources.append("registered")
            if self.bit_feature.isChecked():
                sources.append("feature")
            return sources

        def _selected_decoded_genes(self) -> list[str]:
            """
            Selected decoded genes.

            Returns
            -------
            list[str]
                Computed viewer result.
            """
            selected: list[str] = []
            checked_state = getattr(QtCore.Qt, "CheckState", QtCore.Qt).Checked
            for row in range(self.gene_list.count()):
                item = self.gene_list.item(row)
                if (
                    self._codeword_is_user_checkable(item)
                    and item.checkState() == checked_state
                ):
                    selected.append(self._codeword_item_gene_id(item))
            return selected

        def _codeword_item_gene_id(self, item: Any) -> str:
            """
            Return the source gene id stored on one codeword list item.

            Parameters
            ----------
            item : Any
                Codeword list item.

            Returns
            -------
            str
                Codeword gene id.
            """

            user_role = getattr(QtCore.Qt, "ItemDataRole", QtCore.Qt).UserRole
            gene_id = item.data(user_role)
            return str(gene_id if gene_id is not None else item.text())

        def _codeword_color_icon(self, color_hex: str) -> Any:
            """
            Return a Qt icon swatch for one codeword color.

            Parameters
            ----------
            color_hex : str
                Hex RGB color string.

            Returns
            -------
            Any
                Qt icon.
            """

            pixmap = QtGui.QPixmap(16, 16)
            pixmap.fill(QtGui.QColor(color_hex))
            return QtGui.QIcon(pixmap)

        def _update_codeword_color_key(self) -> None:
            """
            Update codeword list row labels and color swatches.

            Returns
            -------
            None
                Computed viewer result.
            """

            checked_state = getattr(QtCore.Qt, "CheckState", QtCore.Qt).Checked
            selected_genes = self._selected_decoded_genes()
            value_by_gene = {
                gene_id: idx + 1 for idx, gene_id in enumerate(selected_genes)
            }
            value_count = len(selected_genes)
            self.gene_list.blockSignals(True)
            for row in range(self.gene_list.count()):
                item = self.gene_list.item(row)
                if not self._codeword_is_user_checkable(item):
                    continue
                gene_id = self._codeword_item_gene_id(item)
                if item.checkState() == checked_state and gene_id in value_by_gene:
                    value = value_by_gene[gene_id]
                    item.setText(gene_id)
                    item.setIcon(
                        self._codeword_color_icon(
                            codeword_color_hex(value, value_count)
                        )
                    )
                    item.setToolTip(
                        f"{gene_id}: selected transcript color; overlay value {value}"
                    )
                else:
                    item.setText(gene_id)
                    item.setIcon(QtGui.QIcon())
                    item.setToolTip("")
            self.gene_list.blockSignals(False)

        def _apply_lut_names(self, labels: list[str]) -> None:
            """
            Apply lut names.

            Parameters
            ----------
            labels : list[str]
                labels for this viewer operation.

            Returns
            -------
            None
                Computed viewer result.
            """
            if self.array_viewer is None:
                return
            apply_lut_channel_labels(self.array_viewer, labels)

        def _apply_lut_names_callback(self, labels: list[str]) -> Any:
            """
            Return a callback that applies LUT names to the current viewer.

            Parameters
            ----------
            labels : list[str]
                Channel labels to apply when the callback runs.

            Returns
            -------
            Any
                Zero-argument callback for ``QTimer.singleShot``.
            """

            def apply_labels() -> None:
                self._apply_lut_names(labels)

            return apply_labels

        def _start_progress(self, total_steps: int, message: str) -> None:
            """
            Show progress and disable display refresh controls.

            Parameters
            ----------
            total_steps : int
                Number of known steps. Values less than one show an indeterminate bar.
            message : str
                Status text to display while work is running.

            Returns
            -------
            None
                No return value.
            """
            self.display_button.setEnabled(False)
            if total_steps <= 0:
                self.progress_bar.setRange(0, 0)
            else:
                self.progress_bar.setRange(0, max(total_steps, 1))
                self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self.status_label.setText(message)
            QtWidgets.QApplication.processEvents()

        def _finish_progress(self, message: str) -> None:
            """
            Hide progress and restore display refresh controls.

            Parameters
            ----------
            message : str
                Status text to show after work finishes.

            Returns
            -------
            None
                No return value.
            """
            self.display_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_label.setText(message)
            QtWidgets.QApplication.processEvents()

        def _start_display_worker(
            self,
            build_display: Callable[[], ViewerBuildResult],
            message: str,
        ) -> None:
            """
            Start a background display build.

            Parameters
            ----------
            build_display : Callable[[], ViewerBuildResult]
                Background display build callable.
            message : str
                Progress message.

            Returns
            -------
            None
                Computed viewer result.
            """

            if self._display_worker is not None:
                self.status_label.setText("A display update is already running.")
                return
            self._start_progress(0, message)
            self._display_refresh_enabled = False
            worker = DisplayWorker(build_display)
            worker.signals.finished.connect(self._finish_display_worker)
            worker.signals.failed.connect(self._fail_display_worker)
            self._display_worker = worker
            QtCore.QThreadPool.globalInstance().start(worker)

        def _finish_display_worker(self, result: ViewerBuildResult) -> None:
            """
            Show a prepared display result on the Qt UI thread.

            Parameters
            ----------
            result : ViewerBuildResult
                Prepared viewer stack.

            Returns
            -------
            None
                Computed viewer result.
            """

            self._display_worker = None
            self._transcript_refresh_context = result.context
            if result.decoded_spots is not None:
                self._global_decoded_spots = result.decoded_spots
                self._global_decoded_spots_loaded = True
            if (
                result.proseg_run_name is not None
                and result.proseg_transcripts is not None
            ):
                self._proseg_transcripts[result.proseg_run_name] = (
                    result.proseg_transcripts
                )
            self._show_stack(
                result.stack,
                spacing_zyx_um=result.spacing_zyx_um,
                origin_zyx_um=result.origin_zyx_um,
            )
            self._display_refresh_enabled = True
            self._finish_progress(result.status)

        def _fail_display_worker(self, message: str) -> None:
            """
            Finish a failed background display build.

            Parameters
            ----------
            message : str
                User-facing error message.

            Returns
            -------
            None
                Computed viewer result.
            """

            self._display_worker = None
            self._finish_progress(message)

        def _build_global_display_result(
            self,
            include_segmentation: bool,
            include_decoded: bool,
            include_cells: bool,
            include_proseg_transcripts: bool,
            include_proseg_polygons: bool,
            selected_genes: list[str],
            marker_radius: int,
            proseg_run_name: str | None,
        ) -> ViewerBuildResult:
            """
            Build global fused display data off the Qt UI thread.

            Parameters
            ----------
            include_segmentation : bool
                Whether to include the Cellpose mask image.
            include_decoded : bool
                Whether to include datastore decoded transcripts.
            include_cells : bool
                Whether to include Cellpose outlines.
            include_proseg_transcripts : bool
                Whether to include Proseg transcripts.
            include_proseg_polygons : bool
                Whether to include Proseg polygons.
            selected_genes : list[str]
                Selected codeword gene ids.
            marker_radius : int
                Transcript marker radius in pixels.
            proseg_run_name : str | None
                Selected Proseg run name.

            Returns
            -------
            ViewerBuildResult
                Prepared global display.
            """

            global_stack = load_global_image_channels(
                self.datastore,
                include_segmentation=include_segmentation,
            )
            stack = global_stack.stack

            if include_cells:
                cell_overlay = global_cell_outline_overlay(
                    self.datastore,
                    shape_zyx=stack.data.shape[1:],
                    origin_zyx_um=global_stack.origin_zyx_um,
                    spacing_zyx_um=global_stack.spacing_zyx_um,
                )
                stack = append_overlay_channel(
                    stack, cell_overlay, "global Cellpose outlines"
                )

            if include_proseg_polygons and proseg_run_name is not None:
                proseg_overlay = rasterize_global_cell_outlines(
                    self.datastore.load_proseg_cell_polygons_3d(
                        run_name=proseg_run_name
                    ),
                    shape_zyx=stack.data.shape[1:],
                    origin_zyx_um=global_stack.origin_zyx_um,
                    spacing_zyx_um=global_stack.spacing_zyx_um,
                )
                stack = append_overlay_channel(
                    stack,
                    proseg_overlay,
                    f"global Proseg polygons {proseg_run_name}",
                )

            context = {
                "mode": "global",
                "base_stack": stack,
                "origin_zyx_um": global_stack.origin_zyx_um,
                "spacing_zyx_um": global_stack.spacing_zyx_um,
            }
            decoded_spots = None
            proseg_transcripts = None
            if include_decoded:
                decoded_spots = self.datastore.load_global_filtered_decoded_spots()
                overlay = (
                    None
                    if decoded_spots is None
                    else rasterize_global_decoded_spots(
                        decoded_spots,
                        shape_zyx=stack.data.shape[1:],
                        origin_zyx_um=global_stack.origin_zyx_um,
                        spacing_zyx_um=global_stack.spacing_zyx_um,
                        genes=selected_genes,
                        radius=marker_radius,
                    )
                )
                stack = append_overlay_channel(
                    stack,
                    overlay,
                    "global datastore codewords",
                )
            elif include_proseg_transcripts and proseg_run_name is not None:
                proseg_transcripts = self.datastore.load_proseg_transcripts_3d(
                    run_name=proseg_run_name
                )
                overlay = rasterize_global_proseg_transcripts(
                    proseg_transcripts,
                    shape_zyx=stack.data.shape[1:],
                    origin_zyx_um=global_stack.origin_zyx_um,
                    spacing_zyx_um=global_stack.spacing_zyx_um,
                    genes=selected_genes,
                    radius=marker_radius,
                )
                stack = append_overlay_channel(
                    stack,
                    overlay,
                    f"global Proseg transcripts: {proseg_run_name}",
                )

            return ViewerBuildResult(
                stack,
                spacing_zyx_um=global_stack.spacing_zyx_um,
                origin_zyx_um=global_stack.origin_zyx_um,
                context=context,
                status="Displayed: " + ", ".join(stack.labels),
                decoded_spots=decoded_spots,
                proseg_run_name=proseg_run_name,
                proseg_transcripts=proseg_transcripts,
            )

        def _build_local_display_result(
            self,
            tile: str,
            fiducial_sources: list[str],
            fiducial_rounds: list[str],
            bit_ids: list[str],
            bit_sources: list[str],
            warp_options: WarpChainOptions,
            gpu_id: int,
            include_decoded: bool,
            include_cells: bool,
            include_proseg_transcripts: bool,
            include_proseg_polygons: bool,
            selected_genes: list[str],
            marker_radius: int,
            proseg_run_name: str | None,
        ) -> ViewerBuildResult:
            """
            Build local display data off the Qt UI thread.

            Parameters
            ----------
            tile : str
                Tile identifier.
            fiducial_sources : list[str]
                Fiducial sources.
            fiducial_rounds : list[str]
                Selected fiducial rounds.
            bit_ids : list[str]
                Selected bit ids.
            bit_sources : list[str]
                Selected bit sources.
            warp_options : WarpChainOptions
                Local warp options.
            gpu_id : int
                CUDA device id.
            include_decoded : bool
                Whether to include datastore decoded transcripts.
            include_cells : bool
                Whether to include Cellpose outlines.
            include_proseg_transcripts : bool
                Whether to include Proseg transcripts.
            include_proseg_polygons : bool
                Whether to include Proseg polygons.
            selected_genes : list[str]
                Selected codeword gene ids.
            marker_radius : int
                Transcript marker radius in pixels.
            proseg_run_name : str | None
                Selected Proseg run name.

            Returns
            -------
            ViewerBuildResult
                Prepared local display.
            """

            display = load_local_warped_image_channels(
                self.datastore,
                tile=tile,
                fiducial_round_ids=fiducial_rounds,
                fiducial_sources=fiducial_sources,
                bit_ids=bit_ids,
                bit_sources=bit_sources,
                options=warp_options,
                gpu_id=gpu_id,
            )
            stack = display.stack
            if include_cells:
                cell_overlay = cell_outline_overlay_for_tile(
                    self.datastore,
                    tile=tile,
                    shape_zyx=stack.data.shape[1:],
                )
                stack = append_overlay_channel(stack, cell_overlay, "Cellpose outlines")

            tile_transform = None
            if include_decoded or include_proseg_transcripts or include_proseg_polygons:
                affine, origin, spacing = self.datastore.load_global_coord_xforms_um(
                    tile=tile
                )
                if affine is not None and origin is not None and spacing is not None:
                    tile_transform = (
                        np.asarray(affine, dtype=float),
                        np.asarray(origin, dtype=float),
                        np.asarray(spacing, dtype=float),
                    )

            if include_proseg_polygons and proseg_run_name is not None:
                proseg_overlay = None
                if tile_transform is not None:
                    affine, origin, spacing = tile_transform
                    proseg_overlay = rasterize_cell_outlines(
                        self.datastore.load_proseg_cell_polygons_3d(
                            run_name=proseg_run_name
                        ),
                        shape_zyx=stack.data.shape[1:],
                        affine_zyx_um=affine,
                        origin_zyx_um=origin,
                        spacing_zyx_um=spacing,
                    )
                stack = append_overlay_channel(
                    stack,
                    proseg_overlay,
                    f"Proseg polygons {proseg_run_name}",
                )

            context = {
                "mode": "local",
                "base_stack": stack,
                "tile": tile,
                "spacing_zyx_um": self.datastore.voxel_size_zyx_um,
                "tile_transform": tile_transform,
                "proseg_tile_transform": tile_transform,
            }
            decoded_spots = None
            proseg_transcripts = None
            if include_decoded:
                decoded_spots = self.datastore.load_global_filtered_decoded_spots()
                overlay = (
                    None
                    if decoded_spots is None or tile_transform is None
                    else rasterize_local_decoded_spots(
                        decoded_spots,
                        shape_zyx=stack.data.shape[1:],
                        affine_zyx_um=tile_transform[0],
                        origin_zyx_um=tile_transform[1],
                        spacing_zyx_um=tile_transform[2],
                        genes=selected_genes,
                        radius=marker_radius,
                    )
                )
                stack = append_overlay_channel(stack, overlay, "datastore codewords")
            elif include_proseg_transcripts and proseg_run_name is not None:
                proseg_transcripts = self.datastore.load_proseg_transcripts_3d(
                    run_name=proseg_run_name
                )
                overlay = None
                if tile_transform is not None:
                    affine, origin, spacing = tile_transform
                    overlay = rasterize_local_proseg_transcripts(
                        proseg_transcripts,
                        shape_zyx=stack.data.shape[1:],
                        affine_zyx_um=affine,
                        origin_zyx_um=origin,
                        spacing_zyx_um=spacing,
                        genes=selected_genes,
                        radius=marker_radius,
                    )
                stack = append_overlay_channel(
                    stack,
                    overlay,
                    f"Proseg transcripts: {proseg_run_name}",
                )

            warning_text = "; ".join(display.warnings)
            status = "Displayed: " + ", ".join(stack.labels)
            if warning_text:
                status += f"\nWarnings: {warning_text}"
            return ViewerBuildResult(
                stack=stack,
                spacing_zyx_um=np.asarray(self.datastore.voxel_size_zyx_um),
                origin_zyx_um=None,
                context=context,
                status=status,
                decoded_spots=decoded_spots,
                proseg_run_name=proseg_run_name,
                proseg_transcripts=proseg_transcripts,
            )

        def display_selection(self) -> None:
            """
            Start display loading for the current viewer selection.

            Returns
            -------
            None
                Computed viewer result.
            """

            if self.datastore is None:
                self.status_label.setText("Select a datastore first.")
                return
            mode = self.view_mode_combo.currentText()
            if mode == "Select view type...":
                self.status_label.setText("Select a view type first.")
                return

            proseg_run = self._selected_proseg_run()
            proseg_run_name = None if proseg_run is None else proseg_run.name
            selected_genes = self._selected_decoded_genes()
            marker_radius = self.marker_radius_spinbox.value()
            include_decoded = self.decoded_checkbox.isChecked()
            include_cells = self.cells_checkbox.isChecked()
            include_proseg_transcripts = self.proseg_transcripts_checkbox.isChecked()
            include_proseg_polygons = self.proseg_polygons_checkbox.isChecked()

            if mode == "Global fused":
                include_segmentation = (
                    self.global_segmentation_checkbox.isChecked()
                    and self.global_segmentation_checkbox.isEnabled()
                )
                self._start_display_worker(
                    partial(
                        self._build_global_display_result,
                        include_segmentation,
                        include_decoded,
                        include_cells,
                        include_proseg_transcripts,
                        include_proseg_polygons,
                        selected_genes,
                        marker_radius,
                        proseg_run_name,
                    ),
                    "Loading global fused data...",
                )
                return

            tile = self.tile_combo.currentText()
            if not tile:
                self.status_label.setText("No tile available.")
                return

            fiducial_sources = self._fiducial_sources()
            fiducial_rounds = self._selected_fiducial_rounds()
            bit_ids = self._checked_bits()
            bit_sources = self._bit_sources()
            warp_options = self._warp_options()
            gpu_id = self.gpu_spinbox.value()
            self._start_display_worker(
                partial(
                    self._build_local_display_result,
                    tile,
                    fiducial_sources,
                    fiducial_rounds,
                    bit_ids,
                    bit_sources,
                    warp_options,
                    gpu_id,
                    include_decoded,
                    include_cells,
                    include_proseg_transcripts,
                    include_proseg_polygons,
                    selected_genes,
                    marker_radius,
                    proseg_run_name,
                ),
                f"Loading {tile}...",
            )

    qt_app = QtWidgets.QApplication.instance()
    if qt_app is None:
        qt_app = QtWidgets.QApplication([])
    qt_app.setQuitOnLastWindowClosed(False)
    window = DatastoreViewerWindow(initial_path)
    window.adjustSize()
    window.show()
    exec_method = getattr(qt_app, "exec", None) or qt_app.exec_
    exec_method()
