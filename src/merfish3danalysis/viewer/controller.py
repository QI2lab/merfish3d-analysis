"""Qt controller window for the datastore viewer."""

from functools import partial
from pathlib import Path
from typing import Any

from qtpy import QtCore, QtGui, QtWidgets

from merfish3danalysis.viewer.colors import transcript_color_hex
from merfish3danalysis.viewer.datastore import (
    ViewerDatastoreLoadResult,
    ViewerDatastoreOptions,
    load_datastore_for_viewer,
)
from merfish3danalysis.viewer.display import ViewerDisplayModel
from merfish3danalysis.viewer.models import (
    DisplayContext,
    GlobalDisplayRequest,
    LocalDisplayRequest,
    SparseOverlayPayload,
    TranscriptRefreshRequest,
    ViewerBuildResult,
    ViewerState,
    WarpChainOptions,
)
from merfish3danalysis.viewer.view import DatastoreNdvView
from merfish3danalysis.viewer.workers import WorkerCoordinator


class DatastoreViewerWindow(QtWidgets.QMainWindow):
    """Qt controller for datastore selection and display options."""

    def __init__(self, path: Path | None = None) -> None:
        """
        Initialize the datastore viewer controller.

        Parameters
        ----------
        path : Path | None
            Optional experiment root or datastore path to open.
        """
        super().__init__()
        self.setWindowTitle("qi2lab datastore viewer")
        self.display_model = ViewerDisplayModel()
        self.view = DatastoreNdvView()
        self.datastore_options: ViewerDatastoreOptions | None = None
        self.transcript_gene_to_bits: dict[str, list[str]] = {}
        self._control_rows: dict[Any, Any] = {}
        self._sparse_payload = SparseOverlayPayload()
        self._workers = WorkerCoordinator(self)
        self._display_refresh_enabled = False
        self._display_refresh_pending = False
        self._transcript_refresh_pending = False
        self._closing = False
        self._transcript_refresh_context: DisplayContext | None = None
        self._busy_enabled_states: dict[QtWidgets.QWidget, bool] = {}
        self._build_ui()
        self._connect_worker_signals()
        if path is not None:
            self.load_path(path)

    def _connect_worker_signals(self) -> None:
        """Connect background-worker events to controller slots."""
        connection_type = QtCore.Qt.ConnectionType.QueuedConnection
        self._workers.started.connect(
            self._start_progress,
            type=connection_type,
        )
        self._workers.finished.connect(
            self._finish_display_worker,
            type=connection_type,
        )
        self._workers.failed.connect(
            self._fail_display_worker,
            type=connection_type,
        )

    def _build_ui(self) -> None:
        """Build the compact controller UI."""
        control_layout = self._build_root_layout()
        self._build_datastore_controls(control_layout)
        self._build_mode_controls(control_layout)
        selection_mode = self._no_selection_mode()
        self._build_local_controls(control_layout, selection_mode)
        self._build_transcript_controls(control_layout, selection_mode)
        self._build_boundary_controls(control_layout)
        self._build_action_controls(control_layout)
        self._update_sparse_dimension_button_style()
        self._update_view_options()

    def _build_root_layout(self) -> QtWidgets.QVBoxLayout:
        """
        Create the window root and return the controller layout.

        Returns
        -------
        QtWidgets.QVBoxLayout
            Layout used for controller widgets.
        """
        central = QtWidgets.QWidget()
        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(4)
        self.control_panel = QtWidgets.QWidget()
        self.control_panel.setObjectName("viewerControlPanel")
        self.control_panel.setMaximumWidth(360)
        control_layout = QtWidgets.QVBoxLayout(self.control_panel)
        control_layout.setContentsMargins(6, 4, 6, 4)
        control_layout.setSpacing(3)
        root_layout.addWidget(self.control_panel)
        self.setCentralWidget(central)
        return control_layout

    def _build_datastore_controls(self, control_layout: QtWidgets.QVBoxLayout) -> None:
        """
        Create datastore path and component controls.

        Parameters
        ----------
        control_layout : QtWidgets.QVBoxLayout
            Controller widget layout.
        """
        self.path_label = QtWidgets.QLabel("No datastore selected")
        self.path_label.setWordWrap(True)
        self.open_button = QtWidgets.QPushButton("Open datastore")
        self.open_button.clicked.connect(self.open_directory)
        control_layout.addWidget(self.path_label)
        control_layout.addWidget(self.open_button)

        self.component_label = QtWidgets.QLabel("Components: none")
        self.component_label.setWordWrap(True)
        control_layout.addWidget(self.component_label)

    def _build_mode_controls(self, control_layout: QtWidgets.QVBoxLayout) -> None:
        """
        Create view mode and warp controls.

        Parameters
        ----------
        control_layout : QtWidgets.QVBoxLayout
            Controller widget layout.
        """
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
        self.view_mode_row = self._add_control_row(
            control_layout, self.view_mode_label, self.view_mode_combo
        )

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
        self._add_control_row(
            control_layout, self.warp_preset_label, self.warp_preset_combo
        )
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
        self._add_control_row(control_layout, self.gpu_label, self.gpu_spinbox)

    def _build_local_controls(
        self,
        control_layout: QtWidgets.QVBoxLayout,
        selection_mode: Any,
    ) -> None:
        """
        Create local tile, fiducial, and bit controls.

        Parameters
        ----------
        control_layout : QtWidgets.QVBoxLayout
            Controller widget layout.
        selection_mode : Any
            Qt item selection mode.
        """
        self.tile_label = QtWidgets.QLabel("Tile")
        self.tile_combo = QtWidgets.QComboBox()
        self._add_control_row(control_layout, self.tile_label, self.tile_combo)

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

    def _build_transcript_controls(
        self,
        control_layout: QtWidgets.QVBoxLayout,
        selection_mode: Any,
    ) -> None:
        """
        Create transcript source, filter, and marker controls.

        Parameters
        ----------
        control_layout : QtWidgets.QVBoxLayout
            Controller widget layout.
        selection_mode : Any
            Qt item selection mode.
        """
        self._create_image_boundary_widgets()
        self._build_transcript_source_controls(control_layout)
        self._build_transcript_filter_controls(control_layout, selection_mode)
        self._build_sparse_dimension_controls(control_layout)

    def _create_image_boundary_widgets(self) -> None:
        """Create global image and cell-boundary checkboxes."""
        self.global_fused_image_checkbox = QtWidgets.QCheckBox("Fused polyDT image")
        self.global_fused_image_checkbox.toggled.connect(self._request_display_refresh)
        self.cellpose_boundaries_checkbox = QtWidgets.QCheckBox(
            "Cellpose cell boundaries"
        )
        self.cellpose_boundaries_checkbox.toggled.connect(self._request_display_refresh)
        self.cellpose_boundaries_checkbox.toggled.connect(
            self._on_geometry_overlay_toggled
        )
        self.global_segmentation_checkbox = QtWidgets.QCheckBox("Cellpose mask")
        self.global_segmentation_checkbox.toggled.connect(self._request_display_refresh)
        self.proseg_boundaries_checkbox = QtWidgets.QCheckBox("Proseg cell boundaries")
        self.proseg_boundaries_checkbox.toggled.connect(self._request_display_refresh)
        self.proseg_boundaries_checkbox.toggled.connect(
            self._on_geometry_overlay_toggled
        )
        self.baysor_boundaries_checkbox = QtWidgets.QCheckBox("Baysor cell boundaries")
        self.baysor_boundaries_checkbox.toggled.connect(self._request_display_refresh)
        self.baysor_boundaries_checkbox.toggled.connect(
            self._on_geometry_overlay_toggled
        )

    def _build_transcript_source_controls(
        self, control_layout: QtWidgets.QVBoxLayout
    ) -> None:
        """
        Create mutually exclusive transcript source controls.

        Parameters
        ----------
        control_layout : QtWidgets.QVBoxLayout
            Controller widget layout.
        """
        self.datastore_transcripts_checkbox = QtWidgets.QCheckBox(
            "Datastore transcripts"
        )
        self.datastore_transcripts_checkbox.toggled.connect(
            self._on_datastore_transcripts_toggled
        )
        self.proseg_run_combo = QtWidgets.QComboBox()
        self.proseg_run_combo.currentTextChanged.connect(self._on_proseg_run_changed)
        self.proseg_transcripts_checkbox = QtWidgets.QCheckBox("Proseg transcripts")
        self.proseg_transcripts_checkbox.toggled.connect(
            self._on_proseg_transcripts_toggled
        )
        self.baysor_transcripts_checkbox = QtWidgets.QCheckBox("Baysor transcripts")
        self.baysor_transcripts_checkbox.toggled.connect(
            self._on_baysor_transcripts_toggled
        )
        control_layout.addWidget(self.datastore_transcripts_checkbox)
        self.proseg_run_label = QtWidgets.QLabel("Proseg run")
        self._add_control_row(
            control_layout, self.proseg_run_label, self.proseg_run_combo
        )
        control_layout.addWidget(self.proseg_transcripts_checkbox)
        control_layout.addWidget(self.baysor_transcripts_checkbox)

    def _build_transcript_filter_controls(
        self,
        control_layout: QtWidgets.QVBoxLayout,
        selection_mode: Any,
    ) -> None:
        """
        Create transcript filter list and apply controls.

        Parameters
        ----------
        control_layout : QtWidgets.QVBoxLayout
            Controller widget layout.
        selection_mode : Any
            Qt item selection mode.
        """
        self.transcript_filter_label = QtWidgets.QLabel("Transcript filter")
        control_layout.addWidget(self.transcript_filter_label)
        self.transcript_button_panel = QtWidgets.QWidget()
        transcript_button_layout = QtWidgets.QHBoxLayout(self.transcript_button_panel)
        transcript_button_layout.setContentsMargins(0, 0, 0, 0)
        transcript_button_layout.setSpacing(3)
        self.select_all_transcripts_button = QtWidgets.QPushButton("Select all")
        self.deselect_all_transcripts_button = QtWidgets.QPushButton("Deselect all")
        self.apply_transcripts_button = QtWidgets.QPushButton("Apply")
        self.select_all_transcripts_button.clicked.connect(self._select_all_transcripts)
        self.deselect_all_transcripts_button.clicked.connect(
            self._deselect_all_transcripts
        )
        self.apply_transcripts_button.clicked.connect(self._apply_transcript_changes)
        self.apply_transcripts_button.setEnabled(False)
        transcript_button_layout.addWidget(self.select_all_transcripts_button)
        transcript_button_layout.addWidget(self.deselect_all_transcripts_button)
        transcript_button_layout.addWidget(self.apply_transcripts_button)
        control_layout.addWidget(self.transcript_button_panel)
        self.marker_radius_label = QtWidgets.QLabel("Marker radius")
        self.marker_radius_spinbox = QtWidgets.QSpinBox()
        self.marker_radius_spinbox.setMinimum(0)
        self.marker_radius_spinbox.setMaximum(20)
        self.marker_radius_spinbox.setValue(10)
        self.marker_radius_spinbox.valueChanged.connect(
            self._mark_transcript_changes_pending
        )
        self._add_control_row(
            control_layout, self.marker_radius_label, self.marker_radius_spinbox
        )
        self.transcript_gene_list = QtWidgets.QListWidget()
        self.transcript_gene_list.setSelectionMode(selection_mode)
        self.transcript_gene_list.itemChanged.connect(self._on_transcript_gene_checked)
        self.transcript_gene_list.setMaximumHeight(170)

    def _build_sparse_dimension_controls(
        self, control_layout: QtWidgets.QVBoxLayout
    ) -> None:
        """
        Create 2D, max projection, and 3D sparse geometry controls.

        Parameters
        ----------
        control_layout : QtWidgets.QVBoxLayout
            Controller widget layout.
        """
        self.sparse_dimension_panel = QtWidgets.QWidget()
        sparse_dimension_layout = QtWidgets.QHBoxLayout(self.sparse_dimension_panel)
        sparse_dimension_layout.setContentsMargins(0, 0, 0, 0)
        sparse_dimension_layout.setSpacing(3)
        self.sparse_2d_button = QtWidgets.QPushButton("2D")
        self.sparse_max_button = QtWidgets.QPushButton("Max")
        self.sparse_3d_button = QtWidgets.QPushButton("3D")
        self.sparse_2d_button.setCheckable(True)
        self.sparse_max_button.setCheckable(True)
        self.sparse_3d_button.setCheckable(True)
        self.sparse_dimension_group = QtWidgets.QButtonGroup(self)
        self.sparse_dimension_group.setExclusive(True)
        self.sparse_dimension_group.addButton(self.sparse_2d_button)
        self.sparse_dimension_group.addButton(self.sparse_max_button)
        self.sparse_dimension_group.addButton(self.sparse_3d_button)
        self.sparse_2d_button.setChecked(True)
        self.sparse_2d_button.toggled.connect(self._on_sparse_dimension_toggled)
        self.sparse_max_button.toggled.connect(self._on_sparse_dimension_toggled)
        self.sparse_3d_button.toggled.connect(self._on_sparse_dimension_toggled)
        sparse_dimension_layout.addWidget(self.sparse_2d_button)
        sparse_dimension_layout.addWidget(self.sparse_max_button)
        sparse_dimension_layout.addWidget(self.sparse_3d_button)
        control_layout.addWidget(self.sparse_dimension_panel)
        control_layout.addWidget(self.transcript_gene_list)

    def _build_boundary_controls(self, control_layout: QtWidgets.QVBoxLayout) -> None:
        """
        Create image and cell-boundary controls.

        Parameters
        ----------
        control_layout : QtWidgets.QVBoxLayout
            Controller widget layout.
        """
        control_layout.addWidget(self.global_fused_image_checkbox)
        control_layout.addWidget(self.cellpose_boundaries_checkbox)
        control_layout.addWidget(self.global_segmentation_checkbox)
        control_layout.addWidget(self.proseg_boundaries_checkbox)
        control_layout.addWidget(self.baysor_boundaries_checkbox)

    def _build_action_controls(self, control_layout: QtWidgets.QVBoxLayout) -> None:
        """
        Create display, status, and progress controls.

        Parameters
        ----------
        control_layout : QtWidgets.QVBoxLayout
            Controller widget layout.
        """
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

    def _no_selection_mode(self) -> Any:
        """Return the Qt no-selection enum across Qt versions."""
        item_view = getattr(
            QtWidgets.QAbstractItemView,
            "SelectionMode",
            QtWidgets.QAbstractItemView,
        )
        return item_view.NoSelection

    def _add_control_row(
        self,
        control_layout: QtWidgets.QVBoxLayout,
        label: QtWidgets.QLabel,
        widget: QtWidgets.QWidget,
    ) -> QtWidgets.QWidget:
        """
        Add a compact label-widget row to the controller.

        Parameters
        ----------
        control_layout : QtWidgets.QVBoxLayout
            Controller widget layout.
        label : QtWidgets.QLabel
            Row label widget.
        widget : QtWidgets.QWidget
            Row input widget.

        Returns
        -------
        QtWidgets.QWidget
            Row container widget.
        """
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
        return row

    def _on_view_mode_changed(self, _mode: str) -> None:
        """
        Rebuild visible option controls for the selected view mode.

        Parameters
        ----------
        _mode : str
            Selected view mode.

        """
        self._populate_selected_options()
        self._update_view_options()

    def _update_view_options(self) -> None:
        """Update visible controls for the selected view mode."""
        self._enable_view_mode_selection()
        mode = self.view_mode_combo.currentText()
        local_mode = mode in {"Local native", "Local warped"}
        warped_mode = mode == "Local warped"
        global_mode = mode == "Global fused"
        mode_selected = local_mode or global_mode

        self._set_mode_style(mode)
        self._set_warp_option_states(warped_mode)
        self._set_local_option_states(local_mode)
        self._set_proseg_option_states(mode_selected)
        transcript_source_selected = (
            self.datastore_transcripts_checkbox.isChecked()
            or self.proseg_transcripts_checkbox.isChecked()
            or self.baysor_transcripts_checkbox.isChecked()
        )
        sparse_geometry_selected = (
            transcript_source_selected
            or self.cellpose_boundaries_checkbox.isChecked()
            or self.proseg_boundaries_checkbox.isChecked()
            or self.baysor_boundaries_checkbox.isChecked()
        )
        self._set_global_option_states(global_mode)
        self._set_transcript_option_states(mode_selected, transcript_source_selected)
        self._set_boundary_option_states(mode_selected, sparse_geometry_selected)
        self._set_option_state(self.display_button, mode_selected)
        self.display_button.setEnabled(mode_selected)
        self._enable_view_mode_selection()

    def _set_mode_style(self, mode: str) -> None:
        """
        Style the controller edge for the active view mode.

        Parameters
        ----------
        mode : str
            Current view mode.
        """
        style = {
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
        self.control_panel.setStyleSheet(style)

    def _set_warp_option_states(self, visible: bool) -> None:
        """
        Show local warp controls only for warped local views.

        Parameters
        ----------
        visible : bool
            Whether warp controls should be visible.
        """
        for widget in (
            self.warp_preset_label,
            self.warp_preset_combo,
            self.chromatic_checkbox,
            self.stage_affine_checkbox,
            self.sofima_checkbox,
            self.gpu_label,
            self.gpu_spinbox,
        ):
            self._set_option_state(widget, visible, visible)

    def _set_local_option_states(self, visible: bool) -> None:
        """
        Show local image controls for local views.

        Parameters
        ----------
        visible : bool
            Whether local controls should be visible.
        """
        for widget in (
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
        ):
            self._set_option_state(widget, visible, visible)

    def _set_proseg_option_states(self, mode_selected: bool) -> None:
        """
        Show Proseg controls when a datastore has Proseg outputs.

        Parameters
        ----------
        mode_selected : bool
            Whether a display mode is selected.
        """
        proseg_runs = (
            () if self.datastore_options is None else self.datastore_options.proseg_runs
        )
        visible = mode_selected and bool(proseg_runs)
        for widget in (
            self.proseg_run_label,
            self.proseg_run_combo,
            self.proseg_transcripts_checkbox,
            self.proseg_boundaries_checkbox,
        ):
            self._set_option_state(widget, visible, visible)

    def _set_global_option_states(self, global_mode: bool) -> None:
        """
        Show global image controls for global views.

        Parameters
        ----------
        global_mode : bool
            Whether global fused mode is active.
        """
        self._set_option_state(
            self.global_segmentation_checkbox, global_mode, global_mode
        )
        self._set_option_state(
            self.global_fused_image_checkbox, global_mode, global_mode
        )

    def _set_transcript_option_states(
        self,
        mode_selected: bool,
        transcript_source_selected: bool,
    ) -> None:
        """
        Show transcript controls for the selected transcript source.

        Parameters
        ----------
        mode_selected : bool
            Whether a display mode is selected.
        transcript_source_selected : bool
            Whether a transcript source is selected.
        """
        has_baysor = self._baysor_available()
        self._set_option_state(
            self.datastore_transcripts_checkbox, mode_selected, mode_selected
        )
        self._set_option_state(
            self.baysor_transcripts_checkbox,
            mode_selected and has_baysor,
            has_baysor,
        )
        for widget in (
            self.transcript_filter_label,
            self.transcript_button_panel,
            self.marker_radius_label,
            self.marker_radius_spinbox,
            self.transcript_gene_list,
        ):
            enabled = mode_selected and transcript_source_selected
            self._set_option_state(
                widget,
                enabled,
                enabled,
            )

    def _set_boundary_option_states(
        self,
        mode_selected: bool,
        sparse_geometry_selected: bool,
    ) -> None:
        """
        Show cell-boundary and sparse-dimension controls.

        Parameters
        ----------
        mode_selected : bool
            Whether a display mode is selected.
        sparse_geometry_selected : bool
            Whether sparse overlays are selected.
        """
        has_baysor = self._baysor_available()
        self._set_option_state(
            self.sparse_dimension_panel,
            mode_selected,
            mode_selected,
        )
        if mode_selected:
            self._enable_sparse_dimension_selection()
        self._set_option_state(
            self.cellpose_boundaries_checkbox, mode_selected, mode_selected
        )
        self._set_option_state(
            self.baysor_boundaries_checkbox,
            mode_selected and has_baysor,
            has_baysor,
        )

    def _set_option_state(
        self,
        widget: QtWidgets.QWidget,
        visible: bool,
        enabled: bool | None = None,
    ) -> None:
        """
        Set visibility and enabled state for an option row.

        Parameters
        ----------
        widget : QtWidgets.QWidget
            Widget or row member.
        visible : bool
            Whether the option row is visible.
        enabled : bool or None
            Optional enabled state.
        """
        target = self._control_rows.get(widget, widget)
        target.setVisible(visible)
        if enabled is not None:
            target.setEnabled(enabled)
            widget.setEnabled(enabled)
            for child in target.findChildren(QtWidgets.QWidget):
                child.setEnabled(enabled)

    def open_directory(self) -> None:
        """Open a directory picker and load the selected datastore."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Open qi2lab datastore",
            str(Path.home()),
        )
        if directory:
            self.load_path(Path(directory))

    def load_path(self, path: Path) -> None:
        """
        Load a datastore and refresh controller options.

        Parameters
        ----------
        path : Path
            Experiment root or datastore path.
        """
        if self._workers.is_running:
            self.status_label.setText("Wait for the current load to finish.")
            return
        self._start_display_worker(
            partial(load_datastore_for_viewer, path),
            "Loading datastore...",
        )

    def _finish_datastore_load(self, result: ViewerDatastoreLoadResult) -> None:
        """
        Install a loaded datastore and refresh controller options.

        Parameters
        ----------
        result : ViewerDatastoreLoadResult
            Loaded datastore and selector metadata.
        """
        self.display_model.set_datastore(result.datastore)
        self.datastore_options = result.options
        self.transcript_gene_to_bits = result.options.transcript_gene_to_bits
        self.path_label.setText(str(result.datastore_path))
        self.component_label.setText(
            "Components: " + self._component_summary_text(result.options.components)
        )
        self._populate_static_options()
        self._populate_selected_options()
        self._display_refresh_enabled = False
        self._update_view_options()

    def _component_summary_text(self, components: dict[str, bool]) -> str:
        """
        Return compact component availability text.

        Parameters
        ----------
        components : dict[str, bool]
            Component availability flags.

        Returns
        -------
        str
            Comma-separated component names.
        """
        names = [name for name, available in components.items() if available]
        if self._baysor_available():
            names.append("Baysor3D")
        return ", ".join(names) if names else "none"

    def _baysor_available(self) -> bool:
        """Return whether the loaded datastore has Baysor 3D outputs."""
        return bool(
            self.datastore_options is not None
            and self.datastore_options.baysor_available
        )

    def _populate_static_options(self) -> None:
        """Populate datastore-dependent selector widgets."""
        if self.datastore_options is None:
            return
        blockers = (
            QtCore.QSignalBlocker(self.tile_combo),
            QtCore.QSignalBlocker(self.fiducial_round_list),
            QtCore.QSignalBlocker(self.bit_list),
            QtCore.QSignalBlocker(self.proseg_run_combo),
        )
        self.tile_combo.clear()
        self.tile_combo.addItems(self.datastore_options.tile_ids)
        self.fiducial_round_list.clear()
        self.fiducial_round_list.addItems(self.datastore_options.round_ids)
        self.bit_list.clear()
        self.bit_list.addItems(self.datastore_options.bit_ids)
        for list_widget in (self.fiducial_round_list, self.bit_list):
            for row in range(list_widget.count()):
                item = list_widget.item(row)
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.proseg_run_combo.clear()
        self.proseg_run_combo.addItems(self.datastore_options.proseg_runs)
        del blockers

    def _populate_selected_options(self) -> None:
        """Populate controls once the user selects a view type."""
        if self.view_mode_combo.currentText() == "Select view type...":
            return
        fiducial_blocker = QtCore.QSignalBlocker(self.fiducial_round_list)
        bit_blocker = QtCore.QSignalBlocker(self.bit_list)
        if self.fiducial_round_list.count() > 0:
            if not self._checked_items(self.fiducial_round_list):
                self.fiducial_round_list.item(0).setCheckState(
                    QtCore.Qt.CheckState.Checked
                )
        if self.bit_list.count() > 0 and not self._checked_items(self.bit_list):
            self.bit_list.item(0).setCheckState(QtCore.Qt.CheckState.Checked)
        del fiducial_blocker, bit_blocker
        if self.transcript_gene_list.count() == 0:
            self._rebuild_transcript_gene_list(())

    def _set_transcript_source(self, source: str | None) -> None:
        """
        Select one transcript source and clear the other sources.

        Parameters
        ----------
        source : str or None
            Transcript source name to select.
        """
        pairs = (
            ("datastore", self.datastore_transcripts_checkbox),
            ("proseg", self.proseg_transcripts_checkbox),
            ("baysor", self.baysor_transcripts_checkbox),
        )
        blockers = tuple(QtCore.QSignalBlocker(checkbox) for _name, checkbox in pairs)
        for name, checkbox in pairs:
            checkbox.setChecked(source == name)
        del blockers
        self._rebuild_transcript_gene_list(self._selected_transcript_genes())
        self._mark_transcript_changes_pending()
        self._update_view_options()

    def _on_datastore_transcripts_toggled(self, checked: bool) -> None:
        """
        Handle datastore transcript source changes.

        Parameters
        ----------
        checked : bool
            Whether the datastore source is checked.
        """
        self._set_transcript_source("datastore" if checked else None)

    def _on_proseg_transcripts_toggled(self, checked: bool) -> None:
        """
        Handle Proseg transcript source changes.

        Parameters
        ----------
        checked : bool
            Whether the Proseg source is checked.
        """
        self._set_transcript_source("proseg" if checked else None)

    def _on_baysor_transcripts_toggled(self, checked: bool) -> None:
        """
        Handle Baysor transcript source changes.

        Parameters
        ----------
        checked : bool
            Whether the Baysor source is checked.
        """
        self._set_transcript_source("baysor" if checked else None)

    def _on_proseg_run_changed(self, _run_name: str) -> None:
        """
        Refresh display state when the selected Proseg run changes.

        Parameters
        ----------
        _run_name : str
            Selected Proseg run name.
        """
        self._mark_transcript_changes_pending()
        self._request_display_refresh()

    def _on_geometry_overlay_toggled(self, _checked: bool) -> None:
        """
        Update geometry controls after a boundary overlay toggle.

        Parameters
        ----------
        _checked : bool
            Whether the toggled overlay is checked.
        """
        self._update_view_options()

    def _on_transcript_gene_checked(self, _item: Any) -> None:
        """
        Mark transcript display settings as pending.

        Parameters
        ----------
        _item : Any
            Changed transcript list item.
        """
        self._mark_transcript_changes_pending()

    def _select_all_transcripts(self) -> None:
        """Check every selectable transcript in the filter list."""
        self._set_all_transcripts(QtCore.Qt.CheckState.Checked)

    def _deselect_all_transcripts(self) -> None:
        """Uncheck every selectable transcript in the filter list."""
        self._set_all_transcripts(QtCore.Qt.CheckState.Unchecked)

    def _set_all_transcripts(self, state: Any) -> None:
        """
        Set the check state on all transcript items.

        Parameters
        ----------
        state : Any
            Qt check state.
        """
        blocker = QtCore.QSignalBlocker(self.transcript_gene_list)
        for row in range(self.transcript_gene_list.count()):
            item = self.transcript_gene_list.item(row)
            if item.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(state)
        del blocker
        self._mark_transcript_changes_pending()

    def _apply_transcript_changes(self) -> None:
        """Apply pending transcript selection and marker-size changes."""
        selected_genes = self._selected_transcript_genes()
        self._rebuild_transcript_gene_list(selected_genes)
        self.apply_transcripts_button.setEnabled(False)
        self._request_transcript_refresh()

    def _mark_transcript_changes_pending(self, *_args: Any) -> None:
        """
        Enable the transcript apply button.

        Parameters
        ----------
        *_args : Any
            Optional Qt signal arguments.
        """
        if self._current_transcript_source() is not None:
            self.apply_transcripts_button.setEnabled(True)

    def _current_transcript_source(self) -> str | None:
        """Return the selected transcript source."""
        if self.datastore_transcripts_checkbox.isChecked():
            return "datastore"
        if self.proseg_transcripts_checkbox.isChecked():
            return "proseg"
        if self.baysor_transcripts_checkbox.isChecked():
            return "baysor"
        return None

    def _rebuild_transcript_gene_list(self, selected_genes: tuple[str, ...]) -> None:
        """
        Rebuild transcript list with selected genes at the top.

        Parameters
        ----------
        selected_genes : tuple[str, ...]
            Genes that should remain selected.
        """
        genes = sorted(self.transcript_gene_to_bits)
        available = set(genes)
        ordered = list(
            dict.fromkeys(gene for gene in selected_genes if gene in available)
        )
        selected = set(ordered)
        remaining = [gene for gene in genes if gene not in selected]
        blocker = QtCore.QSignalBlocker(self.transcript_gene_list)
        self.transcript_gene_list.clear()
        selected_count = len(ordered)
        for index, gene in enumerate(ordered):
            self._add_transcript_gene_item(
                gene,
                checked=True,
                color_hex=transcript_color_hex(index + 1, selected_count),
            )
        if ordered and remaining:
            separator = QtWidgets.QListWidgetItem("Unselected transcripts")
            separator.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            separator.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
            self.transcript_gene_list.addItem(separator)
        for gene in remaining:
            self._add_transcript_gene_item(gene, checked=False, color_hex=None)
        del blocker

    def _add_transcript_gene_item(
        self,
        gene: str,
        checked: bool,
        color_hex: str | None,
    ) -> None:
        """
        Add one transcript gene row with a stable color tooltip.

        Parameters
        ----------
        gene : str
            Transcript gene name.
        checked : bool
            Whether the row starts checked.
        color_hex : str or None
            Optional selected-gene color.
        """
        item = QtWidgets.QListWidgetItem(gene)
        item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(
            QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked
        )
        if color_hex is not None:
            pixmap = QtGui.QPixmap(14, 14)
            pixmap.fill(QtGui.QColor(color_hex))
            item.setIcon(QtGui.QIcon(pixmap))
            item.setToolTip(color_hex)
        self.transcript_gene_list.addItem(item)

    def _apply_warp_preset(self, preset: str) -> None:
        """
        Apply a named local warp preset.

        Parameters
        ----------
        preset : str
            Preset name from the combo box.
        """
        options = {
            "Decode-equivalent": (True, True, True),
            "Native": (False, False, False),
            "Affine only": (False, True, False),
            "Affine + chromatic": (True, True, False),
            "Full": (True, True, True),
        }.get(preset)
        if options is None:
            return
        chromatic, stage_affine, sofima = options
        self.chromatic_checkbox.setChecked(chromatic)
        self.stage_affine_checkbox.setChecked(stage_affine)
        self.sofima_checkbox.setChecked(sofima)

    def _checked_items(self, list_widget: Any) -> tuple[str, ...]:
        """
        Return checked items from a Qt list widget.

        Parameters
        ----------
        list_widget : Any
            Qt list widget with checkable items.

        Returns
        -------
        tuple[str, ...]
            Checked item text values.
        """
        checked: list[str] = []
        for row in range(list_widget.count()):
            item = list_widget.item(row)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                checked.append(item.text())
        return tuple(checked)

    def _selected_fiducial_rounds(self) -> tuple[str, ...]:
        """Return selected fiducial rounds."""
        return self._checked_items(self.fiducial_round_list)

    def _checked_bits(self) -> tuple[str, ...]:
        """Return selected readout bits."""
        return self._checked_items(self.bit_list)

    def _fiducial_sources(self) -> tuple[str, ...]:
        """Return selected fiducial image sources."""
        sources: list[str] = []
        if self.fiducial_corrected.isChecked():
            sources.append("corrected")
        if self.fiducial_registered.isChecked():
            sources.append("registered")
        return tuple(sources)

    def _bit_sources(self) -> tuple[str, ...]:
        """Return selected bit image sources."""
        sources: list[str] = []
        if self.bit_corrected.isChecked():
            sources.append("corrected")
        if self.bit_registered.isChecked():
            sources.append("registered")
        if self.bit_feature.isChecked():
            sources.append("feature")
        return tuple(sources)

    def _warp_options(self) -> WarpChainOptions:
        """Return selected local warp-chain options."""
        if self.view_mode_combo.currentText() == "Local native":
            return WarpChainOptions(False, False, False)
        return WarpChainOptions(
            chromatic=self.chromatic_checkbox.isChecked(),
            stage_affine=self.stage_affine_checkbox.isChecked(),
            sofima=self.sofima_checkbox.isChecked(),
        )

    def _selected_transcript_genes(self) -> tuple[str, ...]:
        """Return checked transcript genes."""
        genes: list[str] = []
        for row in range(self.transcript_gene_list.count()):
            item = self.transcript_gene_list.item(row)
            if (
                item.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable
                and item.checkState() == QtCore.Qt.CheckState.Checked
            ):
                genes.append(item.text())
        return tuple(genes)

    def _selected_proseg_run(self) -> str | None:
        """Return the selected Proseg run."""
        name = self.proseg_run_combo.currentText()
        return name or None

    def display_selection(self) -> None:
        """Start display loading for the current viewer selection."""
        if self.datastore_options is None:
            self.status_label.setText("Select a datastore first.")
            return
        state = self._viewer_state()
        if state.view_mode == "Select view type...":
            self.status_label.setText("Select a view type first.")
            return
        if state.view_mode == "Global fused":
            request = self._global_display_request(state)
            self._start_display_worker(
                partial(self.display_model.build_global, request),
                "Loading global display...",
            )
            return
        if not state.tile:
            self.status_label.setText("No tile available.")
            return
        request = self._local_display_request(state)
        self._start_display_worker(
            partial(self.display_model.build_local, request),
            f"Loading {request.tile}...",
        )

    def _viewer_state(self) -> ViewerState:
        """Return the current controller selection state."""
        return ViewerState(
            view_mode=self.view_mode_combo.currentText(),
            tile=self.tile_combo.currentText(),
            fiducial_sources=self._fiducial_sources(),
            fiducial_rounds=self._selected_fiducial_rounds(),
            bit_ids=self._checked_bits(),
            bit_sources=self._bit_sources(),
            warp_options=self._warp_options(),
            gpu_id=self.gpu_spinbox.value(),
            include_fused_image=self.global_fused_image_checkbox.isChecked(),
            include_segmentation=self.global_segmentation_checkbox.isChecked(),
            include_cell_boundaries=self.cellpose_boundaries_checkbox.isChecked(),
            include_proseg_boundaries=self.proseg_boundaries_checkbox.isChecked(),
            include_baysor_boundaries=self.baysor_boundaries_checkbox.isChecked(),
            transcript_source=self._current_transcript_source(),
            selected_genes=self._selected_transcript_genes(),
            marker_radius=self.marker_radius_spinbox.value(),
            proseg_run_name=self._selected_proseg_run(),
            max_project=self.sparse_max_button.isChecked(),
        )

    def _global_display_request(self, state: ViewerState) -> GlobalDisplayRequest:
        """
        Return the current global display request.

        Parameters
        ----------
        state : ViewerState
            Current controller state.

        Returns
        -------
        GlobalDisplayRequest
            Global display request.
        """
        return GlobalDisplayRequest(
            include_fused_image=state.include_fused_image,
            include_segmentation=state.include_segmentation,
            include_cell_boundaries=state.include_cell_boundaries,
            include_proseg_boundaries=state.include_proseg_boundaries,
            include_baysor_boundaries=state.include_baysor_boundaries,
            transcript_source=state.transcript_source,
            selected_genes=state.selected_genes,
            marker_radius=state.marker_radius,
            proseg_run_name=state.proseg_run_name,
            max_project=state.max_project,
        )

    def _local_display_request(self, state: ViewerState) -> LocalDisplayRequest:
        """
        Return the current local display request.

        Parameters
        ----------
        state : ViewerState
            Current controller state.

        Returns
        -------
        LocalDisplayRequest
            Local display request.
        """
        return LocalDisplayRequest(
            tile=state.tile,
            fiducial_sources=state.fiducial_sources,
            fiducial_rounds=state.fiducial_rounds,
            bit_ids=state.bit_ids,
            bit_sources=state.bit_sources,
            warp_options=state.warp_options,
            gpu_id=state.gpu_id,
            include_cell_boundaries=state.include_cell_boundaries,
            include_proseg_boundaries=state.include_proseg_boundaries,
            include_baysor_boundaries=state.include_baysor_boundaries,
            transcript_source=state.transcript_source,
            selected_genes=state.selected_genes,
            marker_radius=state.marker_radius,
            proseg_run_name=state.proseg_run_name,
        )

    def _start_display_worker(self, build_display: Any, message: str) -> None:
        """
        Start a background display worker.

        Parameters
        ----------
        build_display : Any
            Callable that builds the display payload.
        message : str
            Status message shown while work runs.
        """
        if self._closing:
            return
        if not self._workers.start(build_display, message):
            self.status_label.setText("Wait for the current load to finish.")

    @QtCore.Slot(str)
    def _start_progress(self, message: str) -> None:
        """
        Show progress UI for a started worker.

        Parameters
        ----------
        message : str
            Status message.
        """
        self.progress_bar.setVisible(True)
        self._set_controls_busy(True)
        self.status_label.setText(message)

    @QtCore.Slot(object)
    def _finish_display_worker(self, result: object) -> None:
        """
        Show a prepared display result.

        Parameters
        ----------
        result : object
            Worker result object.
        """
        if self._closing:
            return
        if not self.isVisible():
            return
        if isinstance(result, ViewerDatastoreLoadResult):
            self._finish_datastore_load(result)
            self._finish_progress("Datastore loaded.")
            self._enable_view_mode_selection()
            return
        if not isinstance(result, ViewerBuildResult):
            self._finish_progress(f"Unexpected worker result: {type(result).__name__}")
            return
        self._transcript_refresh_context = result.context
        self._finish_progress("Opening display...")
        if result.context.refresh_only:
            try:
                self._set_sparse_payload(result.sparse_payload)
            except Exception as exc:
                self.status_label.setText(f"Display failed: {exc}")
                return
            self._restore_controller_interaction()
            self.status_label.setText(result.status)
            return
        try:
            self._show_stack(
                result.stack,
                spacing_zyx_um=result.spacing_zyx_um,
                origin_zyx_um=result.origin_zyx_um,
                sparse_payload=result.sparse_payload,
            )
        except Exception as exc:
            self.status_label.setText(f"Display failed: {exc}")
            return
        self._display_refresh_enabled = True
        self._restore_controller_interaction()
        self.status_label.setText(result.status)

    @QtCore.Slot(str)
    def _fail_display_worker(self, message: str) -> None:
        """
        Finish a failed display worker.

        Parameters
        ----------
        message : str
            Error message.
        """
        if self._closing:
            return
        self._finish_progress(message)

    def _finish_progress(self, message: str) -> None:
        """
        Restore controller controls after a worker completes.

        Parameters
        ----------
        message : str
            Final status message.
        """
        self.progress_bar.setVisible(False)
        self._set_controls_busy(False)
        self.status_label.setText(message)

    def _restore_controller_interaction(self) -> None:
        """Restore usable controller state after NDV or sparse-view setup."""
        self.setEnabled(True)
        self.control_panel.setEnabled(True)
        self._set_controls_busy(False)
        self._update_view_options()

    def _set_controls_busy(self, busy: bool) -> None:
        """
        Disable mutable controller controls while a worker is active.

        Parameters
        ----------
        busy : bool
            Whether controls should be disabled.
        """
        excluded = {
            self.status_label,
            self.progress_bar,
            self.view_mode_row,
            self.view_mode_label,
            self.view_mode_combo,
            self._control_rows.get(self.view_mode_combo),
            self.view_mode_combo.view(),
        }
        if busy:
            if self._busy_enabled_states:
                return
            for widget in self.control_panel.findChildren(QtWidgets.QWidget):
                if widget in excluded:
                    continue
                self._busy_enabled_states[widget] = widget.isEnabled()
                widget.setEnabled(False)
            return

        for widget, was_enabled in self._busy_enabled_states.items():
            widget.setEnabled(was_enabled)
        self._busy_enabled_states.clear()

    def _enable_view_mode_selection(self) -> None:
        """Ensure the view-mode selector is usable after datastore loading."""
        self.view_mode_row.setVisible(True)
        self.view_mode_row.setEnabled(True)
        self.view_mode_label.setVisible(True)
        self.view_mode_label.setEnabled(True)
        self.view_mode_combo.setVisible(True)
        self.view_mode_combo.setEnabled(True)
        self.view_mode_combo.view().setEnabled(True)
        model = self.view_mode_combo.model()
        for row in range(self.view_mode_combo.count()):
            item = model.item(row)
            if item is not None:
                item.setEnabled(True)
                item.setSelectable(True)
            self.view_mode_combo.setItemData(
                row,
                None,
                QtCore.Qt.ItemDataRole.UserRole - 1,
            )

    def _enable_sparse_dimension_selection(self) -> None:
        """Ensure 2D, Max, and 3D sparse mode buttons are usable."""
        self.sparse_dimension_panel.setVisible(True)
        self.sparse_dimension_panel.setEnabled(True)
        for button in (
            self.sparse_2d_button,
            self.sparse_max_button,
            self.sparse_3d_button,
        ):
            button.setVisible(True)
            button.setEnabled(True)

    def _show_stack(
        self,
        stack: Any,
        *,
        spacing_zyx_um: Any,
        origin_zyx_um: Any | None,
        sparse_payload: SparseOverlayPayload | None,
    ) -> None:
        """
        Display an image stack and sparse overlays.

        Parameters
        ----------
        stack : Any
            Image channel stack.
        spacing_zyx_um : Any
            Z, Y, X voxel spacing in microns.
        origin_zyx_um : Any or None
            Optional global origin in microns.
        sparse_payload : SparseOverlayPayload or None
            Sparse overlay payload.
        """
        self.view.show_stack(
            stack,
            spacing_zyx_um=spacing_zyx_um,
            origin_zyx_um=origin_zyx_um,
        )
        self._set_sparse_payload(sparse_payload)

    def _set_sparse_payload(self, payload: SparseOverlayPayload | None) -> None:
        """
        Install sparse geometry on the current NDV canvas.

        Parameters
        ----------
        payload : SparseOverlayPayload or None
            Sparse overlay payload.
        """
        self._sparse_payload = SparseOverlayPayload() if payload is None else payload
        self.view.set_sparse_payload(self._sparse_payload)

    def _request_display_refresh(self, *_args: Any) -> None:
        """
        Refresh the current NDV display after an option change.

        Parameters
        ----------
        *_args : Any
            Optional Qt signal arguments.
        """
        if (
            self._closing
            or not self._display_refresh_enabled
            or self._display_refresh_pending
            or self.progress_bar.isVisible()
        ):
            return
        self._display_refresh_pending = True
        QtCore.QTimer.singleShot(250, self._refresh_current_display)

    def _request_transcript_refresh(self, *_args: Any) -> None:
        """
        Refresh only transcript overlays after transcript display changes.

        Parameters
        ----------
        *_args : Any
            Optional Qt signal arguments.
        """
        if (
            self._closing
            or not self._display_refresh_enabled
            or self._transcript_refresh_pending
            or self.progress_bar.isVisible()
        ):
            return
        self._transcript_refresh_pending = True
        QtCore.QTimer.singleShot(0, self._refresh_transcript_overlay)

    def _refresh_transcript_overlay(self) -> None:
        """Redraw transcript overlays without reloading image channels."""
        self._transcript_refresh_pending = False
        if (
            self._closing
            or not self._display_refresh_enabled
            or self.progress_bar.isVisible()
            or self._transcript_refresh_context is None
        ):
            return
        context = self._transcript_refresh_context
        state = self._viewer_state()
        request = TranscriptRefreshRequest(
            source=state.transcript_source,
            selected_genes=state.selected_genes,
            marker_radius=state.marker_radius,
            proseg_run_name=state.proseg_run_name,
        )
        self._start_display_worker(
            partial(self.display_model.build_transcript_refresh, context, request),
            "Updating transcript overlay...",
        )

    def _refresh_current_display(self) -> None:
        """Redraw the current NDV view after a control change."""
        self._display_refresh_pending = False
        if (
            self._closing
            or not self._display_refresh_enabled
            or self.progress_bar.isVisible()
        ):
            return
        self.display_selection()

    def _update_sparse_dimension_button_style(self) -> None:
        """
        Highlight the active sparse geometry dimension button.

        Returns
        -------
        None
            No return value.
        """
        active_style = (
            "QPushButton {"
            "background-color: #4f7cff;"
            "color: white;"
            "font-weight: 600;"
            "border: 1px solid #2f5edf;"
            "padding: 2px 8px;"
            "}"
        )
        inactive_style = "QPushButton { padding: 2px 8px; }"
        self.sparse_2d_button.setStyleSheet(
            active_style if self.sparse_2d_button.isChecked() else inactive_style
        )
        self.sparse_max_button.setStyleSheet(
            active_style if self.sparse_max_button.isChecked() else inactive_style
        )
        self.sparse_3d_button.setStyleSheet(
            active_style if self.sparse_3d_button.isChecked() else inactive_style
        )

    def _on_sparse_dimension_toggled(self, checked: bool) -> None:
        """
        Switch sparse geometry between current-slice 2D and sparse 3D display.

        Parameters
        ----------
        checked : bool
            Whether the toggled button became active.

        Returns
        -------
        None
            No return value.
        """
        if not checked:
            return
        self._update_sparse_dimension_button_style()
        self.view.set_sparse_mode(
            use_3d=self.sparse_3d_button.isChecked(),
            project_z=self.sparse_max_button.isChecked(),
        )
        if (
            self.view_mode_combo.currentText() == "Global fused"
            and self.global_fused_image_checkbox.isChecked()
            and self._display_refresh_enabled
        ):
            self._request_display_refresh()
            return

    def closeEvent(self, event: Any) -> None:
        """
        Close NDV viewer windows when the controller window closes.

        Parameters
        ----------
        event : Any
            Main window close event.
        """
        self._closing = True
        self.view.close()
        self._display_refresh_enabled = False
        self._display_refresh_pending = False
        self._transcript_refresh_pending = False
        self._workers.cancel()
        super().closeEvent(event)
