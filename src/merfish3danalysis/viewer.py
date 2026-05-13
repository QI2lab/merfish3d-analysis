"""View qi2lab datastores with an ndv/PyQt GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ChannelStack:
    """Channel-stacked image data and labels for display."""

    data: np.ndarray
    labels: list[str]


def stack_with_micron_coords(
    stack: ChannelStack,
    voxel_size_zyx_um: Any,
) -> Any:
    """Attach zyx micron coordinates to a channel stack for ndv display."""

    import xarray as xr

    data = stack.data.astype(np.float32, copy=False)
    voxel = np.asarray(voxel_size_zyx_um, dtype=np.float32)
    if data.ndim != 4 or voxel.shape[0] != 3:
        raise ValueError("Expected channel stack shape (c, z, y, x).")

    return xr.DataArray(
        data,
        dims=("c", "z_um", "y_um", "x_um"),
        coords={
            "c": np.arange(data.shape[0]),
            "z_um": np.arange(data.shape[1], dtype=np.float32) * voxel[0],
            "y_um": np.arange(data.shape[2], dtype=np.float32) * voxel[1],
            "x_um": np.arange(data.shape[3], dtype=np.float32) * voxel[2],
        },
        attrs={"z_spacing_um": float(voxel[0])},
    )


def empty_micron_stack() -> Any:
    """Return an empty micron-coordinate stack for initializing ndv axes."""

    return stack_with_micron_coords(
        ChannelStack(
            data=np.zeros((1, 1, 2, 2), dtype=np.float32),
            labels=["empty"],
        ),
        (1.0, 1.0, 1.0),
    )


def normalize_datastore_path(path: Path) -> Path:
    """Resolve an experiment root or direct datastore path to a datastore path."""

    expanded = path.expanduser().resolve()
    direct_state_path = expanded / "datastore_state.json"
    if direct_state_path.exists():
        return expanded

    nested = expanded / "qi2labdatastore"
    nested_state_path = nested / "datastore_state.json"
    if nested_state_path.exists():
        return nested

    raise FileNotFoundError(
        "Could not find qi2lab datastore. Select an experiment root containing "
        "'qi2labdatastore' or select the datastore directory directly."
    )


def open_datastore(datastore_path: Path) -> Any:
    """Open a qi2lab datastore without expensive validation."""

    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    return qi2labDataStore(datastore_path, validate=False)


def component_summary(datastore: Any) -> dict[str, bool]:
    """Return datastore component availability from existing datastore state."""

    state = datastore.datastore_state or {}
    return {
        "Calibrations": bool(state.get("Calibrations", False)),
        "Corrected": bool(state.get("Corrected", False)),
        "LocalRegistered": bool(state.get("LocalRegistered", False)),
        "GlobalRegistered": bool(state.get("GlobalRegistered", False)),
        "Fused": bool(state.get("Fused", False)),
        "SegmentedCells": bool(state.get("SegmentedCells", False)),
        "DecodedSpots": bool(state.get("DecodedSpots", False)),
        "FilteredSpots": bool(state.get("FilteredSpots", False)),
    }


def _datastore_path(datastore: Any) -> Path | None:
    """Return datastore path from the existing datastore object if available."""

    path = getattr(datastore, "_datastore_path", None)
    return Path(path) if path is not None else None


def decoded_available(datastore: Any) -> bool:
    """Return whether decoded spots are available without requiring fresh state flags."""

    state = component_summary(datastore)
    if state["DecodedSpots"] or state["FilteredSpots"]:
        return True

    datastore_path = _datastore_path(datastore)
    if datastore_path is None:
        return False

    filtered_path = (
        datastore_path
        / "all_tiles_filtered_decoded_features"
        / "decoded_features.parquet"
    )
    if filtered_path.exists():
        return True

    decoded_dir = datastore_path / "decoded"
    return decoded_dir.exists() and any(
        decoded_dir.glob("tile*_decoded_features.parquet")
    )


def cell_outlines_available(datastore: Any) -> bool:
    """Return whether cell outlines are available without requiring fresh state flags."""

    if component_summary(datastore)["SegmentedCells"]:
        return True

    datastore_path = _datastore_path(datastore)
    if datastore_path is None:
        return False

    return (
        datastore_path
        / "segmentation"
        / "cellpose"
        / "imagej_rois"
        / "global_coords_rois.zip"
    ).exists()


def codebook_gene_bits(datastore: Any) -> dict[str, list[str]]:
    """Map codebook genes to existing datastore bit IDs."""

    parsed = datastore.load_codebook_parsed()
    if parsed is None:
        return {}

    gene_ids, codebook_matrix = parsed
    bit_ids = list(datastore.bit_ids or [])
    codebook_array = np.asarray(codebook_matrix)
    gene_to_bits: dict[str, list[str]] = {}
    for gene_id, row in zip(gene_ids, codebook_array, strict=False):
        selected_bits = [
            bit_ids[bit_idx]
            for bit_idx, value in enumerate(np.asarray(row).astype(bool))
            if value and bit_idx < len(bit_ids)
        ]
        gene_to_bits[str(gene_id)] = selected_bits

    return gene_to_bits


def _as_zyx(image: Any) -> np.ndarray:
    """Convert a loaded image to a 3D zyx NumPy array."""

    array = np.asarray(image)
    array = np.squeeze(array)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    if array.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D image, got shape {array.shape}.")
    return array


def _append_channel(
    channels: list[np.ndarray],
    labels: list[str],
    image: Any,
    label: str,
) -> None:
    """Append one image channel if it loaded successfully."""

    if image is None:
        return
    channels.append(_as_zyx(image))
    labels.append(label)


def load_image_channels(
    datastore: Any,
    tile: str,
    fiducial_sources: list[str],
    bit_ids: list[str],
    bit_sources: list[str],
) -> ChannelStack:
    """Load selected datastore images through existing datastore methods."""

    channels: list[np.ndarray] = []
    labels: list[str] = []
    round_ids = list(datastore.round_ids or [])
    round_id = round_ids[0] if round_ids else None

    if round_id is not None:
        if "corrected" in fiducial_sources:
            _append_channel(
                channels,
                labels,
                datastore.load_local_corrected_image(
                    tile=tile, round=round_id, return_future=False
                ),
                f"{tile}:{round_id}:fiducial corrected",
            )
        if "registered" in fiducial_sources:
            _append_channel(
                channels,
                labels,
                datastore.load_local_registered_image(
                    tile=tile, round=round_id, return_future=False
                ),
                f"{tile}:{round_id}:fiducial registered/decon",
            )

    for bit_id in bit_ids:
        if "corrected" in bit_sources:
            _append_channel(
                channels,
                labels,
                datastore.load_local_corrected_image(
                    tile=tile, bit=bit_id, return_future=False
                ),
                f"{tile}:{bit_id}:corrected",
            )
        if "registered" in bit_sources:
            _append_channel(
                channels,
                labels,
                datastore.load_local_registered_image(
                    tile=tile, bit=bit_id, return_future=False
                ),
                f"{tile}:{bit_id}:registered/decon",
            )
        if "feature" in bit_sources:
            _append_channel(
                channels,
                labels,
                datastore.load_local_feature_predictor_image(
                    tile=tile, bit=bit_id, return_future=False
                ),
                f"{tile}:{bit_id}:feature predictor",
            )

    if not channels:
        raise ValueError("No selected image channels were available to display.")

    shape = channels[0].shape
    if any(channel.shape != shape for channel in channels):
        raise ValueError("Selected image channels do not have matching shapes.")

    return ChannelStack(data=np.stack(channels, axis=0), labels=labels)


def selected_image_channel_count(
    fiducial_sources: list[str],
    bit_ids: list[str],
    bit_sources: list[str],
    has_fiducial_round: bool,
) -> int:
    """Return the number of image arrays selected for loading."""

    fiducial_count = len(fiducial_sources) if has_fiducial_round else 0
    return fiducial_count + len(bit_ids) * len(bit_sources)


def unavailable_data_message(error: ValueError) -> str:
    """Return a user-facing message for unavailable viewer data."""

    return f"Data not available: {error}"


def _paint_point(volume: np.ndarray, zyx: np.ndarray, radius: int) -> None:
    """Paint one point into a volume."""

    z, y, x = np.round(zyx).astype(int)
    z_min = max(0, z - radius)
    z_max = min(volume.shape[0], z + radius + 1)
    y_min = max(0, y - radius)
    y_max = min(volume.shape[1], y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(volume.shape[2], x + radius + 1)
    if z_min < z_max and y_min < y_max and x_min < x_max:
        volume[z_min:z_max, y_min:y_max, x_min:x_max] = 1.0


def rasterize_decoded_spots(
    decoded_spots: Any,
    shape_zyx: tuple[int, int, int],
    genes: list[str] | None = None,
    radius: int = 1,
) -> np.ndarray:
    """Rasterize decoded spots with tile pixel coordinates into an overlay."""

    overlay = np.zeros(shape_zyx, dtype=np.float32)
    if decoded_spots is None or len(decoded_spots) == 0:
        return overlay

    required_columns = {"tile_z", "tile_y", "tile_x", "gene_id"}
    if not required_columns.issubset(decoded_spots.columns):
        return overlay

    spots = decoded_spots
    if genes:
        genes_set = {gene.strip() for gene in genes if gene.strip()}
        if genes_set:
            spots = spots.loc[spots["gene_id"].astype(str).isin(genes_set)]

    coords = spots[["tile_z", "tile_y", "tile_x"]].to_numpy(dtype=float, copy=False)
    for coord in coords:
        _paint_point(overlay, coord, radius)

    return overlay


def decoded_overlay_for_tile(
    datastore: Any,
    tile: str,
    shape_zyx: tuple[int, int, int],
    genes: list[str] | None = None,
) -> np.ndarray | None:
    """Load and rasterize decoded spots for one tile using existing datastore APIs."""

    tile_ids = list(datastore.tile_ids or [])
    tile_idx = tile_ids.index(tile) if tile in tile_ids else None

    datastore_path = _datastore_path(datastore)
    filtered_path = None
    if datastore_path is not None:
        filtered_path = (
            datastore_path
            / "all_tiles_filtered_decoded_features"
            / "decoded_features.parquet"
        )

    if component_summary(datastore)["FilteredSpots"] or (
        filtered_path is not None and filtered_path.exists()
    ):
        decoded_spots = datastore.load_global_filtered_decoded_spots()
        if decoded_spots is None:
            return None
        if tile_idx is not None and "tile_idx" in decoded_spots.columns:
            decoded_spots = decoded_spots.loc[decoded_spots["tile_idx"] == tile_idx]
        return rasterize_decoded_spots(decoded_spots, shape_zyx, genes=genes)

    if decoded_available(datastore):
        decoded_spots = datastore.load_local_decoded_spots(tile=tile)
        if decoded_spots is None:
            return None
        return rasterize_decoded_spots(decoded_spots, shape_zyx, genes=genes)

    return None


def _draw_line_2d(image: np.ndarray, start_yx: np.ndarray, end_yx: np.ndarray) -> None:
    """Draw a line into a 2D image using integer interpolation."""

    y_min = min(start_yx[0], end_yx[0])
    y_max = max(start_yx[0], end_yx[0])
    x_min = min(start_yx[1], end_yx[1])
    x_max = max(start_yx[1], end_yx[1])
    if (
        y_max < 0
        or y_min >= image.shape[0]
        or x_max < 0
        or x_min >= image.shape[1]
    ):
        return

    y0, x0 = np.round(start_yx).astype(int)
    y1, x1 = np.round(end_yx).astype(int)
    steps = int(max(abs(y1 - y0), abs(x1 - x0))) + 1
    if steps <= 0:
        return
    ys = np.round(np.linspace(y0, y1, steps)).astype(int)
    xs = np.round(np.linspace(x0, x1, steps)).astype(int)
    valid = (ys >= 0) & (ys < image.shape[0]) & (xs >= 0) & (xs < image.shape[1])
    image[ys[valid], xs[valid]] = 1.0


def _global_xy_to_tile_yx(
    global_xy: np.ndarray,
    affine_zyx_um: np.ndarray,
    origin_zyx_um: np.ndarray,
    spacing_zyx_um: np.ndarray,
) -> np.ndarray:
    """Transform global xy outline coordinates into local tile yx pixels."""

    inverse_affine = np.linalg.inv(np.asarray(affine_zyx_um, dtype=float))
    output = np.zeros((global_xy.shape[0], 2), dtype=float)
    for idx, xy in enumerate(global_xy):
        global_zyx = np.asarray([0.0, xy[1], xy[0], 1.0], dtype=float)
        physical_zyx = (inverse_affine @ global_zyx)[:3]
        pixel_zyx = (physical_zyx - origin_zyx_um) / spacing_zyx_um
        output[idx, :] = pixel_zyx[1:]
    return output


def rasterize_cell_outlines(
    outlines: dict[Any, np.ndarray] | None,
    shape_zyx: tuple[int, int, int],
    affine_zyx_um: np.ndarray,
    origin_zyx_um: np.ndarray,
    spacing_zyx_um: np.ndarray,
) -> np.ndarray:
    """Rasterize global Cellpose outlines into a selected local tile volume."""

    overlay_2d = np.zeros(shape_zyx[1:], dtype=np.float32)
    if not outlines:
        return np.zeros(shape_zyx, dtype=np.float32)

    for outline in outlines.values():
        global_xy = np.asarray(outline, dtype=float)
        if global_xy.ndim != 2 or global_xy.shape[0] < 2 or global_xy.shape[1] != 2:
            continue
        local_yx = _global_xy_to_tile_yx(
            global_xy,
            affine_zyx_um=np.asarray(affine_zyx_um, dtype=float),
            origin_zyx_um=np.asarray(origin_zyx_um, dtype=float),
            spacing_zyx_um=np.asarray(spacing_zyx_um, dtype=float),
        )
        if (
            local_yx[:, 0].max() < 0
            or local_yx[:, 0].min() >= overlay_2d.shape[0]
            or local_yx[:, 1].max() < 0
            or local_yx[:, 1].min() >= overlay_2d.shape[1]
        ):
            continue
        for idx in range(local_yx.shape[0]):
            _draw_line_2d(overlay_2d, local_yx[idx - 1], local_yx[idx])

    return np.repeat(overlay_2d[np.newaxis, :, :], shape_zyx[0], axis=0)


def _load_global_cellpose_roi_zip(datastore: Any) -> dict[int, np.ndarray] | None:
    """Load existing global Cellpose ROI zip when the JSON loader is unavailable."""

    datastore_path = _datastore_path(datastore)
    if datastore_path is None:
        return None

    roi_path = (
        datastore_path
        / "segmentation"
        / "cellpose"
        / "imagej_rois"
        / "global_coords_rois.zip"
    )
    if not roi_path.exists():
        return None

    try:
        from roifile import roiread
    except ImportError:
        return None

    outlines: dict[int, np.ndarray] = {}
    try:
        for idx, roi in enumerate(roiread(roi_path)):
            coordinates = getattr(roi, "subpixel_coordinates", None)
            if coordinates is None:
                coordinates = roi.coordinates()
            if coordinates is None:
                continue
            outlines[idx + 1] = np.asarray(coordinates, dtype=float)
    except (OSError, ValueError):
        return None

    return outlines


def cell_outline_overlay_for_tile(
    datastore: Any,
    tile: str,
    shape_zyx: tuple[int, int, int],
) -> np.ndarray | None:
    """Load and rasterize Cellpose outlines using existing datastore APIs."""

    if not cell_outlines_available(datastore):
        return None

    outlines = _load_global_cellpose_roi_zip(datastore)
    if not outlines:
        outlines = datastore.load_global_cellpose_outlines()
    if outlines is None:
        return None

    affine, origin, spacing = datastore.load_global_coord_xforms_um(tile=tile)
    if affine is None or origin is None or spacing is None:
        return None

    return rasterize_cell_outlines(
        outlines,
        shape_zyx=shape_zyx,
        affine_zyx_um=np.asarray(affine, dtype=float),
        origin_zyx_um=np.asarray(origin, dtype=float),
        spacing_zyx_um=np.asarray(spacing, dtype=float),
    )


def append_overlay_channel(
    stack: ChannelStack,
    overlay: np.ndarray | None,
    label: str,
) -> ChannelStack:
    """Append one overlay channel to an existing channel stack."""

    if overlay is None:
        return stack
    overlay_zyx = _as_zyx(overlay)
    if overlay_zyx.shape != stack.data.shape[1:]:
        raise ValueError("Overlay shape does not match selected image channels.")
    return ChannelStack(
        data=np.concatenate([stack.data, overlay_zyx[np.newaxis, :, :, :]], axis=0),
        labels=[*stack.labels, label],
    )


def apply_lut_channel_labels(array_viewer: Any, labels: list[str]) -> int:
    """Apply human-readable labels and stable fiducial colors to ndv LUT views."""

    controllers = getattr(array_viewer, "_lut_controllers", {})
    applied = 0
    for key, controller in controllers.items():
        if not isinstance(key, int) or key < 0 or key >= len(labels):
            continue
        if ":fiducial " in labels[key]:
            lut_model = getattr(controller, "lut_model", None)
            if lut_model is not None:
                lut_model.cmap = "gray"
        for lut_view in getattr(controller, "lut_views", []):
            set_channel_name = getattr(lut_view, "set_channel_name", None)
            if set_channel_name is None:
                continue
            set_channel_name(labels[key])
            applied += 1
    return applied


class Qi2labViewer:
    """View-only ndv/PyQt GUI for qi2lab datastores."""

    def __init__(self, initial_path: Path | None = None) -> None:
        self.initial_path = initial_path

    def run(self) -> None:
        """Launch the viewer."""

        run_viewer(self.initial_path)


def run_viewer(initial_path: Path | None = None) -> None:
    """Launch the view-only ndv/PyQt datastore viewer."""

    try:
        import ndv
        from qtpy import QtCore, QtWidgets
    except ImportError as exc:
        raise RuntimeError(
            "The qi2lab viewer requires GUI dependencies. Run setup-merfish3d "
            "without --headless to install ndv and Qt support."
        ) from exc

    if hasattr(ndv, "set_gui_backend"):
        ndv.set_gui_backend("qt")
    if hasattr(ndv, "set_canvas_backend"):
        ndv.set_canvas_backend("vispy")

    class DatastoreViewerWindow(QtWidgets.QMainWindow):
        """Small view-only Qt wrapper around ndv.ArrayViewer."""

        def __init__(self, path: Path | None = None) -> None:
            super().__init__()
            self.setWindowTitle("qi2lab datastore viewer")
            self.datastore: Any | None = None
            self.datastore_path: Path | None = None
            self.gene_to_bits: dict[str, list[str]] = {}
            self.channel_labels: list[str] = []
            self.array_viewer: Any | None = None
            self.viewer_layout: Any | None = None
            self._build_ui()
            self._reset_array_viewer(empty_micron_stack())
            if path is not None:
                self.load_path(path)

        def _build_ui(self) -> None:
            central = QtWidgets.QWidget()
            root_layout = QtWidgets.QHBoxLayout(central)
            control_panel = QtWidgets.QWidget()
            control_layout = QtWidgets.QVBoxLayout(control_panel)

            self.path_label = QtWidgets.QLabel("No datastore selected")
            self.path_label.setWordWrap(True)
            open_button = QtWidgets.QPushButton("Open datastore")
            open_button.clicked.connect(self.open_directory)
            control_layout.addWidget(self.path_label)
            control_layout.addWidget(open_button)

            self.component_label = QtWidgets.QLabel("Components: none")
            self.component_label.setWordWrap(True)
            control_layout.addWidget(self.component_label)

            control_layout.addWidget(QtWidgets.QLabel("Tile"))
            self.tile_combo = QtWidgets.QComboBox()
            control_layout.addWidget(self.tile_combo)

            control_layout.addWidget(QtWidgets.QLabel("Fiducial round 1"))
            self.fiducial_corrected = QtWidgets.QCheckBox("corrected")
            self.fiducial_registered = QtWidgets.QCheckBox("registered/decon")
            self.fiducial_registered.setChecked(True)
            control_layout.addWidget(self.fiducial_corrected)
            control_layout.addWidget(self.fiducial_registered)

            control_layout.addWidget(QtWidgets.QLabel("Readout bit sources"))
            self.bit_corrected = QtWidgets.QCheckBox("corrected")
            self.bit_registered = QtWidgets.QCheckBox("registered/decon")
            self.bit_feature = QtWidgets.QCheckBox("feature predictor")
            self.bit_feature.setChecked(True)
            control_layout.addWidget(self.bit_corrected)
            control_layout.addWidget(self.bit_registered)
            control_layout.addWidget(self.bit_feature)

            control_layout.addWidget(QtWidgets.QLabel("Bits"))
            self.bit_list = QtWidgets.QListWidget()
            selection_mode = (getattr(
                    QtWidgets.QAbstractItemView,
                    "SelectionMode",
                    QtWidgets.QAbstractItemView,
                )).NoSelection
            self.bit_list.setSelectionMode(selection_mode)
            control_layout.addWidget(self.bit_list)

            self.decoded_checkbox = QtWidgets.QCheckBox("decoded codebook words")
            self.gene_combo = QtWidgets.QComboBox()
            self.gene_combo.setEditable(False)
            self.gene_combo.activated.connect(self._on_gene_selected)
            self.cells_checkbox = QtWidgets.QCheckBox("cell outlines")
            control_layout.addWidget(self.decoded_checkbox)
            control_layout.addWidget(QtWidgets.QLabel("Decoded gene"))
            control_layout.addWidget(self.gene_combo)
            control_layout.addWidget(self.cells_checkbox)

            self.display_button = QtWidgets.QPushButton("Display")
            self.display_button.clicked.connect(self.display_selection)
            control_layout.addWidget(self.display_button)

            self.status_label = QtWidgets.QLabel("")
            self.status_label.setWordWrap(True)
            control_layout.addWidget(self.status_label)
            self.progress_bar = QtWidgets.QProgressBar()
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setVisible(False)
            control_layout.addWidget(self.progress_bar)
            control_layout.addStretch()

            self.viewer_panel = QtWidgets.QWidget()
            self.viewer_layout = QtWidgets.QVBoxLayout(self.viewer_panel)
            self.viewer_layout.setContentsMargins(0, 0, 0, 0)

            root_layout.addWidget(control_panel, stretch=0)
            root_layout.addWidget(self.viewer_panel, stretch=1)
            self.setCentralWidget(central)

        def _reset_array_viewer(self, data: Any) -> None:
            """Replace ndv's viewer to avoid stale axis/channel state."""

            if self.viewer_layout is not None:
                while self.viewer_layout.count():
                    item = self.viewer_layout.takeAt(0)
                    widget = item.widget()
                    if widget is not None:
                        widget.setParent(None)
                        widget.deleteLater()

            self.array_viewer = ndv.ArrayViewer(
                data,
                channel_axis="c",
                channel_mode="composite",
                visible_axes=("y_um", "x_um"),
                current_index={"z_um": int(data.sizes["z_um"]) // 2},
            )
            if self.viewer_layout is not None:
                self.viewer_layout.addWidget(self.array_viewer.widget())

        def open_directory(self) -> None:
            selected = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Select experiment root or qi2labdatastore",
                str(Path.home()),
            )
            if selected:
                self.load_path(Path(selected))

        def load_path(self, path: Path) -> None:
            try:
                datastore_path = normalize_datastore_path(path)
                self.datastore = open_datastore(datastore_path)
                self.datastore_path = datastore_path
            except Exception as exc:
                self.status_label.setText(str(exc))
                return

            self.path_label.setText(str(datastore_path))
            self._populate_controls()
            self.status_label.setText("Datastore loaded.")

        def _populate_controls(self) -> None:
            if self.datastore is None:
                return

            self.tile_combo.clear()
            for tile_id in self.datastore.tile_ids or []:
                self.tile_combo.addItem(str(tile_id))

            self.bit_list.clear()
            self.gene_to_bits = codebook_gene_bits(self.datastore)
            self.gene_combo.setUpdatesEnabled(False)
            self.gene_combo.blockSignals(True)
            self.gene_combo.clear()
            self.gene_combo.addItem("All decoded genes")
            for gene_id in sorted(self.gene_to_bits):
                self.gene_combo.addItem(gene_id)
            self.gene_combo.blockSignals(False)
            self.gene_combo.setUpdatesEnabled(True)

            item_flag = getattr(QtCore.Qt, "ItemFlag", QtCore.Qt).ItemIsUserCheckable
            checked_state = getattr(QtCore.Qt, "CheckState", QtCore.Qt).Checked
            unchecked_state = getattr(QtCore.Qt, "CheckState", QtCore.Qt).Unchecked
            for idx, bit_id in enumerate(self.datastore.bit_ids or []):
                item = QtWidgets.QListWidgetItem(str(bit_id))
                item.setFlags(item.flags() | item_flag)
                item.setCheckState(checked_state if idx == 0 else unchecked_state)
                self.bit_list.addItem(item)

            state = component_summary(self.datastore)
            enabled = [name for name, value in state.items() if value]
            stale_components: list[str] = []
            if decoded_available(self.datastore) and not (
                state["DecodedSpots"] or state["FilteredSpots"]
            ):
                stale_components.append("DecodedSpots")
            if cell_outlines_available(self.datastore) and not state["SegmentedCells"]:
                stale_components.append("SegmentedCells")
            self.component_label.setText(
                "Components: "
                + (", ".join(enabled) if enabled else "none")
                + (
                    "\nDetected despite stale state: " + ", ".join(stale_components)
                    if stale_components
                    else ""
                )
            )
            self.decoded_checkbox.setEnabled(decoded_available(self.datastore))
            self.cells_checkbox.setEnabled(cell_outlines_available(self.datastore))

        def _on_gene_selected(self, _index: int) -> None:
            gene_id = self.gene_combo.currentText().strip()
            if gene_id not in self.gene_to_bits:
                return

            bit_set = set(self.gene_to_bits[gene_id])
            checked_state = getattr(QtCore.Qt, "CheckState", QtCore.Qt).Checked
            unchecked_state = getattr(QtCore.Qt, "CheckState", QtCore.Qt).Unchecked
            self.bit_list.blockSignals(True)
            for row in range(self.bit_list.count()):
                item = self.bit_list.item(row)
                item.setCheckState(
                    checked_state if item.text() in bit_set else unchecked_state
                )
            self.bit_list.blockSignals(False)
            self.decoded_checkbox.setChecked(True)

        def _checked_bits(self) -> list[str]:
            checked: list[str] = []
            checked_state = getattr(QtCore.Qt, "CheckState", QtCore.Qt).Checked
            for row in range(self.bit_list.count()):
                item = self.bit_list.item(row)
                if item.checkState() == checked_state:
                    checked.append(item.text())
            return checked

        def _fiducial_sources(self) -> list[str]:
            sources: list[str] = []
            if self.fiducial_corrected.isChecked():
                sources.append("corrected")
            if self.fiducial_registered.isChecked():
                sources.append("registered")
            return sources

        def _bit_sources(self) -> list[str]:
            sources: list[str] = []
            if self.bit_corrected.isChecked():
                sources.append("corrected")
            if self.bit_registered.isChecked():
                sources.append("registered")
            if self.bit_feature.isChecked():
                sources.append("feature")
            return sources

        def _selected_decoded_genes(self) -> list[str]:
            gene_id = self.gene_combo.currentText().strip()
            if gene_id and gene_id != "All decoded genes":
                return [gene_id]
            return []

        def _apply_lut_names(self, labels: list[str]) -> None:
            if self.array_viewer is None:
                return
            apply_lut_channel_labels(self.array_viewer, labels)

        def _set_loading(self, is_loading: bool, message: str) -> None:
            self.display_button.setEnabled(not is_loading)
            self.progress_bar.setVisible(is_loading)
            self.status_label.setText(message)
            QtWidgets.QApplication.processEvents()

        def _start_progress(self, total_steps: int, message: str) -> None:
            self.display_button.setEnabled(False)
            self.progress_bar.setRange(0, max(total_steps, 1))
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self.status_label.setText(message)
            QtWidgets.QApplication.processEvents()

        def _advance_progress(self, step: int, message: str) -> None:
            self.progress_bar.setValue(step)
            self.status_label.setText(message)
            QtWidgets.QApplication.processEvents()

        def _finish_progress(self, message: str) -> None:
            self.display_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_label.setText(message)
            QtWidgets.QApplication.processEvents()

        def _load_image_channels_with_progress(
            self,
            tile: str,
            fiducial_sources: list[str],
            bit_ids: list[str],
            bit_sources: list[str],
            step: int,
        ) -> tuple[ChannelStack, int]:
            channels: list[np.ndarray] = []
            labels: list[str] = []
            round_ids = list(self.datastore.round_ids or [])
            round_id = round_ids[0] if round_ids else None

            def append_loaded(image: Any, label: str) -> None:
                nonlocal step
                step += 1
                if image is not None:
                    channels.append(_as_zyx(image))
                    labels.append(label)
                self._advance_progress(step, f"Loaded {label}")

            if round_id is not None:
                if "corrected" in fiducial_sources:
                    label = f"{tile}:{round_id}:fiducial corrected"
                    append_loaded(
                        self.datastore.load_local_corrected_image(
                            tile=tile, round=round_id, return_future=False
                        ),
                        label,
                    )
                if "registered" in fiducial_sources:
                    label = f"{tile}:{round_id}:fiducial registered/decon"
                    append_loaded(
                        self.datastore.load_local_registered_image(
                            tile=tile, round=round_id, return_future=False
                        ),
                        label,
                    )

            for bit_id in bit_ids:
                if "corrected" in bit_sources:
                    label = f"{tile}:{bit_id}:corrected"
                    append_loaded(
                        self.datastore.load_local_corrected_image(
                            tile=tile, bit=bit_id, return_future=False
                        ),
                        label,
                    )
                if "registered" in bit_sources:
                    label = f"{tile}:{bit_id}:registered/decon"
                    append_loaded(
                        self.datastore.load_local_registered_image(
                            tile=tile, bit=bit_id, return_future=False
                        ),
                        label,
                    )
                if "feature" in bit_sources:
                    label = f"{tile}:{bit_id}:feature predictor"
                    append_loaded(
                        self.datastore.load_local_feature_predictor_image(
                            tile=tile, bit=bit_id, return_future=False
                        ),
                        label,
                    )

            if not channels:
                raise ValueError("No selected image channels were available to display.")

            shape = channels[0].shape
            if any(channel.shape != shape for channel in channels):
                raise ValueError("Selected image channels do not have matching shapes.")

            return ChannelStack(data=np.stack(channels, axis=0), labels=labels), step

        def display_selection(self) -> None:
            if self.datastore is None:
                self.status_label.setText("Select a datastore first.")
                return
            tile = self.tile_combo.currentText()
            if not tile:
                self.status_label.setText("No tile available.")
                return

            try:
                fiducial_sources = self._fiducial_sources()
                bit_ids = self._checked_bits()
                bit_sources = self._bit_sources()
                total_steps = selected_image_channel_count(
                    fiducial_sources=fiducial_sources,
                    bit_ids=bit_ids,
                    bit_sources=bit_sources,
                    has_fiducial_round=bool(self.datastore.round_ids),
                )
                if self.decoded_checkbox.isChecked():
                    total_steps += 1
                if self.cells_checkbox.isChecked():
                    total_steps += 1
                total_steps += 1

                self._start_progress(total_steps, f"Loading {tile}...")
                step = 0
                stack, step = self._load_image_channels_with_progress(
                    tile=tile,
                    fiducial_sources=fiducial_sources,
                    bit_ids=bit_ids,
                    bit_sources=bit_sources,
                    step=step,
                )
                if self.decoded_checkbox.isChecked():
                    selected_genes = self._selected_decoded_genes()
                    decoded_overlay = decoded_overlay_for_tile(
                        self.datastore,
                        tile=tile,
                        shape_zyx=stack.data.shape[1:],
                        genes=selected_genes,
                    )
                    decoded_label = (
                        "decoded " + ", ".join(selected_genes)
                        if selected_genes
                        else "decoded codebook words"
                    )
                    stack = append_overlay_channel(
                        stack, decoded_overlay, decoded_label
                    )
                    step += 1
                    self._advance_progress(step, f"Loaded {decoded_label}")
                if self.cells_checkbox.isChecked():
                    cell_overlay = cell_outline_overlay_for_tile(
                        self.datastore,
                        tile=tile,
                        shape_zyx=stack.data.shape[1:],
                    )
                    stack = append_overlay_channel(stack, cell_overlay, "cell outlines")
                    step += 1
                    self._advance_progress(step, "Loaded cell outlines")

                self.channel_labels = stack.labels
                self._reset_array_viewer(
                    stack_with_micron_coords(stack, self.datastore.voxel_size_zyx_um)
                )
                step += 1
                self._advance_progress(step, "Updated viewer")
                self._apply_lut_names(self.channel_labels)
                QtCore.QTimer.singleShot(
                    50, lambda labels=stack.labels: self._apply_lut_names(labels)
                )
                QtCore.QTimer.singleShot(
                    250, lambda labels=stack.labels: self._apply_lut_names(labels)
                )
                self.status_label.setText("Displayed: " + ", ".join(stack.labels))
            except ValueError as exc:
                self.status_label.setText(unavailable_data_message(exc))
            except Exception as exc:
                self.status_label.setText(str(exc))
            finally:
                self._finish_progress(self.status_label.text())

    qt_app = QtWidgets.QApplication.instance()
    if qt_app is None:
        qt_app = QtWidgets.QApplication([])
    window = DatastoreViewerWindow(initial_path)
    window.resize(1400, 900)
    window.show()
    exec_method = getattr(qt_app, "exec", None) or qt_app.exec_
    exec_method()
