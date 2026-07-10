"""Tests for viewer loader and overlay helpers."""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.decode_warping import (
    compose_optional_warp_transform_zyx_um,
)
from merfish3danalysis.viewer.colors import transcript_color_hex
from merfish3danalysis.viewer.display import ViewerDisplayModel
from merfish3danalysis.viewer.models import (
    ChannelStack,
    DisplayContext,
    LazyGlobalChannelData,
    LocalDisplayRequest,
    SparseLineLayer,
    SparseOverlayPayload,
    SparsePointLayer,
    TranscriptRefreshRequest,
    ViewerBuildResult,
    WarpChainOptions,
    warp_chain_label,
)
from merfish3danalysis.viewer.ndv import apply_lut_channel_labels
from merfish3danalysis.viewer.overlays import (
    PointOverlayIndex,
    PolylineGeometry,
    ZPolylineGeometry,
    global_transcript_index,
    local_datastore_transcript_index,
    local_transcript_index,
)
from merfish3danalysis.viewer.sparse import (
    SparseVispyOverlay,
    selected_line_data,
    selected_point_data,
)
from merfish3danalysis.viewer.warping import (
    selected_warp_label,
)


class _FakeLutModel:
    """Minimal LUT model for viewer label tests."""

    def __init__(self) -> None:
        self.cmap: str | None = None
        self.clims: tuple[float, float] | None = None


class _FakeLutView:
    """Minimal LUT view for viewer label tests."""

    def __init__(self) -> None:
        self.channel_name: str | None = None

    def set_channel_name(self, channel_name: str) -> None:
        """Record the assigned channel name."""

        self.channel_name = channel_name


class _FakeLutController:
    """Minimal LUT controller for viewer label tests."""

    def __init__(self) -> None:
        self.lut_model = _FakeLutModel()
        self.lut_views = [_FakeLutView()]


class _FakeArrayViewer:
    """Minimal ArrayViewer object for viewer label tests."""

    def __init__(self, channel_count: int) -> None:
        self._lut_controllers = {
            channel_index: _FakeLutController()
            for channel_index in range(channel_count)
        }
        self.data = np.zeros((channel_count, 1, 2, 2), dtype=np.float32)


class _FakeViewerDisplayDatastore:
    """Minimal datastore object for display-model sparse overlay tests."""

    voxel_size_zyx_um = np.ones(3, dtype=np.float32)

    def __init__(self, boundaries: pd.DataFrame | None = None) -> None:
        self.boundaries = boundaries

    def load_baysor_cell_boundaries_3d(self) -> pd.DataFrame | None:
        """Return fake Baysor boundaries."""

        return self.boundaries


def test_lazy_global_channel_data_supports_list_channel_indexing() -> None:
    """Lazy global image stacks support NumPy-style list channel indexing."""

    fused = np.arange(4, dtype=np.uint16).reshape(1, 2, 2)
    image_channel = fused + 10
    data = LazyGlobalChannelData(
        fused_zyx=fused,
        image_channels=(image_channel,),
        coords={"c": range(2)},
    )

    selected = data[[1, 0], :, :, :]

    assert selected.shape == (2, 1, 2, 2)
    np.testing.assert_array_equal(selected[0], image_channel)
    np.testing.assert_array_equal(selected[1], fused)


def _wkb_polygon_xy(points_xy: np.ndarray) -> bytes:
    """Return little-endian WKB bytes for one polygon exterior ring."""

    import struct

    points = np.asarray(points_xy, dtype=float)
    output = bytearray()
    output.extend(struct.pack("<BII", 1, 3, 1))
    output.extend(struct.pack("<I", points.shape[0]))
    for x_value, y_value in points:
        output.extend(struct.pack("<dd", float(x_value), float(y_value)))
    return bytes(output)


def test_warp_chain_labels_and_optional_affine_composition() -> None:
    """Viewer warp-chain labels and affine toggles reflect selected components."""

    round_transform = np.eye(4, dtype=np.float32)
    round_transform[1, 3] = 2.0
    chromatic_transform = np.eye(4, dtype=np.float32)
    chromatic_transform[2, 3] = 4.0

    affine_only = compose_optional_warp_transform_zyx_um(
        round_transform_zyx_um=round_transform,
        chromatic_transform_zyx_um=None,
    )
    full = compose_optional_warp_transform_zyx_um(
        round_transform_zyx_um=round_transform,
        chromatic_transform_zyx_um=chromatic_transform,
    )

    np.testing.assert_allclose(affine_only, round_transform)
    np.testing.assert_allclose(
        full, np.linalg.inv(chromatic_transform) @ round_transform
    )
    assert warp_chain_label(WarpChainOptions(False, False, False)) == "native"
    assert warp_chain_label(WarpChainOptions(True, True, True)) == (
        "affine+chromatic+sofima"
    )
    assert selected_warp_label("tile0008", "bit003", WarpChainOptions()) == (
        "tile0008:bit003 warped affine+chromatic+sofima"
    )


def test_transcript_color_hex_returns_distinct_hex_colors() -> None:
    """Transcript color key values are hex colors and vary by selected value."""

    first_color = transcript_color_hex(1, 3)
    second_color = transcript_color_hex(2, 3)
    expected_first_color = "#" + "".join(
        f"{round(channel * 255):02x}" for channel in colormaps["turbo"](0.0)[:3]
    )

    assert first_color.startswith("#")
    assert len(first_color) == 7
    assert second_color.startswith("#")
    assert len(second_color) == 7
    assert first_color != second_color
    assert first_color == expected_first_color


def test_apply_lut_channel_labels_sets_image_and_cell_boundary_colors() -> None:
    """Viewer LUT labels apply only to image-backed NDV channels."""

    labels = [
        "global polyDT max projection",
        "global Cellpose mask",
    ]
    array_viewer = _FakeArrayViewer(len(labels))

    applied = apply_lut_channel_labels(array_viewer, labels)

    controllers = array_viewer._lut_controllers
    assert applied == len(labels)
    assert controllers[0].lut_model.cmap == "gray"
    assert controllers[1].lut_model.cmap == "gray"
    assert controllers[0].lut_views[0].channel_name == labels[0]
    assert controllers[1].lut_views[0].channel_name == labels[1]


def test_lazy_max_projection_image_presents_one_z_plane() -> None:
    """Lazy max projection reads selected YX pixels across all Z planes."""

    from merfish3danalysis.viewer.overlays import LazyMaxProjectionImage

    image_zyx = np.asarray(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[6, 5, 4], [3, 2, 1]],
        ],
        dtype=np.uint16,
    )
    projected = LazyMaxProjectionImage(image_zyx)

    assert projected.shape == (1, 2, 3)
    assert projected.dtype == np.dtype(np.uint16)
    np.testing.assert_array_equal(
        projected[0],
        np.asarray([[6, 5, 4], [4, 5, 6]], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        projected[:, :, 1:],
        np.asarray([[[5, 4], [5, 6]]], dtype=np.uint16),
    )


def _write_proseg_run(root: Path) -> None:
    """Write a tiny datastore-like Proseg run."""

    root.mkdir(parents=True)
    transcript_table = "transcript_id,qv,x,y,z,gene\n,inf,2.0,4.0,1.0,GeneA\n"
    with gzip.open(
        root / "transcript_metadata_3D.csv.gz", "wt", encoding="utf-8"
    ) as file:
        file.write(transcript_table)

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"cell": 7},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [[[[1.0, 1.0], [3.0, 1.0], [3.0, 3.0]]]],
                },
            }
        ],
    }
    with gzip.open(
        root / "cell_polygons_3D.geojson.gz", "wt", encoding="utf-8"
    ) as file:
        json.dump(geojson, file)


def test_qi2lab_datastore_lists_default_and_nested_proseg_runs(
    tmp_path: Path,
) -> None:
    """Datastore Proseg discovery finds default and nested 3D run folders."""

    proseg_root = tmp_path / "proseg" / "3D"
    _write_proseg_run(proseg_root)
    _write_proseg_run(proseg_root / "fdr.75")
    datastore = qi2labDataStore.__new__(qi2labDataStore)
    datastore._datastore_path = tmp_path

    runs = datastore.list_proseg_3d_runs()
    polygons = datastore.load_proseg_cell_polygons_3d(run_name=runs[0])

    assert runs == ["default", "fdr.75"]
    assert 7 in polygons
    assert polygons[7].shape == (3, 2)


def test_proseg_transcripts_index_global_and_local_coordinates(tmp_path: Path) -> None:
    """Proseg transcript coordinates index in global and local coordinate frames."""

    run_root = tmp_path / "proseg" / "3D"
    _write_proseg_run(run_root)
    transcripts = pd.read_csv(run_root / "transcript_metadata_3D.csv.gz")
    shape = (4, 8, 8)
    origin = np.zeros(3, dtype=np.float32)
    spacing = np.ones(3, dtype=np.float32)

    global_index = global_transcript_index(
        transcripts,
        shape_zyx=shape,
        origin_zyx_um=origin,
        spacing_zyx_um=spacing,
    )
    local_index = local_transcript_index(
        transcripts,
        shape_zyx=shape,
        affine_zyx_um=np.eye(4, dtype=np.float32),
        origin_zyx_um=origin,
        spacing_zyx_um=spacing,
    )

    np.testing.assert_allclose(global_index.coords_sorted, [[1.0, 4.0, 2.0]])
    np.testing.assert_allclose(local_index.coords_sorted, [[1.0, 4.0, 2.0]])
    assert global_index.genes_sorted.tolist() == ["GeneA"]
    assert local_index.genes_sorted.tolist() == ["GeneA"]


def test_indexed_global_transcripts_reuse_transcript_index(tmp_path: Path) -> None:
    """Transcript coordinate indices are independent of selected genes."""

    run_root = tmp_path / "proseg" / "3D"
    _write_proseg_run(run_root)
    transcripts = pd.read_csv(run_root / "transcript_metadata_3D.csv.gz")
    shape = (4, 8, 8)
    transcript_index = global_transcript_index(
        transcripts,
        shape_zyx=shape,
        origin_zyx_um=np.zeros(3, dtype=np.float32),
        spacing_zyx_um=np.ones(3, dtype=np.float32),
    )

    np.testing.assert_allclose(transcript_index.coords_sorted, [[1.0, 4.0, 2.0]])
    assert transcript_index.genes_sorted.tolist() == ["GeneA"]


def test_sparse_point_data_uses_selected_gene_colors() -> None:
    """Sparse point layers return 2D and 3D marker positions for selected genes."""

    point_index = PointOverlayIndex.from_points(
        np.asarray([[1.0, 4.0, 2.0], [2.0, 5.0, 3.0]], dtype=float),
        np.asarray(["GeneA", "GeneB"], dtype=object),
        shape=(4, 8, 8),
    )
    layer = SparsePointLayer(
        point_index,
        selected_genes=("GeneA",),
        marker_size=10,
        spacing_zyx_um=(2.0, 1.0, 1.0),
        label="points",
    )

    positions_2d, colors_2d, marker_size = selected_point_data(
        layer,
        z_index=1,
        use_3d=False,
    )
    positions_3d, colors_3d, _marker_size = selected_point_data(
        layer,
        z_index=1,
        use_3d=True,
    )

    np.testing.assert_allclose(positions_2d, np.asarray([[2.0, 4.0]]))
    np.testing.assert_allclose(positions_3d, np.asarray([[2.0, 4.0, 2.0]]))
    assert colors_2d.shape == (1, 4)
    assert colors_3d.shape == (1, 4)
    assert marker_size == 10.0


def test_sparse_point_data_projects_selected_genes_across_z() -> None:
    """Sparse point layers can project selected transcript positions across Z."""

    point_index = PointOverlayIndex.from_points(
        np.asarray([[1.0, 4.0, 2.0], [2.0, 5.0, 3.0]], dtype=float),
        np.asarray(["GeneA", "GeneA"], dtype=object),
        shape=(4, 8, 8),
    )
    layer = SparsePointLayer(
        point_index,
        selected_genes=("GeneA",),
        marker_size=10,
        spacing_zyx_um=(2.0, 1.0, 1.0),
        label="points",
    )

    positions, _colors, _marker_size = selected_point_data(
        layer,
        z_index=1,
        use_3d=False,
        project_z=True,
    )

    np.testing.assert_allclose(positions, np.asarray([[2.0, 4.0], [3.0, 5.0]]))


def test_sparse_line_data_handles_2d_and_z_aware_lines() -> None:
    """Sparse line layers return VisPy segment arrays in 2D and 3D."""

    repeated = PolylineGeometry(
        (np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float),),
        shape=(4, 8, 8),
        line_thickness=5,
    )
    z_aware = ZPolylineGeometry(
        ((1, 2, np.asarray([[2.0, 1.0], [4.0, 3.0]], dtype=float)),),
        shape=(4, 8, 8),
        line_thickness=1,
    )

    repeated_layer = SparseLineLayer(
        repeated.polylines_yx,
        shape=repeated.shape,
        width=repeated.line_thickness,
        color="white",
        spacing_zyx_um=(2.0, 1.0, 1.0),
        label="repeated",
    )
    z_layer = SparseLineLayer(
        z_aware.polylines,
        shape=z_aware.shape,
        width=z_aware.line_thickness,
        color="white",
        spacing_zyx_um=(2.0, 1.0, 1.0),
        label="z",
        z_aware=True,
    )

    repeated_positions, _color, width, repeated_connect = selected_line_data(
        repeated_layer,
        z_index=1,
        use_3d=False,
    )
    z_positions_2d, _color, _width, z_connect_2d = selected_line_data(
        z_layer,
        z_index=1,
        use_3d=False,
    )
    z_positions_3d, _color, _width, z_connect_3d = selected_line_data(
        z_layer,
        z_index=1,
        use_3d=True,
    )

    np.testing.assert_allclose(
        repeated_positions,
        np.asarray([[2.0, 1.0], [4.0, 3.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        z_positions_2d,
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        z_positions_3d,
        np.asarray([[1.0, 2.0, 3.0], [3.0, 4.0, 3.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(repeated_connect, np.asarray([[0, 1]]))
    np.testing.assert_array_equal(z_connect_2d, np.asarray([[0, 1]]))
    np.testing.assert_array_equal(z_connect_3d, np.asarray([[0, 1]]))
    assert width == 5


def test_sparse_z_indexed_lines_use_current_z_bin() -> None:
    """Sparse Z-aware line layers use pre-binned 2D contours when available."""

    surface_stack = np.asarray(
        [
            [[1.0, 100.0, 100.0], [1.0, 100.0, 102.0], [1.0, 102.0, 100.0]],
            [[3.0, 100.0, 100.0], [3.0, 100.0, 102.0], [3.0, 102.0, 100.0]],
        ],
        dtype=float,
    )
    visible_ring = np.asarray(
        [[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [0.0, 0.0]],
        dtype=float,
    )
    layer = SparseLineLayer(
        (surface_stack,),
        shape=(5, 8, 8),
        width=1,
        color="white",
        spacing_zyx_um=(1.0, 1.0, 1.0),
        label="z indexed",
        z_aware=True,
        z_polylines_by_index={2: (visible_ring,)},
    )

    positions, _color, _width, connect = selected_line_data(
        layer,
        z_index=2,
        use_3d=False,
    )

    np.testing.assert_allclose(positions, visible_ring[:, [1, 0]])
    np.testing.assert_array_equal(connect, np.asarray([[0, 1], [1, 2], [2, 3]]))


def test_sparse_max_lines_use_projected_boundaries() -> None:
    """Sparse Z-aware line layers can project max boundary outlines."""

    z_ring = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
        dtype=float,
    )
    max_ring = np.asarray(
        [[2.0, 2.0], [2.0, 4.0], [4.0, 4.0], [2.0, 2.0]],
        dtype=float,
    )
    layer = SparseLineLayer(
        (),
        shape=(5, 8, 8),
        width=1,
        color="white",
        spacing_zyx_um=(1.0, 1.0, 1.0),
        label="max",
        z_aware=True,
        z_polylines_by_index={1: (z_ring,)},
        max_polylines_yx=(max_ring,),
    )

    positions, _color, _width, connect = selected_line_data(
        layer,
        z_index=1,
        use_3d=False,
        project_z=True,
    )

    np.testing.assert_allclose(positions, max_ring[:, [1, 0]])
    np.testing.assert_array_equal(connect, np.asarray([[0, 1], [1, 2], [2, 3]]))


def test_transcript_refresh_preserves_sparse_cell_boundaries() -> None:
    """Transcript refreshes keep existing sparse boundary layers."""

    stack = ChannelStack(
        data=np.zeros((1, 1, 4, 4), dtype=np.uint16),
        labels=["global empty canvas"],
    )
    line_layer = SparseLineLayer(
        (np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float),),
        shape=(1, 4, 4),
        width=1,
        color="white",
        spacing_zyx_um=(1.0, 1.0, 1.0),
        label="Baysor cell boundaries",
    )
    context = DisplayContext(
        mode="global",
        base_stack=stack,
        base_sparse_lines=(line_layer,),
        shape_zyx=(1, 4, 4),
        origin_zyx_um=np.zeros(3, dtype=np.float32),
        spacing_zyx_um=np.ones(3, dtype=np.float32),
    )

    result = ViewerDisplayModel().build_transcript_refresh(
        context,
        TranscriptRefreshRequest(
            source=None,
            selected_genes=(),
            marker_radius=10,
            proseg_run_name=None,
        ),
    )

    assert result.sparse_payload is not None
    assert result.sparse_payload.lines == (line_layer,)
    assert result.sparse_payload.points == ()


def test_local_baysor_boundaries_do_not_require_proseg_run() -> None:
    """Local Baysor cell boundaries load even when no Proseg run is selected."""

    boundaries = pd.DataFrame(
        {
            "cell": ["cell1"],
            "layer": ["[0.0, 1.0]"],
            "geometry": [
                _wkb_polygon_xy(
                    np.asarray(
                        [
                            [1.0, 1.0],
                            [3.0, 1.0],
                            [3.0, 3.0],
                            [1.0, 1.0],
                        ],
                        dtype=float,
                    )
                )
            ],
        }
    )
    model = ViewerDisplayModel()
    model.set_datastore(_FakeViewerDisplayDatastore(boundaries))
    request = LocalDisplayRequest(
        tile="tile0000",
        fiducial_sources=(),
        fiducial_rounds=(),
        bit_ids=(),
        bit_sources=(),
        warp_options=WarpChainOptions(),
        gpu_id=0,
        include_cell_boundaries=False,
        include_proseg_boundaries=False,
        include_baysor_boundaries=True,
        transcript_source=None,
        selected_genes=(),
        marker_radius=10,
        proseg_run_name=None,
    )

    payload = model._local_sparse_payload(
        request,
        shape_zyx=(3, 8, 8),
        tile_transform=(
            np.eye(4, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
        ),
    )

    assert [layer.label for layer in payload.lines] == ["Baysor cell boundaries"]


def test_sparse_overlay_recreates_visuals_when_dimension_changes() -> None:
    """Sparse VisPy overlays do not reuse 2D line visuals for 3D data."""

    from vispy import scene

    class _FakeCamera:
        def __init__(self) -> None:
            self.fov = 45.0
            self.depth_value = 1.0

    class _FakeCanvas:
        def update(self) -> None:
            """Accept VisPy canvas update calls."""

    class _FakeCanvasController:
        def __init__(self) -> None:
            self._view = type(
                "View",
                (),
                {"scene": scene.Node(), "camera": _FakeCamera()},
            )()
            self._canvas = _FakeCanvas()
            self.ndims: list[int] = []

        def set_ndim(self, ndim: int) -> None:
            self.ndims.append(ndim)

    class _FakeArrayViewer:
        def __init__(self) -> None:
            self._canvas = _FakeCanvasController()

    payload = SparseOverlayPayload(
        lines=(
            SparseLineLayer(
                (np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float),),
                shape=(4, 8, 8),
                width=1,
                color="white",
                spacing_zyx_um=(2.0, 1.0, 1.0),
                label="line",
            ),
        )
    )
    overlay = SparseVispyOverlay()
    overlay.attach(_FakeArrayViewer())

    overlay.set_payload(payload)
    overlay.update(z_index=1, use_3d=False)
    first_visual = overlay._line_visuals[0]
    overlay.update(z_index=1, use_3d=True)

    assert overlay._visual_ndim == 3
    assert overlay._line_visuals[0] is not first_visual
    assert overlay._array_viewer._canvas._view.camera.fov == 0.0
    assert overlay._array_viewer._canvas._view.camera.depth_value == 1e6


def test_local_datastore_transcripts_index_from_global_coordinates() -> None:
    """Final datastore transcripts index locally from global coordinates."""

    datastore_transcripts = pd.DataFrame(
        {
            "global_z": [1.0],
            "global_y": [4.0],
            "global_x": [2.0],
            "gene_id": ["GeneA"],
            "tile_idx": [999],
            "tile_z": [3.0],
            "tile_y": [6.0],
            "tile_x": [7.0],
        }
    )
    transcript_index = local_datastore_transcript_index(
        datastore_transcripts,
        shape_zyx=(4, 8, 8),
        affine_zyx_um=np.eye(4, dtype=np.float32),
        origin_zyx_um=np.zeros(3, dtype=np.float32),
        spacing_zyx_um=np.ones(3, dtype=np.float32),
    )

    np.testing.assert_allclose(transcript_index.coords_sorted, [[1.0, 4.0, 2.0]])
    assert transcript_index.genes_sorted.tolist() == ["GeneA"]


def test_controller_enables_view_mode_after_datastore_load(tmp_path: Path) -> None:
    """Datastore loading leaves the view-mode selector usable."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy import QtWidgets

    from merfish3danalysis.viewer.controller import DatastoreViewerWindow
    from merfish3danalysis.viewer.datastore import (
        ViewerDatastoreLoadResult,
        ViewerDatastoreOptions,
    )

    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = DatastoreViewerWindow()
    try:
        options = ViewerDatastoreOptions(
            components={},
            tile_ids=("tile0000",),
            round_ids=("round001",),
            bit_ids=("bit001",),
            proseg_runs=(),
            baysor_available=False,
            transcript_gene_to_bits={},
        )
        result = ViewerDatastoreLoadResult(tmp_path, object(), options)

        window._set_controls_busy(True)
        for row in range(window.view_mode_combo.count()):
            window.view_mode_combo.model().item(row).setEnabled(False)
        window.sparse_max_button.setEnabled(False)
        window.sparse_3d_button.setEnabled(False)
        window._finish_datastore_load(result)
        window._finish_progress("Datastore loaded.")
        window._enable_view_mode_selection()

        assert not window.view_mode_combo.isHidden()
        assert window.view_mode_combo.isEnabled()
        assert window.view_mode_combo.count() == 4
        assert all(
            window.view_mode_combo.model().item(row).isEnabled()
            for row in range(window.view_mode_combo.count())
        )
        window.view_mode_combo.setCurrentText("Global fused")
        window._update_view_options()
        assert window.view_mode_combo.currentText() == "Global fused"
        assert window.sparse_2d_button.isEnabled()
        assert window.sparse_max_button.isEnabled()
        assert window.sparse_3d_button.isEnabled()
    finally:
        window.close()
        qt_app.processEvents()


def test_controller_reenables_controls_before_showing_stack() -> None:
    """Prepared display results restore controls before NDV setup begins."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy import QtWidgets

    from merfish3danalysis.viewer.controller import DatastoreViewerWindow

    class _ControllerStateCheckingView:
        """Fake view that checks controller controls before display."""

        def __init__(self, window: DatastoreViewerWindow) -> None:
            self.window = window
            self.sparse_payload: SparseOverlayPayload | None = None
            self.show_stack_called = False

        def show_stack(
            self,
            _stack: ChannelStack,
            *,
            spacing_zyx_um: np.ndarray,
            origin_zyx_um: np.ndarray | None,
        ) -> None:
            """Record display and assert controller interactivity."""
            assert self.window.open_button.isEnabled()
            assert self.window.view_mode_combo.isEnabled()
            assert not self.window.progress_bar.isVisible()
            assert spacing_zyx_um.shape == (3,)
            assert origin_zyx_um is None
            self.show_stack_called = True

        def set_sparse_payload(self, payload: SparseOverlayPayload) -> None:
            """Record sparse overlay payload."""
            self.sparse_payload = payload

        def close(self) -> None:
            """Match the real view close API."""

    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = DatastoreViewerWindow()
    fake_view = _ControllerStateCheckingView(window)
    window.view = fake_view
    try:
        window.show()
        qt_app.processEvents()
        window._set_controls_busy(True)
        window.progress_bar.setVisible(True)
        stack = ChannelStack(
            np.zeros((1, 1, 2, 2), dtype=np.uint16),
            ["global empty canvas"],
        )
        spacing = np.ones(3, dtype=np.float32)
        context = DisplayContext(
            mode="global",
            base_stack=stack,
            base_sparse_lines=(),
            shape_zyx=(1, 2, 2),
            spacing_zyx_um=spacing,
        )
        result = ViewerBuildResult(
            stack=stack,
            spacing_zyx_um=spacing,
            origin_zyx_um=None,
            context=context,
            status="Displayed test stack.",
            sparse_payload=SparseOverlayPayload(),
        )

        window._finish_display_worker(result)

        assert fake_view.show_stack_called
        assert fake_view.sparse_payload == SparseOverlayPayload()
        assert window.status_label.text() == "Displayed test stack."
    finally:
        window.close()
        qt_app.processEvents()


def test_controller_transcript_controls_remain_enabled_after_showing_stack() -> None:
    """Transcript controls stay usable after NDV display setup returns."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy import QtWidgets

    from merfish3danalysis.viewer.controller import DatastoreViewerWindow

    class _NoopView:
        """Fake view with the NDV display API."""

        def show_stack(
            self,
            _stack: ChannelStack,
            *,
            spacing_zyx_um: np.ndarray,
            origin_zyx_um: np.ndarray | None,
        ) -> None:
            """Accept stack display calls."""

        def set_sparse_payload(self, _payload: SparseOverlayPayload) -> None:
            """Accept sparse payload calls."""

        def close(self) -> None:
            """Match the real view close API."""

    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = DatastoreViewerWindow()
    window.view = _NoopView()
    try:
        window.show()
        qt_app.processEvents()
        window.transcript_gene_to_bits = {"Aif1": []}
        window.view_mode_combo.setCurrentText("Global fused")
        window.datastore_transcripts_checkbox.setChecked(True)
        window._rebuild_transcript_gene_list(("Aif1",))
        window._update_view_options()
        window._set_controls_busy(True)
        window.progress_bar.setVisible(True)
        stack = ChannelStack(
            np.zeros((1, 1, 2, 2), dtype=np.uint16),
            ["global empty canvas"],
        )
        spacing = np.ones(3, dtype=np.float32)
        context = DisplayContext(
            mode="global",
            base_stack=stack,
            base_sparse_lines=(),
            shape_zyx=(1, 2, 2),
            spacing_zyx_um=spacing,
        )
        result = ViewerBuildResult(
            stack=stack,
            spacing_zyx_um=spacing,
            origin_zyx_um=None,
            context=context,
            status="Displayed test stack.",
            sparse_payload=SparseOverlayPayload(),
        )

        window._finish_display_worker(result)

        assert window.isEnabled()
        assert window.control_panel.isEnabled()
        assert window.transcript_gene_list.isEnabled()
        assert window.marker_radius_spinbox.isEnabled()
        assert window.apply_transcripts_button.isEnabled()
    finally:
        window.close()
        qt_app.processEvents()
