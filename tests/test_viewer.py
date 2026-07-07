"""Tests for viewer loader and overlay helpers."""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.viewer import (
    ProsegRun,
    WarpChainOptions,
    apply_lut_channel_labels,
    codeword_color_hex,
    compose_viewer_warp_transform_zyx_um,
    discover_proseg_runs,
    empty_transcript_overlay,
    rasterize_global_proseg_transcripts,
    rasterize_local_decoded_spots,
    rasterize_local_proseg_transcripts,
    selected_warp_label,
    warp_chain_label,
)


class _FakeDatastore:
    """Minimal datastore object for viewer helper tests."""

    def __init__(self, path: Path) -> None:
        self._datastore_path = path
        self._proseg_runs = ["default", "fdr.75"]

    def list_proseg_3d_runs(self) -> list[str]:
        """Return fake Proseg run names."""

        return self._proseg_runs

    def load_proseg_cell_polygons_3d(self, run_name: str | None = None) -> dict:
        """Load fake Proseg polygons using datastore implementation."""

        datastore = qi2labDataStore.__new__(qi2labDataStore)
        datastore._datastore_path = self._datastore_path
        return datastore.load_proseg_cell_polygons_3d(run_name=run_name)


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


def test_warp_chain_labels_and_optional_affine_composition() -> None:
    """Viewer warp-chain labels and affine toggles reflect selected components."""

    round_transform = np.eye(4, dtype=np.float32)
    round_transform[1, 3] = 2.0
    chromatic_transform = np.eye(4, dtype=np.float32)
    chromatic_transform[2, 3] = 4.0

    affine_only = compose_viewer_warp_transform_zyx_um(
        round_transform_zyx_um=round_transform,
        chromatic_transform_zyx_um=None,
    )
    full = compose_viewer_warp_transform_zyx_um(
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


def test_codeword_color_hex_returns_distinct_hex_colors() -> None:
    """Codeword color key values are hex colors and vary by selected value."""

    first_color = codeword_color_hex(1, 3)
    second_color = codeword_color_hex(2, 3)
    expected_first_color = "#" + "".join(
        f"{round(channel * 255):02x}" for channel in colormaps["turbo"](0.0)[:3]
    )

    assert first_color.startswith("#")
    assert len(first_color) == 7
    assert second_color.startswith("#")
    assert len(second_color) == 7
    assert first_color != second_color
    assert first_color == expected_first_color


def test_apply_lut_channel_labels_sets_image_and_outline_colors() -> None:
    """Viewer LUT labels use gray image channels and white Cellpose outlines."""

    labels = [
        "global polyDT max projection",
        "global Cellpose outlines",
        "global datastore codewords",
    ]
    array_viewer = _FakeArrayViewer(len(labels))
    array_viewer.data[2, 0] = np.asarray([[np.nan, 1.0], [2.0, 3.0]])

    applied = apply_lut_channel_labels(array_viewer, labels)

    controllers = array_viewer._lut_controllers
    assert applied == len(labels)
    assert controllers[0].lut_model.cmap == "gray"
    assert controllers[1].lut_model.cmap == "white"
    assert controllers[2].lut_model.cmap == "turbo"
    assert controllers[2].lut_model.clims == (1.0, 3.0)
    assert controllers[0].lut_views[0].channel_name == labels[0]
    assert controllers[1].lut_views[0].channel_name == labels[1]
    assert controllers[2].lut_views[0].channel_name == labels[2]


def test_empty_transcript_overlay_uses_nan_background() -> None:
    """Transcript overlays use NaN background so zero is not colorized."""

    overlay = empty_transcript_overlay((1, 2, 3))

    assert np.isnan(overlay).all()


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


def test_discover_proseg_runs_detects_default_and_nested(tmp_path: Path) -> None:
    """Proseg discovery finds default and nested 3D run folders."""

    proseg_root = tmp_path / "proseg" / "3D"
    _write_proseg_run(proseg_root)
    _write_proseg_run(proseg_root / "fdr.75")

    runs = discover_proseg_runs(_FakeDatastore(tmp_path))

    assert [run.name for run in runs] == ["default", "fdr.75"]
    datastore = qi2labDataStore.__new__(qi2labDataStore)
    datastore._datastore_path = tmp_path
    polygons = datastore.load_proseg_cell_polygons_3d(run_name=runs[0].name)
    assert 7 in polygons
    assert polygons[7].shape == (3, 2)


def test_proseg_transcripts_rasterize_global_and_local_canvas(tmp_path: Path) -> None:
    """Proseg transcript coordinates rasterize in global and local coordinate frames."""

    run_root = tmp_path / "proseg" / "3D"
    _write_proseg_run(run_root)
    ProsegRun(name="default")
    transcripts = pd.read_csv(run_root / "transcript_metadata_3D.csv.gz")
    shape = (4, 8, 8)
    origin = np.zeros(3, dtype=np.float32)
    spacing = np.ones(3, dtype=np.float32)

    global_overlay = rasterize_global_proseg_transcripts(
        transcripts,
        shape_zyx=shape,
        origin_zyx_um=origin,
        spacing_zyx_um=spacing,
        genes=["GeneA"],
        radius=0,
    )
    local_overlay = rasterize_local_proseg_transcripts(
        transcripts,
        shape_zyx=shape,
        affine_zyx_um=np.eye(4, dtype=np.float32),
        origin_zyx_um=origin,
        spacing_zyx_um=spacing,
        genes=["GeneA"],
        radius=0,
    )

    assert global_overlay[1, 4, 2] == 1.0
    assert local_overlay[1, 4, 2] == 1.0
    assert np.isnan(global_overlay[0, 0, 0])
    assert np.isnan(local_overlay[0, 0, 0])


def test_local_decoded_transcripts_rasterize_from_global_coordinates() -> None:
    """Final decoded transcripts rasterize locally from global coordinates."""

    decoded_spots = pd.DataFrame(
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
    overlay = rasterize_local_decoded_spots(
        decoded_spots,
        shape_zyx=(4, 8, 8),
        affine_zyx_um=np.eye(4, dtype=np.float32),
        origin_zyx_um=np.zeros(3, dtype=np.float32),
        spacing_zyx_um=np.ones(3, dtype=np.float32),
        genes=["GeneA"],
        radius=0,
    )

    assert overlay[1, 4, 2] == 1.0
    assert np.isnan(overlay[3, 6, 7])
