from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from roifile import ImagejRoi, roiwrite

from merfish3danalysis.viewer import (
    ChannelStack,
    append_overlay_channel,
    apply_lut_channel_labels,
    cell_outline_overlay_for_tile,
    cell_outlines_available,
    codebook_gene_bits,
    component_summary,
    decoded_available,
    decoded_overlay_for_tile,
    global_cell_outline_overlay,
    global_decoded_overlay,
    global_fused_available,
    load_global_image_channels,
    load_image_channels,
    normalize_datastore_path,
    rasterize_cell_outlines,
    rasterize_global_decoded_spots,
    selected_image_channel_count,
    stack_with_micron_coords,
    unavailable_data_message,
)


class FakeLutView:
    def __init__(self) -> None:
        self.name = ""

    def set_channel_name(self, name: str) -> None:
        self.name = name


class FakeLutController:
    def __init__(self) -> None:
        self.lut_views = [FakeLutView()]
        self.lut_model = type("FakeLutModel", (), {"cmap": "green"})()


class FakeArrayViewer:
    def __init__(self) -> None:
        self._lut_controllers = {
            0: FakeLutController(),
            1: FakeLutController(),
            "RGB": FakeLutController(),
        }


class FakeDatastore:
    def __init__(self) -> None:
        self.tile_ids = ["tile0000", "tile0001"]
        self.round_ids = ["round001", "round002"]
        self.bit_ids = ["bit001", "bit002"]
        self.datastore_state = {
            "Calibrations": True,
            "Corrected": True,
            "LocalRegistered": True,
            "GlobalRegistered": True,
            "Fused": False,
            "SegmentedCells": True,
            "DecodedSpots": False,
            "FilteredSpots": True,
        }
        self.image = np.arange(27, dtype=np.uint16).reshape(3, 3, 3)
        self.global_image = np.arange(3 * 8 * 10, dtype=np.uint16).reshape(3, 8, 10)
        self.global_segmentation = np.arange(8 * 10, dtype=np.uint16).reshape(8, 10)
        self.voxel_size_zyx_um = np.asarray([0.5, 0.108, 0.108], dtype=np.float32)
        self.global_origin_zyx_um = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
        self.global_spacing_zyx_um = np.asarray([0.5, 1.0, 2.0], dtype=np.float32)
        self.fiducial_folder_name = "polyDT"

    def load_local_corrected_image(
        self,
        tile: str,
        round: str | None = None,
        bit: str | None = None,
        return_future: bool = False,
    ) -> np.ndarray:
        del tile, round, bit, return_future
        return self.image

    def load_local_registered_image(
        self,
        tile: str,
        round: str | None = None,
        bit: str | None = None,
        return_future: bool = False,
    ) -> np.ndarray:
        del tile, round, bit, return_future
        return self.image + 1

    def load_local_feature_predictor_image(
        self,
        tile: str,
        bit: str,
        return_future: bool = False,
    ) -> np.ndarray:
        del tile, bit, return_future
        return (self.image / self.image.max()).astype(np.float32)

    def load_global_filtered_decoded_spots(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "tile_idx": [0, 1],
                "tile_z": [1, 3],
                "tile_y": [1, 3],
                "tile_x": [1, 3],
                "global_z": [10.5, 11.0],
                "global_y": [22.0, 99.0],
                "global_x": [34.0, 99.0],
                "gene_id": ["GeneA", "GeneB"],
            }
        )

    def load_local_decoded_spots(self, tile: str) -> pd.DataFrame:
        del tile
        return pd.DataFrame()

    def load_global_cellpose_outlines(self) -> dict[int, np.ndarray]:
        return {1: np.asarray([[32, 22], [36, 22], [36, 25], [32, 25]], dtype=float)}

    def load_global_coord_xforms_um(
        self, tile: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del tile
        return np.eye(4), np.zeros(3), np.ones(3)

    def load_codebook_parsed(self) -> tuple[list[str], np.ndarray]:
        return ["GeneA", "GeneB"], np.asarray([[1, 0], [1, 1]], dtype=int)

    def load_global_fidicual_image(
        self,
        return_future: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        del return_future
        return (
            self.global_image,
            np.eye(4, dtype=np.float32),
            self.global_origin_zyx_um,
            self.global_spacing_zyx_um,
        )

    def load_global_cellpose_segmentation_image(
        self,
        return_future: bool = False,
    ) -> np.ndarray:
        del return_future
        return self.global_segmentation


class SyntheticDatastore(FakeDatastore):
    def __init__(self, datastore_path: Path) -> None:
        super().__init__()
        self._datastore_path = datastore_path
        self.bit_ids = ["bit001", "bit002", "bit003"]
        self.datastore_state["FilteredSpots"] = False
        self.datastore_state["DecodedSpots"] = False
        self.datastore_state["SegmentedCells"] = False
        self.datastore_state["Fused"] = True
        self.image = np.arange(3 * 12 * 16, dtype=np.uint16).reshape(3, 12, 16)
        self.decoded_spots = pd.DataFrame(
            {
                "tile_idx": [0, 0, 1],
                "tile_z": [1, 1, 1],
                "tile_y": [3, 8, 4],
                "tile_x": [4, 10, 5],
                "global_z": [10.5, 10.5, 10.5],
                "global_y": [23.0, 28.0, 24.0],
                "global_x": [38.0, 50.0, 40.0],
                "gene_id": ["GeneA", "GeneB", "GeneA"],
            }
        )

    def load_global_filtered_decoded_spots(self) -> pd.DataFrame:
        return self.decoded_spots.copy()

    def load_global_cellpose_outlines(self) -> dict[int, np.ndarray]:
        return {}

    def load_codebook_parsed(self) -> tuple[list[str], np.ndarray]:
        return ["GeneA", "GeneB"], np.asarray([[1, 0, 1], [0, 1, 1]], dtype=int)


class PartiallyCompleteDatastore(SyntheticDatastore):
    def __init__(self, datastore_path: Path) -> None:
        super().__init__(datastore_path)
        self.datastore_state["LocalRegistered"] = False

    def load_local_registered_image(
        self,
        tile: str,
        round: str | None = None,
        bit: str | None = None,
        return_future: bool = False,
    ) -> None:
        del tile, round, bit, return_future
        return None

    def load_local_feature_predictor_image(
        self,
        tile: str,
        bit: str,
        return_future: bool = False,
    ) -> None:
        del tile, bit, return_future
        return None


@pytest.fixture
def synthetic_datastore(tmp_path: Path) -> SyntheticDatastore:
    datastore_path = tmp_path / "experiment" / "qi2labdatastore"
    datastore_path.mkdir(parents=True)
    (datastore_path / "datastore_state.json").write_text("{}", encoding="utf-8")

    filtered_dir = datastore_path / "all_tiles_filtered_decoded_features"
    filtered_dir.mkdir(parents=True)
    (filtered_dir / "decoded_features.parquet").touch()

    roi_dir = datastore_path / "segmentation" / "cellpose" / "imagej_rois"
    roi_dir.mkdir(parents=True)
    in_tile_roi = ImagejRoi.frompoints(
        np.asarray([[2, 2], [7, 2], [7, 6], [2, 6]], dtype=np.float32)
    )
    in_tile_roi.name = "cell_0000001"
    off_tile_roi = ImagejRoi.frompoints(
        np.asarray([[100, 100], [104, 100], [104, 104]], dtype=np.float32)
    )
    off_tile_roi.name = "cell_0000002"
    roiwrite(roi_dir / "global_coords_rois.zip", [in_tile_roi, off_tile_roi])
    masks_dir = datastore_path / "segmentation" / "cellpose" / "masks_polyDT_iso_zyx"
    masks_dir.mkdir(parents=True)

    return SyntheticDatastore(datastore_path)


@pytest.fixture
def partial_datastore(
    synthetic_datastore: SyntheticDatastore,
) -> PartiallyCompleteDatastore:
    return PartiallyCompleteDatastore(synthetic_datastore._datastore_path)


def test_normalize_datastore_path_accepts_direct_and_experiment_root(
    synthetic_datastore: SyntheticDatastore,
) -> None:
    datastore = synthetic_datastore._datastore_path

    assert normalize_datastore_path(datastore) == datastore
    assert normalize_datastore_path(datastore.parent) == datastore


def test_normalize_datastore_path_rejects_missing_datastore(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        normalize_datastore_path(tmp_path)


def test_component_summary_uses_existing_datastore_state() -> None:
    summary = component_summary(FakeDatastore())

    assert summary["Corrected"] is True
    assert summary["Fused"] is False
    assert summary["FilteredSpots"] is True


def test_viewer_detects_decoded_and_cell_outputs_when_state_is_stale(
    synthetic_datastore: SyntheticDatastore,
) -> None:
    assert decoded_available(synthetic_datastore) is True
    assert cell_outlines_available(synthetic_datastore) is True
    assert global_fused_available(synthetic_datastore) is True


def test_codebook_gene_bits_maps_genes_to_existing_bit_ids(
    synthetic_datastore: SyntheticDatastore,
) -> None:
    gene_bits = codebook_gene_bits(synthetic_datastore)

    assert gene_bits == {
        "GeneA": ["bit001", "bit003"],
        "GeneB": ["bit002", "bit003"],
    }


@pytest.mark.parametrize(
    ("fiducial_sources", "bit_ids", "bit_sources", "expected_labels"),
    [
        (
            ["corrected"],
            ["bit001"],
            ["registered", "feature"],
            [
                "tile0000:round001:fiducial corrected",
                "tile0000:bit001:registered/decon",
                "tile0000:bit001:feature predictor",
            ],
        ),
        (
            ["registered"],
            ["bit001", "bit002"],
            ["feature"],
            [
                "tile0000:round001:fiducial registered/decon",
                "tile0000:bit001:feature predictor",
                "tile0000:bit002:feature predictor",
            ],
        ),
        (
            [],
            ["bit002"],
            ["corrected"],
            ["tile0000:bit002:corrected"],
        ),
    ],
)
def test_load_image_channels_stacks_selected_synthetic_sources(
    fiducial_sources: list[str],
    bit_ids: list[str],
    bit_sources: list[str],
    expected_labels: list[str],
) -> None:
    stack = load_image_channels(
        FakeDatastore(),
        tile="tile0000",
        fiducial_sources=fiducial_sources,
        bit_ids=bit_ids,
        bit_sources=bit_sources,
    )

    assert stack.data.shape == (len(expected_labels), 3, 3, 3)
    assert stack.labels == expected_labels


def test_load_image_channels_skips_missing_partial_outputs(
    partial_datastore: PartiallyCompleteDatastore,
) -> None:
    stack = load_image_channels(
        partial_datastore,
        tile="tile0000",
        fiducial_sources=["registered"],
        bit_ids=["bit001"],
        bit_sources=["corrected", "feature"],
    )

    assert stack.data.shape == (1, 3, 12, 16)
    assert stack.labels == ["tile0000:bit001:corrected"]


def test_load_image_channels_errors_when_all_partial_outputs_are_missing(
    partial_datastore: PartiallyCompleteDatastore,
) -> None:
    with pytest.raises(
        ValueError, match="No selected image channels were available to display"
    ):
        load_image_channels(
            partial_datastore,
            tile="tile0000",
            fiducial_sources=["registered"],
            bit_ids=["bit001"],
            bit_sources=["feature"],
        )


def test_unavailable_data_message_is_user_facing() -> None:
    message = unavailable_data_message(
        ValueError("No selected image channels were available to display.")
    )

    assert message == (
        "Data not available: No selected image channels were available to display."
    )


def test_partial_datastore_reports_missing_registered_but_available_overlays(
    partial_datastore: PartiallyCompleteDatastore,
) -> None:
    summary = component_summary(partial_datastore)

    assert summary["LocalRegistered"] is False
    assert decoded_available(partial_datastore) is True
    assert cell_outlines_available(partial_datastore) is True


@pytest.mark.parametrize(
    (
        "fiducial_sources",
        "bit_ids",
        "bit_sources",
        "has_fiducial_round",
        "expected_count",
    ),
    [
        (["registered"], ["bit001", "bit002"], ["registered", "feature"], True, 5),
        (["registered"], ["bit001"], ["feature"], False, 1),
        ([], ["bit001", "bit002"], ["corrected", "registered", "feature"], True, 6),
    ],
)
def test_selected_image_channel_count_matches_selected_arrays(
    fiducial_sources: list[str],
    bit_ids: list[str],
    bit_sources: list[str],
    has_fiducial_round: bool,
    expected_count: int,
) -> None:
    assert (
        selected_image_channel_count(
            fiducial_sources=fiducial_sources,
            bit_ids=bit_ids,
            bit_sources=bit_sources,
            has_fiducial_round=has_fiducial_round,
        )
        == expected_count
    )


def test_decoded_overlay_uses_filtered_global_spots_for_selected_tile(
    synthetic_datastore: SyntheticDatastore,
) -> None:
    overlay = decoded_overlay_for_tile(
        synthetic_datastore,
        tile="tile0000",
        shape_zyx=(3, 12, 16),
        genes=["GeneA"],
    )

    assert overlay is not None
    assert overlay[1, 3, 4] == 1.0
    assert overlay[1, 8, 10] == 0.0
    assert overlay[1, 4, 6] == 0.0


def test_cell_outline_overlay_loads_synthetic_roi_zip(
    synthetic_datastore: SyntheticDatastore,
) -> None:
    overlay = cell_outline_overlay_for_tile(
        synthetic_datastore,
        tile="tile0000",
        shape_zyx=(3, 12, 16),
    )

    assert overlay is not None
    assert overlay.shape == (3, 12, 16)
    projected_outline_yx = np.argwhere(overlay.max(axis=0) > 0)
    np.testing.assert_array_equal(projected_outline_yx.min(axis=0), [2, 2])
    np.testing.assert_array_equal(projected_outline_yx.max(axis=0), [6, 7])


def test_load_global_image_channels_includes_fused_polydt_and_segmentation(
    synthetic_datastore: SyntheticDatastore,
) -> None:
    stack = load_global_image_channels(
        synthetic_datastore,
        include_segmentation=True,
    )

    assert stack.stack.data.shape == (2, 1, 8, 10)
    assert stack.stack.labels == [
        "global polyDT max projection",
        "global polyDT segmentation",
    ]
    np.testing.assert_allclose(stack.origin_zyx_um, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(stack.spacing_zyx_um, [0.5, 1.0, 2.0])
    np.testing.assert_array_equal(
        stack.stack.data[1],
        synthetic_datastore.global_segmentation[np.newaxis, :, :],
    )
    np.testing.assert_array_equal(
        stack.stack.data[0],
        np.max(synthetic_datastore.global_image, axis=0, keepdims=True),
    )


def test_global_decoded_overlay_uses_global_coordinates(
    synthetic_datastore: SyntheticDatastore,
) -> None:
    overlay = global_decoded_overlay(
        synthetic_datastore,
        shape_zyx=(1, 8, 10),
        origin_zyx_um=synthetic_datastore.global_origin_zyx_um,
        spacing_zyx_um=synthetic_datastore.global_spacing_zyx_um,
        genes=["GeneA"],
    )

    assert overlay is not None
    assert overlay[0, 3, 4] == 1.0
    assert overlay[0, 8 - 1, 10 - 1] == 0.0


def test_global_cell_outline_overlay_draws_global_roi_zip(
    synthetic_datastore: SyntheticDatastore,
) -> None:
    overlay = global_cell_outline_overlay(
        synthetic_datastore,
        shape_zyx=(3, 12, 16),
        origin_zyx_um=np.zeros(3),
        spacing_zyx_um=np.ones(3),
    )

    assert overlay is not None
    projected_outline_yx = np.argwhere(overlay.max(axis=0) > 0)
    np.testing.assert_array_equal(projected_outline_yx.min(axis=0), [2, 2])
    np.testing.assert_array_equal(projected_outline_yx.max(axis=0), [6, 7])


def test_rasterize_global_decoded_spots_filters_gene_and_maps_um_to_pixels() -> None:
    decoded_spots = pd.DataFrame(
        {
            "global_y": [12.0, 14.0],
            "global_x": [24.0, 28.0],
            "gene_id": ["GeneA", "GeneB"],
        }
    )

    overlay = rasterize_global_decoded_spots(
        decoded_spots,
        shape_zyx=(1, 6, 6),
        origin_zyx_um=[0.0, 10.0, 20.0],
        spacing_zyx_um=[1.0, 1.0, 2.0],
        genes=["GeneA"],
        radius=0,
    )

    assert overlay[0, 2, 2] == 1.0
    assert overlay[0, 4, 4] == 0.0


def test_rasterize_cell_outlines_skips_off_tile_outlines() -> None:
    overlay = rasterize_cell_outlines(
        {
            1: np.asarray([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float),
            2: np.asarray([[100, 100], [102, 100], [102, 102]], dtype=float),
        },
        shape_zyx=(2, 4, 4),
        affine_zyx_um=np.eye(4),
        origin_zyx_um=np.zeros(3),
        spacing_zyx_um=np.ones(3),
    )

    assert overlay.shape == (2, 4, 4)
    assert overlay[:, 0, 0].sum() == 2.0
    assert overlay.sum() < 20


def test_append_overlay_channel_preserves_labels() -> None:
    stack = load_image_channels(
        FakeDatastore(),
        tile="tile0000",
        fiducial_sources=["corrected"],
        bit_ids=[],
        bit_sources=[],
    )
    overlay = np.ones(stack.data.shape[1:], dtype=np.float32)

    combined = append_overlay_channel(stack, overlay, "overlay")

    assert combined.data.shape[0] == 2
    assert combined.labels[-1] == "overlay"


def test_apply_lut_channel_labels_names_numeric_ndv_channels() -> None:
    viewer = FakeArrayViewer()

    applied = apply_lut_channel_labels(viewer, ["first", "second"])

    assert applied == 2
    assert viewer._lut_controllers[0].lut_views[0].name == "first"
    assert viewer._lut_controllers[1].lut_views[0].name == "second"
    assert viewer._lut_controllers["RGB"].lut_views[0].name == ""
    assert viewer._lut_controllers[0].lut_model.cmap == "green"


def test_apply_lut_channel_labels_sets_fiducials_to_gray() -> None:
    viewer = FakeArrayViewer()

    apply_lut_channel_labels(
        viewer,
        [
            "tile0000:round001:fiducial registered/decon",
            "tile0000:bit001:feature predictor",
        ],
    )

    assert viewer._lut_controllers[0].lut_model.cmap == "gray"
    assert viewer._lut_controllers[1].lut_model.cmap == "green"


def test_stack_with_micron_coords_exposes_zyx_pixel_size() -> None:
    stack = ChannelStack(
        data=np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5),
        labels=["a", "b"],
    )

    wrapped = stack_with_micron_coords(stack, [0.5, 0.1, 0.2])

    assert wrapped.dims == ("c", "z_um", "y_um", "x_um")
    np.testing.assert_allclose(wrapped.coords["z_um"], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(wrapped.coords["y_um"], [0.0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(wrapped.coords["x_um"], [0.0, 0.2, 0.4, 0.6, 0.8])
    assert wrapped.attrs["z_spacing_um"] == 0.5
    np.testing.assert_array_equal(wrapped.values, stack.data)
