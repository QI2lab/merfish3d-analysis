"""Cellpose outline filtering and reuse of saved pixel ROIs."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import typer
from roifile import ImagejRoi, roiread, roiwrite
from shapely.geometry import Polygon
from typer.testing import CliRunner

from merfish3danalysis.cli.qi2lab_microscopes import segment_fiducial


@pytest.mark.integration
@pytest.mark.parametrize(
    "cutoff, expected_areas", [(0, [3, 12, 27]), (12, [12, 27]), (28, [])]
)
def test_global_outline_filter_uses_transformed_area_and_replaces_zip(
    tmp_path, cutoff, expected_areas
):
    pixel_rois = [
        ImagejRoi.frompoints(np.array([[0, 0], [side, 0], [side, side], [0, side]]))
        for side in (1, 2, 3)
    ]
    output_path = tmp_path / "global_coords_rois.zip"
    roiwrite(
        output_path, [ImagejRoi.frompoints([[0, 0], [1, 0], [1, 1]], name="stale")]
    )

    segment_fiducial._save_global_rois(
        pixel_rois,
        output_path,
        spacing_zyx_um=np.array([1, 0.5, 0.25]),
        origin_zyx_um=np.array([0, 100, 200]),
        affine_zyx_um=np.diag([1, 2, -12, 1]),
        min_cell_area_um2=cutoff,
    )

    saved_rois = roiread(output_path)
    assert [Polygon(roi.coordinates()).area for roi in saved_rois] == expected_areas
    assert [roi.name for roi in saved_rois] == [
        f"cell_{index:07d}" for index in range(len(expected_areas))
    ]


@pytest.mark.unit
@pytest.mark.parametrize("cutoff", [-1, np.nan, np.inf, -np.inf])
def test_invalid_area_is_rejected_before_loading_data(cutoff):
    with pytest.raises(typer.BadParameter, match="finite and non-negative"):
        segment_fiducial.run_cellpose(Path("/unused"), min_cell_area_um2=cutoff)


@pytest.fixture
def saved_segmentation(tmp_path, monkeypatch):
    datastore_path = tmp_path / "qi2labdatastore"
    roi_directory = datastore_path / "segmentation" / "cellpose" / "imagej_rois"
    roi_directory.mkdir(parents=True)
    (datastore_path / "datastore_state.json").write_text("{}")
    fused_path = datastore_path / "fused"
    fused_path.mkdir()
    pixel_path = roi_directory / "pixel_spacing_rois.zip"
    roiwrite(pixel_path, [ImagejRoi.frompoints([[0, 0], [2, 0], [2, 2], [0, 2]])])
    datastore = SimpleNamespace(
        voxel_size_zyx_um=[1, 1, 1],
        fiducial_folder_name="fiducial",
        fused_image_path=Mock(return_value=fused_path),
        load_image_metadata=Mock(
            return_value={
                "affine_zyx_um": np.eye(4),
                "origin_zyx_um": np.zeros(3),
                "spacing_zyx_um": np.ones(3),
            }
        ),
        load_global_fiducial_image=Mock(
            side_effect=AssertionError("must not load fused image")
        ),
        save_global_cellpose_segmentation_image=Mock(),
        datastore_state={},
    )
    monkeypatch.setattr(segment_fiducial, "qi2labDataStore", lambda _path: datastore)
    monkeypatch.setattr(
        segment_fiducial.models,
        "CellposeModel",
        Mock(side_effect=AssertionError("must not run inference")),
    )
    return tmp_path, pixel_path, datastore


@pytest.mark.integration
def test_outlines_only_cli_can_remove_all_then_restore_from_raw_rois(
    saved_segmentation,
):
    root, pixel_path, datastore = saved_segmentation
    original_pixel_rois = pixel_path.read_bytes()
    runner = CliRunner()
    for cutoff, expected_count in [(5, 0), (0, 1), (4, 1)]:
        result = runner.invoke(
            segment_fiducial.app,
            [
                str(root),
                "--outlines-only",
                "--min-cell-area-um2",
                str(cutoff),
                "--roi-workers",
                "2",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (
            len(roiread(pixel_path.with_name("global_coords_rois.zip")))
            == expected_count
        )
        assert pixel_path.read_bytes() == original_pixel_rois

    assert datastore.datastore_state["SegmentedCells"] is True
    datastore.save_global_cellpose_segmentation_image.assert_not_called()


@pytest.mark.integration
def test_outlines_only_requires_saved_pixel_rois(saved_segmentation):
    root, pixel_path, _datastore = saved_segmentation
    pixel_path.unlink()
    with pytest.raises(FileNotFoundError, match="Saved pixel-space ROIs not found"):
        segment_fiducial.run_cellpose(root, outlines_only=True)


@pytest.mark.unit
def test_outlines_only_requires_saving_outputs():
    with pytest.raises(typer.BadParameter, match="requires --save-outputs"):
        segment_fiducial.run_cellpose(
            Path("/unused"), outlines_only=True, save_outputs=False
        )


@pytest.mark.integration
@pytest.mark.parametrize("empty_masks", [False, True])
def test_inference_exports_filtered_global_rois(
    saved_segmentation, monkeypatch, empty_masks
):
    root, pixel_path, datastore = saved_segmentation
    masks = np.zeros((20, 20), dtype=np.uint16)
    if not empty_masks:
        masks[1:4, 1:4] = 1
        masks[10:15, 10:15] = 2
    model = SimpleNamespace(
        pretrained_model="fake", eval=Mock(return_value=(masks, None, None))
    )
    monkeypatch.setattr(
        segment_fiducial.models, "CellposeModel", Mock(return_value=model)
    )
    projection_path = pixel_path.parent.parent / "fiducial_max_projection.ome.tiff"
    projection_path.touch()
    monkeypatch.setattr(
        segment_fiducial.io, "imread_2D", lambda _path: np.ones((20, 20))
    )
    monkeypatch.setattr(
        segment_fiducial,
        "roiread",
        Mock(side_effect=AssertionError("must reuse pixel ROIs in memory")),
    )
    roiwrite(pixel_path.with_name("global_coords_rois.zip"), roiread(pixel_path))

    segment_fiducial.run_cellpose(
        root,
        min_cell_area_um2=5,
        use_gpu=False,
        roi_multiprocessing=False,
    )

    assert len(roiread(pixel_path)) == (0 if empty_masks else 2)
    saved_rois = roiread(pixel_path.with_name("global_coords_rois.zip"))
    assert [Polygon(roi.coordinates()).area for roi in saved_rois] == (
        [] if empty_masks else [16]
    )
    np.testing.assert_array_equal(
        datastore.save_global_cellpose_segmentation_image.call_args.args[0], masks
    )


@pytest.mark.unit
def test_invalid_worker_count_is_rejected_before_loading_data():
    with pytest.raises(typer.BadParameter, match="non-negative"):
        segment_fiducial.run_cellpose(Path("/unused"), roi_workers=-1)


@pytest.mark.integration
def test_failed_streamed_export_preserves_existing_zip(tmp_path, monkeypatch):
    output_path = tmp_path / "global_coords_rois.zip"
    roi = ImagejRoi.frompoints([[0, 0], [1, 0], [1, 1]], name="original")
    roiwrite(output_path, [roi])
    original_bytes = output_path.read_bytes()

    def failed_export(*_args, **_kwargs):
        yield roi
        raise ValueError("invalid outline")

    monkeypatch.setattr(segment_fiducial, "global_rois", failed_export)
    with pytest.raises(ValueError, match="invalid outline"):
        segment_fiducial._save_global_rois(
            [roi], output_path, np.ones(3), np.zeros(3), np.eye(4), 0, workers=2
        )
    assert output_path.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [output_path]
