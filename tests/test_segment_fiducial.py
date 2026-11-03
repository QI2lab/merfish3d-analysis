"""Cellpose outline filtering and reuse of saved pixel ROIs."""

from pathlib import Path

import numpy as np
import pytest
import typer
from roifile import ImagejRoi, roiread, roiwrite
from shapely.geometry import Polygon
from typer.testing import CliRunner

from merfish3danalysis.cli.qi2lab_microscopes import segment_fiducial


@pytest.mark.unit
@pytest.mark.parametrize(
    "cutoff, expected_areas", [(0, [2.25, 9, 20.25]), (9, [9, 20.25]), (21, [])]
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
        spacing_zyx_um=np.array([1.5, 1.5, 1.5]),
        origin_zyx_um=np.array([0, 100, 200]),
        affine_zyx_um=np.eye(4),
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
def saved_segmentation(image_datastore):
    datastore = image_datastore
    datastore.voxel_size_zyx_um = [1, 1, 1]
    datastore.save_global_fiducial_projection(
        np.ones((20, 20), dtype=np.uint16), np.eye(4), np.zeros(3), np.ones(3)
    )
    labels = np.zeros((20, 20), dtype=np.uint16)
    labels[:3, :3] = 1
    datastore.save_global_cellpose_segmentation_image(labels, downsampling=[1, 1, 1])
    datastore_path = datastore.datastore_path
    pixel_path = datastore.cellpose_pixel_rois_path()
    pixel_path.parent.mkdir(parents=True, exist_ok=True)
    roiwrite(pixel_path, [ImagejRoi.frompoints([[0, 0], [2, 0], [2, 2], [0, 2]])])
    return datastore_path.parent, pixel_path, datastore


@pytest.mark.integration
def test_outlines_only_cli_can_remove_all_then_restore_from_raw_rois(
    saved_segmentation,
):
    root, pixel_path, datastore = saved_segmentation
    original_pixel_rois = pixel_path.read_bytes()
    original_labels = datastore.load_global_cellpose_segmentation_image(False)
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
        assert len(roiread(datastore.cellpose_global_rois_path())) == expected_count
        assert pixel_path.read_bytes() == original_pixel_rois

    reopened = segment_fiducial.qi2labDataStore(datastore.datastore_path)
    assert reopened.datastore_state["SegmentedCells"] is True
    np.testing.assert_array_equal(
        reopened.load_global_cellpose_segmentation_image(False), original_labels
    )

    # Replacing a populated segmentation with no cells must clear the old
    # global outlines as well as the labels and pixel-space ROI archive.
    empty_labels = np.zeros_like(original_labels)
    reopened.save_global_cellpose_segmentation_image(
        empty_labels, downsampling=[1, 1, 1]
    )
    roiwrite(pixel_path, [], mode="w")
    result = runner.invoke(segment_fiducial.app, [str(root), "--outlines-only"])
    assert result.exit_code == 0, result.output
    assert roiread(pixel_path) == []
    assert roiread(reopened.cellpose_global_rois_path()) == []
    reopened = segment_fiducial.qi2labDataStore(datastore.datastore_path)
    np.testing.assert_array_equal(
        reopened.load_global_cellpose_segmentation_image(False), empty_labels
    )


@pytest.mark.unit
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


@pytest.mark.unit
def test_invalid_worker_count_is_rejected_before_loading_data():
    with pytest.raises(typer.BadParameter, match="non-negative"):
        segment_fiducial.run_cellpose(Path("/unused"), roi_workers=-1)


@pytest.mark.unit
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
