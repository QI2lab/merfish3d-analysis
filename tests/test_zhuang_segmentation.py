"""Zhuang-specific Cellpose parameters and physical ROI coordinates."""

import runpy
from pathlib import Path

import numpy as np
import pytest
from roifile import roiread
from typer.testing import CliRunner

import merfish3danalysis.qi2labDataStore as datastore_module
from merfish3danalysis.cli.qi2lab_microscopes import segment_fiducial

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples/zhuang_lab/03_cellpose_segmentation.py"
)


@pytest.mark.unit
def test_zhuang_example_requires_the_fiducial_projection_scene(image_datastore):
    example = runpy.run_path(str(EXAMPLE_PATH))
    with pytest.raises(FileNotFoundError, match="projection Scene not found"):
        example["run_cellpose"](image_datastore.datastore_path, {})


@pytest.mark.integration
@pytest.mark.gpu
def test_zhuang_example_segments_real_projection(image_datastore):
    import torch
    from skimage import data

    if not torch.cuda.is_available():
        pytest.skip("Cellpose integration requires CUDA")
    datastore = image_datastore
    spacing = [0.107, 0.107, 0.107]
    datastore.voxel_size_zyx_um = spacing
    projection = data.cell()
    datastore.save_global_fiducial_projection(
        projection, np.eye(4), [3, 20, 30], spacing
    )

    example = runpy.run_path(str(EXAMPLE_PATH))
    result = CliRunner().invoke(example["app"], [str(datastore.datastore_path)])
    assert result.exit_code == 0, result.exception
    reopened = datastore_module.qi2labDataStore(datastore.datastore_path)
    masks = reopened.load_global_cellpose_segmentation_image(return_future=False)
    assert masks.shape == projection.shape
    assert np.any(masks)
    pixel_rois = roiread(reopened.cellpose_pixel_rois_path())
    world_rois = roiread(reopened.cellpose_global_rois_path())
    _projection, _affine, origin, spacing = reopened.load_global_fiducial_projection(
        return_future=False
    )
    assert len(pixel_rois) == len(world_rois) > 0
    for index, (pixel_roi, world_roi) in enumerate(
        zip(pixel_rois, world_rois, strict=True)
    ):
        xy = pixel_roi.coordinates()
        expected = np.round(
            np.column_stack(
                (origin[2] + xy[:, 0] * spacing[2], origin[1] + xy[:, 1] * spacing[1])
            ),
            2,
        )
        np.testing.assert_allclose(world_roi.coordinates(), expected, atol=0.011)
        assert world_roi.name == f"cell_{index:07d}"
    assert reopened.datastore_state["SegmentedCells"]

    # Rebuild outlines from the actual segmentation without rerunning Cellpose.
    segment_fiducial.run_cellpose(
        datastore.datastore_path, outlines_only=True, roi_multiprocessing=False
    )
    regenerated = roiread(reopened.cellpose_global_rois_path())
    assert len(regenerated) == len(world_rois)
    for actual, expected in zip(regenerated, world_rois, strict=True):
        np.testing.assert_array_equal(actual.coordinates(), expected.coordinates())
