"""Zhuang-specific Cellpose parameters and physical ROI coordinates."""

import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from cellpose import models
from roifile import roiread
from typer.testing import CliRunner

import merfish3danalysis.qi2labDataStore as datastore_module
from merfish3danalysis.utils.cellpose_rois import global_rois as transform_pixel_rois

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples/zhuang_lab/03_cellpose_segmentation.py"
)


@pytest.mark.unit
def test_zhuang_example_requires_the_fiducial_projection_scene(image_datastore):
    example = runpy.run_path(str(EXAMPLE_PATH))
    with pytest.raises(FileNotFoundError, match="projection Scene not found"):
        example["run_cellpose"](image_datastore.datastore_path, {})


@pytest.mark.unit
@pytest.mark.parametrize("datastore_argument", [False, True])
@pytest.mark.parametrize("empty_masks", [False, True])
def test_zhuang_example_passes_own_parameters_and_exports_scaled_rois(
    image_datastore, monkeypatch, datastore_argument, empty_masks
):
    datastore = image_datastore
    datastore_path = datastore.datastore_path
    volume = np.arange(3 * 12 * 16, dtype=np.uint16).reshape(3, 12, 16)
    projection = volume.max(axis=0)
    spacing = np.array([1.5, 1.501, 1.501])
    origin = np.array([12, 100, 200])
    affine = np.eye(4)
    datastore.voxel_size_zyx_um = [1.5, 0.108, 0.108]
    datastore.save_global_fiducial_projection(projection, affine, origin, spacing)
    masks = np.zeros(projection.shape, dtype=np.uint16)
    if not empty_masks:
        masks[2:7, 3:9] = 1
    model = SimpleNamespace(eval=Mock(return_value=(masks, None, None)))
    constructor = Mock(return_value=model)
    monkeypatch.setattr(models, "CellposeModel", constructor)
    root_path = datastore_path if datastore_argument else datastore_path.parent
    example = runpy.run_path(str(EXAMPLE_PATH))
    result = CliRunner().invoke(example["app"], [str(root_path)])
    assert result.exit_code == 0, result.exception

    constructor.assert_called_once_with(gpu=True)
    np.testing.assert_array_equal(np.squeeze(model.eval.call_args.args[0]), projection)
    assert model.eval.call_args.kwargs == {
        "do_3D": False,
        "diameter": 15,
        "flow_threshold": 0.4,
        "cellprob_threshold": -1.0,
        "niter": 200,
        "normalize": {"normalize": True, "percentile": [0.5, 99.5]},
    }
    reopened = datastore_module.qi2labDataStore(datastore_path)
    np.testing.assert_array_equal(
        reopened.load_global_cellpose_segmentation_image(False), masks
    )
    assert reopened.datastore_state["SegmentedCells"] is True
    pixel_rois = roiread(datastore.cellpose_pixel_rois_path())
    global_rois = roiread(datastore.cellpose_global_rois_path())
    assert len(pixel_rois) == len(global_rois) == (0 if empty_masks else 1)
    for pixel_roi, global_roi in zip(pixel_rois, global_rois, strict=True):
        xy = pixel_roi.coordinates()
        np.testing.assert_array_equal(xy.min(axis=0), [3, 2])
        np.testing.assert_array_equal(xy.max(axis=0), [8, 6])
        expected = np.column_stack((200 + xy[:, 0] * 1.501, 100 + xy[:, 1] * 1.501))
        np.testing.assert_allclose(
            global_roi.coordinates(), np.round(expected, 2), atol=2e-5
        )
        assert global_roi.name == "cell_0000000"


@pytest.mark.unit
def test_zhuang_example_reopens_real_masks_and_replaces_existing_rois(
    image_datastore, monkeypatch
):
    from yaozarrs import open_group, v06

    datastore = image_datastore
    datastore_path = datastore.datastore_path
    datastore.voxel_size_zyx_um = [1.5, 0.108, 0.108]
    volume = np.arange(3 * 12 * 16, dtype=np.uint16).reshape(3, 12, 16)
    spacing = [1.5, 1.501, 1.501]
    origin = [12, 100, 200]
    affine = np.eye(4)
    datastore.save_global_fiducial_image(volume, affine, origin, spacing)
    datastore.save_global_fiducial_projection(
        volume.max(axis=0), affine, origin, spacing
    )
    masks = np.zeros((12, 16), dtype=np.uint16)
    masks[2:7, 3:9] = 1
    model = SimpleNamespace(eval=Mock(return_value=(masks, None, None)))
    monkeypatch.setattr(models, "CellposeModel", Mock(return_value=model))
    example = runpy.run_path(str(EXAMPLE_PATH))
    for empty in [False, False, True]:
        # A second populated run must replace, not append; an empty run clears both archives.
        if empty:
            masks = np.zeros_like(masks)
            model.eval.return_value = (masks, None, None)
        result = CliRunner().invoke(example["app"], [str(datastore_path)])
        assert result.exit_code == 0, result.exception
        reopened = datastore_module.qi2labDataStore(datastore_path)
        np.testing.assert_array_equal(
            reopened.load_global_cellpose_segmentation_image(False), masks
        )
        group = open_group(reopened.fiducial_projection_path()).validate(strict=True)
        product = group.attrs["qi2lab"]["products"]["projection/fiducial"]
        image = group[product["path"]]
        labels = image["labels"]
        label_name = labels.ome_metadata(strict=True).labels[0]
        label_model = labels[label_name].ome_metadata(strict=True)
        assert isinstance(label_model, v06.LabelImage)
        grid = label_model.multiscales[0]
        assert [axis.name for axis in grid.axes] == ["y", "x"]
        assert all(axis.unit == "micrometer" for axis in grid.axes)
        assert (
            grid.datasets[0].transform
            == image.ome_metadata(strict=True).multiscales[0].datasets[0].transform
        )
        pixels = roiread(datastore.cellpose_pixel_rois_path())
        physical = roiread(datastore.cellpose_global_rois_path())
        assert len(pixels) == len(physical) == (0 if empty else 1)
        if not empty:
            xy = pixels[0].coordinates()
            np.testing.assert_array_equal(xy.min(axis=0), [3, 2])
            np.testing.assert_array_equal(xy.max(axis=0), [8, 6])
            expected = np.column_stack((200 + xy[:, 0] * 1.501, 100 + xy[:, 1] * 1.501))
            np.testing.assert_allclose(
                physical[0].coordinates(), np.round(expected, 2), atol=2e-5
            )
        assert reopened.datastore_state["SegmentedCells"] is True


@pytest.mark.integration
@pytest.mark.gpu
def test_zhuang_example_segments_real_projection(image_datastore):
    import torch
    from skimage import data

    if not torch.cuda.is_available():
        pytest.skip("Cellpose integration requires CUDA")
    if not (models.MODEL_DIR / "cpsam_v2").is_file():
        pytest.skip("Cellpose cpsam_v2 model weights are not cached")
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
    _projection, affine, origin, spacing = reopened.load_global_fiducial_projection(
        return_future=False
    )
    expected_rois = list(transform_pixel_rois(pixel_rois, spacing, origin, affine, 0))
    assert len(world_rois) == len(expected_rois) > 0
    for actual, expected in zip(world_rois, expected_rois, strict=True):
        assert actual.name == expected.name
        np.testing.assert_array_equal(actual.coordinates(), expected.coordinates())
