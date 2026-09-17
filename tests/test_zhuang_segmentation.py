"""Zhuang-specific Cellpose parameters and physical ROI coordinates."""

import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from cellpose import models
from roifile import roiread
from tifffile import imwrite
from typer.testing import CliRunner

import merfish3danalysis.qi2labDataStore as datastore_module

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples/zhuang_lab/03_cellpose_segmentation.py"
)


@pytest.mark.unit
def test_warp_point_uses_fused_spacing_origin_and_camera_affine():
    example = runpy.run_path(str(EXAMPLE_PATH))
    affine = np.eye(4)
    affine[1, 1] = -1
    affine[2, 3] = 7
    actual = example["warp_point"](
        np.array([10, 2, 4]),
        np.array([1.5, 1.501, 1.501]),
        np.array([12, 100, 200]),
        affine,
    )
    np.testing.assert_allclose(actual, [27, -103.002, 213.004])


@pytest.mark.integration
@pytest.mark.parametrize("datastore_argument", [False, True])
@pytest.mark.parametrize("saved_projection", [False, True])
@pytest.mark.parametrize("empty_masks", [False, True])
def test_zhuang_example_passes_own_parameters_and_exports_scaled_rois(
    tmp_path, monkeypatch, datastore_argument, saved_projection, empty_masks
):
    datastore_path = tmp_path / "qi2labdatastore"
    cellpose_path = datastore_path / "segmentation" / "cellpose"
    cellpose_path.mkdir(parents=True)
    (datastore_path / "datastore_state.json").write_text("{}")
    volume = np.arange(3 * 12 * 16, dtype=np.uint16).reshape(3, 12, 16)
    projection = volume.max(axis=0)
    spacing = np.array([1.5, 1.501, 1.501])
    origin = np.array([12, 100, 200])
    affine = np.eye(4)
    affine[1, 1] = -1
    affine[2, 3] = 7
    if saved_projection:
        imwrite(cellpose_path / "fiducial_max_projection.ome.tiff", projection)
    datastore = SimpleNamespace(
        fiducial_folder_name="fiducial",
        fused_image_path=Mock(
            return_value=datastore_path / "fused/fused_fiducial_zyx.ome.zarr"
        ),
        load_image_metadata=Mock(
            return_value={
                "affine_zyx_um": affine,
                "origin_zyx_um": origin,
                "spacing_zyx_um": spacing,
            }
        ),
        load_global_fiducial_image=Mock(return_value=(volume, affine, origin, spacing)),
        voxel_size_zyx_um=[1.5, 0.108, 0.108],
        save_global_cellpose_segmentation_image=Mock(),
        datastore_state={},
    )
    loader = Mock(return_value=datastore)
    monkeypatch.setattr(datastore_module, "qi2labDataStore", loader)
    masks = np.zeros(projection.shape, dtype=np.uint16)
    if not empty_masks:
        masks[2:7, 3:9] = 1
    model = SimpleNamespace(eval=Mock(return_value=(masks, None, None)))
    constructor = Mock(return_value=model)
    monkeypatch.setattr(models, "CellposeModel", constructor)
    root_path = datastore_path if datastore_argument else tmp_path
    example = runpy.run_path(str(EXAMPLE_PATH))
    result = CliRunner().invoke(example["app"], [str(root_path)])
    assert result.exit_code == 0, result.exception

    loader.assert_called_once_with(datastore_path)
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
    if saved_projection:
        datastore.load_global_fiducial_image.assert_not_called()
    else:
        datastore.load_global_fiducial_image.assert_called_once_with(
            return_future=False
        )
    saved = datastore.save_global_cellpose_segmentation_image.call_args
    np.testing.assert_array_equal(saved.args[0], masks)
    np.testing.assert_allclose(
        saved.kwargs["downsampling"], [1, 1.501 / 0.108, 1.501 / 0.108]
    )
    assert datastore.datastore_state["SegmentedCells"] is True
    roi_path = cellpose_path / "imagej_rois"
    pixel_rois = roiread(roi_path / "pixel_spacing_rois.zip")
    global_rois = roiread(roi_path / "global_coords_rois.zip")
    assert len(pixel_rois) == len(global_rois) == (0 if empty_masks else 1)
    for pixel_roi, global_roi in zip(pixel_rois, global_rois, strict=True):
        xy = pixel_roi.coordinates()
        np.testing.assert_array_equal(xy.min(axis=0), [3, 2])
        np.testing.assert_array_equal(xy.max(axis=0), [8, 6])
        expected = np.column_stack((207 + xy[:, 0] * 1.501, -100 - xy[:, 1] * 1.501))
        np.testing.assert_allclose(
            global_roi.coordinates(), np.round(expected, 2), atol=2e-5
        )
        assert global_roi.name == "cell_0000000"


@pytest.mark.integration
@pytest.mark.parametrize("saved_projection", [False, True])
def test_zhuang_example_reopens_real_masks_and_replaces_existing_rois(
    tmp_path, monkeypatch, saved_projection
):
    import json

    from yaozarrs import open_group

    datastore_path = tmp_path / "qi2labdatastore"
    datastore = datastore_module.qi2labDataStore(datastore_path)
    # Minimal calibration metadata for reopening a fused-only test acquisition.
    calibration = {
        "num_rounds": 0,
        "num_tiles": 0,
        "num_bits": 0,
        "channels_in_data": [0],
        "tile_overlap": 0.2,
        "binning": 1,
        "e_per_ADU": 1,
        "na": 1.35,
        "ri": 1.4,
        "exp_order": None,
        "codebook": None,
        "microscope_type": "3D",
        "camera_model": "zhuang_orcav3",
        "voxel_size_zyx_um": [1.5, 0.108, 0.108],
    }
    (datastore_path / "calibrations/attributes.json").write_text(
        json.dumps(calibration)
    )
    datastore.datastore_state = {"Calibrations": True}
    volume = np.arange(3 * 12 * 16, dtype=np.uint16).reshape(3, 12, 16)
    spacing = [1.5, 1.501, 1.501]
    origin = [12, 100, 200]
    affine = np.eye(4)
    affine[1, 1] = -1
    affine[2, 3] = 7
    datastore.save_global_fiducial_image(volume, affine, origin, spacing)
    cellpose_path = datastore_path / "segmentation/cellpose"
    cellpose_path.mkdir()
    if saved_projection:
        imwrite(cellpose_path / "fiducial_max_projection.ome.tiff", volume.max(axis=0))
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
        group = open_group(cellpose_path / "masks_fiducial_iso_zyx.ome.zarr")
        grid = group.attrs["ome"]["multiscales"][0]
        assert [axis["name"] for axis in grid["axes"]] == ["y", "x"]
        assert all(axis["unit"] == "micrometer" for axis in grid["axes"])
        assert grid["datasets"][0]["coordinateTransformations"][0]["scale"] == [
            1.501,
            1.501,
        ]
        np.testing.assert_allclose(
            group.attrs["downsampling"], [1, 1.501 / 0.108, 1.501 / 0.108]
        )
        rois = cellpose_path / "imagej_rois"
        pixels = roiread(rois / "pixel_spacing_rois.zip")
        physical = roiread(rois / "global_coords_rois.zip")
        assert len(pixels) == len(physical) == (0 if empty else 1)
        if not empty:
            xy = pixels[0].coordinates()
            np.testing.assert_array_equal(xy.min(axis=0), [3, 2])
            np.testing.assert_array_equal(xy.max(axis=0), [8, 6])
            expected = np.column_stack(
                (207 + xy[:, 0] * 1.501, -100 - xy[:, 1] * 1.501)
            )
            np.testing.assert_allclose(
                physical[0].coordinates(), np.round(expected, 2), atol=2e-5
            )
        assert reopened.datastore_state["SegmentedCells"] is True
