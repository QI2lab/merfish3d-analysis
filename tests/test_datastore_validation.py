"""Calibration and metadata errors fail at their responsible datastore boundary."""

import json
from unittest.mock import Mock

import numpy as np
import pytest

from merfish3danalysis.qi2labDataStore import qi2labDataStore


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        0.4,
        [],
        [0.4, 0.2],
        [[0.4, 0.2, 0.2]],
        [0.4, 0, 0.2],
        [0.4, -1, 0.2],
        [0.4, np.nan, 0.2],
        [0.4, np.inf, 0.2],
        [0.4, 0.0001, 0.2],
    ],
)
def test_invalid_voxel_calibration_does_not_replace_valid_calibration(tmp_path, value):
    datastore = qi2labDataStore(tmp_path / "store")
    datastore.voxel_size_zyx_um = [0.4, 0.2, 0.2]
    path = tmp_path / "store/calibrations/attributes.json"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="voxel"):
        datastore.voxel_size_zyx_um = value
    assert datastore.voxel_size_zyx_um == [0.4, 0.2, 0.2]
    assert path.read_bytes() == before


@pytest.mark.unit
@pytest.mark.parametrize("value", [0.4, [0.4, np.nan, 0.2], [0.4, 0, 0.2]])
def test_invalid_stored_voxel_calibration_is_not_exposed_as_usable(
    image_datastore, value
):
    path = image_datastore.datastore_path / "calibrations/attributes.json"
    metadata = json.loads(path.read_text())
    metadata["voxel_size_zyx_um"] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="voxel"):
        qi2labDataStore(image_datastore.datastore_path)


@pytest.mark.unit
@pytest.mark.parametrize("value", [[], 1, None, "not metadata"])
def test_entity_metadata_must_be_a_json_object(tmp_path, value):
    datastore = qi2labDataStore(tmp_path / "store")
    entity = tmp_path / "entity"
    entity.mkdir()
    (entity / "attributes.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="object"):
        datastore._load_entity_attributes(entity)
    before = (entity / "attributes.json").read_bytes()
    with pytest.raises(ValueError, match="object"):
        datastore._save_entity_attributes(entity, {"deconvolution": True})
    assert (entity / "attributes.json").read_bytes() == before


@pytest.mark.unit
def test_missing_required_calibration_metadata_is_not_optional(tmp_path):
    datastore = qi2labDataStore(tmp_path / "store")
    (tmp_path / "store/calibrations/attributes.json").unlink()
    with pytest.raises(FileNotFoundError, match="Calibration"):
        datastore._load_calibrations_attributes()
    assert datastore._load_entity_attributes(tmp_path / "optional") == {}


@pytest.mark.unit
@pytest.mark.parametrize(
    "name, cache_name, value",
    [
        ("noise_map", "_noise_map", np.ones((3, 4))),
        ("channel_shading_maps", "_shading_maps", np.ones((1, 3, 4))),
        ("channel_psfs", "_psfs", [np.ones((2, 3, 4))]),
    ],
)
def test_failed_calibration_image_write_does_not_update_cache(
    image_datastore, monkeypatch, name, cache_name, value
):
    datastore = image_datastore
    original = [np.zeros((2, 3, 4))] if name == "channel_psfs" else np.zeros((1, 3, 4))
    setattr(datastore, name, original)
    previous = getattr(datastore, cache_name)
    state = dict(datastore.datastore_state)
    monkeypatch.setattr(
        datastore, "_save_to_zarr_array", Mock(side_effect=OSError("disk full"))
    )
    with pytest.raises(OSError, match="disk full"):
        setattr(datastore, name, value)
    assert getattr(datastore, cache_name) is previous
    assert datastore.datastore_state == state


@pytest.mark.unit
def test_failed_state_write_does_not_change_cached_success_flags(
    image_datastore, monkeypatch
):
    datastore = image_datastore
    previous = dict(datastore.datastore_state)
    monkeypatch.setattr(
        datastore, "_save_to_json", Mock(side_effect=OSError("disk full"))
    )
    with pytest.raises(OSError, match="disk full"):
        datastore.datastore_state = {"Fused": True}
    assert datastore.datastore_state == previous


@pytest.mark.unit
def test_failed_voxel_metadata_write_preserves_cached_calibration(
    image_datastore, monkeypatch
):
    datastore = image_datastore
    datastore.voxel_size_zyx_um = [0.4, 0.2, 0.2]
    monkeypatch.setattr(
        datastore, "_save_to_json", Mock(side_effect=OSError("disk full"))
    )
    with pytest.raises(OSError, match="disk full"):
        datastore.voxel_size_zyx_um = [0.8, 0.4, 0.4]
    assert datastore.voxel_size_zyx_um == [0.4, 0.2, 0.2]


@pytest.mark.unit
@pytest.mark.parametrize("metadata_file", ["zarr.json", "0/zarr.json"])
def test_corrupt_image_metadata_is_not_reported_as_a_valid_shape(
    tmp_path, metadata_file
):
    path = tmp_path / "image.ome.zarr"
    qi2labDataStore._save_to_zarr_array(np.ones((2, 3, 4), dtype=np.uint16), path)
    (path / metadata_file).write_text('{"broken":')
    with pytest.raises(ValueError):
        qi2labDataStore.image_shape(path)


@pytest.mark.unit
def test_fused_state_requires_projection_scene_when_reopening(image_datastore):
    datastore = image_datastore
    datastore.save_global_fiducial_image(
        np.ones((2, 3, 4), dtype=np.uint16),
        affine_zyx_um=np.eye(4),
        origin_zyx_um=[0, 0, 0],
        spacing_zyx_um=datastore.voxel_size_zyx_um,
    )
    datastore.datastore_state = {"Fused": True}
    assert not datastore.fiducial_projection_path().exists()

    with pytest.raises(FileNotFoundError, match=r"[Pp]rojection"):
        qi2labDataStore(datastore.datastore_path, validate=True)


@pytest.mark.unit
def test_segmented_state_requires_cellpose_labels_when_reopening(image_datastore):
    from roifile import roiwrite

    datastore = image_datastore
    datastore.save_global_fiducial_projection(
        np.ones((3, 4), dtype=np.uint16),
        affine_zyx_um=np.eye(4),
        origin_zyx_um=[0, 0, 0],
        spacing_zyx_um=datastore.voxel_size_zyx_um,
    )
    roi_path = (
        datastore.datastore_path
        / "segmentation"
        / "cellpose"
        / "imagej_rois"
        / "global_coords_rois.zip"
    )
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    roiwrite(roi_path, [], mode="w")
    datastore.datastore_state = {"SegmentedCells": True}
    assert datastore.load_global_cellpose_segmentation_image(False) is None

    with pytest.raises(FileNotFoundError, match="Cellpose"):
        qi2labDataStore(datastore.datastore_path, validate=True)
