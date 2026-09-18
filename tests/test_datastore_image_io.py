"""Datastore image coordinates and Zarr v3 persistence contracts."""

import json
from pathlib import Path
from unittest.mock import Mock, mock_open

import numpy as np
import pytest
from yaozarrs import open_group
from yaozarrs.write import v05 as write_v05

from merfish3danalysis.qi2labDataStore import qi2labDataStore


@pytest.mark.unit
@pytest.mark.parametrize(
    "values, ndim, fill, expected",
    [
        ([0.4, 0.2, 0.3], 2, 1, [0.2, 0.3]),
        ([12, 100, 200], 2, 0, [100, 200]),
        ([0.4, 0.2, 0.3], 3, 1, [0.4, 0.2, 0.3]),
        ([0.4, 0.2, 0.3], 5, 1, [1, 1, 0.4, 0.2, 0.3]),
        ([12, 100, 200], 5, 0, [0, 0, 12, 100, 200]),
        (None, 2, 1, [1, 1]),
    ],
)
def test_spatial_transform_matches_image_axes(values, ndim, fill, expected):
    assert qi2labDataStore._normalize_transform(values, ndim, fill) == expected


@pytest.mark.unit
@pytest.mark.parametrize("values", [[1], [1, 2], [1, 2, 3, 4]])
def test_incompatible_transform_is_rejected(values):
    with pytest.raises(ValueError, match="transform"):
        qi2labDataStore._normalize_transform(values, 3, 1)


@pytest.mark.unit
def test_write_future_is_from_writing_pixels_without_readback(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "mkdir", Mock())
    store = Mock()
    prepare = Mock(return_value=(tmp_path, {"0": store}))
    open_image = Mock(side_effect=AssertionError("a write must not read the image"))
    monkeypatch.setattr(write_v05, "prepare_image", prepare)
    monkeypatch.setattr("merfish3danalysis.qi2labDataStore.open_group", open_image)
    pixels = np.ones((3, 4), dtype=np.uint16)
    result = qi2labDataStore._save_to_zarr_array(pixels, tmp_path / "image", True)
    assert result is store.write.return_value
    np.testing.assert_array_equal(store.write.call_args.args[0], pixels)
    open_image.assert_not_called()


@pytest.mark.unit
def test_failed_image_write_propagates_before_success_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "mkdir", Mock())
    write = Mock(side_effect=OSError("disk full"))
    monkeypatch.setattr("merfish3danalysis.qi2labDataStore.write_image", write)
    with pytest.raises(OSError, match="disk full"):
        qi2labDataStore._save_to_zarr_array(
            np.ones((3, 4), dtype=np.uint16), tmp_path / "image"
        )


@pytest.mark.integration
@pytest.mark.parametrize("return_future", [False, True])
@pytest.mark.parametrize("spacing", [[0.315, 0.081, 0.081], [0.4, 0.2, 0.3]])
def test_ome_zarr_v3_write_and_all_read_modes(tmp_path, return_future, spacing):
    path = tmp_path / "image.ome.zarr"
    pixels = np.arange(60, dtype=np.uint16).reshape(3, 4, 5)
    result = qi2labDataStore._save_to_zarr_array(
        pixels, path, return_future, ome_scale=spacing
    )
    if return_future:
        assert result.result() is None
    else:
        assert result is None
    group = open_group(path)
    assert json.loads((path / "zarr.json").read_text())["zarr_format"] == 3
    assert group.attrs["ome"]["version"] == "0.5"
    multiscale = group.attrs["ome"]["multiscales"][0]
    assert [axis["name"] for axis in multiscale["axes"]] == list("zyx")
    assert all(axis["unit"] == "micrometer" for axis in multiscale["axes"])
    transforms = multiscale["datasets"][0]["coordinateTransformations"]
    assert next(t["scale"] for t in transforms if t["type"] == "scale") == spacing
    assert not (path / ".zattrs").exists()
    for mode in (False, True, None):
        loaded = qi2labDataStore._load_from_zarr_array(path, mode)
        if mode is True:
            loaded = loaded.result()
        elif mode is None:
            loaded = loaded[1:].read().result()
        np.testing.assert_array_equal(loaded, pixels[1:] if mode is None else pixels)


@pytest.mark.integration
@pytest.mark.parametrize("shape", [(6, 8), (2, 6, 8)])
@pytest.mark.parametrize(
    "native_spacing, downsampling, expected_spacing",
    [
        ([0.4, 0.2, 0.3], [1, 1, 1], [0.4, 0.2, 0.3]),
        ([0.4, 0.2, 0.3], [2, 2.5, 3], [0.8, 0.5, 0.9]),
        ([0.8, 0.25, 0.5], [1, 3, 2.5], [0.8, 0.75, 1.25]),
    ],
)
def test_mask_ome_spacing_matches_saved_downsampling(
    tmp_path, shape, native_spacing, downsampling, expected_spacing
):
    datastore = qi2labDataStore(tmp_path / "qi2labdatastore")
    datastore.voxel_size_zyx_um = native_spacing
    masks = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape)
    datastore.save_global_cellpose_segmentation_image(masks, downsampling)
    path = (
        tmp_path
        / "qi2labdatastore/segmentation/cellpose/masks_fiducial_iso_zyx.ome.zarr"
    )
    group = open_group(path)
    multiscale = group.attrs["ome"]["multiscales"][0]
    assert [axis["name"] for axis in multiscale["axes"]] == list("zyx"[-len(shape) :])
    assert all(axis["unit"] == "micrometer" for axis in multiscale["axes"])
    scale = multiscale["datasets"][0]["coordinateTransformations"][0]["scale"]
    assert scale == expected_spacing[-len(shape) :]
    np.testing.assert_allclose(group.attrs["downsampling"], downsampling)
    np.testing.assert_array_equal(
        datastore.load_global_cellpose_segmentation_image(False), masks
    )


@pytest.mark.integration
@pytest.mark.parametrize("spacing", [[0.4, 0.2, 0.3], [0.8, 0.75, 1.25]])
def test_fused_image_writer_uses_caller_spacing_without_resampling(tmp_path, spacing):
    datastore = qi2labDataStore(tmp_path / "qi2labdatastore")
    datastore.voxel_size_zyx_um = [0.4, 0.2, 0.3]
    pixels = np.arange(60, dtype=np.uint16).reshape(3, 4, 5)
    datastore.save_global_fiducial_image(
        pixels,
        affine_zyx_um=np.eye(4),
        origin_zyx_um=np.zeros(3),
        spacing_zyx_um=spacing,
    )
    group = open_group(datastore.fused_image_path())
    transforms = group.attrs["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ]
    assert next(t["scale"] for t in transforms if t["type"] == "scale") == spacing
    assert list(group.attrs["spacing_zyx_um"]) == spacing
    loaded, _, _, loaded_spacing = datastore.load_global_fiducial_image(
        return_future=False
    )
    np.testing.assert_array_equal(loaded, pixels)
    np.testing.assert_array_equal(loaded_spacing, spacing)


@pytest.mark.unit
@pytest.mark.parametrize("factors", [[1, 2], [1, 0, 2], [1, -2, 2], [1, np.nan, 2]])
def test_invalid_mask_downsampling_is_rejected_before_writing(
    mock_datastore, monkeypatch, factors
):
    datastore = mock_datastore
    write = Mock()
    monkeypatch.setattr(datastore, "_save_to_zarr_array", write)
    datastore.voxel_size_zyx_um = [0.4, 0.2, 0.3]
    with pytest.raises(ValueError, match="Downsampling"):
        datastore.save_global_cellpose_segmentation_image(np.ones((3, 4)), factors)
    write.assert_not_called()


@pytest.mark.unit
def test_missing_optional_json_is_distinct_from_corrupt_json(monkeypatch):
    path = Path("/mock/attributes.json")
    monkeypatch.setattr("builtins.open", Mock(side_effect=FileNotFoundError))
    assert qi2labDataStore._load_from_json(path) == {}
    monkeypatch.setattr("builtins.open", mock_open(read_data='{"incomplete":'))
    with pytest.raises(json.JSONDecodeError):
        qi2labDataStore._load_from_json(path)


@pytest.mark.unit
def test_invalid_metadata_is_rejected_before_opening_output(monkeypatch):
    open_file = mock_open()
    monkeypatch.setattr("builtins.open", open_file)
    with pytest.raises(TypeError):
        qi2labDataStore._save_to_json(
            {"invalid": object()}, Path("/mock/attributes.json")
        )
    open_file.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("tile, round_index", [(-1, 0), (2, 0), (0, -1), (0, 2)])
def test_invalid_tile_or_round_index_returns_none(tile, round_index):
    datastore = qi2labDataStore.__new__(qi2labDataStore)
    datastore._num_tiles = 2
    datastore._num_rounds = 2
    datastore._tile_ids = ["tile0000", "tile0001"]
    datastore._round_ids = ["round001", "round002"]
    assert datastore.load_local_bit_linker(tile, round_index) is None


@pytest.mark.integration
def test_extra_metadata_replacement_preserves_ome_schema(tmp_path):
    path = tmp_path / "image.ome.zarr"
    qi2labDataStore._save_to_zarr_array(
        np.ones((3, 4), dtype=np.uint16),
        path,
        ome_scale=[0.2, 0.3],
        extra_attributes={"old": True},
    )
    original_ome = dict(open_group(path).attrs["ome"])
    qi2labDataStore.save_image_metadata(path, {"new": True}, merge=False)
    attrs = open_group(path).attrs
    assert attrs["ome"] == original_ome
    assert attrs["new"] is True
    assert "old" not in attrs
    assert not (path / ".zattrs").exists()


@pytest.mark.integration
def test_public_mask_writer_returns_write_completion(tmp_path):
    datastore = qi2labDataStore(tmp_path / "qi2labdatastore")
    datastore.voxel_size_zyx_um = [0.4, 0.2, 0.3]
    masks = np.ones((3, 4), dtype=np.uint16)
    completed = datastore.save_global_cellpose_segmentation_image(
        masks, [1, 2, 2], return_future=True
    )
    assert completed.result() is None
    np.testing.assert_array_equal(
        datastore.load_global_cellpose_segmentation_image(False), masks
    )


@pytest.mark.unit
def test_public_mask_write_failure_does_not_report_success(mock_datastore, monkeypatch):
    datastore = mock_datastore
    datastore.voxel_size_zyx_um = [0.4, 0.2, 0.3]
    monkeypatch.setattr(
        datastore, "_save_to_zarr_array", Mock(side_effect=OSError("disk full"))
    )
    with pytest.raises(OSError, match="disk full"):
        datastore.save_global_cellpose_segmentation_image(np.ones((3, 4)), [1, 2, 2])
    assert datastore.datastore_state["SegmentedCells"] is False


@pytest.mark.unit
@pytest.mark.parametrize("tile, selection", [(-1, 0), (2, 0), (0, -1), (0, 2)])
@pytest.mark.parametrize("kind", ["round", "bit"])
def test_local_image_path_rejects_out_of_range_indices(tile, selection, kind):
    datastore = qi2labDataStore.__new__(qi2labDataStore)
    datastore._tile_ids = ["tile0000", "tile0001"]
    datastore._round_ids = ["round001", "round002"]
    datastore._bit_ids = ["bit001", "bit002"]
    with pytest.raises(ValueError, match="Invalid"):
        datastore.local_image_path(tile, "corrected_data", **{kind: selection})


@pytest.mark.integration
def test_image_shape_uses_metadata_without_opening_array_backend(tmp_path, monkeypatch):
    path = tmp_path / "image.ome.zarr"
    qi2labDataStore._save_to_zarr_array(np.ones((2, 3, 4), dtype=np.uint16), path)
    array_type = type(open_group(path)["0"])
    backend = Mock(side_effect=AssertionError("Shape must come from metadata"))
    monkeypatch.setattr(array_type, "to_tensorstore", backend)
    assert qi2labDataStore.image_shape(path) == (2, 3, 4)
    backend.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("kind", ["round", "bit"])
def test_local_image_path_accepts_first_and_last_indices(tmp_path, index, kind):
    datastore = qi2labDataStore.__new__(qi2labDataStore)
    datastore._tile_ids = ["tile0000", "tile0001"]
    datastore._round_ids = ["round001", "round002"]
    datastore._bit_ids = ["bit001", "bit002"]
    datastore._fiducial_root_path = tmp_path / "fiducial"
    datastore._readouts_root_path = tmp_path / "readouts"
    path = datastore.local_image_path(index, "corrected_data", **{kind: index})
    assert path.parent.parent.name == f"tile{index:04}"
    assert path.parent.name == f"{kind}{index + 1:03}"
    assert path.name == "corrected_data.ome.zarr"
