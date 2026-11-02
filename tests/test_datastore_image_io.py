"""Datastore image coordinates and Zarr v3 persistence contracts."""

import json
from unittest.mock import Mock

import numpy as np
import pytest
from yaozarrs import open_group

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
def test_failed_image_write_propagates_before_success_metadata(
    image_datastore, monkeypatch
):
    write = Mock(side_effect=OSError("disk full"))
    monkeypatch.setattr("merfish3danalysis.storage.ome.write_image", write)
    with pytest.raises(OSError, match="disk full"):
        image_datastore.save_local_corrected_image(
            np.ones((3, 4), dtype=np.uint16), tile=0, round=0
        )
    assert image_datastore.load_local_corrected_image(0, round=0) is None


@pytest.mark.unit
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
    model = group.ome_metadata(strict=True)
    assert model.version == "0.6"
    multiscale = model.multiscales[0]
    assert [axis.name for axis in multiscale.axes] == list("zyx")
    assert all(axis.unit == "micrometer" for axis in multiscale.axes)
    assert multiscale.datasets[0].transform.scale == spacing
    assert not (path / ".zattrs").exists()
    for mode in (False, True, None):
        loaded = qi2labDataStore._load_from_zarr_array(path, mode)
        if mode is True:
            loaded = loaded.result()
        elif mode is None:
            loaded = loaded[1:].read().result()
        np.testing.assert_array_equal(loaded, pixels[1:] if mode is None else pixels)


@pytest.mark.unit
@pytest.mark.parametrize(
    "native_spacing, downsampling, expected_spacing",
    [
        ([0.4, 0.2, 0.3], [1, 1, 1], [0.4, 0.2, 0.3]),
        ([0.4, 0.2, 0.3], [2, 2.5, 3], [0.8, 0.5, 0.9]),
        ([0.8, 0.25, 0.5], [1, 3, 2.5], [0.8, 0.75, 1.25]),
    ],
)
def test_mask_ome_spacing_matches_saved_downsampling(
    image_datastore, native_spacing, downsampling, expected_spacing
):
    from yaozarrs import v06

    datastore = image_datastore
    datastore.voxel_size_zyx_um = native_spacing
    masks = np.arange(48, dtype=np.uint32).reshape(6, 8)
    datastore.save_global_fiducial_projection(
        np.zeros_like(masks), np.eye(4), [0, 0, 0], expected_spacing
    )
    datastore.save_global_cellpose_segmentation_image(masks, downsampling)
    scene = open_group(datastore.fiducial_projection_path()).validate(strict=True)
    image_path = next(
        t.input.path
        for t in scene.ome_metadata().scene.coordinateTransformations
        if t.input.path is not None
    )
    labels = scene[image_path]["labels"]
    labels_model = labels.ome_metadata(strict=True)
    assert isinstance(labels_model, v06.LabelsGroup)
    model = labels[labels_model.labels[0]].ome_metadata(strict=True)
    assert isinstance(model, v06.LabelImage)
    multiscale = model.multiscales[0]
    assert [axis.name for axis in multiscale.axes] == list("yx")
    assert all(axis.unit == "micrometer" for axis in multiscale.axes)
    assert multiscale.datasets[0].transform.scale == expected_spacing[1:]
    np.testing.assert_array_equal(
        datastore.load_global_cellpose_segmentation_image(False), masks
    )


@pytest.mark.unit
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
    multiscale = group.ome_metadata(strict=True).multiscales[0]
    assert multiscale.datasets[0].transform.scale == spacing
    assert list(group.attrs["spacing_zyx_um"]) == spacing
    loaded, _, _, loaded_spacing = datastore.load_global_fiducial_image(
        return_future=False
    )
    np.testing.assert_array_equal(loaded, pixels)
    np.testing.assert_array_equal(loaded_spacing, spacing)


@pytest.mark.unit
@pytest.mark.parametrize("factors", [[1, 2], [1, 0, 2], [1, -2, 2], [1, np.nan, 2]])
def test_invalid_mask_downsampling_is_rejected_before_writing(image_datastore, factors):
    datastore = image_datastore
    with pytest.raises(ValueError, match="Downsampling"):
        datastore.save_global_cellpose_segmentation_image(np.ones((3, 4)), factors)
    assert not datastore.fiducial_projection_path().exists()


@pytest.mark.unit
def test_missing_optional_json_is_distinct_from_corrupt_json(tmp_path):
    path = tmp_path / "attributes.json"
    assert qi2labDataStore._load_from_json(path) == {}
    path.write_text('{"incomplete":')
    with pytest.raises(json.JSONDecodeError):
        qi2labDataStore._load_from_json(path)


@pytest.mark.unit
def test_invalid_metadata_is_rejected_before_opening_output(tmp_path):
    path = tmp_path / "attributes.json"
    with pytest.raises(TypeError):
        qi2labDataStore._save_to_json({"invalid": object()}, path)
    assert not path.exists()


@pytest.mark.unit
@pytest.mark.parametrize("tile, round_index", [(-1, 0), (2, 0), (0, -1), (0, 2)])
def test_invalid_tile_or_round_index_returns_none(image_datastore, tile, round_index):
    assert image_datastore.load_local_bit_linker(tile, round_index) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "masks", [np.ones((1, 3, 4), dtype=np.uint16), np.ones((3, 4), dtype=float)]
)
def test_invalid_mask_does_not_publish_labels(image_datastore, masks):
    datastore = image_datastore
    datastore.save_global_fiducial_projection(
        np.zeros((3, 4), dtype=np.uint16),
        np.eye(4),
        [0, 0, 0],
        datastore.voxel_size_zyx_um,
    )
    with pytest.raises(ValueError, match="integer YX"):
        datastore.save_global_cellpose_segmentation_image(masks, [1, 1, 1])
    assert datastore.load_global_cellpose_segmentation_image(False) is None


@pytest.mark.unit
@pytest.mark.parametrize("tile, selection", [(-1, 0), (2, 0), (0, -1), (0, 2)])
@pytest.mark.parametrize("kind", ["round", "bit"])
def test_local_image_path_rejects_out_of_range_indices(
    image_datastore, tile, selection, kind
):
    datastore = image_datastore
    if kind == "bit" and selection == 2:
        selection = len(datastore.bit_ids)
    with pytest.raises(ValueError, match="Invalid"):
        datastore.local_image_path(tile, "corrected_data", **{kind: selection})


@pytest.mark.unit
@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("kind", ["round", "bit"])
def test_local_image_path_accepts_first_and_last_indices(image_datastore, index, kind):
    datastore = image_datastore
    pixels = np.full((2, 3, 4), index + 1, dtype=np.uint16)
    datastore.save_local_corrected_image(pixels, tile=index, **{kind: index})
    path = datastore.local_image_path(index, "corrected_data", **{kind: index})
    named_path = datastore.local_image_path(
        f"tile{index:04}", "corrected_data", **{kind: f"{kind}{index + 1:03}"}
    )
    assert path == named_path
    group = open_group(path)
    dataset = group.ome_metadata(strict=True).multiscales[0].datasets[0]
    np.testing.assert_array_equal(
        group[dataset.path].to_tensorstore().read().result(), pixels
    )
