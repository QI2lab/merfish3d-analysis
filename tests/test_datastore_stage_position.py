"""Legacy 2D stage positions must be exposed as ZYX throughout registration."""

import json
from pathlib import Path
from unittest.mock import Mock, sentinel

import numpy as np
import pytest

import merfish3danalysis.DataRegistration as registration_module
from merfish3danalysis.DataRegistration import DataRegistration
from merfish3danalysis.qi2labDataStore import qi2labDataStore


@pytest.fixture
def stage_datastore(tmp_path):
    datastore = qi2labDataStore.__new__(qi2labDataStore)
    datastore._tile_ids = ["tile0000"]
    datastore._round_ids = ["round001", "round002"]
    datastore._num_tiles = 1
    datastore._num_rounds = 2
    datastore._fiducial_root_path = tmp_path / "fiducial"
    datastore._voxel_size_zyx_um = [0.50012, 0.199999995, 0.30031]
    affine = np.eye(4, dtype=np.float32)
    affine[1, 1] = -1
    affine[2, 3] = 12.5
    attributes = {
        "round001": {"stage_zyx_um": [125.5, -34.25], "affine_zyx_px": affine.tolist()},
        "round002": {"stage_zyx_um": [81.25, 210.5], "affine_zyx_px": affine.tolist()},
    }
    datastore._load_entity_attributes = lambda path: attributes[path.name]
    datastore._save_entity_attributes = Mock()
    datastore.load_local_round_linker = Mock(return_value=2)
    return datastore, attributes, affine


@pytest.mark.unit
def test_loading_legacy_yx_preserves_lateral_coordinates_and_metadata(stage_datastore):
    datastore, attributes, affine = stage_datastore

    position, loaded_affine = datastore.load_local_stage_position_zyx_um(0, 0)

    np.testing.assert_array_equal(position, [0, 125.5, -34.25])
    assert position.dtype == np.float32
    np.testing.assert_array_equal(loaded_affine, affine)
    assert attributes["round001"]["stage_zyx_um"] == [125.5, -34.25]
    datastore._save_entity_attributes.assert_not_called()


@pytest.mark.integration
@pytest.mark.parametrize(
    "position, expected",
    [
        ([125.5, -34.25], [0, 125.5, -34.25]),
        ([7.5, 125.5, -34.25], [7.5, 125.5, -34.25]),
    ],
)
def test_saving_stage_position_writes_three_coordinates_in_attributes_and_ome(
    stage_datastore, position, expected
):
    datastore, _attributes, affine = stage_datastore
    image_path = (
        datastore._fiducial_root_path
        / "tile0000"
        / "round001"
        / "corrected_data.ome.zarr"
    )
    metadata_path = image_path / "zarr.json"
    datastore._save_to_zarr_array(
        np.zeros((2, 3, 4), dtype=np.uint16),
        image_path,
        ome_scale=[0.5, 0.2, 0.3],
        ome_translation=[0, 0, 0],
    )

    datastore.save_local_stage_position_zyx_um(position, affine, tile=0, round=0)

    updates = datastore._save_entity_attributes.call_args.kwargs["updates"]
    assert updates["stage_zyx_um"] == expected
    np.testing.assert_array_equal(updates["affine_zyx_px"], affine)
    metadata = json.loads(metadata_path.read_text())
    transforms = metadata["attributes"]["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ]
    assert (
        next(t["translation"] for t in transforms if t["type"] == "translation")
        == expected
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "selection, expected",
    [
        ({"round_id": "round001"}, [0, 125.5, -34.25]),
        ({"bit_id": "bit001"}, [0, 81.25, 210.5]),
        ({}, [0, 125.5, -34.25]),
    ],
)
def test_derived_image_origins_use_three_coordinates(
    stage_datastore, selection, expected
):
    datastore, _attributes, _affine = stage_datastore
    assert (
        datastore._resolve_original_tile_position_zyx_um("tile0000", **selection)
        == expected
    )


@pytest.mark.unit
@pytest.mark.parametrize("use_stored_global_transforms", [False, True])
@pytest.mark.parametrize(
    "position, expected_z", [([125.5, -34.25], 0), ([7.5, 125.5, -34.25], 7.5)]
)
def test_global_registration_and_fusion_load_legacy_and_3d_stage_positions(
    stage_datastore, monkeypatch, use_stored_global_transforms, position, expected_z
):
    monkeypatch.setattr(Path, "exists", Mock(return_value=False))
    datastore, attributes, affine = stage_datastore
    attributes["round001"]["stage_zyx_um"] = position
    stored_transform = np.eye(4, dtype=np.float32)
    stored_transform[1, 3] = 5
    datastore.load_global_coord_xforms_um = Mock(
        return_value=(stored_transform, None, None)
    )
    registration = DataRegistration.__new__(DataRegistration)
    registration._datastore = datastore
    registration._tile_ids = datastore._tile_ids
    registration._round_ids = datastore._round_ids
    registration._verbose = 0
    read_sim = Mock(return_value=sentinel.sim)
    monkeypatch.setattr(registration_module, "_read_fiducial_sim", read_sim)
    from multiview_stitcher import msi_utils

    monkeypatch.setattr(
        msi_utils, "get_msim_from_sim", Mock(return_value=sentinel.msim)
    )
    monkeypatch.setattr(msi_utils, "set_affine_transform", Mock())

    result = registration.load_global_fiducial_views(
        use_stored_global_transforms=use_stored_global_transforms,
    )

    assert result == [sentinel.msim]
    assert read_sim.call_args.kwargs["scale"] == {"z": 0.5, "y": 0.2, "x": 0.3}
    assert read_sim.call_args.kwargs["translation"] == {
        "z": expected_z,
        "y": 125.5,
        "x": -34.25,
    }
    np.testing.assert_array_equal(read_sim.call_args.kwargs["affine_zyx_px"], affine)
    if use_stored_global_transforms:
        np.testing.assert_array_equal(
            np.asarray(msi_utils.set_affine_transform.call_args.args[1]).squeeze(),
            stored_transform,
        )
        assert (
            msi_utils.set_affine_transform.call_args.kwargs["transform_key"]
            == "global_registered"
        )
    else:
        msi_utils.set_affine_transform.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("position", [[], [1], [1, 2, 3, 4], [[1, 2, 3]]])
def test_invalid_stage_shape_is_not_interpreted_as_coordinates(position):
    with pytest.raises(ValueError, match=r"two .* or three"):
        qi2labDataStore._normalize_stage_position_zyx_um(position)
