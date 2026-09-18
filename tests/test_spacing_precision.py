"""Pixel-size precision across datastore, global fusion, and OME metadata."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import numpy as np
import pytest

from merfish3danalysis.DataRegistration import DataRegistration
from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.spacing import round_pixel_size_um, round_spacing_um


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_spacing_rounding_and_serialization_do_not_leak_float32_digits(dtype):
    spacing = np.array([0.31561, 0.107999995, 0.09831], dtype=dtype)
    rounded = round_spacing_um(spacing)
    assert rounded.tolist() == [0.316, 0.108, 0.098]
    assert json.dumps(rounded.tolist()) == "[0.316, 0.108, 0.098]"
    assert round_pixel_size_um(spacing[1]) == 0.108
    np.testing.assert_array_equal(round_spacing_um(rounded), rounded)


@pytest.mark.unit
def test_datastore_rounds_new_and_legacy_voxel_calibrations():
    datastore = qi2labDataStore.__new__(qi2labDataStore)
    datastore._set_calibration_attribute = Mock()
    assert datastore.voxel_size_zyx_um is None
    datastore._voxel_size_zyx_um = [0.31561, 0.107999995, 0.09831]
    assert datastore.voxel_size_zyx_um == [0.316, 0.108, 0.098]
    datastore.voxel_size_zyx_um = np.array(
        [0.31561, 0.107999995, 0.09831], dtype=np.float32
    )
    datastore._set_calibration_attribute.assert_called_once_with(
        "voxel_size_zyx_um", [0.316, 0.108, 0.098]
    )


@pytest.mark.unit
def test_chromatic_calibration_spacing_rounds_on_read_and_write():
    datastore = qi2labDataStore.__new__(qi2labDataStore)
    datastore._set_calibration_attribute = Mock()
    calibration = {"voxel_size_zyx_um": [0.31561, 0.107999995, 0.09831]}
    datastore.save_chromatic_affine_transforms_zyx_um(calibration)
    datastore._set_calibration_attribute.assert_called_once_with(
        "chromatic_affine_transforms_zyx_um",
        {"voxel_size_zyx_um": [0.316, 0.108, 0.098]},
    )
    datastore._load_calibrations_attributes = Mock(
        return_value={"chromatic_affine_transforms_zyx_um": calibration}
    )
    assert datastore.load_chromatic_affine_transforms_zyx_um() == {
        "voxel_size_zyx_um": [0.316, 0.108, 0.098]
    }


@pytest.mark.integration
def test_image_writer_rounds_spatial_scale_and_preserves_time_scale(tmp_path):
    image_path = tmp_path / "image.ome.zarr"
    qi2labDataStore._save_to_zarr_array(
        np.ones((1, 1, 2, 3, 4), dtype=np.uint16),
        image_path,
        ome_scale=[0.123456, 1, 0.31561, 0.107999995, 0.09831],
    )
    metadata = json.loads((image_path / "zarr.json").read_text())
    transforms = metadata["attributes"]["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ]
    scale = next(t["scale"] for t in transforms if t["type"] == "scale")
    assert scale == [0.123456, 1, 0.316, 0.108, 0.098]


@pytest.mark.unit
def test_global_transform_metadata_rounds_spacing_only(tmp_path):
    datastore = qi2labDataStore.__new__(qi2labDataStore)
    datastore._num_tiles = 1
    datastore._tile_ids = ["tile0000"]
    datastore._round_ids = ["round001"]
    datastore._fiducial_root_path = tmp_path
    datastore._save_entity_attributes = Mock()
    spacing = np.array([0.31561, 0.107999995, 0.09831], dtype=np.float32)
    affine = np.eye(4, dtype=np.float32)
    affine[1, 2] = 0.123456
    origin = np.array([1.23456, -2.34567, 3.45678], dtype=np.float32)
    datastore.save_global_coord_xforms_um(affine, origin, spacing, tile=0)
    attributes = datastore._save_entity_attributes.call_args.kwargs["updates"]
    assert attributes["spacing_zyx_um"] == [0.316, 0.108, 0.098]
    np.testing.assert_array_equal(attributes["affine_zyx_um"], affine)
    np.testing.assert_array_equal(attributes["origin_zyx_um"], origin)
    # Older metadata with extra digits is normalized on read too.
    attributes["spacing_zyx_um"] = spacing.tolist()
    datastore._load_entity_attributes = Mock(return_value=attributes)
    loaded_affine, loaded_origin, loaded_spacing = (
        datastore.load_global_coord_xforms_um(0)
    )
    assert loaded_spacing.tolist() == [0.316, 0.108, 0.098]
    np.testing.assert_array_equal(loaded_affine, affine)
    np.testing.assert_array_equal(loaded_origin, origin)


@pytest.mark.unit
def test_fused_image_ome_scale_matches_rounded_fused_spacing(tmp_path):
    datastore = qi2labDataStore.__new__(qi2labDataStore)
    datastore._voxel_size_zyx_um = [0.31561, 0.107999995, 0.09831]
    datastore._fused_root_path = tmp_path
    datastore.fiducial_folder_name = "fiducial"
    datastore._save_to_zarr_array = Mock()
    datastore.save_global_fiducial_image(
        np.ones((2, 3, 4), dtype=np.uint16),
        np.eye(4),
        np.zeros(3),
        np.array([1.26261, 0.43189, 0.39231], dtype=np.float32),
    )
    spec = datastore._save_to_zarr_array.call_args.kwargs
    assert spec["ome_scale"] == [1.263, 0.432, 0.392]
    assert spec["extra_attributes"]["spacing_zyx_um"] == spec["ome_scale"]


@pytest.mark.integration
def test_external_fusion_metadata_rounds_every_pyramid_level(tmp_path):
    output_path = tmp_path / "fused.ome.zarr"
    output_path.mkdir()
    origin = [0, 0, 1.234567, 2.345678, 3.456789]
    datasets = [
        {
            "path": str(level),
            "coordinateTransformations": [
                {
                    "type": "scale",
                    "scale": [
                        0.123456,
                        1,
                        0.31561,
                        0.107999995 * factor,
                        0.09831 * factor,
                    ],
                },
                {"type": "translation", "translation": origin},
            ],
        }
        for level, factor in enumerate([1, 2])
    ]
    ome = {
        "multiscales": [
            {
                "axes": [
                    {"name": "t", "type": "time"},
                    {"name": "c", "type": "channel"},
                    *[{"name": axis, "type": "space"} for axis in "zyx"],
                ],
                "datasets": datasets,
            }
        ]
    }
    metadata_path = output_path / "zarr.json"
    metadata_path.write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {"ome": ome}})
    )
    qi2labDataStore.save_image_metadata(
        output_path,
        {"spacing_zyx_um": np.array([0.31561, 0.107999995, 0.09831], dtype=np.float32)},
    )
    metadata = json.loads(metadata_path.read_text())
    attributes = metadata["attributes"]
    assert attributes["spacing_zyx_um"] == [0.316, 0.108, 0.098]
    stored_ome = attributes["ome"]
    for level, expected in zip(
        stored_ome["multiscales"][0]["datasets"],
        [[0.316, 0.108, 0.098], [0.316, 0.216, 0.197]],
        strict=True,
    ):
        scale, translation = level["coordinateTransformations"]
        assert scale["scale"] == [0.123456, 1, *expected]
        assert translation["translation"] == origin


@pytest.mark.unit
@pytest.mark.parametrize("fail_projection", [False, True])
def test_global_fusion_receives_rounded_downsampled_spacing(
    tmp_path, monkeypatch, fail_projection
):
    datastore = SimpleNamespace(
        datastore_path=tmp_path,
        fiducial_folder_name="fiducial",
        fused_image_path=Mock(return_value=tmp_path / "fused.ome.zarr"),
        voxel_size_zyx_um=[0.31561, 0.107999995, 0.09831],
        datastore_state={},
    )
    registration = DataRegistration.__new__(DataRegistration)
    registration._datastore = datastore
    registration._verbose = 0
    from multiview_stitcher import fusion, msi_utils
    from multiview_stitcher import spatial_image_utils as si_utils

    monkeypatch.setattr(fusion, "fuse", Mock(return_value=sentinel.fused))
    monkeypatch.setattr(
        msi_utils,
        "get_transform_from_msim",
        Mock(return_value=SimpleNamespace(data=np.eye(4))),
    )
    monkeypatch.setattr(msi_utils, "get_sim_from_msim", Mock(return_value=sentinel.sim))
    monkeypatch.setattr(si_utils, "get_origin_from_sim", Mock(return_value=np.zeros(3)))
    monkeypatch.setattr(
        si_utils,
        "get_spacing_from_sim",
        Mock(return_value=np.array([0.316, 0.313, 0.314], dtype=np.float32)),
    )
    write_attributes = Mock()
    monkeypatch.setattr(qi2labDataStore, "save_image_metadata", write_attributes)
    if fail_projection:
        import yaozarrs

        import merfish3danalysis.DataRegistration as registration_module

        monkeypatch.setattr(Path, "mkdir", Mock())
        monkeypatch.setattr(yaozarrs, "open_group", Mock(return_value={"0": Mock()}))
        monkeypatch.setattr(
            registration_module,
            "_write_zarr_max_projection_tiff",
            Mock(side_effect=OSError("disk full")),
        )
        with pytest.raises(OSError, match="disk full"):
            registration.fuse_global_fiducial_views(
                msims=[sentinel.tile], create_max_proj_tiff=True
            )
        assert not datastore.datastore_state.get("Fused", False)
    else:
        registration.fuse_global_fiducial_views(
            msims=[sentinel.tile], create_max_proj_tiff=False
        )
        assert datastore.datastore_state["Fused"] is True
    assert fusion.fuse.call_args.kwargs["output_spacing"] == {
        "z": 0.316,
        "y": 0.313,
        "x": 0.314,
    }
    assert write_attributes.call_args.kwargs["extra_attributes"]["spacing_zyx_um"] == [
        0.316,
        0.313,
        0.314,
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    "spacing", [1.5, [1.5, 0.108], [1.5, 0, 0.108], [np.nan, 0.108, 0.108]]
)
def test_global_registration_rejects_invalid_voxel_calibration(spacing):
    registration = DataRegistration.__new__(DataRegistration)
    registration._datastore = SimpleNamespace(voxel_size_zyx_um=spacing)
    with pytest.raises(ValueError, match="three positive finite voxel sizes"):
        registration.load_global_fiducial_views()
