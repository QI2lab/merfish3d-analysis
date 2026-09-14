"""Pixel-size precision across datastore, global fusion, and OME metadata."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import numpy as np
import pytest

from merfish3danalysis.DataRegistration import DataRegistration
from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.spacing import round_pixel_size_um, round_spacing_um


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_spacing_rounding_and_serialization_do_not_leak_float32_digits(dtype):
    spacing = np.array([0.31561, 0.107999995, 0.09831], dtype=dtype)
    rounded = round_spacing_um(spacing)
    assert rounded.tolist() == [0.316, 0.108, 0.098]
    assert json.dumps(rounded.tolist()) == "[0.316, 0.108, 0.098]"
    assert round_pixel_size_um(spacing[1]) == 0.108
    np.testing.assert_array_equal(round_spacing_um(rounded), rounded)


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


def test_image_writer_rounds_spatial_scale_and_preserves_time_scale(tmp_path):
    image_path = tmp_path / "image.ome.zarr"
    qi2labDataStore._save_to_zarr_array(
        np.ones((1, 1, 2, 3, 4), dtype=np.uint16),
        {"driver": "file", "path": str(image_path)},
        {"ome_scale": [0.123456, 1, 0.31561, 0.107999995, 0.09831]},
    )
    metadata = json.loads((image_path / "zarr.json").read_text())
    transforms = metadata["attributes"]["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ]
    scale = next(t["scale"] for t in transforms if t["type"] == "scale")
    assert scale == [0.123456, 1, 0.316, 0.108, 0.098]


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


def test_fused_image_ome_scale_matches_rounded_fused_spacing(tmp_path):
    datastore = qi2labDataStore.__new__(qi2labDataStore)
    datastore._voxel_size_zyx_um = [0.31561, 0.107999995, 0.09831]
    datastore._zarrv2_spec = {"metadata": {}}
    datastore._fused_root_path = tmp_path
    datastore.fiducial_folder_name = "fiducial"
    datastore._save_to_zarr_array = Mock()
    native_spec = datastore._build_image_write_spec()
    assert native_spec["ome_scale"] == [0.316, 0.108, 0.098]
    datastore.save_global_fiducial_image(
        np.ones((2, 3, 4), dtype=np.uint16),
        np.eye(4),
        np.zeros(3),
        np.array([1.26261, 0.43189, 0.39231], dtype=np.float32),
    )
    spec = datastore._save_to_zarr_array.call_args.args[2]
    assert spec["ome_scale"] == [1.263, 0.432, 0.392]
    assert spec["extra_attributes"]["spacing_zyx_um"] == spec["ome_scale"]


@pytest.mark.parametrize("zarr_version", [2, 3])
def test_external_fusion_metadata_rounds_every_pyramid_level(tmp_path, zarr_version):
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
    metadata_path = output_path / ("zarr.json" if zarr_version == 3 else ".zattrs")
    metadata_path.write_text(
        json.dumps({"attributes": {"ome": ome}} if zarr_version == 3 else ome)
    )
    qi2labDataStore._write_extra_attributes(
        output_path,
        {"spacing_zyx_um": np.array([0.31561, 0.107999995, 0.09831], dtype=np.float32)},
    )
    metadata = json.loads(metadata_path.read_text())
    attributes = metadata["attributes"] if zarr_version == 3 else metadata
    assert attributes["spacing_zyx_um"] == [0.316, 0.108, 0.098]
    stored_ome = attributes["ome"] if zarr_version == 3 else attributes
    for level, expected in zip(
        stored_ome["multiscales"][0]["datasets"],
        [[0.316, 0.108, 0.098], [0.316, 0.216, 0.197]],
        strict=True,
    ):
        scale, translation = level["coordinateTransformations"]
        assert scale["scale"] == [0.123456, 1, *expected]
        assert translation["translation"] == origin


def test_global_fusion_receives_explicit_rounded_spacing(tmp_path, monkeypatch):
    datastore = SimpleNamespace(
        _datastore_path=tmp_path,
        _fused_root_path=tmp_path,
        fiducial_folder_name="fiducial",
        _image_store_path=lambda path: path,
        voxel_size_zyx_um=[0.31561, 0.107999995, 0.09831],
        datastore_state={},
    )
    registration = DataRegistration.__new__(DataRegistration)
    registration._datastore = datastore
    registration._verbose = 0
    fusion = SimpleNamespace(fuse=Mock(return_value=sentinel.fused))
    msi_utils = SimpleNamespace(
        get_transform_from_msim=Mock(return_value=SimpleNamespace(data=np.eye(4))),
        get_sim_from_msim=Mock(return_value=sentinel.sim),
    )
    si_utils = SimpleNamespace(
        get_origin_from_sim=Mock(return_value=np.zeros(3)),
        get_spacing_from_sim=Mock(
            return_value=np.array([0.316, 0.108, 0.098], dtype=np.float32)
        ),
    )
    write_attributes = Mock()
    monkeypatch.setattr(qi2labDataStore, "_write_extra_attributes", write_attributes)
    registration._fuse_global_registered_msims(
        msims=[sentinel.tile],
        create_max_proj_tiff=False,
        fusion=fusion,
        misc_utils=SimpleNamespace(process_batch_using_joblib=Mock()),
        msi_utils=msi_utils,
        si_utils=si_utils,
        TiffWriter=sentinel.writer,
        zarr_module=sentinel.zarr,
    )
    assert fusion.fuse.call_args.kwargs["output_spacing"] == {
        "z": 0.316,
        "y": 0.108,
        "x": 0.098,
    }
    assert write_attributes.call_args.kwargs["extra_attributes"]["spacing_zyx_um"] == [
        0.316,
        0.108,
        0.098,
    ]
