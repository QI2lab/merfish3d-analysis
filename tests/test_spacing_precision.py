"""Pixel-size precision across datastore, global fusion, and OME metadata."""

import json
from unittest.mock import Mock

import numpy as np
import pytest

from merfish3danalysis.DataRegistration import (
    DataRegistration,
    GlobalRegistrationConfig,
)
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
def test_datastore_rounds_voxel_calibration_on_write_and_reopen(image_datastore):
    datastore = image_datastore
    datastore.voxel_size_zyx_um = np.array(
        [0.31561, 0.107999995, 0.09831], dtype=np.float32
    )
    assert datastore.voxel_size_zyx_um == [0.316, 0.108, 0.098]
    reopened = qi2labDataStore(datastore.datastore_path, validate=False)
    assert reopened.voxel_size_zyx_um == [0.316, 0.108, 0.098]


@pytest.mark.unit
def test_chromatic_calibration_spacing_rounds_on_read_and_write(image_datastore):
    datastore = image_datastore
    calibration = {"voxel_size_zyx_um": [0.31561, 0.107999995, 0.09831]}
    datastore.save_chromatic_affine_transforms_zyx_um(calibration)
    reopened = qi2labDataStore(datastore.datastore_path, validate=False)
    assert reopened.load_chromatic_affine_transforms_zyx_um() == {
        "voxel_size_zyx_um": [0.316, 0.108, 0.098]
    }


@pytest.mark.unit
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
def test_global_transform_metadata_rounds_spacing_only(image_datastore):
    datastore = image_datastore
    spacing = np.array([0.31561, 0.107999995, 0.09831], dtype=np.float32)
    affine = np.eye(4, dtype=np.float32)
    affine[1, 2] = 0.123456
    origin = np.array([1.23456, -2.34567, 3.45678], dtype=np.float32)
    datastore.save_global_coord_xforms_um(affine, origin, spacing, tile=0)
    loaded_affine, loaded_origin, loaded_spacing = (
        datastore.load_global_coord_xforms_um(0)
    )
    assert loaded_spacing.tolist() == [0.316, 0.108, 0.098]
    np.testing.assert_array_equal(loaded_affine, affine)
    np.testing.assert_array_equal(loaded_origin, origin)


@pytest.mark.unit
def test_fused_image_ome_scale_matches_rounded_fused_spacing(image_datastore):
    from yaozarrs import open_group

    datastore = image_datastore
    datastore.save_global_fiducial_image(
        np.ones((2, 3, 4), dtype=np.uint16),
        np.eye(4),
        np.zeros(3),
        np.array([1.26261, 0.43189, 0.39231], dtype=np.float32),
    )
    group = open_group(datastore.fused_image_path())
    transform = group.ome_metadata(strict=True).multiscales[0].datasets[0].transform
    assert transform.scale == [1.263, 0.432, 0.392]
    assert group.attrs["spacing_zyx_um"] == transform.scale


@pytest.mark.parametrize(
    "fail_projection",
    [
        pytest.param(False, marks=pytest.mark.integration),
        pytest.param(True, marks=pytest.mark.unit),
    ],
)
def test_global_fusion_receives_rounded_downsampled_spacing(
    tmp_path, monkeypatch, fail_projection
):
    import merfish3danalysis.DataRegistration as registration_module

    datastore = qi2labDataStore(tmp_path / "qi2labdatastore")
    datastore.channels_in_data = ["fiducial", "readout"]
    datastore.voxel_size_zyx_um = [0.31561, 0.107999995, 0.09831]
    datastore.num_tiles = 1
    datastore.experiment_order = np.array([[1, 1]])
    datastore.initialize_tile(0)
    datastore.save_local_stage_position_zyx_um([0, 0, 0], np.eye(4), tile=0, round=0)
    datastore.save_local_corrected_image(
        np.arange(4 * 24 * 24, dtype=np.uint16).reshape(4, 24, 24),
        tile=0,
        round=0,
    )
    registration = DataRegistration(
        datastore,
        decon_fiducial=False,
        verbose=0,
        global_registration_config=GlobalRegistrationConfig(fusion_workers=1),
    )
    if fail_projection:
        monkeypatch.setattr(
            registration_module,
            "_write_zarr_max_projection_tiff",
            Mock(side_effect=OSError("disk full")),
        )
        with pytest.raises(OSError, match="disk full"):
            registration.fuse_global_registered(create_max_proj_tiff=True)
        assert not datastore.datastore_state.get("Fused", False)
    else:
        registration.fuse_global_registered(create_max_proj_tiff=False)
        assert datastore.datastore_state["Fused"] is True
    _, _, _, spacing = datastore.load_global_fiducial_image(return_future=False)
    np.testing.assert_array_equal(spacing, [0.316, 0.313, 0.314])
