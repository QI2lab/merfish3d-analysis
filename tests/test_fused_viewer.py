"""Fused images retain their channels and coordinates when viewed lazily."""

import numpy as np
import pytest
import tensorstore as ts

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.viewer.fused import load_fused_channels


@pytest.mark.integration
def test_viewer_loads_real_scene_owned_fusion(image_datastore):
    from merfish3danalysis.DataRegistration import (
        DataRegistration,
        GlobalRegistrationConfig,
    )

    datastore = image_datastore
    for tile in range(2):
        pixels = np.arange(4 * 12 * 16, dtype=np.uint16).reshape(4, 12, 16) + 100
        datastore.save_local_corrected_image(pixels, tile=tile, round=0)
        stage = [2, 4, tile * 3.2]
        datastore.save_local_stage_position_zyx_um(stage, np.eye(4), tile=tile, round=0)
        datastore.save_global_coord_xforms_um(
            np.eye(4), stage, datastore.voxel_size_zyx_um, tile=tile
        )
    registration = DataRegistration(
        datastore,
        decon_fiducial=False,
        verbose=0,
        global_registration_config=GlobalRegistrationConfig(fusion_workers=1),
    )
    registration.fuse_global_registered(create_max_proj_tiff=False)

    fused, _, origin, spacing = datastore.load_global_fiducial_image(False)
    stacks = load_fused_channels(datastore.datastore_path)
    assert len(stacks) == 1
    np.testing.assert_array_equal(stacks[0][0], fused.squeeze())
    assert list(stacks[0].coords["c"]) == ["fused_fiducial_zyx: 0"]
    for index, axis in enumerate("zyx"):
        np.testing.assert_allclose(
            stacks[0].coords[axis],
            origin[index] + np.arange(fused.shape[index + 2]) * spacing[index],
        )


@pytest.mark.unit
def test_all_channels_setter_replaces_its_own_scene_image(image_datastore):
    from yaozarrs import open_group, v06

    datastore = image_datastore
    fiducial = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    datastore.save_global_fiducial_image(fiducial, np.eye(4), [1, 2, 3], [1, 1, 1])
    image = np.arange(48, dtype=np.uint16).reshape(1, 2, 2, 3, 4)
    affine = np.eye(4)
    for expected in (image, image + 1):
        datastore.save_global_fiducial_image(
            expected,
            affine,
            [7, 8, 9],
            [1, 2, 3],
            fusion_type="all_channels",
        )
        path = datastore.fused_image_path("fused_all_channels_zyx")
        group = open_group(path).validate(strict=True)
        model = group.ome_metadata(strict=True)
        assert isinstance(model, v06.Image)
        stored = group[model.multiscales[0].datasets[0].path].to_tensorstore()
        np.testing.assert_array_equal(stored.read().result(), expected)
        np.testing.assert_array_equal(group.attrs["affine_zyx_um"], affine)
        scene = (
            open_group(datastore.datastore_path / "fused.ome.zarr")
            .validate(strict=True)
            .ome_metadata()
        )
        relative = str(path.relative_to(datastore.datastore_path / "fused.ome.zarr"))
        assert any(
            t.input.path == relative and t.output.name == "world"
            for t in scene.scene.coordinateTransformations
        )
    np.testing.assert_array_equal(
        datastore.load_global_fiducial_image(False)[0], fiducial
    )
    stacks = load_fused_channels(datastore.datastore_path)
    channels = next(
        s for s in stacks if s.coords["c"][0] == "fused_all_channels_zyx: 0"
    )
    np.testing.assert_array_equal(channels[:], (image + 1)[0])


@pytest.mark.unit
@pytest.mark.parametrize("same_grid", [False, True])
def test_fused_channels_remain_lazy_and_respect_each_grid(image_datastore, same_grid):
    datastore = image_datastore
    fiducial = np.ones((2, 3, 4), dtype=np.uint16)
    channels = np.full((1, 2, 2, 3, 4), 3, dtype=np.uint16)
    origin = [12, 100, 200]
    spacing = [1.5, 0.2, 0.2]
    channel_spacing = spacing if same_grid else [1.5, 0.4, 0.4]
    datastore.save_global_fiducial_image(fiducial, np.eye(4), origin, spacing)
    datastore.save_global_fiducial_image(
        channels, np.eye(4), origin, channel_spacing, fusion_type="all_channels"
    )

    stacks = load_fused_channels(datastore.datastore_path)
    assert len(stacks) == (1 if same_grid else 2)
    assert all(isinstance(stack.fused_zyx, ts.TensorStore) for stack in stacks)
    np.testing.assert_array_equal(stacks[0][0], fiducial)
    np.testing.assert_array_equal(stacks[-1][-2:], channels[0])
    np.testing.assert_allclose(
        stacks[-1].coords["x"], 200 + np.arange(4) * channel_spacing[-1]
    )


@pytest.mark.unit
def test_fused_viewer_displays_distinct_channel_pixels_and_physical_coordinates(
    tmp_path, monkeypatch
):
    import ndv
    from qtpy import QtWidgets

    from merfish3danalysis.viewer.fused import view_fused_channels

    datastore = qi2labDataStore(tmp_path / "qi2labdatastore")
    pixels = np.arange(120, dtype=np.uint16).reshape(1, 3, 2, 4, 5)
    datastore.save_global_fiducial_image(
        pixels,
        np.eye(4),
        [12, 100, 200],
        [1.5, 0.2, 0.4],
        fusion_type="all_channels",
    )
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    created = []
    constructor = ndv.ArrayViewer

    def record(*args, **kwargs):
        viewer = constructor(*args, **kwargs)
        created.append(viewer)
        return viewer

    def inspect():
        app.processEvents()
        assert len(created) == 1
        wrapper = created[0].data_wrapper
        assert wrapper.data.shape == (3, 2, 4, 5)
        np.testing.assert_array_equal(wrapper.isel({}), pixels[0])
        np.testing.assert_allclose(wrapper.coords["z"], [12, 13.5])
        np.testing.assert_allclose(wrapper.coords["y"], [100, 100.2, 100.4, 100.6])
        np.testing.assert_allclose(
            wrapper.coords["x"], [200, 200.4, 200.8, 201.2, 201.6]
        )
        assert list(wrapper.coords["c"]) == [
            "fused_all_channels_zyx: 0",
            "fused_all_channels_zyx: 1",
            "fused_all_channels_zyx: 2",
        ]
        assert created[0].display_model.channel_mode.value == "composite"

    monkeypatch.setattr(ndv, "ArrayViewer", record)
    monkeypatch.setattr(app, "exec", inspect)
    view_fused_channels(datastore.datastore_path)
