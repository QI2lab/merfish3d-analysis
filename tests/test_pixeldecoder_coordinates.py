import numpy as np
import pytest

from merfish3danalysis.PixelDecoder import PixelDecoder


@pytest.mark.unit
@pytest.mark.parametrize("camera_transform", [False, True])
def test_warp_pixel_applies_spacing_origin_camera_and_global_in_order(camera_transform):
    # All inputs are exactly representable; native physical ZYX is (12, 21.5, 31).
    pixel = np.array([4, 6, 8])
    spacing = np.array([0.5, 0.25, 0.125])
    origin = np.array([10, 20, 30])
    camera_to_stage = np.array(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    )
    global_affine = np.array(
        [[1, 0, 0, 0.5], [0, 1, 0, 1.25], [0, 0, 1, -2], [0, 0, 0, 1]]
    )

    observed = PixelDecoder._warp_pixel(
        pixel,
        spacing,
        origin,
        global_affine,
        camera_to_stage if camera_transform else None,
    )

    # Camera rotates physical XY 90 degrees, then global translation is added.
    expected = [12.5, -29.75, 19.5] if camera_transform else [12.5, 22.75, 29]
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    "tile, origin, global_shift, expected",
    [
        (0, [10, -4, 6], [3, 5, 7], [15, -3, 5]),
        (1, [-2, 8, 4], [-1, 2, 3], [-1, -4, 13]),
    ],
)
def test_persisted_stage_camera_and_global_coordinates(
    image_datastore, tile, origin, global_shift, expected
):
    """The decoder maps reopened tile coordinates to the same physical world."""
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = image_datastore
    camera = np.array(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    global_affine = np.eye(4, dtype=np.float32)
    global_affine[:3, 3] = global_shift
    spacing = [2, 1, 0.5]
    datastore.save_local_stage_position_zyx_um(origin, camera, tile=tile, round=0)
    datastore.save_global_coord_xforms_um(global_affine, origin, spacing, tile=tile)

    reopened = qi2labDataStore(datastore.datastore_path, validate=False)
    loaded_origin, loaded_camera = reopened.load_local_stage_position_zyx_um(tile, 0)
    loaded_global, global_origin, global_spacing = reopened.load_global_coord_xforms_um(
        tile
    )
    np.testing.assert_array_equal(loaded_origin, global_origin)
    observed = PixelDecoder._warp_pixel(
        np.array([1, 2, 4]),
        global_spacing,
        global_origin,
        loaded_global,
        loaded_camera,
    )
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.unit
@pytest.mark.parametrize("global_first", [False, True])
def test_distinct_global_grid_and_stage_survive_reopen(image_datastore, global_first):
    """Global registration and measured stage remain independent Scene values."""
    from yaozarrs import open_group, v06

    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = image_datastore
    stage = np.array([10, 20, 30], dtype=np.float32)
    camera = np.array(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    global_affine = np.eye(4, dtype=np.float32)
    global_affine[:3, 3] = [4, -2, 6]
    global_origin = np.array([3, 8, -2], dtype=np.float32)
    global_spacing = np.array([2, 1, 0.5], dtype=np.float32)
    assert not np.array_equal(global_spacing, datastore.voxel_size_zyx_um)

    if global_first:
        datastore.save_global_coord_xforms_um(
            global_affine, global_origin, global_spacing, tile=0
        )
    datastore.save_local_stage_position_zyx_um(stage, camera, tile=0, round=0)
    if not global_first:
        datastore.save_global_coord_xforms_um(
            global_affine, global_origin, global_spacing, tile=0
        )

    reopened = qi2labDataStore(datastore.datastore_path, validate=False)
    loaded_stage, loaded_camera = reopened.load_local_stage_position_zyx_um(0, 0)
    loaded_affine, loaded_origin, loaded_spacing = reopened.load_global_coord_xforms_um(
        0
    )
    np.testing.assert_array_equal(loaded_stage, stage)
    np.testing.assert_array_equal(loaded_camera, camera)
    np.testing.assert_array_equal(loaded_affine, global_affine)
    np.testing.assert_array_equal(loaded_origin, global_origin)
    np.testing.assert_array_equal(loaded_spacing, global_spacing)
    np.testing.assert_array_equal(
        PixelDecoder._warp_pixel(
            np.array([1, 2, 4]),
            loaded_spacing,
            loaded_origin,
            loaded_affine,
            loaded_camera,
        ),
        [9, -2, 16],
    )

    group = open_group(datastore.tile_scene_path).validate(strict=True)
    scene = group.ome_metadata(strict=True)
    transforms = {t.name: t for t in scene.scene.coordinateTransformations}
    stage_model = transforms["tile0000/round001/stage"]
    camera_model = transforms["tile0000/round001/camera"]
    global_model = transforms["tile0000/global"]
    grid_spacing = transforms["tile0000/decoder_spacing"]
    grid_origin = transforms["tile0000/decoder_origin"]
    assert isinstance(stage_model, v06.TranslationTransformation)
    assert isinstance(camera_model, v06.AffineTransformation)
    assert isinstance(global_model, v06.AffineTransformation)
    assert isinstance(grid_spacing, v06.ScaleTransformation)
    assert isinstance(grid_origin, v06.TranslationTransformation)
    np.testing.assert_array_equal(stage_model.translation, stage)
    np.testing.assert_array_equal(camera_model.affine, camera[:3])
    np.testing.assert_array_equal(global_model.affine, global_affine[:3])
    np.testing.assert_array_equal(grid_spacing.scale, global_spacing)
    np.testing.assert_array_equal(grid_origin.translation, global_origin)
    assert grid_spacing.output == grid_origin.input
    assert grid_origin.output == camera_model.input
    assert camera_model.output == global_model.input
