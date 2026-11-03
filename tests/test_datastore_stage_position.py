"""Stage and camera transforms persist as NGFF Scene relationships."""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pytest
from yaozarrs import open_group, v06

from merfish3danalysis.qi2labDataStore import qi2labDataStore


@pytest.mark.unit
@pytest.mark.parametrize(
    "position, expected",
    [
        ([125.5, -34.25], [0, 125.5, -34.25]),
        ([7.5, 125.5, -34.25], [7.5, 125.5, -34.25]),
    ],
)
def test_stage_and_camera_roundtrip_through_scene(image_datastore, position, expected):
    camera = np.eye(4, dtype=np.float32)
    camera[1, 1] = -1
    camera[2, 3] = 12.5
    assert image_datastore.load_local_stage_position_zyx_um(tile=0, round=0) is None
    image_datastore.save_local_stage_position_zyx_um(position, camera, tile=0, round=0)

    reopened = qi2labDataStore(image_datastore.datastore_path)
    stage, actual_camera = reopened.load_local_stage_position_zyx_um(tile=0, round=0)
    np.testing.assert_array_equal(stage, expected)
    np.testing.assert_array_equal(actual_camera, camera)
    scene = (
        open_group(reopened.tile_scene_path)
        .validate(strict=True)
        .ome_metadata(strict=True)
    )
    transforms = {
        transform.name: transform for transform in scene.scene.coordinateTransformations
    }
    assert isinstance(
        transforms["tile0000/round001/stage"], v06.TranslationTransformation
    )
    assert isinstance(transforms["tile0000/round001/camera"], v06.AffineTransformation)
    assert isinstance(transforms["tile0000/round002/stage"], v06.IdentityTransformation)


@pytest.mark.unit
@pytest.mark.parametrize("position", [[], [1], [1, 2, 3, 4], [[1, 2, 3]]])
def test_invalid_stage_position_does_not_publish(image_datastore, position):
    with pytest.raises(ValueError, match=r"two .* or three"):
        image_datastore.save_local_stage_position_zyx_um(
            position, np.eye(4), tile=0, round=0
        )
    assert image_datastore.load_local_stage_position_zyx_um(tile=0, round=0) is None


@pytest.mark.unit
def test_concurrent_tile_stage_writes_preserve_both_transforms(image_datastore):
    camera = np.eye(4, dtype=np.float32)
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        futures = [
            pool.submit(
                image_datastore.save_local_stage_position_zyx_um,
                position,
                camera,
                tile=tile,
                round=0,
            )
            for tile, position in enumerate(([1, 2, 3], [4, 5, 6]))
        ]
        for future in futures:
            future.result()

    reopened = qi2labDataStore(image_datastore.datastore_path)
    for tile, expected in enumerate(([1, 2, 3], [4, 5, 6])):
        stage, actual_camera = reopened.load_local_stage_position_zyx_um(
            tile=tile, round=0
        )
        np.testing.assert_array_equal(stage, expected)
        np.testing.assert_array_equal(actual_camera, camera)
    open_group(reopened.tile_scene_path).validate(strict=True)
