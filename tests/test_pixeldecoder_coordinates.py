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
