import numpy as np

from merfish3danalysis.PixelDecoder import PixelDecoder


def test_warp_pixel_applies_camera_to_stage_affine_before_global_affine() -> None:
    pixel = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    spacing = np.array([0.32, 0.098, 0.098], dtype=np.float32)
    origin = np.array([2761.3, 107.81, 0.0], dtype=np.float32)
    camera_to_stage = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -0.07, -1.0, 0.0],
            [0.0, -1.0, 0.07, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    global_affine = np.array(
        [
            [1.0, 0.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, 1.25],
            [0.0, 0.0, 1.0, -2.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    physical = pixel * spacing + origin
    camera_space = (camera_to_stage @ np.array([*physical, 1.0]))[:3]
    expected = (global_affine @ np.array([*camera_space, 1.0]))[:3]

    observed = PixelDecoder._warp_pixel(
        pixel,
        spacing,
        origin,
        global_affine,
        camera_to_stage,
    )

    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-5)
