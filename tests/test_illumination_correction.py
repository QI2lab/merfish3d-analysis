import numpy as np
import pytest

from merfish3danalysis.utils.imageprocessing import estimate_shading


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize(
    ("image_datastore", "height", "width"),
    [("shading_16", 64, 96), ("shading_100", 2048, 2048)],
    indirect=["image_datastore"],
    ids=["16-64-96", "100-2048-2048"],
)
def test_estimate_shading_recovers_known_illumination(image_datastore, height, width):
    """Fit real BaSiC at half resolution, including a full 100-image camera stack."""
    cp = pytest.importorskip("cupy")
    pytest.importorskip("basicpy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("requires CUDA")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("requires CUDA")

    y, x = np.mgrid[-1 : 1 : complex(height), -1 : 1 : complex(width)]
    illumination = (1 - 0.2 * y**2 - 0.15 * x**2).astype(np.float32)
    illumination /= illumination.max()

    for tile, intensity in enumerate(
        np.linspace(1000, 2500, image_datastore.num_tiles)
    ):
        image_datastore.save_local_corrected_image(
            np.stack([0.5 * intensity * illumination, intensity * illumination]).astype(
                np.uint16
            ),
            tile=tile,
            round=0,
        )
    shading = estimate_shading(
        image_datastore.load_local_corrected_image(tile=tile, round=0)
        for tile in range(image_datastore.num_tiles)
    )

    assert shading.shape == (height, width)
    assert shading.dtype == np.float32
    assert np.all(np.isfinite(shading)) and np.all(shading > 0)
    assert shading.max() == pytest.approx(1)
    np.testing.assert_allclose(shading, illumination, rtol=0, atol=0.025)
    # A uniform fluorescent specimen should become uniform after division.
    corrected = 1800 * illumination / shading
    assert corrected.std() / corrected.mean() < 0.01
