"""Signal screening against known backgrounds and optical point sources."""

import numpy as np
import pytest

from merfish3danalysis.utils.imageprocessing import image_has_signal


@pytest.mark.unit
@pytest.mark.parametrize(
    "background", ["constant", "gaussian", "poisson", "low_poisson"]
)
@pytest.mark.parametrize("signal", [False, True])
def test_signal_screen_distinguishes_spots_from_background(background, signal):
    rng = np.random.default_rng(4)
    shape = (4, 64, 96)
    if background == "constant":
        image = np.full(shape, 100.0)
    elif background == "gaussian":
        image = rng.normal(100, 5, shape)
    else:
        image = rng.poisson(10 if background == "poisson" else 0.2, shape).astype(float)
    if signal:
        y, x = np.mgrid[:64, :96]
        image[2] += 100 * np.exp(-((y - 30) ** 2 + (x - 40) ** 2) / (2 * 1.5**2))
    assert image_has_signal(image) is signal


@pytest.mark.unit
def test_signal_screen_rejects_hot_pixels_and_uniform_bright_images():
    image = np.full((3, 64, 96), 500, dtype=np.uint16)
    assert not image_has_signal(image)
    image[:, 20, 30] = 65535
    assert not image_has_signal(image)
    image[:, 40, 60] = 65535
    assert not image_has_signal(image)


@pytest.mark.unit
def test_signal_screen_respects_spatial_mask_and_intensity_units():
    y, x = np.mgrid[:64, :96]
    image = 100 + 100 * np.exp(-((y - 30) ** 2 + (x - 40) ** 2) / (2 * 1.5**2))
    mask = np.zeros((64, 96), dtype=bool)
    mask[:, 60:] = True
    assert not image_has_signal(image, mask=mask)
    mask[:, :60] = True
    assert image_has_signal(image, mask=mask)
    assert image_has_signal(0.46 * image + 1000, mask=mask)
    assert not image_has_signal(image, mask=np.zeros_like(mask))


@pytest.mark.unit
def test_signal_screen_rejects_empty_or_invalid_input():
    assert not image_has_signal(np.empty((0, 64, 96)))
    with pytest.raises(ValueError, match="finite"):
        image_has_signal(np.full((64, 96), np.nan))
    with pytest.raises(ValueError, match="YX or ZYX"):
        image_has_signal(np.ones(16))


@pytest.mark.unit
@pytest.mark.parametrize("background", ["gaussian", "poisson", "low_poisson"])
def test_signal_screen_rejects_camera_sized_noise(background):
    rng = np.random.default_rng(5)
    shape = (4, 2048, 2048)
    if background == "gaussian":
        image = rng.normal(100, 5, shape).astype(np.float32)
    else:
        image = rng.poisson(10 if background == "poisson" else 0.2, shape).astype(
            np.uint16
        )
    assert not image_has_signal(image)
