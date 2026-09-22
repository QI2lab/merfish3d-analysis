import sys
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from merfish3danalysis.utils.imageprocessing import estimate_shading


class _FutureImage:
    def __init__(self, image: np.ndarray) -> None:
        self._image = image

    def result(self) -> np.ndarray:
        return self._image


@pytest.mark.unit
@pytest.mark.parametrize("lazy", [False, True])
def test_estimate_shading_uses_half_resolution_basic_working_size(
    monkeypatch: pytest.MonkeyPatch,
    lazy: bool,
) -> None:
    calls = []
    live_images = []

    class _MemoryPool:
        def free_all_blocks(self) -> None:
            pass

    class _Stream:
        def synchronize(self) -> None:
            pass

    fake_cupy = SimpleNamespace(
        asnumpy=lambda images: np.asarray(images),
        max=np.max,
        squeeze=np.squeeze,
        cuda=SimpleNamespace(Stream=SimpleNamespace(null=_Stream())),
        get_default_memory_pool=lambda: _MemoryPool(),
        get_default_pinned_memory_pool=lambda: _MemoryPool(),
    )

    class _FakeBaSiC:
        def __init__(self, **kwargs) -> None:
            assert all(ref() is None for ref in live_images)
            calls.append(("init", kwargs))
            self.flatfield = np.ones((10, 14), dtype=np.float32)

        def autotune(self, images: np.ndarray) -> None:
            calls.append(("autotune", images.copy()))

        def fit(self, images: np.ndarray) -> None:
            calls.append(("fit", images.copy()))

    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(
        sys.modules,
        "basicpy",
        SimpleNamespace(BaSiC=_FakeBaSiC),
    )

    image = np.stack(
        [
            np.full((10, 14), 1, dtype=np.uint16),
            np.full((10, 14), 7, dtype=np.uint16),
            np.full((10, 14), 3, dtype=np.uint16),
        ]
    )

    def read_images():
        for _ in range(2):
            assert all(ref() is None for ref in live_images)
            pixels = image.copy()
            live_images.append(weakref.ref(pixels))
            yield _FutureImage(pixels)
            del pixels

    shading = estimate_shading(
        read_images() if lazy else [_FutureImage(image), _FutureImage(image)]
    )

    assert calls[0] == (
        "init",
        {
            "get_darkfield": False,
            "sort_intensity": True,
            "working_size": [5, 7],
        },
    )
    expected_max_projections = np.full((2, 10, 14), 7, dtype=np.uint16)
    assert calls[1][0] == "autotune"
    assert calls[2][0] == "fit"
    np.testing.assert_array_equal(calls[1][1], expected_max_projections)
    np.testing.assert_array_equal(calls[2][1], expected_max_projections)
    np.testing.assert_array_equal(shading, np.ones((10, 14), dtype=np.float32))


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("count,height,width", [(16, 64, 96), (100, 2048, 2048)])
def test_estimate_shading_recovers_known_illumination(count, height, width):
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
    images = (
        _FutureImage(
            np.stack([0.5 * intensity * illumination, intensity * illumination]).astype(
                np.uint16
            )
        )
        for intensity in np.linspace(1000, 2500, count)
    )

    shading = estimate_shading(images)

    assert shading.shape == (height, width)
    assert shading.dtype == np.float32
    assert np.all(np.isfinite(shading)) and np.all(shading > 0)
    assert shading.max() == pytest.approx(1)
    np.testing.assert_allclose(shading, illumination, rtol=0, atol=0.025)
    # A uniform fluorescent specimen should become uniform after division.
    corrected = 1800 * illumination / shading
    assert corrected.std() / corrected.mean() < 0.01
