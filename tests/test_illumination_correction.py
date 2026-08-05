import sys
from types import SimpleNamespace

import numpy as np
import pytest

from merfish3danalysis.utils.imageprocessing import estimate_shading


class _FutureImage:
    def __init__(self, image: np.ndarray) -> None:
        self._image = image

    def result(self) -> np.ndarray:
        return self._image


def test_estimate_shading_uses_half_resolution_basic_working_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

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
    shading = estimate_shading([_FutureImage(image), _FutureImage(image)])

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
