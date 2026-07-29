from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import numpy as np

from merfish3danalysis.DataRegistration import _read_fiducial_sim


class _FakeMetadata:
    dimension_names = ("z", "y", "x")

    @staticmethod
    def to_dict() -> dict:
        return {
            "zarr_format": 3,
            "shape": [2, 3, 4],
            "dimension_names": ["z", "y", "x"],
        }


class _FakeSourceArray:
    metadata = _FakeMetadata()
    ndim = 3
    store_path = sentinel.store_path


class _FakeAsyncArray:
    def __init__(self, metadata, store_path) -> None:
        self.metadata = metadata
        self.store_path = store_path


class _FakeArray:
    def __init__(self, async_array) -> None:
        self.async_array = async_array


def test_read_fiducial_sim_builds_zarr_backed_three_dimensional_view() -> None:
    open_array = Mock(return_value=_FakeSourceArray())
    get_sim_from_array = Mock(return_value=sentinel.spatial_image)
    affine_zyx_px = np.eye(4)

    result = _read_fiducial_sim(
        input_path=Path("/unused/input.ome.zarr"),
        scale={"z": 2.0, "y": 0.5, "x": 0.5},
        translation={"z": 3.0, "y": 4.0, "x": 5.0},
        affine_zyx_px=affine_zyx_px,
        transform_key="stage_metadata",
        zarr_module=SimpleNamespace(
            open_array=open_array,
            AsyncArray=_FakeAsyncArray,
            Array=_FakeArray,
        ),
        si_utils=SimpleNamespace(get_sim_from_array=get_sim_from_array),
    )

    assert result is sentinel.spatial_image
    open_array.assert_called_once_with(
        Path("/unused/input.ome.zarr/0"),
        mode="r",
    )
    zarr_view = get_sim_from_array.call_args.args[0]
    assert isinstance(zarr_view, _FakeArray)
    assert zarr_view.async_array.store_path is sentinel.store_path
    assert "dimension_names" not in zarr_view.async_array.metadata
    get_sim_from_array.assert_called_once_with(
        zarr_view,
        dims=("z", "y", "x"),
        scale={"z": 2.0, "y": 0.5, "x": 0.5},
        translation={"z": 3.0, "y": 4.0, "x": 5.0},
        affine=affine_zyx_px,
        transform_key="stage_metadata",
        c_coords=None,
        t_coords=None,
    )
