from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import numpy as np
from tifffile import TiffWriter, imread

from merfish3danalysis.DataRegistration import (
    _read_fiducial_sim,
    _write_zarr_max_projection_tiff,
)


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


class _ProjectionMetadata:
    dimension_names = ("t", "c", "z", "y", "x")


class _ProjectionArray:
    def __init__(self, data: np.ndarray, chunks: tuple[int, ...]) -> None:
        self._data = data
        self.shape = data.shape
        self.ndim = data.ndim
        self.dtype = data.dtype
        self.chunks = chunks
        self.metadata = _ProjectionMetadata()
        self.selections: list[tuple[int | slice, ...]] = []

    def __getitem__(self, selection: tuple[int | slice, ...]) -> np.ndarray:
        self.selections.append(selection)
        return self._data[selection]


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


def test_write_zarr_max_projection_tiff_streams_spatial_tiles(
    tmp_path: Path,
) -> None:
    data = np.arange(1 * 1 * 5 * 19 * 21, dtype=np.uint16).reshape(
        1,
        1,
        5,
        19,
        21,
    )
    array = _ProjectionArray(data, chunks=(1, 1, 2, 8, 8))
    output_path = tmp_path / "projection.ome.tiff"

    _write_zarr_max_projection_tiff(
        array=array,
        filename_path=output_path,
        spacing_zyx_um=np.asarray((0.32, 0.098, 0.098), dtype=np.float32),
        TiffWriter=TiffWriter,
        tile_shape_yx=(16, 16),
    )

    np.testing.assert_array_equal(imread(output_path), data.max(axis=2).squeeze())
    assert len(array.selections) == 12
    for selection in array.selections:
        z_slice = selection[2]
        y_slice = selection[3]
        x_slice = selection[4]
        assert isinstance(z_slice, slice)
        assert isinstance(y_slice, slice)
        assert isinstance(x_slice, slice)
        assert z_slice.stop - z_slice.start <= 2
        assert y_slice.stop - y_slice.start <= 16
        assert x_slice.stop - x_slice.start <= 16
