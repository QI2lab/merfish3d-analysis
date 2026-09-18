from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from tifffile import TiffFile, imread

from merfish3danalysis.DataRegistration import (
    _read_fiducial_sim,
    _write_zarr_max_projection_tiff,
)


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


@pytest.mark.integration
@pytest.mark.parametrize("storage_dims", [("z", "y", "x"), ("y", "x", "z")])
def test_fiducial_view_preserves_generated_pixels_and_physical_coordinates(
    monkeypatch, storage_dims
):
    import yaozarrs
    import zarr
    from multiview_stitcher import spatial_image_utils as si_utils

    pixels = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    axes = tuple("zyx".index(dim) for dim in storage_dims)
    source = zarr.create_array(
        store=zarr.storage.MemoryStore(),
        data=pixels.transpose(axes),
        dimension_names=storage_dims,
        zarr_format=3,
    )
    monkeypatch.setattr(
        yaozarrs,
        "open_group",
        Mock(return_value={"0": SimpleNamespace(to_zarr_python=lambda: source)}),
    )
    affine = np.diag([1.0, -1.0, 1.0, 1.0])
    affine[2, 3] = 7
    sim = _read_fiducial_sim(
        input_path=Path("/mock/fiducial.ome.zarr"),
        scale={"z": 2.00039, "y": 0.49989, "x": 0.50041},
        translation={"z": 3.0, "y": 4.0, "x": 5.0},
        affine_zyx_px=affine,
        transform_key="stage_metadata",
    )
    np.testing.assert_array_equal(
        sim.transpose("t", "c", "z", "y", "x"), pixels[None, None]
    )
    np.testing.assert_allclose(sim.z, [3.0, 5.0])
    np.testing.assert_allclose(sim.y, [4.0, 4.5, 5.0])
    np.testing.assert_allclose(sim.x, [5.0, 5.5, 6.0, 6.5])
    stored_affine = si_utils.get_affine_from_sim(sim, "stage_metadata").values.squeeze()
    np.testing.assert_allclose(
        stored_affine @ [5.0, 4.5, 6.0, 1.0], [5.0, -4.5, 13.0, 1.0]
    )
    # Reading must leave the source axes and bytes unchanged.
    assert source.metadata.dimension_names == storage_dims
    np.testing.assert_array_equal(source[:], pixels.transpose(axes))


@pytest.mark.integration
def test_write_zarr_max_projection_tiff_streams_spatial_tiles(
    tmp_path: Path,
) -> None:
    expected = np.arange(19 * 21, dtype=np.uint16).reshape(19, 21) + 10
    data = np.zeros((1, 1, 5, 19, 21), dtype=np.uint16)
    yy, xx = np.indices(expected.shape)
    data[0, 0, (yy + xx) % 5, yy, xx] = expected
    array = _ProjectionArray(data, chunks=(1, 1, 2, 8, 8))
    output_path = tmp_path / "projection.ome.tiff"

    _write_zarr_max_projection_tiff(
        array=array,
        filename_path=output_path,
        spacing_zyx_um=np.asarray((0.32, 0.098, 0.098), dtype=np.float32),
        tile_shape_yx=(16, 16),
    )

    np.testing.assert_array_equal(imread(output_path), expected)
    with TiffFile(output_path) as tiff:
        from xml.etree import ElementTree

        pixels = ElementTree.fromstring(tiff.ome_metadata).find(".//{*}Pixels")
        assert pixels.attrib["PhysicalSizeX"] == "0.098"
        assert pixels.attrib["PhysicalSizeY"] == "0.098"
        assert pixels.attrib["PhysicalSizeXUnit"] == "µm"
        assert pixels.attrib["PhysicalSizeYUnit"] == "µm"
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
