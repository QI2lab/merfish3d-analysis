from pathlib import Path

import numpy as np
import pytest
from tifffile import TiffFile, imread

from merfish3danalysis.DataRegistration import (
    _read_fiducial_sim,
    _write_zarr_max_projection_tiff,
)


@pytest.mark.unit
def test_fiducial_view_preserves_generated_pixels_and_physical_coordinates(
    image_datastore,
):
    from multiview_stitcher import spatial_image_utils as si_utils

    pixels = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    image_datastore.save_local_corrected_image(pixels, tile=0, round=0)
    array = image_datastore.load_local_fusion_image(0, round=0)
    affine = np.diag([1.0, -1.0, 1.0, 1.0])
    affine[2, 3] = 7
    sim = _read_fiducial_sim(
        array=array,
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
    assert array.metadata.dimension_names == tuple("zyx")
    np.testing.assert_array_equal(
        image_datastore.load_local_corrected_image(0, round=0, return_future=False),
        pixels,
    )


@pytest.mark.unit
def test_write_zarr_max_projection_tiff_streams_spatial_tiles(
    image_datastore,
    tmp_path: Path,
) -> None:
    expected = np.arange(19 * 21, dtype=np.uint16).reshape(19, 21) + 10
    data = np.zeros((1, 1, 5, 19, 21), dtype=np.uint16)
    yy, xx = np.indices(expected.shape)
    data[0, 0, (yy + xx) % 5, yy, xx] = expected
    image_datastore.save_global_fiducial_image(
        data, np.eye(4), [0, 0, 0], [0.32, 0.098, 0.098]
    )
    array, _ = image_datastore.load_global_fused_image(return_future=None)
    output_path = tmp_path / "projection.ome.tiff"

    _write_zarr_max_projection_tiff(
        array=array,
        dims=array.domain.labels,
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
