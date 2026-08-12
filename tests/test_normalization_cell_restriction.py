from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from merfish3danalysis.PixelDecoder import PixelDecoder


def test_cell_mask_rasterizes_global_segmentation_into_tile_pixels() -> None:
    datastore = SimpleNamespace(
        load_global_cellpose_roi_zip=Mock(
            return_value={
                1: np.asarray(((1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)))
            }
        ),
        load_global_cellpose_outlines=Mock(return_value=None),
        load_local_stage_position_zyx_um=Mock(return_value=(np.zeros(3), np.eye(4))),
        load_global_coord_xforms_um=Mock(
            return_value=(np.eye(4), np.zeros(3), np.ones(3))
        ),
        voxel_size_zyx_um=np.ones(3),
    )
    decoder = PixelDecoder.__new__(PixelDecoder)
    decoder._datastore = datastore
    decoder._z_range = [0, None]

    mask = decoder._normalization_cell_mask_for_tile(
        tile_id="tile0000",
        image_shape_zyx=(2, 6, 6),
    )

    expected = np.zeros((6, 6), dtype=bool)
    expected[1:4, 1:4] = True
    np.testing.assert_array_equal(mask, expected)
    datastore.load_global_cellpose_roi_zip.assert_called_once_with()


def test_global_normalization_pixel_selection_excludes_noncell_voxels() -> None:
    image = np.asarray(
        [
            [[1000.0, 1000.0], [1000.0, 2.0]],
            [[1000.0, 1000.0], [1000.0, 4.0]],
        ],
        dtype=np.float32,
    )
    cell_mask = np.asarray([[False, False], [False, True]])

    selected = PixelDecoder._normalization_pixels(image, cell_mask)

    np.testing.assert_array_equal(selected, np.asarray((2.0, 4.0)))


def test_present_empty_segmentation_does_not_fall_back_to_all_pixels() -> None:
    datastore = SimpleNamespace(
        load_global_cellpose_roi_zip=Mock(return_value={}),
        load_global_cellpose_outlines=Mock(return_value={}),
        load_local_stage_position_zyx_um=Mock(return_value=(np.zeros(2), np.eye(4))),
        load_global_coord_xforms_um=Mock(
            return_value=(np.eye(4), np.zeros(2), np.ones(3))
        ),
        voxel_size_zyx_um=np.ones(3),
    )
    decoder = PixelDecoder.__new__(PixelDecoder)
    decoder._datastore = datastore
    decoder._z_range = [0, None]

    mask = decoder._normalization_cell_mask_for_tile(
        tile_id="tile0000",
        image_shape_zyx=(2, 4, 4),
    )

    assert mask is not None
    assert not mask.any()


def test_iterative_normalization_loads_only_transcripts_inside_cells(
    tmp_path: Path,
) -> None:
    first_tile = pd.DataFrame(
        {
            "gene_id": ["inside", "outside"],
            "distance_min": [0.1, 0.1],
            "global_x": [2.0, 20.0],
            "global_y": [2.0, 20.0],
        }
    )
    second_tile = pd.DataFrame(
        {
            "gene_id": ["boundary", "outside_2"],
            "distance_min": [0.1, 0.1],
            "global_x": [4.0, -2.0],
            "global_y": [2.0, -2.0],
        }
    )
    first_tile.to_parquet(tmp_path / "tile000_temp_decoded.parquet")
    second_tile.to_parquet(tmp_path / "tile001_temp_decoded.parquet")

    decoder = PixelDecoder.__new__(PixelDecoder)
    decoder._optimize_normalization_weights = True
    decoder._temp_dir = tmp_path
    decoder._verbose = 0
    decoder._normalization_cell_polygons = [
        Polygon(((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)))
    ]

    decoder._load_all_barcodes()

    assert decoder._df_barcodes_loaded["gene_id"].tolist() == [
        "inside",
        "boundary",
    ]
