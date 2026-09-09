from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon
from typer.testing import CliRunner

import merfish3danalysis.PixelDecoder as pixel_decoder_module
from merfish3danalysis.cli.qi2lab_microscopes import pixeldecode
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
        load_global_cellpose_outlines=Mock(
            side_effect=AssertionError("must not restore outlines from legacy JSON")
        ),
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


@pytest.mark.parametrize(
    "features, expected",
    [
        ("cells", ["inside", "boundary"]),
        ("all", ["inside", "outside", "boundary", "outside_2"]),
    ],
)
def test_iterative_normalization_selects_requested_features(
    tmp_path: Path,
    features: str,
    expected: list[str],
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
    decoder._normalization_features = features
    decoder._normalization_cell_polygons = [
        Polygon(((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)))
    ]

    decoder._load_all_barcodes()

    assert decoder._df_barcodes_loaded["gene_id"].tolist() == expected


def test_all_features_bypasses_mask_loading_and_selects_all_voxels():
    decoder = PixelDecoder.__new__(PixelDecoder)
    decoder._normalization_features = "all"
    decoder._datastore = SimpleNamespace(
        load_global_cellpose_roi_zip=Mock(
            side_effect=AssertionError("must not read masks")
        ),
        load_global_cellpose_outlines=Mock(
            side_effect=AssertionError("must not read masks")
        ),
    )
    mask = decoder._normalization_cell_mask_for_tile(
        tile_id="tile0000", image_shape_zyx=(2, 3, 3)
    )
    assert mask is None
    image = np.arange(18).reshape(2, 3, 3)
    np.testing.assert_array_equal(
        decoder._normalization_pixels(image, mask), np.arange(18)
    )


def test_cells_without_segmentation_uses_all_features(tmp_path):
    decoder = PixelDecoder.__new__(PixelDecoder)
    decoder._normalization_features = "cells"
    decoder._datastore = SimpleNamespace(
        _segmentation_root_path=tmp_path,
        load_global_cellpose_roi_zip=Mock(return_value=None),
        load_global_cellpose_outlines=Mock(return_value=None),
    )
    assert (
        decoder._normalization_cell_mask_for_tile(
            tile_id="tile0000", image_shape_zyx=(2, 3, 3)
        )
        is None
    )
    barcodes = pd.DataFrame({"gene_id": ["GeneA"]})
    assert decoder._restrict_barcodes_to_segmented_cells(barcodes) is barcodes


@pytest.mark.parametrize("features", [None, "cells", "all"])
def test_cli_passes_normalization_feature_selection(monkeypatch, tmp_path, features):
    datastore = SimpleNamespace(microscope_type="3D", num_bits=4)
    decoder = Mock(spec=PixelDecoder)
    factory = Mock(return_value=decoder)
    monkeypatch.setattr(pixeldecode, "qi2labDataStore", Mock(return_value=datastore))
    monkeypatch.setattr(pixeldecode, "PixelDecoder", factory)
    args = [str(tmp_path)]
    if features is not None:
        args.extend(["--normalization-features", features])
    result = CliRunner().invoke(pixeldecode.app, args)
    assert result.exit_code == 0, result.output
    assert factory.call_args.kwargs["normalization_features"] == (features or "cells")
    decoder.optimize_normalization_by_decoding.assert_called_once()
    decoder.decode_all_tiles.assert_called_once()


def test_cli_rejects_unknown_normalization_features(monkeypatch, tmp_path):
    datastore_factory = Mock()
    monkeypatch.setattr(pixeldecode, "qi2labDataStore", datastore_factory)
    result = CliRunner().invoke(
        pixeldecode.app, [str(tmp_path), "--normalization-features", "invalid"]
    )
    assert result.exit_code == 2
    datastore_factory.assert_not_called()


@pytest.fixture
def normalization_cache(monkeypatch):
    pool = SimpleNamespace(free_all_blocks=lambda: None)
    fake_cp = SimpleNamespace(
        asarray=np.asarray,
        cuda=SimpleNamespace(
            Device=lambda _id: nullcontext(),
            Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda: None)),
        ),
        get_default_memory_pool=lambda: pool,
        get_default_pinned_memory_pool=lambda: pool,
    )
    monkeypatch.setattr(pixel_decoder_module, "cp", fake_cp)
    decoder = PixelDecoder.__new__(PixelDecoder)
    decoder._decode_run_key = None
    decoder._datastore = SimpleNamespace(
        load_decode_normalization_vectors=Mock(return_value=(np.ones(4), np.zeros(4))),
        load_decode_normalization_metadata=Mock(),
    )
    decoder._global_normalization_vectors = Mock()
    return decoder


@pytest.mark.parametrize(
    "features, cached_features, matches",
    [
        ("all", "all", True),
        ("cells", "cells", True),
        ("cells", None, True),
        ("all", "cells", False),
        ("cells", "all", False),
        ("all", None, False),
    ],
)
def test_normalization_cache_respects_feature_selection(
    normalization_cache, features, cached_features, matches
):
    decoder = normalization_cache
    decoder._normalization_features = features
    decoder._datastore.load_decode_normalization_metadata.return_value = (
        {"normalization_features": cached_features}
        if cached_features is not None
        else None
    )
    decoder._load_global_normalization_vectors()
    assert decoder._global_normalization_vectors.called is not matches
    if matches:
        decoder._load_iterative_normalization_vectors()
        np.testing.assert_array_equal(
            decoder._iterative_normalization_vector, np.ones(4)
        )
    else:
        with pytest.raises(ValueError, match="without --skip-optimization"):
            decoder._load_iterative_normalization_vectors()
