import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from roifile import ImagejRoi, roiwrite
from typer.testing import CliRunner

from merfish3danalysis.cli.qi2lab_microscopes import pixeldecode
from merfish3danalysis.PixelDecoder import PixelDecoder


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize("reflected", [False, True])
def test_cell_mask_rasterizes_global_segmentation_into_tile_pixels(
    image_datastore, reflected
) -> None:
    datastore = image_datastore
    camera = np.diag([1, -1 if reflected else 1, 1, 1])
    datastore.save_local_stage_position_zyx_um(np.zeros(3), camera, tile=0, round=0)
    datastore.save_global_coord_xforms_um(np.eye(4), np.zeros(3), np.ones(3), tile=0)
    roi_path = (
        datastore.datastore_path
        / "segmentation/cellpose/imagej_rois/global_coords_rois.zip"
    )
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    sign = -1 if reflected else 1
    roi = ImagejRoi.frompoints([[1, sign], [3, sign], [3, 3 * sign], [1, 3 * sign]])
    roiwrite(roi_path, [roi], mode="w")
    decoder = PixelDecoder(datastore, merfish_bits=4, num_gpus=0, verbose=0)

    mask = decoder._normalization_cell_mask_for_tile(
        tile_id="tile0000",
        image_shape_zyx=(2, 6, 6),
    )

    expected = np.zeros((6, 6), dtype=bool)
    expected[1:4, 1:4] = True
    np.testing.assert_array_equal(mask, expected)


@pytest.mark.unit
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


@pytest.mark.unit
@pytest.mark.gpu
def test_present_empty_segmentation_does_not_fall_back_to_all_pixels(
    image_datastore,
) -> None:
    roi_path = (
        image_datastore.datastore_path
        / "segmentation/cellpose/imagej_rois/global_coords_rois.zip"
    )
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    roiwrite(roi_path, [], mode="w")
    (roi_path.parent.parent / "cell_outlines.json").write_text(
        json.dumps(
            {
                "features": [
                    {
                        "properties": {"cell_id": 1},
                        "geometry": {"coordinates": [[[0, 0], [2, 0], [2, 2], [0, 0]]]},
                    }
                ]
            }
        )
    )
    assert image_datastore.load_global_cellpose_outlines()
    decoder = PixelDecoder(image_datastore, merfish_bits=4, num_gpus=0, verbose=0)

    mask = decoder._normalization_cell_mask_for_tile(
        tile_id="tile0000",
        image_shape_zyx=(2, 4, 4),
    )

    assert mask is not None
    assert not mask.any()


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize(
    "features, expected",
    [
        ("cells", ["inside", "boundary"]),
        ("all", ["inside", "outside", "boundary", "outside_2"]),
    ],
)
def test_iterative_normalization_selects_requested_features(
    tmp_path: Path,
    image_datastore,
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

    roi_path = (
        image_datastore.datastore_path
        / "segmentation/cellpose/imagej_rois/global_coords_rois.zip"
    )
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    roiwrite(
        roi_path, [ImagejRoi.frompoints([[0, 0], [4, 0], [4, 4], [0, 4]])], mode="w"
    )
    decoder = PixelDecoder(
        image_datastore,
        merfish_bits=4,
        num_gpus=0,
        verbose=0,
        normalization_features=features,
    )
    decoder._optimize_normalization_weights = True
    decoder._temp_dir = tmp_path

    decoder._load_all_barcodes()

    assert decoder._df_barcodes_loaded["gene_id"].tolist() == expected


@pytest.mark.unit
@pytest.mark.gpu
def test_all_features_bypasses_mask_loading_and_selects_all_voxels(image_datastore):
    roi_path = (
        image_datastore.datastore_path
        / "segmentation/cellpose/imagej_rois/global_coords_rois.zip"
    )
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    roiwrite(
        roi_path, [ImagejRoi.frompoints([[0, 0], [1, 0], [1, 1], [0, 1]])], mode="w"
    )
    decoder = PixelDecoder(
        image_datastore,
        merfish_bits=4,
        num_gpus=0,
        verbose=0,
        normalization_features="all",
    )
    mask = decoder._normalization_cell_mask_for_tile(
        tile_id="tile0000", image_shape_zyx=(2, 3, 3)
    )
    assert mask is None
    image = np.arange(18).reshape(2, 3, 3)
    np.testing.assert_array_equal(
        decoder._normalization_pixels(image, mask), np.arange(18)
    )


@pytest.mark.unit
@pytest.mark.gpu
def test_cells_without_segmentation_uses_all_features(image_datastore):
    decoder = PixelDecoder(image_datastore, merfish_bits=4, num_gpus=0, verbose=0)
    assert (
        decoder._normalization_cell_mask_for_tile(
            tile_id="tile0000", image_shape_zyx=(2, 3, 3)
        )
        is None
    )
    barcodes = pd.DataFrame({"gene_id": ["GeneA"]})
    assert decoder._restrict_barcodes_to_segmented_cells(barcodes) is barcodes


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize("features", [None, "cells", "all"])
def test_cli_passes_normalization_feature_selection(
    monkeypatch, image_datastore, features
):
    factory = Mock(wraps=PixelDecoder)
    optimize = Mock()
    decode = Mock()
    monkeypatch.setattr(PixelDecoder, "optimize_normalization_by_decoding", optimize)
    monkeypatch.setattr(PixelDecoder, "decode_all_tiles", decode)
    monkeypatch.setattr(pixeldecode, "PixelDecoder", factory)
    args = [str(image_datastore.datastore_path)]
    if features is not None:
        args.extend(["--normalization-features", features])
    result = CliRunner().invoke(pixeldecode.app, args)
    assert result.exit_code == 0, result.output
    assert factory.call_args.kwargs["normalization_features"] == (features or "cells")
    optimize.assert_called_once()
    decode.assert_called_once()


@pytest.mark.unit
def test_cli_rejects_unknown_normalization_features(tmp_path):
    datastore_path = tmp_path / "datastore"
    result = CliRunner().invoke(
        pixeldecode.app, [str(datastore_path), "--normalization-features", "invalid"]
    )
    assert result.exit_code == 2
    assert not datastore_path.exists()


@pytest.fixture
def normalization_cache(image_datastore):
    decoder = PixelDecoder(image_datastore, merfish_bits=4, num_gpus=0, verbose=0)
    decoder._global_normalization_vectors = Mock()
    return decoder


@pytest.mark.unit
@pytest.mark.gpu
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
    metadata = (
        {"normalization_features": cached_features}
        if cached_features is not None
        else None
    )
    for kind in ("global", "iterative"):
        decoder._datastore.save_decode_normalization_vectors(
            None, kind, np.ones(4), np.zeros(4), metadata=metadata
        )
    decoder._load_global_normalization_vectors()
    assert decoder._global_normalization_vectors.called is not matches
    if matches:
        decoder._load_iterative_normalization_vectors()
        np.testing.assert_array_equal(
            decoder._iterative_normalization_vector.get(), np.ones(4)
        )
    else:
        with pytest.raises(ValueError, match="without --skip-optimization"):
            decoder._load_iterative_normalization_vectors()


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["global", "iterative"])
@pytest.mark.parametrize(
    "norm, background",
    [
        ([1, 0, 1, 1], [0, 0, 0, 0]),
        ([1, np.nan, 1, 1], [0, 0, 0, 0]),
        ([1, np.inf, 1, 1], [0, 0, 0, 0]),
        ([1, -1, 1, 1], [0, 0, 0, 0]),
        ([1, 1, 1, 1], [0, np.nan, 0, 0]),
        ([[1, 1, 1, 1]], [0, 0, 0, 0]),
        ([1, 1, 1], [0, 0, 0, 0]),
    ],
)
def test_invalid_cached_normalization_cannot_remain_loaded(
    normalization_cache, kind, norm, background
):
    decoder = normalization_cache
    decoder._n_merfish_bits = 4
    decoder._datastore.save_decode_normalization_vectors(
        None,
        kind,
        np.asarray(norm),
        np.asarray(background),
        metadata={"normalization_features": "cells"},
    )
    setattr(decoder, f"_{kind}_normalization_loaded", True)

    with pytest.raises(ValueError, match="normalization vectors"):
        getattr(decoder, f"_load_{kind}_normalization_vectors")()

    assert not getattr(decoder, f"_{kind}_normalization_loaded")
