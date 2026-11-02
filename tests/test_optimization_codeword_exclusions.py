from pathlib import Path
from unittest.mock import Mock

import cupy as cp
import numpy as np
import pandas as pd
import pytest
import typer

import merfish3danalysis.PixelDecoder as pixel_decoder_module
from merfish3danalysis.cli.qi2lab_microscopes.pixeldecode import (
    _load_optimization_exclusions_file,
    _optimization_exclusions_path,
    decode_pixels,
)
from merfish3danalysis.PixelDecoder import PixelDecoder
from merfish3danalysis.qi2labDataStore import qi2labDataStore


@pytest.fixture
def decoder(image_datastore) -> PixelDecoder:
    image_datastore.codebook = pd.DataFrame(
        [["GeneA", 1, 1, 1, 0], ["GeneB", 1, 1, 0, 1], ["GeneC", 1, 0, 1, 1]],
        columns=["gene_id", "bit01", "bit02", "bit03", "bit04"],
    )
    return PixelDecoder(image_datastore, merfish_bits=4, num_gpus=1, verbose=0)


@pytest.mark.unit
def test_load_optimization_exclusions_file_ignores_comments_and_blank_lines(
    tmp_path: Path,
) -> None:
    exclusions_path = tmp_path / "bad_codewords.txt"
    exclusions_path.write_text("# known failures\n\n GeneB \nGeneC\n")

    assert _load_optimization_exclusions_file(exclusions_path) == ["GeneB", "GeneC"]


@pytest.mark.unit
def test_load_optimization_exclusions_file_rejects_empty_input(
    tmp_path: Path,
) -> None:
    exclusions_path = tmp_path / "bad_codewords.txt"
    exclusions_path.write_text("\n# no entries\n")

    with pytest.raises(typer.BadParameter, match="contains no gene IDs"):
        _load_optimization_exclusions_file(exclusions_path)


@pytest.mark.unit
def test_relative_exclusions_file_resolves_inside_datastore(tmp_path: Path) -> None:
    datastore_path = tmp_path / "qi2labdatastore"

    assert (
        _optimization_exclusions_path(
            datastore_path,
            Path("bad_codewords.txt"),
        )
        == datastore_path / "bad_codewords.txt"
    )


@pytest.mark.unit
def test_absolute_exclusions_file_is_preserved(tmp_path: Path) -> None:
    exclusions_path = tmp_path / "bad_codewords.txt"

    assert (
        _optimization_exclusions_path(tmp_path / "qi2labdatastore", exclusions_path)
        == exclusions_path
    )


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["skip", "reprocess"])
def test_cli_rejects_exclusions_when_optimization_will_not_run(
    tmp_path: Path,
    mode: str,
) -> None:
    exclusions_path = tmp_path / "bad_codewords.txt"

    kwargs = {
        "skip_optimization": mode == "skip",
        "reprocess_existing": mode == "reprocess",
    }
    with pytest.raises(typer.BadParameter, match="cannot be used"):
        decode_pixels(
            root_path=tmp_path,
            optimization_exclusions_file=exclusions_path,
            **kwargs,
        )


@pytest.mark.unit
@pytest.mark.gpu
def test_resolve_exclusions_is_exact_deduplicated_and_index_stable(decoder) -> None:
    gene_ids, indices = decoder._resolve_excluded_gene_ids([" GeneB ", "GeneB"])

    assert gene_ids == ("GeneB",)
    assert indices == (1,)
    with pytest.raises(ValueError, match="case-sensitive"):
        decoder._resolve_excluded_gene_ids(["geneb"])


@pytest.mark.unit
@pytest.mark.gpu
def test_resolve_exclusions_rejects_removing_entire_codebook(decoder) -> None:
    with pytest.raises(ValueError, match="every codeword"):
        decoder._resolve_excluded_gene_ids(["GeneA", "GeneB", "GeneC"])


@pytest.mark.unit
def test_excluded_winner_becomes_background_without_index_fallback() -> None:
    decoded = np.asarray([0, 1, 2, 1, -1], dtype=np.int16)
    nearest = np.asarray([0, 1, 2, 1, 1], dtype=np.int16)

    PixelDecoder._suppress_excluded_codeword_assignments(decoded, nearest, (1,))

    np.testing.assert_array_equal(decoded, np.asarray([0, -1, 2, -1, -1]))


@pytest.mark.unit
@pytest.mark.gpu
def test_exclusion_indices_are_converted_for_array_module() -> None:
    decoded = cp.asarray([0, 1, 2], dtype=cp.int16)
    nearest = cp.asarray([0, 1, 2], dtype=cp.int16)

    PixelDecoder._suppress_excluded_codeword_assignments(decoded, nearest, (1,))

    assert decoded.dtype == nearest.dtype
    np.testing.assert_array_equal(cp.asnumpy(decoded), np.asarray([0, -1, 2]))


@pytest.mark.unit
@pytest.mark.parametrize("z_support", [1, 3])
def test_plane_wise_centroid_statistics_match_known_moments(z_support) -> None:
    labels = np.asarray(
        [
            [[0, 1, 0], [0, 0, 2]],
            [[0, 0, 0], [3, 0, 0]],
            [[0, 1, 0], [0, 0, 2]],
        ],
        dtype=np.int32,
    )
    intensity = np.arange(1, labels.size + 1, dtype=np.float32).reshape(labels.shape)
    intensity[0, 0, 0] = -5.0
    minlength = int(labels.max()) + 1

    observed = PixelDecoder._plane_wise_weighted_centroid_statistics(
        labels,
        intensity,
        z_support=z_support,
        minlength=minlength,
    )

    # Columns are background, labels 1, 2, 3; rows are summed intensity,
    # intensity-weighted Z/Y/X moments, and peak inside the original labels.
    # For support=3, label 1 receives intensities 2+8+14=24 and Z moment
    # 0*2+1*8+2*14=36. Negative background contributes zero weight.
    expected = (
        [
            [120, 16, 24, 10],
            [169, 28, 36, 10],
            [65, 0, 24, 10],
            [119, 16, 48, 0],
            [17, 14, 18, 10],
        ]
        if z_support == 1
        else [
            [80, 24, 36, 30],
            [117, 36, 48, 42],
            [33, 0, 36, 30],
            [87, 24, 72, 0],
            [17, 14, 18, 10],
        ]
    )

    for observed_values, expected_values in zip(observed, expected, strict=True):
        np.testing.assert_allclose(observed_values, expected_values)


@pytest.mark.unit
@pytest.mark.parametrize("label_ids", [(1, 2), (2, 1)])
def test_centroid_support_is_nearest_and_independent_of_label_numbers(label_ids):
    labels = np.zeros((5, 1, 1), dtype=np.int32)
    labels[1, 0, 0], labels[3, 0, 0] = label_ids
    intensity = np.array([1, 2, 8, 16, 32], dtype=np.float32).reshape(5, 1, 1)

    sums, z_moments, y_moments, x_moments, peaks = (
        PixelDecoder._plane_wise_weighted_centroid_statistics(
            labels, intensity, z_support=5, minlength=3
        )
    )

    # Own labeled voxels cannot be stolen. Planes 0/4 belong to the nearest
    # object; plane 2 is equidistant and must not favor either numeric label.
    order = [0, *label_ids]
    np.testing.assert_array_equal(sums[order], [8, 3, 48])
    np.testing.assert_array_equal(z_moments[order], [16, 2, 176])
    np.testing.assert_array_equal(y_moments, 0)
    np.testing.assert_array_equal(x_moments, 0)
    np.testing.assert_array_equal(peaks[order], [32, 2, 16])


@pytest.mark.unit
def test_centroid_support_does_not_resolve_a_near_tie_using_a_farther_label():
    labels = np.array([3, 1, 0, 2, 3], dtype=np.int32).reshape(5, 1, 1)
    intensity = np.array([0, 0, 8, 0, 0], dtype=np.float32).reshape(5, 1, 1)

    sums, z_moments, *_ = PixelDecoder._plane_wise_weighted_centroid_statistics(
        labels, intensity, z_support=5, minlength=4
    )

    # The center is tied between labels 1 and 2 one plane away. Label 3,
    # two planes away on both sides, cannot claim those eight photons.
    np.testing.assert_array_equal(sums, [8, 0, 0, 0])
    np.testing.assert_array_equal(z_moments, [16, 0, 0, 0])


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize("normalization_features", ["all", "cells"])
def test_optimizer_passes_resolved_exclusions_to_gpu_worker(
    monkeypatch: pytest.MonkeyPatch,
    decoder,
    normalization_features: str,
) -> None:
    for tile in range(2):
        fiducial = np.full((1, 32, 48), 100, dtype=np.uint16)
        if tile == 1:
            fiducial[:, 12:20, 18:26] = 1000
        decoder._datastore.save_local_corrected_image(fiducial, tile=tile, round=0)
    monkeypatch.setattr(
        pixel_decoder_module, "sample", lambda values, count: list(values)
    )
    decoder._normalization_features = normalization_features
    decoder._global_normalization_vector = np.ones(4, dtype=np.float32)
    decoder._cleanup = Mock()
    decoder._load_all_barcodes = Mock()
    decoder._iterative_normalization_vectors = Mock()
    decoder._load_global_normalization_vectors = Mock()
    start_worker = Mock()
    monkeypatch.setattr(
        pixel_decoder_module,
        "_start_gpu_worker_process",
        start_worker,
    )
    monkeypatch.setattr(pixel_decoder_module, "_join_gpu_workers", Mock())

    decoder.optimize_normalization_by_decoding(
        n_random_tiles=1,
        n_iterations=1,
        minimum_pixels=1,
        excluded_gene_ids=["GeneB", "GeneB"],
    )

    start_worker.assert_called_once()
    assert decoder._load_global_normalization_vectors.call_args_list[0].kwargs[
        "tile_indices"
    ] == [1]
    args = start_worker.call_args.kwargs["args"]
    assert args[0] == decoder._datastore.datastore_path
    assert args[1] == [1]
    assert args[-2] == ("GeneB",)
    assert args[-1] == normalization_features
    assert (
        decoder._iterative_normalization_metadata()["normalization_features"]
        == normalization_features
    )


@pytest.mark.unit
def test_run_scoped_normalization_metadata_round_trips(image_datastore) -> None:
    datastore = image_datastore
    metadata = {
        "scope": "iterative_optimization",
        "excluded_gene_ids": ["GeneB"],
        "codebook_sha256": "abc123",
    }

    datastore.save_decode_normalization_vectors(
        "run1",
        "iterative",
        np.ones(4, dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        decode_mode="3d",
        metadata=metadata,
    )

    reopened = qi2labDataStore(datastore.datastore_path, validate=False)
    assert reopened.load_decode_normalization_metadata("run1", "iterative") == metadata
    norm, background = reopened.load_decode_normalization_vectors("run1", "iterative")
    np.testing.assert_array_equal(norm, np.ones(4))
    np.testing.assert_array_equal(background, np.zeros(4))
