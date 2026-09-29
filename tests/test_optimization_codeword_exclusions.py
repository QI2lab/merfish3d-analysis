from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
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


def _decoder_with_codebook() -> PixelDecoder:
    decoder = PixelDecoder.__new__(PixelDecoder)
    decoder._gene_ids = ["GeneA", "GeneB", "GeneC"]
    decoder._codebook_matrix = np.asarray(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
        ],
        dtype=np.int8,
    )
    return decoder


@pytest.mark.unit
def test_load_optimization_exclusions_file_ignores_comments_and_blank_lines(
    tmp_path: Path,
    monkeypatch,
) -> None:
    exclusions_path = tmp_path / "bad_codewords.txt"
    monkeypatch.setattr(
        Path, "read_text", Mock(return_value="# known failures\n\n GeneB \nGeneC\n")
    )

    assert _load_optimization_exclusions_file(exclusions_path) == ["GeneB", "GeneC"]


@pytest.mark.unit
def test_load_optimization_exclusions_file_rejects_empty_input(
    tmp_path: Path,
    monkeypatch,
) -> None:
    exclusions_path = tmp_path / "bad_codewords.txt"
    monkeypatch.setattr(Path, "read_text", Mock(return_value="\n# no entries\n"))

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
def test_resolve_exclusions_is_exact_deduplicated_and_index_stable() -> None:
    decoder = _decoder_with_codebook()

    gene_ids, indices = decoder._resolve_excluded_gene_ids([" GeneB ", "GeneB"])

    assert gene_ids == ("GeneB",)
    assert indices == (1,)
    with pytest.raises(ValueError, match="case-sensitive"):
        decoder._resolve_excluded_gene_ids(["geneb"])


@pytest.mark.unit
def test_resolve_exclusions_rejects_removing_entire_codebook() -> None:
    decoder = _decoder_with_codebook()

    with pytest.raises(ValueError, match="every codeword"):
        decoder._resolve_excluded_gene_ids(["GeneA", "GeneB", "GeneC"])


@pytest.mark.unit
def test_excluded_winner_becomes_background_without_index_fallback() -> None:
    decoded = np.asarray([0, 1, 2, 1, -1], dtype=np.int16)
    nearest = np.asarray([0, 1, 2, 1, 1], dtype=np.int16)

    PixelDecoder._suppress_excluded_codeword_assignments(decoded, nearest, (1,))

    np.testing.assert_array_equal(decoded, np.asarray([0, -1, 2, -1, -1]))


@pytest.mark.unit
def test_exclusion_indices_are_converted_for_array_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoded = np.asarray([0, 1, 2], dtype=np.int16)
    nearest = np.asarray([0, 1, 2], dtype=np.int16)
    converted = []

    class _CupyLikeArrayModule:
        @staticmethod
        def asarray(values, dtype=None):
            result = np.asarray(values, dtype=dtype)
            converted.append(result)
            return result

        @staticmethod
        def isin(element, test_elements):
            assert isinstance(test_elements, np.ndarray)
            return np.isin(element, test_elements)

    fake_cp = SimpleNamespace(
        get_array_module=lambda _array: _CupyLikeArrayModule,
    )
    monkeypatch.setattr(pixel_decoder_module, "cp", fake_cp)

    PixelDecoder._suppress_excluded_codeword_assignments(decoded, nearest, (1,))

    assert converted[0].dtype == nearest.dtype
    np.testing.assert_array_equal(decoded, np.asarray([0, -1, 2]))


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
@pytest.mark.parametrize("normalization_features", ["all", "cells"])
def test_optimizer_passes_resolved_exclusions_to_gpu_worker(
    monkeypatch: pytest.MonkeyPatch,
    normalization_features: str,
) -> None:
    decoder = _decoder_with_codebook()
    decoder._num_gpus = 1
    decoder._verbose = 0
    fiducials = np.full((3, 1, 32, 48), 100, dtype=np.uint16)
    fiducials[1:, :, 12:20, 18:26] = 1000
    decoder._datastore = SimpleNamespace(
        tile_ids=["tile000", "tile001", "tile002"],
        load_local_fiducial_image=lambda *, tile, round, return_future: fiducials[
            int(tile[-3:])
        ],
    )
    decoder._z_slice = slice(None)
    decoder._normalization_cell_mask_for_tile = lambda **kwargs: None
    monkeypatch.setattr(
        pixel_decoder_module, "sample", lambda values, count: list(values)
    )
    decoder._datastore_path = Path("/unused/datastore")
    decoder._decode_run_key = None
    decoder._decode_mode = "3d"
    decoder._effective_decode_mode = "3d"
    decoder._n_merfish_bits = 4
    decoder._estimate_chromatic_affines = False
    decoder._normalization_features = normalization_features
    decoder._is_3D = True
    decoder._cleanup = lambda: None
    decoder._load_all_barcodes = lambda: None
    decoder._iterative_normalization_vectors = lambda gpu_id=0: None

    global_calls = []

    def _load_global_normalization_vectors(**_kwargs) -> None:
        global_calls.append(_kwargs)
        decoder._global_background_vector = np.ones(4, dtype=np.float32)
        decoder._global_normalization_vector = np.ones(4, dtype=np.float32)

    decoder._load_global_normalization_vectors = _load_global_normalization_vectors

    captured_args = []

    def _capture_worker(*, target, args, physical_gpu_id):
        captured_args.append((target, args, physical_gpu_id))
        return SimpleNamespace()

    class _Device:
        def __init__(self, _gpu_id: int) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    memory_pool = SimpleNamespace(free_all_blocks=lambda: None)
    fake_cp = SimpleNamespace(
        cuda=SimpleNamespace(
            Device=_Device,
            Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda: None)),
        ),
        get_default_memory_pool=lambda: memory_pool,
        get_default_pinned_memory_pool=lambda: memory_pool,
    )
    monkeypatch.setattr(pixel_decoder_module, "cp", fake_cp)
    monkeypatch.setattr(
        pixel_decoder_module,
        "_start_gpu_worker_process",
        _capture_worker,
    )
    monkeypatch.setattr(pixel_decoder_module, "_join_gpu_workers", lambda *_args: None)

    decoder.optimize_normalization_by_decoding(
        n_random_tiles=2,
        n_iterations=1,
        minimum_pixels=1,
        excluded_gene_ids=["GeneB", "GeneB"],
    )

    assert len(captured_args) == 1
    assert global_calls[0]["tile_indices"] == [1, 2]
    assert captured_args[0][1][1] == [1, 2]
    assert captured_args[0][1][-2] == ("GeneB",)
    assert captured_args[0][1][-1] == normalization_features
    assert (
        decoder._iterative_normalization_metadata()["normalization_features"]
        == normalization_features
    )


@pytest.mark.unit
def test_run_scoped_normalization_metadata_round_trips() -> None:
    datastore = qi2labDataStore.__new__(qi2labDataStore)
    attributes: dict[str, object] = {}
    datastore._load_calibrations_attributes = lambda: dict(attributes)

    def _save(updated: dict[str, object]) -> None:
        attributes.clear()
        attributes.update(updated)

    datastore._save_calibrations_attributes = _save
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

    assert datastore.load_decode_normalization_metadata("run1", "iterative") == metadata
