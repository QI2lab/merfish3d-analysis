from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import typer
from scipy.ndimage import grey_dilation

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


def test_load_optimization_exclusions_file_ignores_comments_and_blank_lines(
    tmp_path: Path,
) -> None:
    exclusions_path = tmp_path / "bad_codewords.txt"
    exclusions_path.write_text(
        "# known failures\n\n GeneB \nGeneC\n",
        encoding="utf-8",
    )

    assert _load_optimization_exclusions_file(exclusions_path) == ["GeneB", "GeneC"]


def test_load_optimization_exclusions_file_rejects_empty_input(
    tmp_path: Path,
) -> None:
    exclusions_path = tmp_path / "bad_codewords.txt"
    exclusions_path.write_text("\n# no entries\n", encoding="utf-8")

    with pytest.raises(typer.BadParameter, match="contains no gene IDs"):
        _load_optimization_exclusions_file(exclusions_path)


def test_relative_exclusions_file_resolves_inside_datastore(tmp_path: Path) -> None:
    datastore_path = tmp_path / "qi2labdatastore"

    assert (
        _optimization_exclusions_path(
            datastore_path,
            Path("bad_codewords.txt"),
        )
        == datastore_path / "bad_codewords.txt"
    )


def test_absolute_exclusions_file_is_preserved(tmp_path: Path) -> None:
    exclusions_path = tmp_path / "bad_codewords.txt"

    assert (
        _optimization_exclusions_path(tmp_path / "qi2labdatastore", exclusions_path)
        == exclusions_path
    )


@pytest.mark.parametrize("mode", ["skip", "reprocess"])
def test_cli_rejects_exclusions_when_optimization_will_not_run(
    tmp_path: Path,
    mode: str,
) -> None:
    exclusions_path = tmp_path / "bad_codewords.txt"
    exclusions_path.write_text("GeneB\n", encoding="utf-8")

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


def test_resolve_exclusions_is_exact_deduplicated_and_index_stable() -> None:
    decoder = _decoder_with_codebook()

    gene_ids, indices = decoder._resolve_excluded_gene_ids([" GeneB ", "GeneB"])

    assert gene_ids == ("GeneB",)
    assert indices == (1,)
    with pytest.raises(ValueError, match="case-sensitive"):
        decoder._resolve_excluded_gene_ids(["geneb"])


def test_resolve_exclusions_rejects_removing_entire_codebook() -> None:
    decoder = _decoder_with_codebook()

    with pytest.raises(ValueError, match="every codeword"):
        decoder._resolve_excluded_gene_ids(["GeneA", "GeneB", "GeneC"])


def test_excluded_winner_becomes_background_without_index_fallback() -> None:
    decoded = np.asarray([0, 1, 2, 1, -1], dtype=np.int16)
    nearest = np.asarray([0, 1, 2, 1, 1], dtype=np.int16)

    PixelDecoder._suppress_excluded_codeword_assignments(decoded, nearest, (1,))

    np.testing.assert_array_equal(decoded, np.asarray([0, -1, 2, -1, -1]))


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


def test_plane_wise_centroid_statistics_match_full_volume_reference() -> None:
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
        z_support=3,
        minlength=minlength,
    )

    centroid_labels = grey_dilation(labels, size=(3, 1, 1))
    weights = np.maximum(intensity, np.float32(0))
    z_coords = np.arange(labels.shape[0], dtype=np.float32)[:, None, None]
    y_coords = np.arange(labels.shape[1], dtype=np.float32)[None, :, None]
    x_coords = np.arange(labels.shape[2], dtype=np.float32)[None, None, :]
    expected = [
        np.bincount(
            centroid_labels.ravel(),
            weights=weights.ravel(),
            minlength=minlength,
        ),
        np.bincount(
            centroid_labels.ravel(),
            weights=(weights * z_coords).ravel(),
            minlength=minlength,
        ),
        np.bincount(
            centroid_labels.ravel(),
            weights=(weights * y_coords).ravel(),
            minlength=minlength,
        ),
        np.bincount(
            centroid_labels.ravel(),
            weights=(weights * x_coords).ravel(),
            minlength=minlength,
        ),
    ]
    expected_peak = np.zeros(minlength, dtype=np.float32)
    np.maximum.at(expected_peak, labels.ravel(), weights.ravel())
    expected.append(expected_peak)

    for observed_values, expected_values in zip(observed, expected, strict=True):
        np.testing.assert_allclose(observed_values, expected_values)


def test_optimizer_passes_resolved_exclusions_to_gpu_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = _decoder_with_codebook()
    decoder._num_gpus = 1
    decoder._verbose = 0
    decoder._datastore = SimpleNamespace(tile_ids=["tile000"])
    decoder._datastore_path = Path("/unused/datastore")
    decoder._decode_run_key = None
    decoder._decode_mode = "3d"
    decoder._effective_decode_mode = "3d"
    decoder._n_merfish_bits = 4
    decoder._estimate_chromatic_affines = False
    decoder._is_3D = True
    decoder._cleanup = lambda: None
    decoder._load_all_barcodes = lambda: None
    decoder._iterative_normalization_vectors = lambda gpu_id=0: None

    def _load_global_normalization_vectors(**_kwargs) -> None:
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
        n_iterations=1,
        minimum_pixels=1,
        excluded_gene_ids=["GeneB", "GeneB"],
    )

    assert len(captured_args) == 1
    assert captured_args[0][1][-1] == ("GeneB",)


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
