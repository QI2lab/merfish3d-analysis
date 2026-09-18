import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

import merfish3danalysis.PixelDecoder as pixel_decoder_module
from merfish3danalysis.PixelDecoder import PixelDecoder


@pytest.fixture
def worker_dependencies(monkeypatch):
    """Replace GPU and datastore operations while running the real worker loop."""
    device = Mock()
    cupy = SimpleNamespace(
        cuda=SimpleNamespace(
            Device=device,
            Stream=SimpleNamespace(null=Mock()),
        ),
        get_default_memory_pool=Mock(),
        get_default_pinned_memory_pool=Mock(),
    )
    torch = SimpleNamespace(cuda=SimpleNamespace(set_device=Mock()))
    local_decoder = Mock(spec=PixelDecoder)
    decoder_factory = Mock(return_value=local_decoder)
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(pixel_decoder_module, "preload_cuda_libraries", Mock())
    monkeypatch.setattr(pixel_decoder_module, "qi2labDataStore", Mock())
    monkeypatch.setattr(pixel_decoder_module, "PixelDecoder", decoder_factory)
    return SimpleNamespace(
        device=device,
        set_device=torch.cuda.set_device,
        decoder=local_decoder,
        decoder_factory=decoder_factory,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("visible_devices", "gpu_id", "expected_label"),
    [
        ("0", 0, "0"),
        ("1", 0, "1"),
        (None, 1, "1"),
        ("3,5", 1, "5"),
        ("GPU-assigned-device", 0, "GPU-assigned-device"),
    ],
)
def test_worker_logs_visible_gpu_but_computes_on_local_device(
    monkeypatch,
    capsys,
    tmp_path,
    worker_dependencies,
    visible_devices,
    gpu_id,
    expected_label,
):
    if visible_devices is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)

    pixel_decoder_module.decode_tiles_worker(
        datastore_path=tmp_path,
        tile_indices=[7],
        gpu_id=gpu_id,
        merfish_bits=4,
        verbose=1,
        decode_mode="3d",
        lowpass_sigma=(3, 1, 1),
        magnitude_threshold=(1.5, 10),
        minimum_pixels=1,
        feature_predictor_threshold=0.1,
        normalization_method="iterative",
    )

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert f"GPU {expected_label}: starting tile 1 of 1 (tile index: 7)." in lines[0]
    assert (
        f"GPU {expected_label}: decoded and saved tile 1 of 1 (tile index: 7)."
        in lines[1]
    )
    worker_dependencies.device.assert_called_once_with(gpu_id)
    worker_dependencies.set_device.assert_called_once_with(gpu_id)
    assert (
        worker_dependencies.decoder._load_global_normalization_vectors.call_args.kwargs[
            "gpu_id"
        ]
        == gpu_id
    )
    worker_dependencies.decoder._load_iterative_normalization_vectors.assert_called_once_with(
        gpu_id=gpu_id
    )
    assert (
        worker_dependencies.decoder.decode_one_tile.call_args.kwargs["gpu_id"] == gpu_id
    )


@pytest.mark.unit
@pytest.mark.parametrize("normalization_method", ["global", "none"])
def test_worker_does_not_load_unused_iterative_normalization(
    tmp_path, worker_dependencies, normalization_method
):
    pixel_decoder_module.decode_tiles_worker(
        datastore_path=tmp_path,
        tile_indices=[0],
        gpu_id=0,
        merfish_bits=4,
        verbose=0,
        decode_mode="3d",
        lowpass_sigma=(3, 1, 1),
        magnitude_threshold=(1.5, 10),
        minimum_pixels=1,
        feature_predictor_threshold=0.1,
        normalization_method=normalization_method,
        normalization_features="all",
    )
    assert (
        worker_dependencies.decoder_factory.call_args.kwargs["normalization_features"]
        == "all"
    )
    worker_dependencies.decoder._load_iterative_normalization_vectors.assert_not_called()
    assert worker_dependencies.decoder._load_global_normalization_vectors.called == (
        normalization_method == "global"
    )


@pytest.mark.unit
@pytest.mark.parametrize("verbose", [0, 1])
@pytest.mark.parametrize("normalization_features", ["all", "cells"])
def test_two_gpu_decode_logs_each_assignment_without_changing_local_device(
    monkeypatch, capsys, tmp_path, worker_dependencies, verbose, normalization_features
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    started_masks = []

    class InlineProcess:
        """Run workers with the environment present when Process.start is called."""

        def __init__(self, *, target, args):
            self.target = target
            self.args = args
            self.exitcode = None

        def start(self):
            started_masks.append(os.environ["CUDA_VISIBLE_DEVICES"])
            self.target(*self.args)
            self.exitcode = 0

        def join(self):
            assert self.exitcode == 0

    monkeypatch.setattr(pixel_decoder_module.mp, "Process", InlineProcess)
    decoder = PixelDecoder.__new__(PixelDecoder)
    decoder._num_gpus = 2
    decoder._datastore = SimpleNamespace(tile_ids=["tile000", "tile001"])
    decoder._datastore_path = tmp_path
    decoder._n_merfish_bits = 4
    decoder._verbose = verbose
    decoder._decode_mode = "3d"
    decoder._is_3D = True
    decoder._normalization_features = normalization_features
    decoder._df_barcodes_loaded = []
    for method in (
        "_load_all_barcodes",
        "_apply_filter_method",
        "_remove_duplicates_in_tile_overlap",
        "_save_barcodes",
    ):
        monkeypatch.setattr(decoder, method, Mock())

    decoder.decode_all_tiles(assign_to_cells=False, minimum_pixels=1)

    assert started_masks == ["0", "1"]
    assert [
        invocation.kwargs["normalization_features"]
        for invocation in worker_dependencies.decoder_factory.call_args_list
    ] == [normalization_features, normalization_features]
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert worker_dependencies.device.call_args_list == [call(0), call(0)]
    assert worker_dependencies.set_device.call_args_list == [call(0), call(0)]
    assert [
        (invocation.kwargs["tile_idx"], invocation.kwargs["gpu_id"])
        for invocation in worker_dependencies.decoder.decode_one_tile.call_args_list
    ] == [(0, 0), (1, 0)]
    gpu_lines = [
        line for line in capsys.readouterr().out.splitlines() if "GPU " in line
    ]
    if verbose:
        assert len(gpu_lines) == 4
        for gpu_id in range(2):
            assert (
                f"GPU {gpu_id}: starting tile 1 of 1 (tile index: {gpu_id})."
                in gpu_lines[2 * gpu_id]
            )
            assert (
                f"GPU {gpu_id}: decoded and saved tile 1 of 1 (tile index: {gpu_id})."
                in gpu_lines[2 * gpu_id + 1]
            )
    else:
        assert gpu_lines == []
