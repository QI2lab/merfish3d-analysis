import os
import subprocess
import sys

import pytest

import merfish3danalysis.PixelDecoder as pixel_decoder_module


@pytest.mark.unit
@pytest.mark.parametrize("initial_visible", [None, "8,9"])
def test_spawn_workers_receive_distinct_gpu_assignments(
    monkeypatch, tmp_path, initial_visible
):
    if initial_visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", initial_visible)

    paths = [tmp_path / "gpu0.txt", tmp_path / "gpu1.txt"]
    processes = []
    for gpu_id, path in enumerate(paths):
        command = [
            sys.executable,
            "-c",
            "import os, pathlib, sys; "
            "pathlib.Path(sys.argv[1]).write_text(os.environ['CUDA_VISIBLE_DEVICES'])",
            str(path),
        ]
        processes.append(
            pixel_decoder_module._start_gpu_worker_process(
                target=subprocess.check_call,
                args=(command,),
                physical_gpu_id=gpu_id,
            )
        )
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == initial_visible

    pixel_decoder_module._join_gpu_workers(processes, "decode")
    assert [path.read_text() for path in paths] == ["0", "1"]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == initial_visible
