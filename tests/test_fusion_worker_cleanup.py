"""Regression coverage for standard semaphore cleanup inside Loky workers."""

import os
import subprocess
import sys
from unittest.mock import Mock

import pytest

from merfish3danalysis.DataRegistration import _cleanup_fusion_worker_semaphore

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX named semaphores")


@pytest.mark.parametrize("missing", [False, True])
def test_semaphore_cleanup_unregisters_present_and_already_removed_names(
    monkeypatch, missing
):
    import multiprocessing.resource_tracker as tracker
    import multiprocessing.synchronize as synchronize

    unlink = Mock(side_effect=FileNotFoundError() if missing else None)
    unregister = Mock()
    monkeypatch.setattr(synchronize, "sem_unlink", unlink)
    monkeypatch.setattr(tracker, "unregister", unregister)
    _cleanup_fusion_worker_semaphore("/loky-test")
    unlink.assert_called_once_with("/loky-test")
    unregister.assert_called_once_with("/loky-test", "semaphore")


def test_semaphore_cleanup_preserves_other_errors_and_tracking(monkeypatch):
    import multiprocessing.resource_tracker as tracker
    import multiprocessing.synchronize as synchronize

    unregister = Mock()
    monkeypatch.setattr(synchronize, "sem_unlink", Mock(side_effect=PermissionError()))
    monkeypatch.setattr(tracker, "unregister", unregister)
    with pytest.raises(PermissionError):
        _cleanup_fusion_worker_semaphore("/loky-test")
    unregister.assert_not_called()


def test_loky_worker_cleanup_race_exits_without_stale_tracker_warnings():
    # A subprocess lets us check stderr after the resource trackers exit too.
    # Removing a name before finalization reproduces Python 3.12's stale-entry
    # bug while the underlying lock remains usable by its current owner.
    script = """
import gc
import multiprocessing as mp
from types import SimpleNamespace
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from merfish3danalysis.DataRegistration import _direct_zarr_fusion_kwargs

def work(remove_first):
    from multiprocessing.synchronize import sem_unlink
    lock = mp.get_context("spawn").RLock()
    if remove_first:
        sem_unlink(lock._semlock.name)
    with lock:
        pass
    del lock
    gc.collect()
    return remove_first

if __name__ == "__main__":
    options = _direct_zarr_fusion_kwargs(
        misc_utils=SimpleNamespace(process_batch_using_joblib=None), fusion_workers=2
    )
    result = Parallel(**options["batch_options"]["batch_func_kwargs"])(
        delayed(work)(remove_first) for remove_first in [False, True, False, True]
    )
    assert result == [False, True, False, True]
    get_reusable_executor().shutdown(wait=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "resource_tracker:" not in result.stderr, result.stderr
    assert "Exception ignored" not in result.stderr, result.stderr
