"""Qt worker objects for datastore viewer display building."""

from collections.abc import Callable

from qtpy import QtCore


class DisplayWorkerSignals(QtCore.QObject):
    """Signals emitted by a background display worker."""

    finished = QtCore.Signal(int, object)
    failed = QtCore.Signal(int, str)


class DisplayWorker(QtCore.QRunnable):
    """Run display-stack preparation off the Qt UI thread."""

    def __init__(
        self,
        generation: int,
        build_display: Callable[[], object],
    ) -> None:
        """
        Initialize a display worker.

        Parameters
        ----------
        generation : int
            Controller generation token for this worker.
        build_display : Callable[[], object]
            Callable that prepares the worker result.
        """
        super().__init__()
        self.generation = generation
        self.build_display = build_display
        self.signals = DisplayWorkerSignals()
        self._cancelled = False

    def cancel(self) -> None:
        """Prevent this worker from emitting a completed result."""
        self._cancelled = True

    def run(self) -> None:
        """Build display data and emit the result."""
        try:
            result = self.build_display()
            if not self._cancelled:
                self.signals.finished.emit(self.generation, result)
        except ValueError as exc:
            if not self._cancelled:
                self.signals.failed.emit(
                    self.generation,
                    f"Data not available: {exc}",
                )
        except Exception as exc:
            if not self._cancelled:
                self.signals.failed.emit(self.generation, str(exc))


class WorkerCoordinator(QtCore.QObject):
    """Own background display workers and expose simple Qt signals."""

    started = QtCore.Signal(str)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        """
        Initialize an idle worker coordinator.

        Parameters
        ----------
        parent : QtCore.QObject or None
            Optional Qt parent object.
        """
        super().__init__(parent)
        self._generation = 0
        self._worker: DisplayWorker | None = None
        self._thread_pool = QtCore.QThreadPool(self)

    @property
    def is_running(self) -> bool:
        """Return whether a background worker is active."""
        return self._worker is not None

    def start(self, build_display: Callable[[], object], message: str) -> bool:
        """
        Start a worker and return whether it was accepted.

        Parameters
        ----------
        build_display : Callable[[], object]
            Callable executed in the worker thread.
        message : str
            Status message emitted when the worker starts.

        Returns
        -------
        bool
            True when the worker was started.
        """
        if self._worker is not None:
            return False
        self._generation += 1
        worker = DisplayWorker(self._generation, build_display)
        connection_type = QtCore.Qt.ConnectionType.QueuedConnection
        worker.signals.finished.connect(
            self._finish_worker,
            type=connection_type,
        )
        worker.signals.failed.connect(
            self._fail_worker,
            type=connection_type,
        )
        self._worker = worker
        self.started.emit(message)
        self._thread_pool.start(worker)
        return True

    def cancel(self) -> None:
        """Cancel the current worker and clear queued work."""
        self._generation += 1
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None
        self._thread_pool.clear()

    @QtCore.Slot(int, object)
    def _finish_worker(self, generation: int, result: object) -> None:
        """
        Emit a finished result when it belongs to the active worker.

        Parameters
        ----------
        generation : int
            Worker generation token.
        result : object
            Worker result object.
        """
        if generation != self._generation:
            return
        self._worker = None
        self.finished.emit(result)

    @QtCore.Slot(int, str)
    def _fail_worker(self, generation: int, message: str) -> None:
        """
        Emit a failed result when it belongs to the active worker.

        Parameters
        ----------
        generation : int
            Worker generation token.
        message : str
            Error message.
        """
        if generation != self._generation:
            return
        self._worker = None
        self.failed.emit(message)
