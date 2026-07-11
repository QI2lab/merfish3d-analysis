"""Public launcher for the datastore viewer."""

from pathlib import Path


class Qi2labViewer:
    """View-only ndv/PyQt GUI for qi2lab datastores."""

    def __init__(self, initial_path: Path | None = None) -> None:
        """
        Initialize a viewer launcher.

        Parameters
        ----------
        initial_path : Path | None
            Optional experiment root or datastore path opened at launch.
        """
        self.initial_path = initial_path

    def run(self) -> None:
        """Launch the viewer event loop."""
        run_viewer(self.initial_path)


def run_viewer(initial_path: Path | None = None) -> None:
    """
    Launch the view-only ndv/PyQt datastore viewer.

    Parameters
    ----------
    initial_path : Path | None
        Optional experiment root or datastore path opened at launch.
    """
    try:
        import ndv
        from qtpy import QtWidgets
    except ImportError as exc:
        raise RuntimeError(
            "The qi2lab viewer requires GUI dependencies. Run "
            "`uv sync` to install ndv and Qt support."
        ) from exc

    if hasattr(ndv, "set_gui_backend"):
        ndv.set_gui_backend("qt")
    if hasattr(ndv, "set_canvas_backend"):
        ndv.set_canvas_backend("vispy")

    from merfish3danalysis.viewer.controller import DatastoreViewerWindow

    qt_app = QtWidgets.QApplication.instance()
    if qt_app is None:
        qt_app = QtWidgets.QApplication([])
    qt_app.setQuitOnLastWindowClosed(True)
    window = DatastoreViewerWindow(initial_path)
    window.show()
    exec_method = getattr(qt_app, "exec", None) or qt_app.exec_
    exec_method()
