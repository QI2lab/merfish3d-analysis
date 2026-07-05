"""Shared helpers for qi2lab microscope CLI commands."""

from pathlib import Path


def qi2lab_datastore_path(root_path: Path) -> Path:
    """Return the standard qi2lab datastore path for an experiment root.

    Parameters
    ----------
    root_path : Path
        Experiment root directory.

    Returns
    -------
    Path
        Standard ``qi2labdatastore`` path.
    """

    return root_path / "qi2labdatastore"
