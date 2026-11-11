"""
Data I/O functions for qi2lab 3D MERFISH.

This module provides utilities for reading and writing data in various formats
used by qi2lab 3D MERFISH datasets.

"""

import gzip
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd
import scipy.io as sio
import scipy.sparse as sparse
from numpy.typing import ArrayLike


def resolve_datastore_path(path: Path | str) -> Path:
    """Find an existing datastore from its root or its experiment directory.

    Parameters
    ----------
    path : Path or str
        Datastore directory or experiment directory containing qi2labdatastore.

    Returns
    -------
    Path
        Absolute datastore directory identified by its state metadata.

    Raises
    ------
    FileNotFoundError
        Neither the supplied directory nor its qi2labdatastore child has state
        metadata. This function never creates directories.
    """
    path = Path(path).expanduser().resolve()
    for candidate in (path, path / "qi2labdatastore"):
        if (candidate / "datastore_state" / "state.json").is_file():
            return candidate
    raise FileNotFoundError(
        f"No qi2lab datastore found at {path} or its qi2labdatastore child."
    )


def read_metadatafile(fname: str | Path) -> dict:
    """Read metadata from csv file.

    Parameters
    ----------
    fname : str or Path
        Acquisition CSV path containing a header and one metadata row.

    Returns
    -------
    metadata : Dict
        metadata dictionary
    """
    return pd.read_csv(fname, nrows=1).iloc[0].to_dict()


def read_config_file(config_path: Path | str) -> dict:
    """Read config data from csv file.

    Parameters
    ----------
    config_path : Path or str
        Headerless, two-column CSV path containing configuration keys and values.

    Returns
    -------
    dict_from_csv : dict
        instrument configuration metadata
    """
    dict_from_csv = (
        pd.read_csv(config_path, header=None, index_col=0).squeeze("columns").to_dict()
    )

    return dict_from_csv


def write_metadata(data_dict: dict, save_path: str | Path) -> None:
    """Write dictionary as CSV file.

    Parameters
    ----------
    data_dict : dict
        Metadata field names and values written as one CSV row.
    save_path : str or Path
        Destination CSV path.
    """
    pd.DataFrame([data_dict]).to_csv(save_path, index=False)


def time_stamp() -> str:
    """Generate timestamp string.

    Returns
    -------
    timestamp : str
        timestamp formatted as string
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_sparse_mtx(
    output_dir_path: Path | str,
    matrix: ArrayLike,
    cells: Sequence[str],
    features: Sequence[str],
) -> None:
    """Write sparse matrix in MTX format.

    Parameters
    ----------
    output_dir_path : Path or str
        Existing directory receiving the compressed matrix and TSV files.
    matrix : ArrayLike
        Count matrix written to Matrix Market format without transposition.
    cells : Sequence[str]
        Ordered cell IDs written to barcodes.tsv.gz with a ``cell_`` prefix.
    features : Sequence[str]
        Ordered feature IDs written to features.tsv.gz. IDs starting with
        ``Blank`` receive the ``Blank Codeword`` feature type.
    """
    output_dir_path = Path(output_dir_path)
    with gzip.open(output_dir_path / "matrix.mtx.gz", "wb", compresslevel=6) as stream:
        sio.mmwrite(stream, sparse.coo_matrix(matrix))
    pd.Series(["cell_" + str(cell) for cell in cells]).to_csv(
        output_dir_path / "barcodes.tsv.gz",
        sep="\t",
        index=False,
        header=False,
        compression={"method": "gzip", "compresslevel": 6},
    )
    pd.DataFrame(
        [
            [
                str(f),
                str(f),
                "Blank Codeword" if str(f).startswith("Blank") else "Gene Expression",
            ]
            for f in features
        ],
    ).to_csv(
        output_dir_path / "features.tsv.gz",
        sep="\t",
        index=False,
        header=False,
        compression={"method": "gzip", "compresslevel": 6},
    )
