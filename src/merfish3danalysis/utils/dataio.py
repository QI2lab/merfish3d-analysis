"""
Data I/O functions for qi2lab 3D MERFISH.

This module provides utilities for reading and writing data in various formats
used by qi2lab 3D MERFISH datasets.

"""

import csv
import re
import subprocess
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
    fname: str or Path
        filename

    Returns
    -------
    metadata: Dict
        metadata dictionary
    """
    scan_data_raw_lines = []

    with open(fname) as f:
        for line in f:
            scan_data_raw_lines.append(line.replace("\n", ""))

    titles = scan_data_raw_lines[0].split(",")

    # convert values to appropriate datatypes
    vals = scan_data_raw_lines[1].split(",")
    for ii in range(len(vals)):
        if re.fullmatch(r"\d+", vals[ii]):
            vals[ii] = int(vals[ii])
        elif re.fullmatch(r"\d*.\d+", vals[ii]):
            vals[ii] = float(vals[ii])
        elif vals[ii].lower() == "False".lower():
            vals[ii] = False
        elif vals[ii].lower() == "True".lower():
            vals[ii] = True
        else:
            # otherwise, leave as string
            pass

    # convert to dictionary
    metadata = {}
    for t, v in zip(titles, vals, strict=False):
        metadata[t] = v

    return metadata


def read_config_file(config_path: Path | str) -> dict:
    """Read config data from csv file.

    Parameters
    ----------
    config_path: Path
        Location of configuration file

    Returns
    -------
    dict_from_csv: dict
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
    data_dict: dict
        metadata dictionary
    save_path: str or Path
        path for file
    """
    pd.DataFrame([data_dict]).to_csv(save_path)


def time_stamp() -> str:
    """Generate timestamp string.

    Returns
    -------
    timestamp: str
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
    output_dir_path: Path or str
        Path to output directory
    matrix: ArrayLike
        Sparse matrix
    cells: Sequence[str]
        Cell names
    features: Sequence[str]
        Feature names
    """
    sparse_mat = sparse.coo_matrix(matrix.values)
    sio.mmwrite(str(output_dir_path / "matrix.mtx"), sparse_mat)
    write_tsv(output_dir_path / "barcodes.tsv", ["cell_" + str(cell) for cell in cells])
    write_tsv(
        output_dir_path / "features.tsv",
        [
            [
                str(f),
                str(f),
                "Blank Codeword" if str(f).startswith("Blank") else "Gene Expression",
            ]
            for f in features
        ],
    )
    subprocess.run(f"gzip -f {output_dir_path!s}/*", shell=True)


def write_tsv(filename: str | Path, data: Sequence[str | Sequence[str]]) -> None:
    """Write data to TSV file.

    Parameters
    ----------
    filename: str or Path
        Filename
    data: Sequence[str or Sequence[str]]
        Data to write
    """
    with open(filename, "w", newline="") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
        for item in data:
            writer.writerow([item] if isinstance(item, str) else item)
