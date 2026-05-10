"""
Data I/O functions for qi2lab 3D MERFISH.

This module provides utilities for reading and writing data in various formats
used by qi2lab 3D MERFISH datasets.

History:
---------
- **2024/12**: Refactored repo structure.
- **2024/12**: Updated docstrings.
- **2024/07**: Removed native NDTiff reading package; integrated tifffile/zarr.
               Reduced dask dependencies.
"""

import csv
import re
import subprocess
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sparse
import zarr
from numpy.typing import ArrayLike
from tifffile import imread


def read_metadatafile(fname: str | Path) -> dict:
    """Read metadata from csv file.

    Parameters
    ----------
    fname: Union[str,Path]
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
    save_path: Union[str,Path]
        path for file
    """

    pd.DataFrame([data_dict]).to_csv(save_path)


def return_data_zarr(
    dataset_path: Path | str, ch_idx: int, ch_idx_offset: int | None = 0
) -> ArrayLike:
    """Return NDTIFF data as a numpy array via tiffile.

    Parameters
    ----------
    dataset_path: Dataset
        pycromanager dataset object
    ch_idx: int
        channel index in ZarrTiffStore file
    ch_idx_offset: int
        channel index offset for unused phase channels

    Returns
    -------
    data: ArrayLike
        data stack
    """

    ndtiff_zarr_store = imread(dataset_path, mode="r+", aszarr=True)
    ndtiff_zarr = zarr.open(ndtiff_zarr_store, mode="r+")
    first_dim = str(ndtiff_zarr.attrs["_ARRAY_DIMENSIONS"][0])

    if first_dim == "C":
        data = np.asarray(ndtiff_zarr[ch_idx - ch_idx_offset, :], dtype=np.uint16)
    else:
        data = np.asarray(ndtiff_zarr[:, ch_idx - ch_idx_offset, :], dtype=np.uint16)
    del ndtiff_zarr_store, ndtiff_zarr

    return np.squeeze(data)


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
    output_dir_path: Union[Path,str]
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
    filename: Union[str, Path]
        Filename
    data: Sequence[Union[str, Sequence[str]]]
        Data to write
    """

    with open(filename, "w", newline="") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
        for item in data:
            writer.writerow([item] if isinstance(item, str) else item)
