"""File exports produced by the shared data I/O utilities."""

import gzip

import numpy as np
import pandas as pd
import pytest
from scipy.io import mmread
from scipy.sparse import csr_matrix

from merfish3danalysis.utils.dataio import write_sparse_mtx


@pytest.mark.unit
@pytest.mark.parametrize("matrix_type", [np.asarray, pd.DataFrame, csr_matrix])
def test_sparse_export_accepts_documented_matrix_and_path_types(tmp_path, matrix_type):
    output = tmp_path / "expression export"
    output.mkdir()
    existing = output / "notes.txt"
    existing.write_text("leave this file alone")
    values = np.array([[1, 0], [0, 2]], dtype=np.int32)

    write_sparse_mtx(str(output), matrix_type(values), ["1", "2"], ["A", "Blank-1"])

    with gzip.open(output / "matrix.mtx.gz", "rb") as stream:
        np.testing.assert_array_equal(mmread(stream).toarray(), values)
    with gzip.open(output / "barcodes.tsv.gz", "rt") as stream:
        assert stream.read().splitlines() == ["cell_1", "cell_2"]
    with gzip.open(output / "features.tsv.gz", "rt") as stream:
        assert stream.read().splitlines() == [
            "A\tA\tGene Expression",
            "Blank-1\tBlank-1\tBlank Codeword",
        ]
    assert existing.read_text() == "leave this file alone"
