"""Bulk RNA-seq comparisons of decoded transcript tables."""

import xml.etree.ElementTree as ET

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from merfish3danalysis.cli.qi2lab_microscopes.bulkseq_correlation import app


@pytest.mark.integration
def test_bulkseq_cli_compares_both_decoded_tables_after_prefix_filtering(
    image_datastore, tmp_path
):
    """Both Parquet inputs retain their known log correlations after filtering.

    Parameters
    ----------
    image_datastore : qi2labDataStore
        Datastore fixture used to save and export decoded transcripts.
    tmp_path : pathlib.Path
        Temporary directory for the external tables and correlation figure.
    """
    genes = ["Actb", "Gapdh", "Slc17a7", "Gad1"]
    decoded = pd.DataFrame(
        {
            "gene_id": np.repeat([*genes, "DropControl"], [2, 4, 8, 16, 1]),
            "cell_id": 1,
        }
    )
    image_datastore.save_global_filtered_decoded_spots(decoded)
    decoded_path = tmp_path / "merfish3d.parquet"
    image_datastore.load_global_filtered_decoded_spots().to_parquet(
        decoded_path, index=False
    )
    merlin = pd.DataFrame(
        {
            "gene_id": np.repeat([*genes, "DropControl"], [16, 8, 4, 2, 1]),
            "cell_id": 1,
        }
    )
    merlin_path = tmp_path / "merlin.parquet"
    merlin.to_parquet(merlin_path, index=False)
    bulk_path = tmp_path / "bulk.txt"
    pd.DataFrame({"gene": [*genes, "DropControl"], "fpkm": [1, 2, 4, 8, 1000]}).to_csv(
        bulk_path, sep="\t", index=False
    )
    plot_path = tmp_path / "correlations.svg"

    plt.switch_backend("Agg")
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        result = CliRunner().invoke(
            app,
            [
                str(decoded_path),
                str(bulk_path),
                "gene_id",
                "gene",
                "fpkm",
                "--file1b",
                str(merlin_path),
                "--gene-col1b",
                "gene_id",
                "--drop-prefix",
                "dRoP",
                "--plot-out",
                str(plot_path),
            ],
        )

    assert result.exit_code == 0, (result.output, result.exception)
    assert "DropControl" not in result.output
    texts = [element.text for element in ET.parse(plot_path).findall(".//{*}text")]
    assert "merfish3d: r = 1.000" in texts
    assert "merlin: r = -1.000" in texts
