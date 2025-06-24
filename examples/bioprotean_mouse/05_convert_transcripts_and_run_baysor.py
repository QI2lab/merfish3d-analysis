"""Convert decoded transcripts to CSV and run Baysor.

This script converts the final product of decoding located in
``all_tiles_filtered_decoded_features/transcripts.parquet``
to ``transcripts.csv`` and runs Baysor using that CSV file as input.

The Baysor binary path, options file, and number of Julia threads are
read from the ``qi2labDataStore`` saved in ``qi2labdatastore``.
"""

from pathlib import Path
import pandas as pd
import subprocess
from merfish3danalysis.qi2labDataStore import qi2labDataStore


def convert_parquet_to_csv_and_run_baysor(root_path: Path) -> None:
    """Convert ``transcripts.parquet`` to CSV and run Baysor.

    Parameters
    ----------
    root_path : Path
        Path to the experiment directory that contains ``qi2labdatastore``.
    """

    datastore_path = root_path / "qi2labdatastore"
    datastore = qi2labDataStore(datastore_path)

    baysor_input_parquet = (
        datastore_path
        / "all_tiles_filtered_decoded_features"
        / "transcripts.parquet"
    )
    baysor_input_csv = (
        datastore_path
        / "all_tiles_filtered_decoded_features"
        / "transcripts.csv"
    )

    # Convert Parquet to CSV for Baysor
    df = pd.read_parquet(baysor_input_parquet)
    baysor_input_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(baysor_input_csv, index=False)

    baysor_output_path = datastore._segmentation_root_path / "baysor"
    baysor_output_path.mkdir(exist_ok=True)

    julia_prefix = f"JULIA_NUM_THREADS={datastore.julia_threads} "
    preview_options = f"preview -c {datastore.baysor_options}"
    preview_cmd = (
        julia_prefix
        + str(datastore.baysor_path)
        + f" {preview_options} {baysor_input_csv} -o {baysor_output_path}"
    )

    try:
        subprocess.run(preview_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as exc:
        print("Baysor preview failed:", exc)

    run_options = f"run -p -c {datastore.baysor_options}"
    run_cmd = (
        julia_prefix
        + str(datastore.baysor_path)
        + f" {run_options} {baysor_input_csv} -o {baysor_output_path}"
        + " --polygon-format GeometryCollectionLegacy --count-matrix-format tsv :cell_id"
    )

    try:
        subprocess.run(run_cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        fallback_cmd = (
            julia_prefix
            + str(datastore.baysor_path)
            + f" {run_options} {baysor_input_csv} -o {baysor_output_path}"
            + " --count-matrix-format tsv"
        )
        try:
            subprocess.run(fallback_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as exc:
            print("Baysor run failed:", exc)


if __name__ == "__main__":
    root = Path(r"/path/to/root")
    convert_parquet_to_csv_and_run_baysor(root)

