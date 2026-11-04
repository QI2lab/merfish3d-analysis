"""
Calculate F1-score between qi2lab filtered decoded spots assigned to cells and MERLIN.

We preserve the existing qi2lab coordinate swap when assembling the qi2lab
coordinate array so the comparison stays aligned with MERLIN output.

Contributors: Shepherd, Codex.
"""

from pathlib import Path
from pprint import pp

import pandas as pd
import typer

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.dataio import resolve_datastore_path
from merfish3danalysis.utils.metrics import calculate_F1_with_radius

app = typer.Typer(pretty_exceptions_enable=False)


def load_qi2lab_transcripts_within_cells(specified_dir: Path) -> pd.DataFrame:
    """Load filtered decoded features and keep only transcripts assigned to cells."""
    if specified_dir.is_file():
        if specified_dir.suffixes[-2:] == [".csv", ".gz"]:
            transcripts_df = pd.read_csv(specified_dir)
        else:
            transcripts_df = pd.read_parquet(specified_dir)
    else:
        datastore = qi2labDataStore(
            resolve_datastore_path(specified_dir), validate=False
        )
        transcripts_df = datastore.load_global_filtered_decoded_spots()
        if transcripts_df is None:
            raise FileNotFoundError("Filtered decoded features not found in datastore.")

    required_columns = {
        "gene_id",
        "global_z",
        "global_y",
        "global_x",
        "cell_id",
    }
    missing_columns = sorted(required_columns - set(transcripts_df.columns))
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {specified_dir}: {missing_columns}"
        )

    transcripts_df = transcripts_df.copy()
    transcripts_df["cell_id"] = pd.to_numeric(
        transcripts_df["cell_id"], errors="coerce"
    )
    transcripts_df = transcripts_df.loc[transcripts_df["cell_id"] > 0].copy()
    transcripts_df = transcripts_df.loc[
        ~transcripts_df["gene_id"].astype(str).str.lower().str.startswith("blank")
    ].copy()

    return transcripts_df


def calculate_F1(
    specified_dir: Path,
    merlin_path: Path,
    search_radius: float,
) -> dict:
    """Calculate F1 using filtered decoded spots and MERLIN ground truth."""
    qi2lab_spots = load_qi2lab_transcripts_within_cells(specified_dir)
    qi2lab_coords = qi2lab_spots[["global_z", "global_x", "global_y"]].to_numpy()
    qi2lab_gene_ids = qi2lab_spots["gene_id"].to_numpy()

    merlin_spots = pd.read_csv(merlin_path)
    merlin_coords = merlin_spots[["global_z", "global_y", "global_x"]].to_numpy()
    merlin_gene_ids = merlin_spots["target_molecule_name"].to_numpy()

    results, _, _, _ = calculate_F1_with_radius(
        qi2lab_coords, qi2lab_gene_ids, merlin_coords, merlin_gene_ids, search_radius
    )

    print(f"Number of qi2lab transcripts within cells: {len(qi2lab_spots)}")
    print(f"Number of Zhuang spots: {len(merlin_spots)}")

    return results


@app.command()
def main(root_path: Path) -> None:
    """Compare Zhuang decoded spots with MERLIN using a 3 µm search radius."""
    root_path = root_path.expanduser().resolve()
    merlin_spots_path = root_path / Path(
        r"mop/mouse_sample1_raw/zhuang_decoded_codewords/spots_mouse1sample1.csv"
    )
    results = calculate_F1(root_path, merlin_spots_path, search_radius=3.0)
    pp(results)


if __name__ == "__main__":
    app()
