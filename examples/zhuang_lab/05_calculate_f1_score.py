"""
Calculate F1-score between qi2lab filtered decoded spots assigned to cells and MERLIN.

We preserve the existing qi2lab coordinate swap when assembling the qi2lab
coordinate array so the comparison stays aligned with MERLIN output.

Shepherd 2025/01 - create script.
Codex 2026/05 - switch qi2lab input to filtered decoded features within cells.
"""

from pathlib import Path
from pprint import pp

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree


def calculate_F1_with_radius(
    qi2lab_coords: ArrayLike,
    qi2lab_gene_ids: ArrayLike,
    merlin_coords: ArrayLike,
    merlin_gene_ids: ArrayLike,
    radius: float,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Greedy closest-first matching within `radius`, with strict same-gene and one-to-one constraints.
    Returns F1 metrics and (TP coords, FP coords, FN coords).
    """
    qi2lab_coords = np.asarray(qi2lab_coords, float)
    merlin_coords = np.asarray(merlin_coords, float)
    qi2lab_gene_ids = np.asarray(qi2lab_gene_ids).astype(str)
    merlin_gene_ids = np.asarray(merlin_gene_ids).astype(str)

    Nq = qi2lab_coords.shape[0]
    Ng = merlin_coords.shape[0]

    if Nq == 0 and Ng == 0:
        return (
            {
                "F1 Score": 1.0,
                "Precision": 1.0,
                "Recall": 1.0,
                "True Positives": 0,
                "False Positives": 0,
                "False Negatives": 0,
            },
            np.empty((0, 3), float),
            np.empty((0, 3), float),
            np.empty((0, 3), float),
        )
    if Nq == 0:
        return (
            {
                "F1 Score": 0.0,
                "Precision": 0.0,
                "Recall": 0.0 if Ng > 0 else 1.0,
                "True Positives": 0,
                "False Positives": 0,
                "False Negatives": int(Ng),
            },
            np.empty((0, 3), float),
            np.empty((0, 3), float),
            merlin_coords.copy(),
        )
    if Ng == 0:
        return (
            {
                "F1 Score": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "True Positives": 0,
                "False Positives": int(Nq),
                "False Negatives": 0,
            },
            np.empty((0, 3), float),
            qi2lab_coords.copy(),
            np.empty((0, 3), float),
        )

    pair_q_idx_all: list[np.ndarray] = []
    pair_g_idx_all: list[np.ndarray] = []
    pair_dist_all: list[np.ndarray] = []

    common_genes = np.intersect1d(np.unique(qi2lab_gene_ids), np.unique(merlin_gene_ids))
    for gene in common_genes:
        q_idx = np.flatnonzero(qi2lab_gene_ids == gene)
        g_idx = np.flatnonzero(merlin_gene_ids == gene)
        if q_idx.size == 0 or g_idx.size == 0:
            continue

        q_pts = qi2lab_coords[q_idx]
        g_pts = merlin_coords[g_idx]

        q_tree = cKDTree(q_pts)
        g_tree = cKDTree(g_pts)
        try:
            dist_coo = q_tree.sparse_distance_matrix(
                g_tree, max_distance=radius, output_type="coo_matrix"
            )
        except TypeError:
            dist_coo = q_tree.sparse_distance_matrix(
                g_tree, max_distance=radius
            ).tocoo()

        if dist_coo.nnz == 0:
            continue

        pair_q_idx_all.append(q_idx[dist_coo.row])
        pair_g_idx_all.append(g_idx[dist_coo.col])
        pair_dist_all.append(dist_coo.data)

    if not pair_q_idx_all:
        q_used = np.zeros(Nq, dtype=bool)
        g_used = np.zeros(Ng, dtype=bool)
        tp_coords = np.empty((0, 3), float)
        fp_coords = qi2lab_coords.copy()
        fn_coords = merlin_coords.copy()
        tp = 0
    else:
        pair_q_idx = np.concatenate(pair_q_idx_all)
        pair_g_idx = np.concatenate(pair_g_idx_all)
        pair_dist = np.concatenate(pair_dist_all)

        order = np.argsort(pair_dist, kind="stable")
        pair_q_idx = pair_q_idx[order]
        pair_g_idx = pair_g_idx[order]

        q_used = np.zeros(Nq, dtype=bool)
        g_used = np.zeros(Ng, dtype=bool)
        matched_q = []
        matched_g = []

        for qi, gi in zip(pair_q_idx, pair_g_idx, strict=False):
            if q_used[qi] or g_used[gi]:
                continue
            if qi2lab_gene_ids[qi] != merlin_gene_ids[gi]:
                continue
            q_used[qi] = True
            g_used[gi] = True
            matched_q.append(qi)
            matched_g.append(gi)

        matched_q = np.asarray(matched_q, dtype=int)
        matched_g = np.asarray(matched_g, dtype=int)

        if matched_q.size:
            if not np.all(qi2lab_gene_ids[matched_q] == merlin_gene_ids[matched_g]):
                raise RuntimeError("Gene-ID mismatch detected in accepted matches.")

        tp = int(matched_q.size)
        tp_coords = qi2lab_coords[matched_q]
        fp_coords = qi2lab_coords[~q_used]
        fn_coords = merlin_coords[~g_used]

    fp = int((~q_used).sum())
    fn = int((~g_used).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    results = {
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "True Positives": tp,
        "False Positives": fp,
        "False Negatives": fn,
    }
    return results, tp_coords, fp_coords, fn_coords


def resolve_qi2lab_transcripts_path(specified_dir: Path) -> Path:
    """Resolve the qi2lab filtered decoded output path from a directory or file path."""

    if specified_dir.is_file():
        return specified_dir

    if specified_dir.name == "qi2labdatastore":
        parquet_path = (
            specified_dir
            / Path("all_tiles_filtered_decoded_features")
            / Path("decoded_features.parquet")
        )
        csv_gz_path = (
            specified_dir
            / Path("all_tiles_filtered_decoded_features")
            / Path("decoded_features.csv.gz")
        )
    else:
        parquet_path = (
            specified_dir
            / Path("qi2labdatastore")
            / Path("all_tiles_filtered_decoded_features")
            / Path("decoded_features.parquet")
        )
        csv_gz_path = (
            specified_dir
            / Path("qi2labdatastore")
            / Path("all_tiles_filtered_decoded_features")
            / Path("decoded_features.csv.gz")
        )

    if parquet_path.exists():
        return parquet_path
    if csv_gz_path.exists():
        return csv_gz_path

    raise FileNotFoundError(
        "Could not find filtered decoded features at "
        f"{parquet_path} or {csv_gz_path}"
    )


def load_qi2lab_transcripts_within_cells(specified_dir: Path) -> pd.DataFrame:
    """Load filtered decoded features and keep only transcripts assigned to cells."""

    transcripts_path = resolve_qi2lab_transcripts_path(specified_dir)
    if transcripts_path.suffixes[-2:] == [".csv", ".gz"]:
        transcripts_df = pd.read_csv(transcripts_path)
    else:
        transcripts_df = pd.read_parquet(transcripts_path)

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
            f"Missing required columns in {transcripts_path}: {missing_columns}"
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


if __name__ == "__main__":
    root_path = Path(r"/media/dps/data/zhuang")
    merlin_spots_path = root_path / Path(
        r"mop/mouse_sample1_raw/zhuang_decoded_codewords/spots_mouse1sample1.csv"
    )
    results = calculate_F1(root_path, merlin_spots_path, search_radius=3.0)
    pp(results)
