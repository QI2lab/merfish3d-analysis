#!/usr/bin/env python3
"""Compare z-strided proseg outputs against the no-stride run."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from merfish3danalysis.cli.statphysbio_simulation.calculate_F1 import (
    calculate_F1_with_radius,
)
from merfish3danalysis.qi2labDataStore import qi2labDataStore

RUN_RE = re.compile(r"^zstride(?P<stride>\d+)_(?P<mode>2d|3d)$", re.IGNORECASE)
RNA_F1_RADIUS_SPACING_MULTIPLIER = 3.0


@dataclass(frozen=True)
class RunInfo:
    name: str
    path: Path
    zstride: int
    decode_mode: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare z-strided proseg RNA detection and max-projection cell-count "
            "metrics against a no-stride reference run."
        )
    )
    parser.add_argument(
        "--proseg-root",
        type=Path,
        default=Path(
            "/home/dps/server_data2/bioprotean/20250513_Bartelle_MERFISH_control/"
            "qi2labdatastore/proseg"
        ),
        help="Path to the qi2labdatastore/proseg directory.",
    )
    parser.add_argument(
        "--reference-run",
        default="zstride01_3d",
        help="No-stride proseg run folder to use as reference.",
    )
    parser.add_argument(
        "--min-cell-iou",
        type=float,
        default=0.25,
        help=(
            "Minimum 2D polygon IoU when --cell-match-mode=iou. Ignored for "
            "the default cell-ID matching mode."
        ),
    )
    parser.add_argument(
        "--cell-match-mode",
        choices=("id", "iou"),
        default="id",
        help=(
            "How to match max-projection cells. Use 'id' when runs share the "
            "same starting segmentation; use 'iou' only for independent "
            "segmentations."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default="zstride_f1_summary",
        help="Output prefix written under --proseg-root.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a compact summary table after writing outputs.",
    )
    return parser.parse_args()


def _load_run_metadata(datastore: qi2labDataStore) -> dict[str, dict[str, Any]]:
    try:
        attrs = datastore._load_calibrations_attributes()
    except Exception:
        return {}
    runs = attrs.get("decode_normalization_runs", {})
    return runs if isinstance(runs, dict) else {}


def _discover_runs(proseg_root: Path, datastore: qi2labDataStore) -> list[RunInfo]:
    run_metadata = _load_run_metadata(datastore)
    runs: list[RunInfo] = []
    for child in sorted(proseg_root.iterdir()):
        if not child.is_dir():
            continue
        match = RUN_RE.match(child.name)
        if match is None:
            continue
        zstride = int(match.group("stride"))
        decode_mode = match.group("mode").lower()

        metadata_key = f"zstride_{zstride:02d}_{decode_mode}"
        metadata = run_metadata.get(metadata_key, {})
        if isinstance(metadata, dict):
            zstride = int(metadata.get("zstride_level", zstride))
            decode_mode = str(metadata.get("decode_mode", decode_mode)).lower()

        runs.append(
            RunInfo(
                name=child.name,
                path=child,
                zstride=max(zstride, 1),
                decode_mode=decode_mode,
            )
        )
    return runs


def _is_blank_gene(series: pd.Series) -> pd.Series:
    return series.astype(str).str.match(r"^[Bb]lank")


def _load_transcripts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"x", "y", "z", "gene"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    if "background" in df.columns:
        background = df["background"].astype(str).str.lower().isin({"true", "1"})
        df = df.loc[~background].copy()
    df = df.loc[~_is_blank_gene(df["gene"])].copy()
    return df


def _decoded_run_key(run: RunInfo) -> str:
    return f"zstride_{run.zstride:02d}_{run.decode_mode}"


def _decoded_features_path(datastore_root: Path, run: RunInfo) -> Path:
    decoded_root = datastore_root / "all_tiles_filtered_decoded_features"
    if run.zstride <= 1:
        default_csv = decoded_root / "decoded_features.csv.gz"
        if default_csv.exists():
            return default_csv
        default_parquet = decoded_root / "decoded_features.parquet"
        if default_parquet.exists():
            return default_parquet

    run_dir = decoded_root / _decoded_run_key(run)
    run_csv = run_dir / "decoded_features.csv.gz"
    if run_csv.exists():
        return run_csv
    run_parquet = run_dir / "decoded_features.parquet"
    if run_parquet.exists():
        return run_parquet

    raise FileNotFoundError(
        "Could not find decoded features for "
        f"{run.name}. Checked default and {run_dir}."
    )


def _load_decoded_features(datastore_root: Path, run: RunInfo) -> pd.DataFrame:
    path = _decoded_features_path(datastore_root, run)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    required = {"global_x", "global_y", "global_z", "gene_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.loc[~_is_blank_gene(df["gene_id"])].copy()
    return df


def _gene_set(df: pd.DataFrame, gene_column: str) -> set[str]:
    return set(df[gene_column].astype(str).dropna())


def _coordinate_extent_metrics(
    prefix: str,
    df: pd.DataFrame,
    coordinate_columns: list[str],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for column in coordinate_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        axis = column.removeprefix("global_")
        metrics[f"{prefix}_{axis}_min"] = float(values.min())
        metrics[f"{prefix}_{axis}_max"] = float(values.max())
    return metrics


def _scaled_rna_coords(
    df: pd.DataFrame,
    z_scale: float,
    coordinate_columns: list[str],
    gene_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    coords = df[coordinate_columns].to_numpy(dtype=np.float64)
    coords[:, 0] *= z_scale
    genes = df[gene_column].to_numpy()
    return coords, genes


def _rna_f1(
    run: RunInfo,
    reference_df: pd.DataFrame,
    datastore_root: Path,
    voxel_zyx_um: np.ndarray,
) -> dict[str, Any]:
    run_decoded_path = _decoded_features_path(datastore_root, run)
    run_df = _load_decoded_features(datastore_root, run)
    comparable_reference_df = reference_df
    base_z_step_um = float(voxel_zyx_um[0])
    effective_z_step_um = float(voxel_zyx_um[0]) * float(max(run.zstride, 1))
    base_xy_radius_um = float(max(base_z_step_um, voxel_zyx_um[1], voxel_zyx_um[2]))
    xy_radius_um = RNA_F1_RADIUS_SPACING_MULTIPLIER * base_xy_radius_um
    z_radius_um = RNA_F1_RADIUS_SPACING_MULTIPLIER * effective_z_step_um
    if z_radius_um <= 0 or xy_radius_um <= 0:
        raise ValueError(f"Invalid voxel metadata: {voxel_zyx_um!r}")

    z_scale = xy_radius_um / z_radius_um
    coordinate_columns = ["global_z", "global_y", "global_x"]
    run_coords, run_genes = _scaled_rna_coords(
        run_df, z_scale, coordinate_columns, "gene_id"
    )
    reference_coords, reference_genes = _scaled_rna_coords(
        comparable_reference_df, z_scale, coordinate_columns, "gene_id"
    )
    results, _, _, _ = calculate_F1_with_radius(
        qi2lab_coords=run_coords,
        qi2lab_gene_ids=run_genes,
        gt_coords=reference_coords,
        gt_gene_ids=reference_genes,
        radius=xy_radius_um,
    )
    run_genes_set = _gene_set(run_df, "gene_id")
    reference_genes_set = _gene_set(comparable_reference_df, "gene_id")
    shared_genes = run_genes_set & reference_genes_set

    return {
        "rna_f1": float(results["F1 Score"]),
        "rna_precision": float(results["Precision"]),
        "rna_recall": float(results["Recall"]),
        "rna_true_positives": int(results["True Positives"]),
        "rna_false_positives": int(results["False Positives"]),
        "rna_false_negatives": int(results["False Negatives"]),
        "rna_xy_radius_um": xy_radius_um,
        "rna_z_radius_um": z_radius_um,
        "rna_radius_spacing_multiplier": RNA_F1_RADIUS_SPACING_MULTIPLIER,
        "rna_effective_z_step_um": effective_z_step_um,
        "rna_z_scale": z_scale,
        "rna_reference_plane_rule": "all",
        "run_rna_count": len(run_df),
        "reference_rna_count": len(reference_df),
        "comparable_reference_rna_count": len(comparable_reference_df),
        "run_decoded_features_path": str(run_decoded_path),
        "run_unique_genes": len(run_genes_set),
        "reference_unique_genes": len(reference_genes_set),
        "shared_unique_genes": len(shared_genes),
        "run_gene_overlap_fraction": (
            len(shared_genes) / len(run_genes_set) if run_genes_set else 0.0
        ),
        "reference_gene_overlap_fraction": (
            len(shared_genes) / len(reference_genes_set) if reference_genes_set else 0.0
        ),
        **_coordinate_extent_metrics("run_rna", run_df, coordinate_columns),
        **_coordinate_extent_metrics(
            "reference_rna", comparable_reference_df, coordinate_columns
        ),
    }


def _load_polygons(path: Path) -> dict[int, Any]:
    try:
        from shapely.geometry import shape
    except ImportError as exc:
        raise RuntimeError(
            "Max-projection cell matching requires shapely. Install shapely in "
            "the analysis environment, then rerun this script."
        ) from exc

    with gzip.open(path, "rt") as handle:
        data = json.load(handle)

    polygons: dict[int, Any] = {}
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        if "cell" not in props:
            continue
        cell_id = int(props["cell"])
        geom = shape(feature.get("geometry"))
        if geom.is_empty or geom.area <= 0:
            continue
        polygons[cell_id] = geom
    return polygons


def _query_tree(tree: Any, geom: Any, indexed_geoms: list[Any]) -> list[int]:
    result = tree.query(geom)
    if len(result) == 0:
        return []
    first = result[0]
    if isinstance(first, (int, np.integer)):
        return [int(i) for i in result]

    by_id = {id(candidate): idx for idx, candidate in enumerate(indexed_geoms)}
    return [by_id[id(candidate)] for candidate in result if id(candidate) in by_id]


def _match_polygons_by_iou(
    reference_polygons: dict[int, Any],
    run_polygons: dict[int, Any],
    min_iou: float,
) -> pd.DataFrame:
    try:
        from shapely.strtree import STRtree
    except ImportError as exc:
        raise RuntimeError(
            "Max-projection cell matching requires shapely. Install shapely in "
            "the analysis environment, then rerun this script."
        ) from exc

    run_items = list(run_polygons.items())
    if not reference_polygons or not run_items:
        return pd.DataFrame(columns=["reference_cell", "run_cell", "iou"])

    run_ids = [cell_id for cell_id, _geom in run_items]
    run_geoms = [geom for _cell_id, geom in run_items]
    tree = STRtree(run_geoms)
    candidates: list[tuple[float, int, int]] = []

    for reference_cell, reference_geom in reference_polygons.items():
        for run_idx in _query_tree(tree, reference_geom, run_geoms):
            run_geom = run_geoms[run_idx]
            intersection_area = reference_geom.intersection(run_geom).area
            if intersection_area <= 0:
                continue
            union_area = reference_geom.union(run_geom).area
            if union_area <= 0:
                continue
            iou = float(intersection_area / union_area)
            if iou >= min_iou:
                candidates.append((iou, reference_cell, run_ids[run_idx]))

    candidates.sort(key=lambda item: item[0], reverse=True)
    used_reference: set[int] = set()
    used_run: set[int] = set()
    matches: list[dict[str, Any]] = []
    for iou, reference_cell, run_cell in candidates:
        if reference_cell in used_reference or run_cell in used_run:
            continue
        used_reference.add(reference_cell)
        used_run.add(run_cell)
        matches.append(
            {
                "reference_cell": int(reference_cell),
                "run_cell": int(run_cell),
                "iou": float(iou),
            }
        )
    return pd.DataFrame(matches)


def _cell_gene_counts(path: Path) -> dict[int, dict[str, int]]:
    df = _load_transcripts(path)
    if "assignment" not in df.columns:
        raise ValueError(f"{path} is missing required column: assignment")

    assigned = pd.to_numeric(df["assignment"], errors="coerce")
    df = df.loc[assigned.notna()].copy()
    df["cell_id"] = assigned.loc[assigned.notna()].astype(int).to_numpy()
    grouped = df.groupby(["cell_id", "gene"], observed=True).size()

    counts: dict[int, dict[str, int]] = {}
    for (cell_id, gene), count in grouped.items():
        counts.setdefault(int(cell_id), {})[str(gene)] = int(count)
    return counts


def _count_f1_from_matches(
    matches: pd.DataFrame,
    reference_counts: dict[int, dict[str, int]],
    run_counts: dict[int, dict[str, int]],
) -> dict[str, Any]:
    tp = fp = fn = 0
    for row in matches.itertuples(index=False):
        reference = reference_counts.get(int(row.reference_cell), {})
        run = run_counts.get(int(row.run_cell), {})
        genes = set(reference) | set(run)
        for gene in genes:
            reference_count = int(reference.get(gene, 0))
            run_count = int(run.get(gene, 0))
            tp += min(reference_count, run_count)
            fp += max(run_count - reference_count, 0)
            fn += max(reference_count - run_count, 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "max_proj_cell_count_f1": float(f1),
        "max_proj_cell_count_precision": float(precision),
        "max_proj_cell_count_recall": float(recall),
        "max_proj_cell_count_tp": int(tp),
        "max_proj_cell_count_fp": int(fp),
        "max_proj_cell_count_fn": int(fn),
    }


def _match_cells_by_id(
    reference_cell_ids: set[int],
    run_cell_ids: set[int],
) -> pd.DataFrame:
    shared_cells = sorted(reference_cell_ids & run_cell_ids)
    return pd.DataFrame(
        {
            "reference_cell": shared_cells,
            "run_cell": shared_cells,
            "iou": np.nan,
        }
    )


def _max_projection_cell_count_f1(
    run: RunInfo,
    reference_polygons: dict[int, Any],
    reference_counts: dict[int, dict[str, int]],
    min_cell_iou: float,
    cell_match_mode: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    run_polygons = _load_polygons(
        run.path / "max_proj" / "cell_polygons_max_proj.geojson.gz"
    )
    run_counts = _cell_gene_counts(
        run.path / "max_proj" / "transcript_metadata_max_proj.csv.gz"
    )

    if cell_match_mode == "id":
        matches = _match_cells_by_id(set(reference_polygons), set(run_polygons))
    elif cell_match_mode == "iou":
        matches = _match_polygons_by_iou(reference_polygons, run_polygons, min_cell_iou)
    else:
        raise ValueError("cell_match_mode must be 'id' or 'iou'.")

    count_metrics = _count_f1_from_matches(matches, reference_counts, run_counts)

    matched_reference = (
        set(matches["reference_cell"].astype(int)) if not matches.empty else set()
    )
    matched_run = set(matches["run_cell"].astype(int)) if not matches.empty else set()
    ious = (
        matches["iou"].dropna().to_numpy(dtype=float)
        if not matches.empty
        else np.array([])
    )

    metrics = {
        **count_metrics,
        "matched_cells": len(matches),
        "unmatched_reference_cells": int(
            len(reference_polygons) - len(matched_reference)
        ),
        "unmatched_run_cells": int(len(run_polygons) - len(matched_run)),
        "mean_cell_iou": float(np.mean(ious)) if ious.size else 0.0,
        "median_cell_iou": float(np.median(ious)) if ious.size else 0.0,
        "reference_cell_count": len(reference_polygons),
        "run_cell_count": len(run_polygons),
        "reference_cells_with_counts": len(reference_counts),
        "run_cells_with_counts": len(run_counts),
        "cell_match_mode": cell_match_mode,
    }
    matches = matches.copy()
    matches.insert(0, "run", run.name)
    return metrics, matches


def _write_outputs(
    proseg_root: Path,
    output_prefix: str,
    summary: pd.DataFrame,
    matches: pd.DataFrame,
) -> tuple[Path, Path | None, Path]:
    csv_path = proseg_root / f"{output_prefix}.csv"
    parquet_path = proseg_root / f"{output_prefix}.parquet"
    matches_path = proseg_root / f"{output_prefix}_cell_matches.csv"

    summary.to_csv(csv_path, index=False)
    try:
        summary.to_parquet(parquet_path, index=False)
    except Exception:
        parquet_path = None
    matches.to_csv(matches_path, index=False)
    return csv_path, parquet_path, matches_path


def _print_summary(summary: pd.DataFrame) -> None:
    columns = [
        "run",
        "cell_match_mode",
        "rna_f1",
        "rna_precision",
        "rna_recall",
        "run_rna_count",
        "comparable_reference_rna_count",
        "max_proj_cell_count_f1",
        "max_proj_cell_count_precision",
        "max_proj_cell_count_recall",
        "matched_cells",
        "unmatched_reference_cells",
        "unmatched_run_cells",
    ]
    if (
        "cell_match_mode" in summary.columns
        and (summary["cell_match_mode"] == "iou").any()
    ):
        columns.append("median_cell_iou")
    printable = summary.loc[
        :, [col for col in columns if col in summary.columns]
    ].copy()
    float_cols = printable.select_dtypes(include=[float]).columns
    printable.loc[:, float_cols] = printable.loc[:, float_cols].round(4)
    print()
    print("Z-stride proseg F1 summary")
    print(printable.to_string(index=False))


def main() -> None:
    args = _parse_args()
    proseg_root = args.proseg_root.expanduser().resolve()
    datastore_root = proseg_root.parent
    datastore = qi2labDataStore(datastore_root)
    voxel_zyx_um = np.asarray(datastore.voxel_size_zyx_um, dtype=np.float64)
    if voxel_zyx_um.shape[0] != 3:
        raise ValueError(f"Expected 3 voxel dimensions, got {voxel_zyx_um!r}")

    runs = _discover_runs(proseg_root, datastore)
    runs_by_name = {run.name: run for run in runs}
    if args.reference_run not in runs_by_name:
        raise ValueError(
            f"Reference run {args.reference_run!r} was not found under {proseg_root}."
        )

    reference_run = runs_by_name[args.reference_run]
    reference_decoded_path = _decoded_features_path(datastore_root, reference_run)
    reference_df = _load_decoded_features(datastore_root, reference_run)
    reference_polygons = _load_polygons(
        reference_run.path / "max_proj" / "cell_polygons_max_proj.geojson.gz"
    )
    reference_counts = _cell_gene_counts(
        reference_run.path / "max_proj" / "transcript_metadata_max_proj.csv.gz"
    )

    rows: list[dict[str, Any]] = []
    match_tables: list[pd.DataFrame] = []
    for run in runs:
        rna_metrics = _rna_f1(run, reference_df, datastore_root, voxel_zyx_um)
        cell_metrics, matches = _max_projection_cell_count_f1(
            run,
            reference_polygons,
            reference_counts,
            args.min_cell_iou,
            args.cell_match_mode,
        )
        rows.append(
            {
                "run": run.name,
                "zstride": run.zstride,
                "decode_mode": run.decode_mode,
                "reference_run": reference_run.name,
                "reference_decoded_features_path": str(reference_decoded_path),
                "voxel_z_um": float(voxel_zyx_um[0]),
                "voxel_y_um": float(voxel_zyx_um[1]),
                "voxel_x_um": float(voxel_zyx_um[2]),
                "min_cell_iou": float(args.min_cell_iou),
                "cell_match_mode": args.cell_match_mode,
                **rna_metrics,
                **cell_metrics,
            }
        )
        match_tables.append(matches)

    summary = (
        pd.DataFrame(rows)
        .sort_values(["zstride", "decode_mode"])
        .reset_index(drop=True)
    )
    all_matches = (
        pd.concat(match_tables, ignore_index=True)
        if match_tables
        else pd.DataFrame(columns=["run", "reference_cell", "run_cell", "iou"])
    )
    csv_path, parquet_path, matches_path = _write_outputs(
        proseg_root, args.output_prefix, summary, all_matches
    )

    print(f"Wrote summary CSV: {csv_path}")
    if parquet_path is not None:
        print(f"Wrote summary Parquet: {parquet_path}")
    print(f"Wrote cell matches CSV: {matches_path}")
    if args.print_summary:
        _print_summary(summary)


if __name__ == "__main__":
    main()
