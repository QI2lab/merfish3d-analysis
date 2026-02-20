#!/usr/bin/env python3
# correlation.py
#
# Usage:
#   python correlation.py file1a file2 gene_col1a gene_col2 fpkm_col \
#       [--file1b FILE] [--gene-col1b COL] [--cellid-col1b COL] [...]
#
# Example (two datasets):
#   python correlation.py spots_a.csv bulk.txt target_molecule_name gene_id FPKM \
#     --file1b spots_b.csv --gene-col1b target_gene \
#     --file2-sep "\t" --min-fpkm 0.1 --only-cell-id-positive
#
# Notes:
# - file1a/file1b: .txt / .csv / .parquet (only the specified gene column is used; counts = occurrences)
# - file2: .txt (contains gene + FPKM)
# - Optional: filter file1a/file1b rows to keep only those with cell_id > 0

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _load_file1(path: str, sep_override: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        typer.secho(f"ERROR: file1 not found: {p}", fg=typer.colors.RED)
        raise typer.Exit(code=2)
    ext = p.suffix.lower()
    try:
        if ext == ".parquet":
            df = pd.read_parquet(p)
        elif ext == ".csv":
            df = pd.read_csv(p, sep=sep_override if sep_override else ",")
        elif ext == ".txt":
            if sep_override:
                df = pd.read_csv(p, sep=sep_override)
            else:
                df = pd.read_csv(p, sep=None, engine="python")  # sniff
        else:
            typer.secho(
                f"ERROR: file1 must be .txt, .csv, or .parquet (got {ext}).",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=2)
    except Exception as e:  # noqa: BLE001
        typer.secho(
            f"ERROR: failed to read file1 ({p}): {e}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)
    return df


def _load_file2_txt(path: str, sep_override: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        typer.secho(f"ERROR: file2 not found: {p}", fg=typer.colors.RED)
        raise typer.Exit(code=2)
    if p.suffix.lower() != ".txt":
        typer.secho(
            f"ERROR: file2 must be .txt (got {p.suffix}).",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)
    try:
        if sep_override:
            return pd.read_csv(p, sep=sep_override)
        return pd.read_csv(p, sep=None, engine="python")  # sniff
    except Exception as e:  # noqa: BLE001
        typer.secho(
            f"ERROR: failed to read file2 ({p}): {e}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

def _drop_gene_prefixes(s: pd.Series, prefixes: list[str]) -> pd.Series:
    """
    Mark genes whose names begin with any prefix in `prefixes` as NaN.

    This returns a Series with the SAME index/length as the input,
    so it can be used safely for boolean indexing via .notna().
    """
    if not prefixes:
        return s

    prefixes_upper = tuple(p.upper() for p in prefixes)
    mask = ~s.str.upper().str.startswith(prefixes_upper)
    # Keep non-matching entries; turn matches into NaN
    return s.where(mask)

def _strip_trailing_dash_number(s: pd.Series) -> pd.Series:
    """
    Remove trailing -1, -2, -3, ... from gene names.

    Examples
    --------
    OR10A2-1 -> OR10A2
    OR10A2-2 -> OR10A2
    ABC-XYZ  -> ABC-XYZ  (unchanged; no trailing number)
    """
    return s.str.replace(r"-\d+$", "", regex=True)


def _counts_vs_fpkm(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    gene_col1: str,
    gene_col2: str,
    fpkm_col: str,
    only_pos_cell: bool,
    cell_col: str,
    label_for_errors: str,
    drop_prefix: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Return merged counts-vs-FPKM table and genes in file1 not found in file2.

    Both sides are normalized by stripping trailing -1/-2/... from gene names.
    """
    if gene_col1 not in df1.columns:
        typer.secho(
            f"ERROR: '{gene_col1}' not in {label_for_errors}. "
            f"Columns: {list(df1.columns)}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)
    for col in (gene_col2, fpkm_col):
        if col not in df2.columns:
            typer.secho(
                f"ERROR: '{col}' not in file2. Columns: {list(df2.columns)}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=2)

    df1_work = df1
    if only_pos_cell:
        if cell_col not in df1.columns:
            typer.secho(
                "ERROR: Requested cell_id filtering but "
                f"'{cell_col}' not found in {label_for_errors}. "
                f"Columns: {list(df1.columns)}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=2)
        df1_work = df1.copy()
        # robust numeric filter: cell_id > 0
        df1_work[cell_col] = pd.to_numeric(df1_work[cell_col], errors="coerce")
        df1_work = df1_work[df1_work[cell_col] > 0]
        if df1_work.empty:
            typer.secho(
                f"ERROR: After '{cell_col}' > 0 filtering, "
                f"{label_for_errors} has no rows.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=2)

    # ---- File1 normalization & counting ----
    genes1 = (
        df1_work[gene_col1]
        .dropna()
        .astype(str)
        .pipe(_strip_trailing_dash_number)
    )

    # Drop genes starting with any of the requested prefixes
    genes1 = _drop_gene_prefixes(genes1, drop_prefix)

    # Remove the NaNs created by prefix filtering
    genes1 = genes1.dropna()

    # Ignore "Blank" (after normalization)
    genes1 = genes1[genes1.str.upper() != "BLANK"]

    counts_df = (
        genes1.value_counts()
        .rename_axis("gene_id")
        .reset_index(name="count")
    )
    counts_df["count"] = counts_df["count"].astype("int64")
    print(genes1)

    # ---- File2 normalization & FPKM aggregation ----
    df2 = df2.copy()
    df2[gene_col2] = (
        df2[gene_col2]
        .astype(str)
        .pipe(_strip_trailing_dash_number)
    )

    # Apply prefix dropping (creates NaNs for dropped genes)
    df2[gene_col2] = _drop_gene_prefixes(df2[gene_col2], drop_prefix)

    # Remove rows where gene was dropped by prefix filter
    df2 = df2[df2[gene_col2].notna()]

    # Ignore "Blank" after normalization
    df2 = df2[df2[gene_col2].str.upper() != "BLANK"]
    df2 = df2[df2[fpkm_col] >= 1e-2]

    fpkm_by_gene = (
        df2.groupby(gene_col2, dropna=False, as_index=False)[fpkm_col]
        .mean()
        .rename(columns={gene_col2: "gene_id", fpkm_col: "fpkm"})
    )

    # ---- Overlap + filtering ----
    merged = counts_df.merge(fpkm_by_gene, on="gene_id", how="inner")
    merged = merged.dropna(subset=["count", "fpkm"])
    merged = merged[(merged["count"] > 0) & (merged["fpkm"] > 0)]

    if merged.empty:
        typer.secho(
            "ERROR: No overlapping genes with positive counts and "
            f"positive FPKM for {label_for_errors}.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    # ---- Genes in file1 not found in file2 (after normalization) ----
    file1_genes = set(counts_df["gene_id"])
    file2_genes = set(fpkm_by_gene["gene_id"])
    unmatched = sorted(file1_genes - file2_genes)

    return merged, unmatched


def _pearson_loglog_x_fpkm_y_counts(merged: pd.DataFrame) -> float:
    # Pearson correlation in log10 space with x=FPKM, y=counts
    x = merged["fpkm"].to_numpy(dtype=float)
    y = merged["count"].to_numpy(dtype=float)
    logx = np.log10(x)
    logy = np.log10(y)
    return float(np.corrcoef(logx, logy)[0, 1])


@app.command(
    help=(
        "One or two file1 inputs vs one file2: FPKM on X, counts on Y; "
        "filter by FPKM and optional cell_id>0; no trend lines; 2× styling; only r."
    ),
)
def main(
    file1a: str = typer.Argument(
        ...,
        help="First file1 (.txt/.csv/.parquet).",
    ),
    file2: str = typer.Argument(
        ...,
        help="file2 (.txt) with gene IDs and FPKM.",
    ),
    gene_col1a: str = typer.Argument(
        ...,
        help="Column in file1a containing gene IDs.",
    ),
    gene_col2: str = typer.Argument(
        ...,
        help="Column in file2 containing gene IDs.",
    ),
    fpkm_col: str = typer.Argument(
        ...,
        help="Column in file2 containing FPKM.",
    ),
    # Optional second dataset (file1b)
    file1b: str | None = typer.Option(
        None,
        "--file1b",
        help="Optional second file1 (.txt/.csv/.parquet).",
    ),
    gene_col1b: str | None = typer.Option(
        None,
        "--gene-col1b",
        help="Column in file1b containing gene IDs (required if --file1b is set).",
    ),
    plot_out: str = typer.Option(
        "fpkm_vs_counts_dual.png",
        help="Output plot filename.",
    ),
    file1a_sep: str = typer.Option(
        "",
        help="Optional delimiter for file1a (.txt/.csv). E.g., '\\t' or ','.",
    ),
    file1b_sep: str = typer.Option(
        "",
        help="Optional delimiter for file1b (.txt/.csv). E.g., '\\t' or ','.",
    ),
    file2_sep: str = typer.Option(
        "",
        help="Optional delimiter for file2 (.txt). E.g., '\\t' or ','.",
    ),
    min_fpkm: float = typer.Option(
        0.0,
        help=(
            "Minimum FPKM threshold to include (e.g., 0.1 for 10^-1). "
            "Genes with FPKM <= threshold are excluded."
        ),
    ),
    only_cell_id_positive: bool = typer.Option(
        False,
        "--only-cell-id-positive / --no-only-cell-id-positive",
        help=(
            "If set, keep only rows with cell_id > 0 in file1a (and file1b, "
            "if provided) before counting."
        ),
    ),
    cellid_col1a: str = typer.Option(
        "cell_id",
        help="Cell ID column name in file1a (used only if --only-cell-id-positive).",
    ),
    cellid_col1b: str = typer.Option(
        "cell_id",
        help=(
            "Cell ID column name in file1b "
            "(used only if --file1b and --only-cell-id-positive)."
        ),
    ),
    drop_prefix: list[str] = typer.Option(
        [],
        "--drop-prefix",
        help="Remove any genes whose name begins with this prefix (case-insensitive). "
            "May be passed multiple times. Applied after normalization.",
    )
):
    # Validate second dataset arguments
    if file1b is not None and gene_col1b is None:
        typer.secho(
            "ERROR: --file1b was provided but --gene-col1b is missing.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    # Load
    df1a = _load_file1(file1a, file1a_sep)
    df2 = _load_file2_txt(file2, file2_sep)
    df1b: pd.DataFrame | None = None
    if file1b is not None:
        df1b = _load_file1(file1b, file1b_sep)
    min_fpkm = float(min_fpkm)

    # Compute merged tables (apply cell_id filter if requested)
    merged_a, unmatched_a = _counts_vs_fpkm(
        df1a,
        df2,
        gene_col1a,
        gene_col2,
        fpkm_col,
        only_cell_id_positive,
        cellid_col1a,
        "file1a",
        drop_prefix
    )

    merged_b: pd.DataFrame | None = None
    if df1b is not None and gene_col1b is not None:
        merged_b, unmatched_b = _counts_vs_fpkm(
            df1b,
            df2,
            gene_col1b,
            gene_col2,
            fpkm_col,
            only_cell_id_positive,
            cellid_col1b,
            "file1b",
        )
        # If you also want to see unmatched for file1b, you can print unmatched_b here.

    # Apply FPKM threshold filter
    if min_fpkm > 0.0:
        merged_a = merged_a[merged_a["fpkm"] > min_fpkm]
        if merged_b is not None:
            merged_b = merged_b[merged_b["fpkm"] > min_fpkm]

    if merged_a.empty:
        typer.secho(
            f"ERROR: After filtering (min_fpkm={min_fpkm}), "
            "no genes remain for file1a.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    if merged_b is not None and merged_b.empty:
        typer.secho(
            f"ERROR: After filtering (min_fpkm={min_fpkm}), "
            "no genes remain for file1b.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    # Correlations (log10 space; x=FPKM, y=counts)
    rA = _pearson_loglog_x_fpkm_y_counts(merged_a)
    rB = _pearson_loglog_x_fpkm_y_counts(merged_b) if merged_b is not None else None

    # Fixed labels
    labelA = "merfish3d"
    labelB = "merlin"

    # ---- Styling: ~2× bigger everything ----
    scale = 2.0
    base_font = 12.0
    plt.rcParams.update(
        {
            "font.size": base_font * scale,
            "axes.labelsize": base_font * scale,
            "xtick.labelsize": base_font * scale,
            "ytick.labelsize": base_font * scale,
            "legend.fontsize": base_font * scale,
        },
    )
    marker_area = 12 * (scale**2)
    spine_lw = 1.0 * scale
    tick_len = 3.5 * scale
    tick_w = 1.0 * scale
    ann_font = base_font * scale

    # Plot
    fig, ax = plt.subplots(figsize=(7.5 * scale, 5.5 * scale), dpi=150)

    colorA = "C0"
    colorB = "C1"

    # Scatter: x = FPKM, y = counts
    ax.scatter(
        merged_a["fpkm"],
        merged_a["count"],
        s=marker_area,
        alpha=0.7,
        marker="o",
        label=labelA if merged_b is not None else None,
        edgecolors="none",
        c=colorA,
    )

    if merged_b is not None:
        ax.scatter(
            merged_b["fpkm"],
            merged_b["count"],
            s=marker_area,
            alpha=0.7,
            marker="s",
            label=labelB,
            edgecolors="none",
            c=colorB,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Bulk RNA-Seq (FPKM)")
    ax.set_ylabel("MERFISH (total counts in ROI)")

    # Minimal look: remove top/right box, thicken remaining axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(spine_lw)
    ax.spines["bottom"].set_linewidth(spine_lw)
    ax.tick_params(axis="both", which="both", length=tick_len, width=tick_w)

    # Only correlation coefficients on the plot
    ax.text(
        0.03,
        0.97,
        f"{labelA}: r = {rA:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color=colorA,
        fontsize=ann_font,
    )

    if rB is not None:
        ax.text(
            0.03,
            0.90,
            f"{labelB}: r = {rB:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            color=colorB,
            fontsize=ann_font,
        )

    # Legend only if we actually have labels
    handles, labels = ax.get_legend_handles_labels()
    if any(labels):
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.tight_layout()
    fig.savefig(plot_out, bbox_inches="tight")
    plt.close(fig)
    typer.secho(f"Wrote plot: {plot_out}", fg=typer.colors.GREEN)

    # ---- Output genes from file1a not found in reference (after normalization) ----
    if unmatched_a:
        typer.secho(
            f"{len(unmatched_a)} genes in file1a (after normalization) "
            "were not found in file2:",
            fg=typer.colors.YELLOW,
        )
        for g in unmatched_a:
            typer.echo(g)
    else:
        typer.secho(
            "All genes in file1a (after normalization) were found in file2.",
            fg=typer.colors.GREEN,
        )


if __name__ == "__main__":
    app()
