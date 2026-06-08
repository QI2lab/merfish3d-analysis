#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROSEG_ROOT="${PROSEG_ROOT:-/home/dps/server_data2/bioprotean/20250513_Bartelle_MERFISH_control/qi2labdatastore/proseg}"
REFERENCE_RUN="${REFERENCE_RUN:-zstride01_3d}"
MIN_CELL_IOU="${MIN_CELL_IOU:-0.25}"
CELL_MATCH_MODE="${CELL_MATCH_MODE:-id}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-zstride_f1_summary}"
PYTHON_BIN="${PYTHON_BIN:-/media/dps/data/miniforg3/envs/merfish3d/bin/python}"

cd "$REPO_ROOT"

PYTHONPATH=src "$PYTHON_BIN" scripts/compare_zstride_proseg_f1.py \
  --proseg-root "$PROSEG_ROOT" \
  --reference-run "$REFERENCE_RUN" \
  --min-cell-iou "$MIN_CELL_IOU" \
  --cell-match-mode "$CELL_MATCH_MODE" \
  --output-prefix "$OUTPUT_PREFIX" \
  --print-summary
