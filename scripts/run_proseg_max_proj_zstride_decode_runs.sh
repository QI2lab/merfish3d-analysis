#!/usr/bin/env bash
set -euo pipefail

# Run proseg max-projection resegmentation for every z-strided decode run.
#
# Defaults match the Bartelle control example. Override any setting at launch:
#   DATASTORE=/path/to/qi2labdatastore \
#   PROSEG_BIN=/path/to/proseg \
#   scripts/run_proseg_max_proj_zstride_decode_runs.sh
#
# To run only selected decode runs, pass run keys as arguments:
#   scripts/run_proseg_max_proj_zstride_decode_runs.sh zstride_02_3d zstride_06_3d

PROSEG_BIN="${PROSEG_BIN:-$HOME/Documents/github/proseg/target/release/proseg}"
DATASTORE="${DATASTORE:-/media/dps/data2/bioprotean/20250513_Bartelle_MERFISH_control/qi2labdatastore}"
DECODED_ROOT="${DECODED_ROOT:-$DATASTORE/all_tiles_filtered_decoded_features}"
PROSEG_ROOT="${PROSEG_ROOT:-$DATASTORE/proseg}"

GENE_COLUMN="${GENE_COLUMN:-gene_id}"
X_COLUMN="${X_COLUMN:-global_x}"
Y_COLUMN="${Y_COLUMN:-global_y}"
Z_COLUMN="${Z_COLUMN:-global_z}"
FOV_COLUMN="${FOV_COLUMN:-tile_idx}"
CELL_ID_COLUMN="${CELL_ID_COLUMN:-cell_id}"
CELL_ID_UNASSIGNED="${CELL_ID_UNASSIGNED:-0}"
EXCLUDED_GENES_REGEX="${EXCLUDED_GENES_REGEX:-^[Bb]lank.*$}"

DENSITY_BINS="${DENSITY_BINS:-1}"
BURNIN_SAMPLES="${BURNIN_SAMPLES:-1000}"
SAMPLES="${SAMPLES:-2000}"
VOXEL_SIZE="${VOXEL_SIZE:-1.0}"
BURNIN_VOXEL_SIZE="${BURNIN_VOXEL_SIZE:-4.0}"
DIFFUSION_PROBABILITY="${DIFFUSION_PROBABILITY:-0.0}"
NTHREADS="${NTHREADS:-}"
OVERWRITE="${OVERWRITE:-1}"

if [[ ! -x "$PROSEG_BIN" ]]; then
  echo "proseg executable not found or not executable: $PROSEG_BIN" >&2
  exit 1
fi

if [[ ! -d "$DECODED_ROOT" ]]; then
  echo "Decoded root not found: $DECODED_ROOT" >&2
  exit 1
fi

run_keys=("$@")
if [[ ${#run_keys[@]} -eq 0 ]]; then
  while IFS= read -r run_dir; do
    run_keys+=("$(basename "$run_dir")")
  done < <(find "$DECODED_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'zstride_[0-9][0-9]_*' | sort)
fi

if [[ ${#run_keys[@]} -eq 0 ]]; then
  echo "No z-strided decoded runs found under $DECODED_ROOT" >&2
  exit 1
fi

mkdir -p "$PROSEG_ROOT"

for run_key in "${run_keys[@]}"; do
  input_transcripts="$DECODED_ROOT/$run_key/decoded_features.csv.gz"
  if [[ ! -f "$input_transcripts" ]]; then
    echo "Skipping $run_key: missing $input_transcripts" >&2
    continue
  fi

  # Keep the existing proseg naming convention from the example:
  # decoded key zstride_06_3d -> proseg folder zstride06_3d.
  proseg_run_key="${run_key/zstride_/zstride}"
  output_dir="$PROSEG_ROOT/$proseg_run_key/max_proj"

  mkdir -p "$output_dir"

  spatialdata_out="$output_dir/spatialdata_max_proj.zarr"
  counts_out="$output_dir/counts_max_proj.mtx.gz"
  polygons_out="$output_dir/cell_polygons_max_proj.geojson.gz"
  transcript_metadata_out="$output_dir/transcript_metadata_max_proj.csv.gz"

  cmd=(
    "$PROSEG_BIN"
    --gene-column "$GENE_COLUMN"
    -x "$X_COLUMN"
    -y "$Y_COLUMN"
    -z "$Z_COLUMN"
    --fov-column "$FOV_COLUMN"
    --cell-id-column "$CELL_ID_COLUMN"
    --cell-id-unassigned "$CELL_ID_UNASSIGNED"
    --excluded-genes "$EXCLUDED_GENES_REGEX"
    --ignore-z-coord
    --density-bins "$DENSITY_BINS"
    --burnin-samples "$BURNIN_SAMPLES"
    --samples "$SAMPLES"
    --voxel-size "$VOXEL_SIZE"
    --burnin-voxel-size "$BURNIN_VOXEL_SIZE"
    --enforce-connectivity
    --diffusion-probability "$DIFFUSION_PROBABILITY"
    --output-spatialdata "$spatialdata_out"
    --output-counts "$counts_out"
    --output-cell-polygons "$polygons_out"
    --output-transcript-metadata "$transcript_metadata_out"
  )

  if [[ -n "$NTHREADS" ]]; then
    cmd+=(--nthreads "$NTHREADS")
  fi

  if [[ "$OVERWRITE" == "1" ]]; then
    cmd+=(--overwrite)
  fi

  cmd+=("$input_transcripts")

  echo "Running max-projection proseg for $run_key"
  echo "  input:  $input_transcripts"
  echo "  output: $output_dir"
  "${cmd[@]}"
done
