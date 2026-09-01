# merfish3d-analysis

_WARNING: under active development._ Please expect breaking changes.

GPU accelerated post-processing for 2D / 3D iterative barcoded FISH data. This package is currently **NVIDIA only** and **Linux only**. Documentation, including examples, is available at [https://qi2lab.github.io/merfish3d-analysis/](https://qi2lab.github.io/merfish3d-analysis/).

## Associated preprint publication
[GPU-accelerated, self-optimizing processing for 3D multiplexed iterative RNA-FISH experiments](https://www.biorxiv.org/content/10.1101/2025.10.10.681751v1).

## Try it without installation
You can try out the package in the cloud on simulated data using a [Google Colab notebook](https://colab.research.google.com/github/QI2lab/merfish3d-analysis/blob/main/examples/notebooks/Simulation_example.ipynb) that demonstrates data preprocessing and decoding.

## Sample data

The GPU regression workflow uses the StatPhysBio MERFISH simulations archived
on Zenodo. Download the
[simulation test data](https://zenodo.org/records/17274305/files/merfish3d_analysis-simulation.zip?download=1)
or follow the complete [synthetic-data example](https://qi2lab.github.io/merfish3d-analysis/examples/statphysbio_synthetic/)
for checksum verification, extraction, model setup, and test commands.

## Installation

This project uses one `uv` environment based on CUDA 12.9 runtime/toolkit. You need a compatible NVIDIA driver installed on the machine.

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and enter it:

```bash
git clone https://github.com/QI2lab/merfish3d-analysis
cd merfish3d-analysis
```

Create and sync the environment:

```bash
uv sync
```

For development tools, include the `dev` group:

```bash
uv sync --group dev
```

For documentation tools, include the `docs` group:

```bash
uv sync --group docs
```

Run package entry points through `uv run`, for example:

```bash
uv run qi2lab-preprocess /path/to/experiment
```

## Registration conventions

Local fiducial registration uses the qi2lab GPU registration path: lateral XY
registration on max-Z projections, XYZ registration, and optional SOFIMA
residual flow fields after affine fiducial alignment. Global registration and
fused fiducial OME-Zarr creation follow the multiview-stitcher workflow; GPU
acceleration is used in the direct fusion step.

## View a qi2lab datastore

The standard `uv sync` install includes the NDV, VisPy, and PyQt dependencies
for the view-only datastore GUI.

Open an experiment root:

```bash
uv run viewer /path/to/experiment
```

or open a datastore directly:

```bash
uv run viewer /path/to/experiment/qi2labdatastore
```

The viewer only reads existing datastore contents. It uses NDV for Zarr-backed
image channels and VisPy for sparse transcript and cell-boundary overlays. It
can inspect local native tiles, local tiles warped with selected chromatic,
stage/round affine, and SOFIMA components, and global fused views. Transcript
sources are mutually exclusive: datastore transcripts, Proseg transcripts, or
Baysor transcripts. Cell boundaries can be shown from Cellpose, Proseg, or
Baysor when those outputs are present.

The documentation includes a [viewer guide with screenshots](https://qi2lab.github.io/merfish3d-analysis/viewer/) and a [current API notes](https://qi2lab.github.io/merfish3d-analysis/api/) page covering the current RLGC, PSF, and U-FISH defaults.

## Proseg segmentation optimization

Follow the installation instructions at [proseg](https://github.com/dcjones/proseg).

Example calls to proseg:

2D segmentation optimization (ignore z coordinate)
```bash
mkdir /path/to/qi2labdatastore/proseg/2D

/path/to/proseg --gene-column gene_id -x global_x -y global_y -z global_z --fov-column tile_idx --cell-id-column cell_id --cell-id-unassigned 0 --excluded-genes ^[Bb]lank.*$ --ignore-z-coord --density-bins 1 --burnin-samples 1000 --samples 2000 --voxel-size 1.0 --burnin-voxel-size 4.0 --enforce-connectivity --diffusion-probability 0.0 --output-spatialdata /path/to/data/qi2labdatastore/proseg/2D/spatialdata_2D.zarr --output-counts /path/to/data/qi2labdatastore/proseg/2D/counts_2D.mtx.gz --output-cell-polygons /path/to/data/qi2labdatastore/proseg/2D/cell_polygons_2D.geojson.gz --output-transcript-metadata /path/to/data/qi2labdatastore/proseg/2D/transcript_metadata_2D.csv.gz /path/to/data/qi2labdatastore/all_tiles_filtered_decoded_features/decoded_features.csv.gz
```

3D segmentation optimization. Here `--voxel-layers` should be roughly set to the height of the imaged volume in microns. For example, a 15 micron thick sample should have `--voxel-layers 15`.
```bash
mkdir /path/to/qi2labdatastore/proseg/3D

/path/to/proseg --gene-column gene_id -x global_x -y global_y -z global_z --fov-column tile_idx --cell-id-column cell_id --cell-id-unassigned 0 --excluded-genes ^[Bb]lank.*$ --voxel-layers 15 --density-bins 1 --burnin-samples 1000 --samples 2000 --voxel-size 1.0 --burnin-voxel-size 4.0 --enforce-connectivity --diffusion-probability 0.0 --output-spatialdata /path/to/data/qi2labdatastore/proseg/3D/spatialdata_3D.zarr --output-counts /path/to/data/qi2labdatastore/proseg/3D/counts_3D.mtx.gz --output-cell-polygons-layers /path/to/data/qi2labdatastore/proseg/3D/cell_polygons_3D.geojson.gz --output-transcript-metadata /path/to/data/qi2labdatastore/proseg/3D/transcript_metadata_3D.csv.gz /path/to/data/qi2labdatastore/all_tiles_filtered_decoded_features/decoded_features.csv.gz
```

## Documentation

To build the documentation, install using `uv sync --group docs`.
Then execute `uv run mkdocs build --clean` or `uv run mkdocs serve`. The
documentation is available in your web browser at `http://127.0.0.1:8000/`.

Pushes to `main` automatically build and deploy the documentation through the
`Documentation` GitHub Actions workflow. The workflow installs only the
documentation dependency group and runs on a standard CPU runner; it does not
require CUDA or GPU hardware. Configure the repository's Pages source as
**GitHub Actions** under **Settings → Pages**.

## Testing

The simulation tests require an NVIDIA GPU, the extracted Zenodo sample data,
and the default `smfish` U-FISH weights cached by U-FISH. Pass the extracted
dataset directory with `--simulation-data-root`. See the
[synthetic-data example](https://qi2lab.github.io/merfish3d-analysis/examples/statphysbio_synthetic/)
for setup instructions.

Standard simulation matrix:

```bash
uv run pytest tests/test_simulation_example_pipeline.py -vv \
  --simulation-data-root simulation-data/merfish3d_analysis-simulation
```

Full repository suite, including the exhaustive simulation matrix:

```bash
uv run pytest -vv --run-simulation-exhaustive \
  --simulation-data-root simulation-data/merfish3d_analysis-simulation
```

The standard simulation matrix uses CUDA device 0 and runs paired affine and
SOFIMA preprocessing for every default dataset, axial spacing, and
chromatic-aberration setting. The exhaustive mode adds deconvolution-on and
deconvolution-off coverage using SOFIMA and synthetic chromatic aberration. It
writes measured performance records to
`tests/data/simulation_performance.json`. The tests verify the Zenodo image
fingerprints and enforce the documented F1 baselines with an absolute tolerance
of 0.02.
