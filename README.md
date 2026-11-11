# merfish3d-analysis

_WARNING: under active development._ Please expect breaking changes.

GPU accelerated post-processing for 2D / 3D iterative barcoded FISH data. This package is currently **NVIDIA only** and **Linux only**. Documentation, including examples, is available at [https://qi2lab.github.io/merfish3d-analysis/](https://qi2lab.github.io/merfish3d-analysis/).

## Associated preprint publication
[GPU-accelerated, self-optimizing processing for 3D multiplexed iterative RNA-FISH experiments](https://www.biorxiv.org/content/10.1101/2025.10.10.681751v1).

## Try it without installation
You can try out the package in the cloud on simulated data using a [Google Colab notebook](https://colab.research.google.com/github/QI2lab/merfish3d-analysis/blob/main/examples/notebooks/Simulation_example.ipynb) that demonstrates data preprocessing and decoding.

## Sample data

The GPU integration test workflow uses the StatPhysBio MERFISH simulations archived
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
fused fiducial OME-Zarr creation follow the multiview-stitcher workflow, using
CPU registration and CPU direct-to-Zarr fusion.

## Fuse all channels

After local and global registration, fuse the first-round fiducial and codebook
bits into a multichannel image in `qi2labdatastore/fused.ome.zarr`:

```bash
uv run qi2lab-fuseall /path/to/experiment \
  --output-chunk-zyx 32 2048 2048 \
  --fusion-workers 30 \
  --compression blosc-lz4 \
  --compression-level 1
```

Fusion uses lossless compression. The default is `blosc-zstd` with bitshuffle
at level 1; `blosc-lz4` and `zstd` are also available at levels 1 through 9.

The output is an OME-NGFF v0.6 image, with the fiducial first and then bits in
numeric codebook order. The command uses stored local-round, stage and global
transforms; chromatic correction is applied during decoding and SOFIMA is
omitted from fusion.
See the [fusion workflow](https://qi2lab.github.io/merfish3d-analysis/workflow/#fuse-all-channels)
for resource settings and restart behavior.

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

The viewer displays native and registered tiles, fused images, transcripts, and
cell boundaries. It can display transcripts and boundaries from the datastore,
Proseg, or Baysor.

The documentation includes a [viewer guide with screenshots](https://qi2lab.github.io/merfish3d-analysis/viewer/) and a [current API notes](https://qi2lab.github.io/merfish3d-analysis/api/) page covering the current RLGC, PSF, and U-FISH defaults.

## Proseg segmentation optimization

Follow the installation instructions at [proseg](https://github.com/dcjones/proseg).

Example calls to proseg:

2D segmentation optimization (ignore z coordinate)
```bash
mkdir /path/to/qi2labdatastore/proseg/2D

/path/to/proseg --gene-column gene_id -x global_x -y global_y -z global_z --fov-column tile_idx --cell-id-column cell_id --cell-id-unassigned 0 --excluded-genes ^[Bb]lank.*$ --ignore-z-coord --density-bins 1 --burnin-samples 1000 --samples 2000 --voxel-size 1.0 --burnin-voxel-size 4.0 --enforce-connectivity --diffusion-probability 0.0 --output-spatialdata /path/to/data/qi2labdatastore/proseg/2D/spatialdata_2D.zarr --output-counts /path/to/data/qi2labdatastore/proseg/2D/counts_2D.mtx.gz --output-cell-polygons /path/to/data/qi2labdatastore/proseg/2D/cell_polygons_2D.geojson.gz --output-transcript-metadata /path/to/data/qi2labdatastore/proseg/2D/transcript_metadata_2D.csv.gz /path/to/data/qi2labdatastore/codewords/global_filtered/decoded_features.csv.gz
```

3D segmentation optimization. Here `--voxel-layers` should be roughly set to the height of the imaged volume in microns. For example, a 15 micron thick sample should have `--voxel-layers 15`.
```bash
mkdir /path/to/qi2labdatastore/proseg/3D

/path/to/proseg --gene-column gene_id -x global_x -y global_y -z global_z --fov-column tile_idx --cell-id-column cell_id --cell-id-unassigned 0 --excluded-genes ^[Bb]lank.*$ --voxel-layers 15 --density-bins 1 --burnin-samples 1000 --samples 2000 --voxel-size 1.0 --burnin-voxel-size 4.0 --enforce-connectivity --diffusion-probability 0.0 --output-spatialdata /path/to/data/qi2labdatastore/proseg/3D/spatialdata_3D.zarr --output-counts /path/to/data/qi2labdatastore/proseg/3D/counts_3D.mtx.gz --output-cell-polygons-layers /path/to/data/qi2labdatastore/proseg/3D/cell_polygons_3D.geojson.gz --output-transcript-metadata /path/to/data/qi2labdatastore/proseg/3D/transcript_metadata_3D.csv.gz /path/to/data/qi2labdatastore/codewords/global_filtered/decoded_features.csv.gz
```

## Documentation

To build the documentation, install using `uv sync --group docs`.
Then execute `uv run mkdocs build --clean` or `uv run mkdocs serve`. The
documentation is available in your web browser at `http://127.0.0.1:8000/`.

The `Documentation` GitHub Actions workflow checks documentation changes on
pull requests and deploys the site after relevant pushes to `main`. Set the
repository's Pages source to **GitHub Actions**.

## Testing

The suite consists of unit tests and integration tests. Every test has exactly
one of these markers. Run either category with:

```bash
uv run pytest -m unit
uv run pytest -m integration
```

See the [testing guidelines](docs/contributing.md#testing) for the required
assertions and how to classify tests.

The simulation tests require an NVIDIA GPU and the extracted Zenodo sample data.
Pass the dataset directory with `--simulation-data-root`. See the
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
