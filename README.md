# merfish3d-analysis

_WARNING: under active development._ Please expect breaking changes.

GPU accelerated post-processing for 2D / 3D iterative barcoded FISH data. This package is currently **NVIDIA only** and **Linux only**. Documentation, including examples, is available at [https://qi2lab.github.io/merfish3d-analysis/](https://qi2lab.github.io/merfish3d-analysis/).

## Associated preprint publication
[GPU-accelerated, self-optimizing processing for 3D multiplexed iterative RNA-FISH experiments](https://www.biorxiv.org/content/10.1101/2025.10.10.681751v1).

## Try it without installation
You can try out the package in the cloud on simulated data using a [Google Colab notebook](https://colab.research.google.com/github/QI2lab/merfish3d-analysis/blob/main/examples/notebooks/Simulation_example.ipynb) that demonstrates data preprocessing and decoding.

## Installation

This project uses one `uv` environment for preprocessing, decoding, registration, stitching, viewing, development, and documentation. The Python environment installs CUDA 12.9 runtime/toolkit wheels through the project dependencies. You still need a compatible NVIDIA driver installed on the machine.

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

The standard `uv sync` install includes the ndv/PyQt dependencies for the view-only datastore GUI.

Open an experiment root:

```bash
uv run viewer /path/to/experiment
```

or open a datastore directly:

```bash
uv run viewer /path/to/experiment/qi2labdatastore
```

The viewer only reads existing datastore contents. It can display selected tiles, round-1 fiducials, selected bits, feature predictor probability images, decoded codebook-word overlays, and cell-outline overlays when those components are already present. When the datastore contains fused global polyDT data, globally decoded features, cell outlines, and the global polyDT segmentation image, enable the global fused view to inspect the downsampled polyDT max projection with selected RNA identities on the global coordinate canvas.

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

## Testing

Standard simulation matrix:

```bash
uv run pytest tests/test_simulation_example_pipeline.py -vv
```

Full feature-prediction probability threshold sweep:

```bash
uv run pytest tests/test_simulation_example_pipeline.py -vv --run-simulation-exhaustive
```

The simulation suite currently runs the default `simfish` U-FISH model. The
exhaustive mode expands the feature-prediction probability threshold and
preprocessing matrix for that default model.

### Simulation Results

F1 scores for the default `simfish` model are summarized below using the consensus feature-prediction probability threshold selected from the full sweep.

| Dataset | Axial spacing (um) | No deconvolution | Deconvolution |
| --- | --- | --- | --- |
| cells | 0.315 | 0.988 | 0.985 |
| cells | 1.0 | 0.954 | 0.953 |
| cells | 1.5 | 0.905 | 0.377 |
| uniform | 0.315 | 0.988 | 0.990 |
| uniform | 1.0 | 0.960 | 0.967 |
| uniform | 1.5 | 0.790 | 0.616 |

<details>
<summary>exhaustive feature prediction testing</summary>

Columns are feature-prediction probability thresholds. Values are F1 scores.

#### cells, axial spacing 0.315 um

##### No deconvolution

| U-FISH model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| --- | --- | --- | --- | --- | --- |
| exseq | 0.987 | 0.987 | 0.987 | 0.987 | 0.987 |
| dnafish | 0.989 | 0.989 | 0.989 | 0.989 | 0.989 |
| deepspot | 0.988 | 0.988 | 0.988 | 0.988 | 0.988 |
| deepblink | 0.987 | 0.987 | 0.987 | 0.987 | 0.987 |
| suntag | 0.989 | 0.989 | 0.989 | 0.989 | 0.989 |
| rca | 0.990 | 0.990 | 0.990 | 0.990 | 0.990 |
| seqfish | 0.987 | 0.987 | 0.987 | 0.987 | 0.987 |
| simfish | 0.988 | 0.988 | 0.988 | 0.988 | 0.988 |
| merfish | 0.074 | 0.074 | 0.074 | 0.074 | 0.074 |

##### Deconvolution

| U-FISH model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| --- | --- | --- | --- | --- | --- |
| exseq | 0.989 | 0.989 | 0.989 | 0.989 | 0.989 |
| dnafish | 0.990 | 0.990 | 0.990 | 0.990 | 0.990 |
| deepspot | 0.988 | 0.988 | 0.988 | 0.988 | 0.988 |
| deepblink | 0.745 | 0.745 | 0.745 | 0.745 | 0.745 |
| suntag | 0.989 | 0.989 | 0.989 | 0.989 | 0.989 |
| rca | 0.989 | 0.989 | 0.989 | 0.989 | 0.989 |
| seqfish | 0.987 | 0.987 | 0.987 | 0.987 | 0.987 |
| simfish | 0.985 | 0.985 | 0.985 | 0.985 | 0.985 |
| merfish | 0.986 | 0.986 | 0.986 | 0.986 | 0.986 |

#### cells, axial spacing 1.0 um

##### No deconvolution

| U-FISH model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| --- | --- | --- | --- | --- | --- |
| exseq | 0.938 | 0.938 | 0.938 | 0.938 | 0.938 |
| dnafish | 0.940 | 0.940 | 0.940 | 0.940 | 0.940 |
| deepspot | 0.954 | 0.954 | 0.954 | 0.954 | 0.954 |
| deepblink | 0.927 | 0.927 | 0.927 | 0.927 | 0.927 |
| suntag | 0.961 | 0.961 | 0.961 | 0.961 | 0.961 |
| rca | 0.956 | 0.956 | 0.956 | 0.956 | 0.956 |
| seqfish | 0.952 | 0.952 | 0.952 | 0.952 | 0.952 |
| simfish | 0.954 | 0.954 | 0.954 | 0.954 | 0.954 |
| merfish | 0.695 | 0.695 | 0.695 | 0.695 | 0.695 |

##### Deconvolution

| U-FISH model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| --- | --- | --- | --- | --- | --- |
| exseq | 0.941 | 0.941 | 0.941 | 0.941 | 0.941 |
| dnafish | 0.960 | 0.960 | 0.960 | 0.960 | 0.960 |
| deepspot | 0.961 | 0.961 | 0.961 | 0.961 | 0.961 |
| deepblink | 0.880 | 0.880 | 0.880 | 0.880 | 0.880 |
| suntag | 0.965 | 0.965 | 0.965 | 0.965 | 0.965 |
| rca | 0.961 | 0.961 | 0.961 | 0.961 | 0.961 |
| seqfish | 0.956 | 0.956 | 0.956 | 0.956 | 0.956 |
| simfish | 0.953 | 0.953 | 0.953 | 0.953 | 0.953 |
| merfish | 0.884 | 0.884 | 0.884 | 0.884 | 0.884 |

#### cells, axial spacing 1.5 um

##### No deconvolution

| U-FISH model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| --- | --- | --- | --- | --- | --- |
| exseq | 0.922 | 0.922 | 0.922 | 0.922 | 0.922 |
| dnafish | 0.886 | 0.886 | 0.886 | 0.886 | 0.886 |
| deepspot | 0.936 | 0.936 | 0.936 | 0.936 | 0.936 |
| deepblink | 0.917 | 0.917 | 0.917 | 0.917 | 0.917 |
| suntag | 0.888 | 0.888 | 0.888 | 0.888 | 0.888 |
| rca | 0.939 | 0.939 | 0.939 | 0.939 | 0.939 |
| seqfish | 0.920 | 0.920 | 0.920 | 0.920 | 0.920 |
| simfish | 0.905 | 0.905 | 0.905 | 0.905 | 0.905 |
| merfish | 0.464 | 0.464 | 0.464 | 0.464 | 0.464 |

##### Deconvolution

| U-FISH model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| --- | --- | --- | --- | --- | --- |
| exseq | 0.539 | 0.539 | 0.539 | 0.539 | 0.539 |
| dnafish | 0.596 | 0.596 | 0.596 | 0.596 | 0.596 |
| deepspot | 0.593 | 0.593 | 0.593 | 0.593 | 0.593 |
| deepblink | 0.530 | 0.530 | 0.530 | 0.530 | 0.530 |
| suntag | 0.358 | 0.358 | 0.358 | 0.358 | 0.358 |
| rca | 0.457 | 0.457 | 0.457 | 0.457 | 0.457 |
| seqfish | 0.588 | 0.588 | 0.588 | 0.588 | 0.588 |
| simfish | 0.377 | 0.377 | 0.377 | 0.377 | 0.377 |
| merfish | 0.417 | 0.417 | 0.417 | 0.417 | 0.417 |

#### uniform, axial spacing 0.315 um

##### No deconvolution

| U-FISH model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| --- | --- | --- | --- | --- | --- |
| exseq | 0.988 | 0.988 | 0.988 | 0.988 | 0.988 |
| dnafish | 0.988 | 0.988 | 0.988 | 0.988 | 0.988 |
| deepspot | 0.988 | 0.988 | 0.988 | 0.988 | 0.988 |
| deepblink | 0.987 | 0.987 | 0.987 | 0.987 | 0.987 |
| suntag | 0.988 | 0.988 | 0.988 | 0.988 | 0.988 |
| rca | 0.988 | 0.988 | 0.988 | 0.988 | 0.988 |
| seqfish | 0.988 | 0.988 | 0.988 | 0.988 | 0.988 |
| simfish | 0.988 | 0.988 | 0.988 | 0.988 | 0.988 |
| merfish | 0.988 | 0.988 | 0.988 | 0.988 | 0.988 |

##### Deconvolution

| U-FISH model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| --- | --- | --- | --- | --- | --- |
| exseq | 0.990 | 0.990 | 0.990 | 0.990 | 0.990 |
| dnafish | 0.989 | 0.989 | 0.989 | 0.989 | 0.989 |
| deepspot | 0.990 | 0.990 | 0.990 | 0.990 | 0.990 |
| deepblink | 0.984 | 0.984 | 0.984 | 0.984 | 0.984 |
| suntag | 0.991 | 0.991 | 0.991 | 0.991 | 0.991 |
| rca | 0.990 | 0.990 | 0.990 | 0.990 | 0.990 |
| seqfish | 0.987 | 0.987 | 0.987 | 0.987 | 0.987 |
| simfish | 0.990 | 0.990 | 0.990 | 0.990 | 0.990 |
| merfish | 0.990 | 0.990 | 0.990 | 0.990 | 0.990 |

#### uniform, axial spacing 1.0 um

##### No deconvolution

| U-FISH model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| --- | --- | --- | --- | --- | --- |
| exseq | 0.959 | 0.959 | 0.959 | 0.959 | 0.959 |
| dnafish | 0.962 | 0.962 | 0.962 | 0.962 | 0.962 |
| deepspot | 0.961 | 0.961 | 0.961 | 0.961 | 0.961 |
| deepblink | 0.946 | 0.946 | 0.946 | 0.946 | 0.946 |
| suntag | 0.965 | 0.965 | 0.965 | 0.965 | 0.965 |
| rca | 0.964 | 0.964 | 0.964 | 0.964 | 0.964 |
| seqfish | 0.960 | 0.960 | 0.960 | 0.960 | 0.960 |
| simfish | 0.960 | 0.960 | 0.960 | 0.960 | 0.960 |
| merfish | 0.951 | 0.951 | 0.951 | 0.951 | 0.951 |

##### Deconvolution

| U-FISH model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| --- | --- | --- | --- | --- | --- |
| exseq | 0.966 | 0.966 | 0.966 | 0.966 | 0.966 |
| dnafish | 0.965 | 0.965 | 0.965 | 0.965 | 0.965 |
| deepspot | 0.872 | 0.872 | 0.872 | 0.872 | 0.872 |
| deepblink | 0.952 | 0.952 | 0.952 | 0.952 | 0.952 |
| suntag | 0.958 | 0.958 | 0.958 | 0.958 | 0.958 |
| rca | 0.967 | 0.967 | 0.967 | 0.967 | 0.967 |
| seqfish | 0.963 | 0.963 | 0.963 | 0.963 | 0.963 |
| simfish | 0.967 | 0.967 | 0.967 | 0.967 | 0.967 |
| merfish | 0.966 | 0.966 | 0.966 | 0.966 | 0.966 |

#### uniform, axial spacing 1.5 um

##### No deconvolution

| U-FISH model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| --- | --- | --- | --- | --- | --- |
| exseq | 0.924 | 0.924 | 0.924 | 0.924 | 0.924 |
| dnafish | 0.935 | 0.935 | 0.935 | 0.935 | 0.935 |
| deepspot | 0.938 | 0.938 | 0.938 | 0.938 | 0.938 |
| deepblink | 0.920 | 0.920 | 0.920 | 0.920 | 0.920 |
| suntag | 0.914 | 0.914 | 0.914 | 0.914 | 0.914 |
| rca | 0.939 | 0.939 | 0.939 | 0.939 | 0.939 |
| seqfish | 0.927 | 0.927 | 0.927 | 0.927 | 0.927 |
| simfish | 0.790 | 0.790 | 0.790 | 0.790 | 0.790 |
| merfish | 0.903 | 0.903 | 0.903 | 0.903 | 0.903 |

##### Deconvolution

| U-FISH model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| --- | --- | --- | --- | --- | --- |
| exseq | 0.584 | 0.584 | 0.584 | 0.584 | 0.584 |
| dnafish | 0.665 | 0.665 | 0.665 | 0.665 | 0.665 |
| deepspot | 0.659 | 0.659 | 0.659 | 0.659 | 0.659 |
| deepblink | 0.581 | 0.581 | 0.581 | 0.581 | 0.581 |
| suntag | 0.624 | 0.624 | 0.624 | 0.624 | 0.624 |
| rca | 0.644 | 0.644 | 0.644 | 0.644 | 0.644 |
| seqfish | 0.627 | 0.627 | 0.627 | 0.627 | 0.627 |
| simfish | 0.616 | 0.616 | 0.616 | 0.616 | 0.616 |
| merfish | 0.594 | 0.594 | 0.594 | 0.594 | 0.594 |

</details>
