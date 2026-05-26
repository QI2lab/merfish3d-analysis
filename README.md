# merfish3d-analysis

_WARNING: under active development._ Please expect breaking changes.

GPU accelerated post-processing for 2D / 3D iterative barcoded FISH data. This package currently **Nvidia only** and **Linux only** due to RAPIDS.AI package availability. Documentation, including examples, is available at [https://qi2lab.github.io/merfish3d-analysis/](https://qi2lab.github.io/merfish3d-analysis/).

## Associated preprint publication
[GPU-accelerated, self-optimizing processing for 3D multiplexed iterative RNA-FISH experiments](https://www.biorxiv.org/content/10.1101/2025.10.10.681751v1).

## Try it without installation
You can try out the package in the cloud on simulated data using a [Google Colab notebook](https://colab.research.google.com/github/QI2lab/merfish3d-analysis/blob/main/examples/notebooks/Simulation_example.ipynb) that demonstrates data preprocessing and decoding.

## Installation

Create a python 3.12 environment using your favorite package manager, e.g.
```
conda create -n merfish3d python=3.12
```

Activate the environment and install the GPU dependencies. This install method assumes an Nvidia GPU capable of running CUDA 12.8.
```
conda activate merfish3d
```

Next, clone the repository in your location of choice and enter the directory using
```
git clone https://github.com/QI2lab/merfish3d-analysis
cd merfish3d-analysis
``` 

and install using 
```
pip install .
```

For interactive editing use 
```
pip install -e .
``` 

Finally, install the `merfish3d-analysis` package using the command 
```
setup-merfish3d
```` 

This will automatically setup the correct CUDA libraries and other packages in the conda environment. **Note**: Due to package incompatibility, the install script currently creates a second conda/mamba environment called `merfish3d-stitcher`. In this environment, we install the minimal packages required to read the datastore used by `merfish3d-analysis` and [multiview-stitcher](https://github.com/multiview-stitcher/multiview-stitcher). The reason for this change is that one of the `multiview-stitcher` sub-dependencies (`xarray-dataclass`) now requires `numpy>2.0`, which is incompatible with the scientific computing packages used for `merfish3d-analysis`.

The `merfish3d-stitcher` environment is only used when individual tiles are registered into a global coordinate system. The code automatically invokes this second environment, but it is important to note that the current install strategy does create a new conda/mamba environment beyond what you as the user creates. As soon as the dependency issue is solved, we will remove this work around.

## View a qi2lab datastore

The standard `setup-merfish3d` install includes the ndv/PyQt dependencies for the
view-only datastore GUI. If you run `setup-merfish3d --headless`, GUI dependencies
are skipped and `qi2lab-viewer` will not be usable until the GUI dependencies are
installed.

Open an experiment root:

```bash
qi2lab-viewer /path/to/experiment
```

or open a datastore directly:

```bash
qi2lab-viewer /path/to/experiment/qi2labdatastore
```

The viewer only reads existing datastore contents. It can display selected tiles,
round-1 fiducials, selected bits, feature predictor probability images, decoded
codebook-word overlays, and cell-outline overlays when those components are
already present. When the datastore contains fused global polyDT data, globally
decoded features, cell outlines, and the global polyDT segmentation image, enable
the global fused view to inspect the downsampled polyDT max projection with
selected RNA identities on the global coordinate canvas.

## Proseg segmentation optimization

Follow the installation instructions at [proseg](https://github.com/dcjones/proseg).

Example calls to proseg:

2D segmentation optimization (ignore z coordinate)
```bash
mkdir /path/to/qi2labdatastore/proseg/3D

/path/to/proseg --gene-column gene_id -x global_x -y global_y -z global_z --fov-column tile_idx --cell-id-column cell_id --cell-id-unassigned 0 --excluded-genes ^[Bb]lank.*$ --ignore-z-coord --density-bins 1 --burnin-samples 1000 --samples 2000 --voxel-size 1.0 --burnin-voxel-size 4.0 --enforce-connectivity --diffusion-probability 0.0 --output-spatialdata /path/to/data/qi2labdatastore/proseg/2D/spatialdata_2D.zarr --output-counts /path/to/data/qi2labdatastore/proseg/2D/counts_2D.mtx.gz --output-cell-polygons /path/to/data/qi2labdatastore/proseg/2D/cell_polygons_2D.geojson.gz --output-transcript-metadata /path/to/data/qi2labdatastore/proseg/2D/transcript_metadata_2D.csv.gz /path/to/data/qi2labdatastore/all_tiles_filtered_decoded_features/decoded_features.csv.gz
```

3D segmentation optimization. Here `--voxel-layers` should be roughly set to the height of the imaged volume in microns. For example, a 15 micron thick sample should have `--voxel-layers 15`.
```bash
mkdir /path/to/qi2labdatastore/proseg/3D

/path/to/proseg --gene-column gene_id -x global_x -y global_y -z global_z --fov-column tile_idx --cell-id-column cell_id --cell-id-unassigned 0 --excluded-genes ^[Bb]lank.*$ --voxel-layers 15 --density-bins 1 --burnin-samples 1000 --samples 2000 --voxel-size 1.0 --burnin-voxel-size 4.0 --enforce-connectivity --diffusion-probability 0.0 --output-spatialdata /path/to/data/qi2labdatastore/proseg/3D/spatialdata_3D.zarr --output-counts /path/to/data/qi2labdatastore/proseg/3D/counts_3D.mtx.gz --output-cell-polygons-layers /path/to/data/qi2labdatastore/proseg/3D/cell_polygons_3D.geojson.gz --output-transcript-metadata /path/to/data/qi2labdatastore/proseg/3D/transcript_metadata_3D.csv.gz /path/to/data/qi2labdatastore/all_tiles_filtered_decoded_features/decoded_features.csv.gz
```

## Documentation

To build the documentation, install using `pip install .[docs]`. Then execute `mkdocs build --clean` and `mkdocs serve`. The documentation is available in your web browser at `http://127.0.0.1:8000/`.

## Testing

The test coverage for this repository is the local simulation integration matrix in
[tests/test_simulation_example_pipeline.py](tests/test_simulation_example_pipeline.py). These tests exercise a end-to-end simulation workflow:

- convert simulation data into a fake acquisition
- convert the acquisition into a datastore
- preprocess with registration and optional readout deconvolution
- decode transcripts
- calculate F1 against the simulation ground truth

The simulation dataset root is configured directly in the test file:

```python
LOCAL_SIMULATION_DATA_ROOT = Path("/media/dps/data/merfish3d_analysis-simulation")
```

If your local simulation data lives elsewhere, update that constant before running the tests.

### Standard integration matrix

Run the required standard matrix with:

```bash
python -m pytest tests/test_simulation_example_pipeline.py -q
```

This runs the standard simulation policy across:

- `cells` and `uniform`
- axial spacings `0.315`, `1.0`, and `1.5`
- `decon=True`
- `feature_predictor_threshold=0.5` for the pinned regression matrix
- every locally cached U-FISH model, with pinned regression assertions for the
  package default feature-prediction model, `simfish`

The default decode policy is locked to:

- `minimum_pixels_per_rna = 28` for `0.315` simulations
- `minimum_pixels_per_rna = 7` for `1.0` and `1.5` simulations
- wrapper feature-predictor threshold defaults:
  - `0.315` or non-deconvolved data -> `0.5`
  - `1.0` deconvolved 2D data -> `0.3`
  - `1.5` deconvolved 2D data -> `0.2`
- sampling-aware magnitude threshold defaults:
  - `0.315` -> `0.9`
  - `1.0` -> `0.7`
  - `1.5` -> `0.2`

### Exhaustive regression matrix

Run the optional exhaustive regression matrix with:

```bash
python -m pytest tests/test_simulation_example_pipeline.py -q --run-simulation-exhaustive
```

This expands the matrix to include:

- `decon` and `no-decon`
- feature predictor thresholds `0.1`, `0.2`, `0.3`, `0.4`, and `0.5`
- every locally cached U-FISH model

### Exhaustive U-FISH Model Sweep

These F1 scores are from `python -m pytest --run-simulation-exhaustive -W always -ra` using the locally cached U-FISH model weights. `simfish` is the package default and also covers the `smfish` alias because both resolve to `v1.0.1-simfish_model.onnx`.

Values are rounded to three decimal places. The bold value marks the selected best model and threshold for that simulation condition and deconvolution state; ties after rounding prefer `simfish`, then `merfish`.

#### cells, axial spacing 0.315 um

##### No deconvolution

| U-FISH model | fp=0.1 | fp=0.2 | fp=0.3 | fp=0.4 | fp=0.5 |
| --- | --- | --- | --- | --- | --- |
| merfish | 0.992 | 0.992 | 0.992 | 0.995 | 0.995 |
| seqfish | 0.992 | 0.994 | 0.995 | 0.996 | 0.995 |
| simfish | 0.994 | 0.992 | 0.995 | 0.995 | 0.996 |
| deepspot | 0.994 | 0.996 | 0.996 | 0.995 | 0.996 |
| exseq | 0.992 | 0.992 | 0.994 | 0.996 | 0.995 |
| dnafish | 0.993 | 0.994 | 0.991 | 0.989 | 0.982 |
| rca | 0.991 | 0.994 | 0.995 | 0.995 | **0.998** |
| deepblink | 0.992 | 0.992 | 0.991 | 0.994 | 0.993 |
| suntag | 0.993 | 0.995 | 0.996 | 0.996 | 0.995 |

##### Deconvolution

| U-FISH model | fp=0.1 | fp=0.2 | fp=0.3 | fp=0.4 | fp=0.5 |
| --- | --- | --- | --- | --- | --- |
| merfish | 0.981 | 0.988 | 0.992 | 0.994 | 0.994 |
| seqfish | 0.982 | 0.991 | 0.993 | 0.992 | 0.993 |
| simfish | 0.941 | 0.978 | 0.993 | 0.992 | 0.994 |
| deepspot | 0.983 | 0.989 | 0.994 | 0.993 | 0.991 |
| exseq | 0.982 | 0.989 | 0.988 | 0.993 | 0.995 |
| dnafish | 0.987 | 0.988 | 0.990 | 0.991 | 0.985 |
| rca | 0.906 | 0.978 | 0.992 | 0.997 | **0.998** |
| deepblink | 0.937 | 0.946 | 0.962 | 0.977 | 0.979 |
| suntag | 0.984 | 0.992 | 0.991 | 0.990 | 0.991 |

#### cells, axial spacing 1.0 um

##### No deconvolution

| U-FISH model | fp=0.1 | fp=0.2 | fp=0.3 | fp=0.4 | fp=0.5 |
| --- | --- | --- | --- | --- | --- |
| merfish | 0.714 | 0.767 | 0.779 | 0.823 | 0.854 |
| seqfish | 0.759 | 0.786 | 0.831 | 0.862 | 0.877 |
| simfish | 0.780 | 0.835 | 0.886 | 0.920 | 0.951 |
| deepspot | 0.786 | 0.820 | 0.851 | 0.879 | 0.926 |
| exseq | 0.745 | 0.785 | 0.815 | 0.852 | 0.877 |
| dnafish | 0.891 | 0.912 | 0.919 | 0.914 | 0.902 |
| rca | 0.834 | 0.874 | 0.910 | 0.935 | 0.969 |
| deepblink | 0.703 | 0.719 | 0.705 | 0.734 | 0.704 |
| suntag | 0.927 | 0.947 | 0.953 | 0.964 | **0.977** |

##### Deconvolution

| U-FISH model | fp=0.1 | fp=0.2 | fp=0.3 | fp=0.4 | fp=0.5 |
| --- | --- | --- | --- | --- | --- |
| merfish | 0.913 | 0.938 | 0.967 | 0.969 | 0.969 |
| seqfish | 0.907 | 0.923 | 0.928 | 0.933 | 0.936 |
| simfish | 0.922 | 0.962 | 0.964 | 0.978 | 0.978 |
| deepspot | 0.908 | 0.927 | 0.920 | 0.911 | 0.878 |
| exseq | 0.936 | 0.943 | 0.968 | 0.971 | 0.971 |
| dnafish | 0.910 | 0.897 | 0.877 | 0.861 | 0.838 |
| rca | 0.942 | 0.968 | **0.984** | 0.976 | 0.981 |
| deepblink | 0.729 | 0.516 | 0.425 | 0.383 | 0.344 |
| suntag | 0.966 | 0.968 | 0.969 | 0.956 | 0.932 |

#### cells, axial spacing 1.5 um

##### No deconvolution

| U-FISH model | fp=0.1 | fp=0.2 | fp=0.3 | fp=0.4 | fp=0.5 |
| --- | --- | --- | --- | --- | --- |
| merfish | 0.828 | 0.856 | 0.878 | 0.899 | 0.922 |
| seqfish | 0.901 | 0.924 | 0.943 | 0.947 | 0.957 |
| simfish | 0.921 | 0.938 | 0.963 | 0.971 | **0.974** |
| deepspot | 0.941 | 0.951 | 0.958 | 0.955 | 0.965 |
| exseq | 0.891 | 0.914 | 0.928 | 0.936 | 0.940 |
| dnafish | 0.901 | 0.875 | 0.850 | 0.829 | 0.788 |
| rca | 0.970 | 0.971 | 0.969 | 0.959 | 0.935 |
| deepblink | 0.891 | 0.890 | 0.883 | 0.884 | 0.879 |
| suntag | 0.896 | 0.883 | 0.866 | 0.850 | 0.840 |

##### Deconvolution

| U-FISH model | fp=0.1 | fp=0.2 | fp=0.3 | fp=0.4 | fp=0.5 |
| --- | --- | --- | --- | --- | --- |
| merfish | 0.903 | 0.941 | 0.953 | 0.957 | 0.950 |
| seqfish | 0.958 | 0.950 | 0.944 | 0.934 | 0.913 |
| simfish | 0.936 | 0.959 | 0.969 | **0.974** | 0.968 |
| deepspot | 0.939 | 0.930 | 0.920 | 0.894 | 0.862 |
| exseq | 0.946 | 0.944 | 0.942 | 0.930 | 0.922 |
| dnafish | 0.698 | 0.688 | 0.669 | 0.660 | 0.626 |
| rca | 0.947 | 0.939 | 0.922 | 0.903 | 0.886 |
| deepblink | 0.788 | 0.612 | 0.509 | 0.444 | 0.390 |
| suntag | 0.840 | 0.827 | 0.804 | 0.782 | 0.763 |

#### uniform, axial spacing 0.315 um

##### No deconvolution

| U-FISH model | fp=0.1 | fp=0.2 | fp=0.3 | fp=0.4 | fp=0.5 |
| --- | --- | --- | --- | --- | --- |
| merfish | 0.993 | 0.996 | 0.996 | 0.997 | 0.996 |
| seqfish | 0.995 | 0.997 | 0.996 | 0.997 | 0.996 |
| simfish | 0.995 | 0.997 | 0.996 | 0.996 | **0.998** |
| deepspot | 0.995 | 0.996 | 0.995 | 0.997 | 0.996 |
| exseq | 0.995 | 0.997 | 0.996 | 0.997 | 0.997 |
| dnafish | 0.997 | 0.997 | 0.996 | 0.995 | 0.993 |
| rca | 0.995 | 0.997 | 0.995 | 0.997 | 0.997 |
| deepblink | 0.995 | 0.995 | 0.995 | 0.995 | 0.995 |
| suntag | 0.996 | 0.996 | 0.997 | 0.995 | 0.998 |

##### Deconvolution

| U-FISH model | fp=0.1 | fp=0.2 | fp=0.3 | fp=0.4 | fp=0.5 |
| --- | --- | --- | --- | --- | --- |
| merfish | 0.990 | 0.993 | 0.994 | 0.995 | **0.996** |
| seqfish | 0.991 | 0.993 | 0.993 | 0.995 | 0.996 |
| simfish | 0.989 | 0.991 | 0.991 | 0.995 | 0.995 |
| deepspot | 0.991 | 0.992 | 0.991 | 0.992 | 0.993 |
| exseq | 0.990 | 0.993 | 0.993 | 0.994 | 0.995 |
| dnafish | 0.993 | 0.993 | 0.994 | 0.990 | 0.986 |
| rca | 0.989 | 0.993 | 0.994 | 0.995 | 0.996 |
| deepblink | 0.936 | 0.976 | 0.980 | 0.987 | 0.988 |
| suntag | 0.992 | 0.993 | 0.994 | 0.996 | 0.994 |

#### uniform, axial spacing 1.0 um

##### No deconvolution

| U-FISH model | fp=0.1 | fp=0.2 | fp=0.3 | fp=0.4 | fp=0.5 |
| --- | --- | --- | --- | --- | --- |
| merfish | 0.810 | 0.858 | 0.883 | 0.910 | 0.943 |
| seqfish | 0.834 | 0.875 | 0.882 | 0.904 | 0.935 |
| simfish | 0.815 | 0.865 | 0.905 | 0.945 | 0.958 |
| deepspot | 0.862 | 0.900 | 0.911 | 0.942 | 0.957 |
| exseq | 0.831 | 0.874 | 0.886 | 0.916 | 0.936 |
| dnafish | 0.921 | 0.944 | 0.958 | 0.956 | 0.931 |
| rca | 0.849 | 0.894 | 0.932 | 0.960 | **0.982** |
| deepblink | 0.697 | 0.702 | 0.676 | 0.685 | 0.686 |
| suntag | 0.929 | 0.943 | 0.963 | 0.978 | 0.980 |

##### Deconvolution

| U-FISH model | fp=0.1 | fp=0.2 | fp=0.3 | fp=0.4 | fp=0.5 |
| --- | --- | --- | --- | --- | --- |
| merfish | 0.969 | 0.979 | 0.988 | 0.985 | 0.983 |
| seqfish | 0.922 | 0.941 | 0.944 | 0.942 | 0.937 |
| simfish | 0.972 | 0.979 | 0.983 | 0.979 | 0.986 |
| deepspot | 0.901 | 0.861 | 0.797 | 0.671 | 0.445 |
| exseq | 0.978 | 0.987 | **0.990** | 0.987 | 0.986 |
| dnafish | 0.821 | 0.800 | 0.777 | 0.759 | 0.724 |
| rca | 0.978 | 0.983 | 0.988 | 0.982 | 0.985 |
| deepblink | 0.720 | 0.527 | 0.438 | 0.403 | 0.338 |
| suntag | 0.982 | 0.981 | 0.973 | 0.961 | 0.936 |

#### uniform, axial spacing 1.5 um

##### No deconvolution

| U-FISH model | fp=0.1 | fp=0.2 | fp=0.3 | fp=0.4 | fp=0.5 |
| --- | --- | --- | --- | --- | --- |
| merfish | 0.831 | 0.866 | 0.894 | 0.918 | 0.935 |
| seqfish | 0.877 | 0.899 | 0.913 | 0.940 | 0.956 |
| simfish | 0.899 | 0.932 | 0.959 | 0.976 | 0.979 |
| deepspot | 0.929 | 0.939 | 0.951 | 0.963 | 0.974 |
| exseq | 0.865 | 0.885 | 0.912 | 0.932 | 0.948 |
| dnafish | 0.958 | 0.963 | 0.946 | 0.923 | 0.885 |
| rca | 0.960 | 0.972 | 0.979 | **0.986** | 0.980 |
| deepblink | 0.851 | 0.854 | 0.842 | 0.831 | 0.818 |
| suntag | 0.953 | 0.936 | 0.920 | 0.911 | 0.902 |

##### Deconvolution

| U-FISH model | fp=0.1 | fp=0.2 | fp=0.3 | fp=0.4 | fp=0.5 |
| --- | --- | --- | --- | --- | --- |
| merfish | 0.969 | 0.979 | 0.981 | 0.979 | 0.972 |
| seqfish | 0.941 | 0.960 | 0.964 | 0.959 | 0.955 |
| simfish | 0.958 | 0.975 | 0.978 | 0.982 | 0.978 |
| deepspot | 0.795 | 0.750 | 0.683 | 0.568 | 0.393 |
| exseq | 0.968 | **0.984** | 0.980 | 0.979 | 0.970 |
| dnafish | 0.718 | 0.688 | 0.662 | 0.637 | 0.610 |
| rca | 0.977 | 0.971 | 0.966 | 0.960 | 0.957 |
| deepblink | 0.865 | 0.740 | 0.675 | 0.638 | 0.606 |
| suntag | 0.899 | 0.885 | 0.872 | 0.854 | 0.838 |
