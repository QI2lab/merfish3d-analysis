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

## Proseg segmentation optimization

Follow the installation instructions at [proseg](https://github.com/dcjones/proseg).

Example calls to proseg:

2D segmentation optimization (ignore z coordinate)
```bash
/path/to/proseg --gene-column gene_id -x global_x -y global_y -z global_z --fov-column tile_idx --cell-id-column cell_id --cell-id-unassigned 0 --excluded-genes ^[Bb]lank.*$ --ignore-z-coord --density-bins 1 --burnin-samples 1000 --samples 2000 --voxel-size 1.0 --burnin-voxel-size 4.0 --enforce-connectivity --output-spatialdata /path/to/data/qi2labdatastore/proseg/spatialdata_2D.zarr --output-counts /path/to/data/qi2labdatastore/proseg/counts_2D.mtx.gz --output-cell-polygons /path/to/data/qi2labdatastore/proseg/cell_polygons_2D.geojson.gz --output-transcript-metadata /path/to/data/qi2labdatastore/proseg/transcript_metadata_2D.csv.gz /path/to/data/qi2labdatastore/all_tiles_filtered_decoded_features/decoded_features.csv.gz
```

3D segmentation optimization. Here `--voxel-layers` should be roughly set to the height of the imaged volume in microns. For example, a 15 micron thick sample should have `--voxel-layers 15`.
```bash
/path/to/proseg --gene-column gene_id -x global_x -y global_y -z global_z --fov-column tile_idx --cell-id-column cell_id --cell-id-unassigned 0 --excluded-genes ^[Bb]lank.*$ --voxel-layers 15 --density-bins 1 --burnin-samples 1000 --samples 2000 --voxel-size 1.0 --burnin-voxel-size 4.0 --enforce-connectivity --output-spatialdata /path/to/data/qi2labdatastore/proseg/spatialdata_3D.zarr --output-counts /path/to/data/qi2labdatastore/proseg/counts_3D.mtx.gz --output-cell-polygons-layers /path/to/data/qi2labdatastore/proseg/cell_polygons_3D.geojson.gz --output-transcript-metadata /path/to/data/qi2labdatastore/proseg/transcript_metadata_3D.csv.gz /path/to/data/qi2labdatastore/all_tiles_filtered_decoded_features/decoded_features.csv.gz
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

This runs the default simulation policy across:

- `cells` and `uniform`
- axial spacings `0.315`, `1.0`, and `1.5`
- `decon=True`
- `feature_predictor_threshold=0.5`

The default decode policy is locked to:

- `minimum_pixels_per_rna = 28` for `0.315` simulations
- `minimum_pixels_per_rna = 7` for `1.0` and `1.5` simulations
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