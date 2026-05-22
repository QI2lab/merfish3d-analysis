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

#### cells, axial spacing 0.315 um

| U-FISH model | no-decon fp=0.1 | no-decon fp=0.2 | no-decon fp=0.3 | no-decon fp=0.4 | no-decon fp=0.5 | decon fp=0.1 | decon fp=0.2 | decon fp=0.3 | decon fp=0.4 | decon fp=0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| merfish | 0.9924 | 0.9916 | 0.9916 | 0.9950 | 0.9950 | 0.9808 | 0.9883 | 0.9916 | 0.9941 | 0.9941 |
| seqfish | 0.9924 | 0.9941 | 0.9950 | 0.9958 | 0.9950 | 0.9823 | 0.9908 | 0.9933 | 0.9924 | 0.9933 |
| simfish | 0.9941 | 0.9916 | 0.9950 | 0.9950 | 0.9958 | 0.9412 | 0.9781 | 0.9925 | 0.9916 | 0.9942 |
| deepspot | 0.9941 | 0.9958 | 0.9958 | 0.9950 | 0.9958 | 0.9825 | 0.9891 | 0.9941 | 0.9933 | 0.9907 |
| exseq | 0.9924 | 0.9916 | 0.9941 | 0.9958 | 0.9950 | 0.9824 | 0.9891 | 0.9883 | 0.9925 | 0.9950 |
| dnafish | 0.9925 | 0.9941 | 0.9908 | 0.9890 | 0.9822 | 0.9867 | 0.9883 | 0.9900 | 0.9908 | 0.9849 |
| rca | 0.9908 | 0.9941 | 0.9950 | 0.9950 | 0.9975 | 0.9056 | 0.9783 | 0.9916 | 0.9967 | 0.9975 |
| deepblink | 0.9924 | 0.9916 | 0.9908 | 0.9941 | 0.9933 | 0.9374 | 0.9459 | 0.9619 | 0.9765 | 0.9788 |
| suntag | 0.9925 | 0.9950 | 0.9958 | 0.9958 | 0.9950 | 0.9841 | 0.9916 | 0.9908 | 0.9899 | 0.9907 |

#### cells, axial spacing 1.0 um

| U-FISH model | no-decon fp=0.1 | no-decon fp=0.2 | no-decon fp=0.3 | no-decon fp=0.4 | no-decon fp=0.5 | decon fp=0.1 | decon fp=0.2 | decon fp=0.3 | decon fp=0.4 | decon fp=0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| merfish | 0.7136 | 0.7667 | 0.7786 | 0.8229 | 0.8536 | 0.9134 | 0.9384 | 0.9668 | 0.9692 | 0.9690 |
| seqfish | 0.7588 | 0.7865 | 0.8310 | 0.8621 | 0.8773 | 0.9074 | 0.9229 | 0.9284 | 0.9334 | 0.9363 |
| simfish | 0.7801 | 0.8347 | 0.8858 | 0.9203 | 0.9508 | 0.9224 | 0.9619 | 0.9642 | 0.9775 | 0.9783 |
| deepspot | 0.7862 | 0.8195 | 0.8510 | 0.8789 | 0.9258 | 0.9082 | 0.9266 | 0.9200 | 0.9115 | 0.8776 |
| exseq | 0.7445 | 0.7853 | 0.8154 | 0.8518 | 0.8770 | 0.9360 | 0.9426 | 0.9682 | 0.9707 | 0.9707 |
| dnafish | 0.8911 | 0.9122 | 0.9193 | 0.9143 | 0.9022 | 0.9097 | 0.8966 | 0.8771 | 0.8609 | 0.8383 |
| rca | 0.8343 | 0.8740 | 0.9104 | 0.9350 | 0.9691 | 0.9420 | 0.9675 | 0.9841 | 0.9758 | 0.9807 |
| deepblink | 0.7032 | 0.7193 | 0.7054 | 0.7341 | 0.7040 | 0.7290 | 0.5165 | 0.4247 | 0.3826 | 0.3445 |
| suntag | 0.9272 | 0.9468 | 0.9525 | 0.9640 | 0.9765 | 0.9664 | 0.9679 | 0.9685 | 0.9559 | 0.9320 |

#### cells, axial spacing 1.5 um

| U-FISH model | no-decon fp=0.1 | no-decon fp=0.2 | no-decon fp=0.3 | no-decon fp=0.4 | no-decon fp=0.5 | decon fp=0.1 | decon fp=0.2 | decon fp=0.3 | decon fp=0.4 | decon fp=0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| merfish | 0.8283 | 0.8560 | 0.8777 | 0.8994 | 0.9223 | 0.9032 | 0.9413 | 0.9528 | 0.9570 | 0.9503 |
| seqfish | 0.9006 | 0.9245 | 0.9430 | 0.9465 | 0.9570 | 0.9582 | 0.9500 | 0.9441 | 0.9336 | 0.9127 |
| simfish | 0.9211 | 0.9377 | 0.9627 | 0.9709 | 0.9741 | 0.9360 | 0.9589 | 0.9694 | 0.9741 | 0.9680 |
| deepspot | 0.9405 | 0.9511 | 0.9577 | 0.9551 | 0.9649 | 0.9386 | 0.9299 | 0.9201 | 0.8937 | 0.8617 |
| exseq | 0.8909 | 0.9138 | 0.9280 | 0.9365 | 0.9396 | 0.9455 | 0.9438 | 0.9415 | 0.9301 | 0.9224 |
| dnafish | 0.9009 | 0.8753 | 0.8504 | 0.8293 | 0.7876 | 0.6977 | 0.6878 | 0.6689 | 0.6600 | 0.6256 |
| rca | 0.9703 | 0.9707 | 0.9694 | 0.9585 | 0.9349 | 0.9466 | 0.9389 | 0.9224 | 0.9028 | 0.8864 |
| deepblink | 0.8909 | 0.8902 | 0.8832 | 0.8844 | 0.8793 | 0.7881 | 0.6121 | 0.5093 | 0.4444 | 0.3899 |
| suntag | 0.8958 | 0.8831 | 0.8660 | 0.8500 | 0.8401 | 0.8396 | 0.8265 | 0.8044 | 0.7822 | 0.7634 |

#### uniform, axial spacing 0.315 um

| U-FISH model | no-decon fp=0.1 | no-decon fp=0.2 | no-decon fp=0.3 | no-decon fp=0.4 | no-decon fp=0.5 | decon fp=0.1 | decon fp=0.2 | decon fp=0.3 | decon fp=0.4 | decon fp=0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| merfish | 0.9933 | 0.9958 | 0.9958 | 0.9967 | 0.9958 | 0.9899 | 0.9929 | 0.9941 | 0.9946 | 0.9958 |
| seqfish | 0.9946 | 0.9967 | 0.9958 | 0.9967 | 0.9962 | 0.9908 | 0.9933 | 0.9925 | 0.9946 | 0.9958 |
| simfish | 0.9954 | 0.9967 | 0.9958 | 0.9962 | 0.9975 | 0.9887 | 0.9912 | 0.9912 | 0.9950 | 0.9954 |
| deepspot | 0.9954 | 0.9962 | 0.9954 | 0.9971 | 0.9962 | 0.9908 | 0.9921 | 0.9908 | 0.9921 | 0.9925 |
| exseq | 0.9950 | 0.9967 | 0.9958 | 0.9967 | 0.9967 | 0.9895 | 0.9929 | 0.9929 | 0.9937 | 0.9950 |
| dnafish | 0.9967 | 0.9967 | 0.9958 | 0.9950 | 0.9933 | 0.9933 | 0.9933 | 0.9937 | 0.9895 | 0.9857 |
| rca | 0.9954 | 0.9967 | 0.9946 | 0.9971 | 0.9967 | 0.9887 | 0.9929 | 0.9937 | 0.9954 | 0.9962 |
| deepblink | 0.9946 | 0.9950 | 0.9950 | 0.9950 | 0.9950 | 0.9365 | 0.9755 | 0.9800 | 0.9870 | 0.9878 |
| suntag | 0.9958 | 0.9958 | 0.9967 | 0.9946 | 0.9975 | 0.9917 | 0.9929 | 0.9937 | 0.9962 | 0.9937 |

#### uniform, axial spacing 1.0 um

| U-FISH model | no-decon fp=0.1 | no-decon fp=0.2 | no-decon fp=0.3 | no-decon fp=0.4 | no-decon fp=0.5 | decon fp=0.1 | decon fp=0.2 | decon fp=0.3 | decon fp=0.4 | decon fp=0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| merfish | 0.8103 | 0.8576 | 0.8832 | 0.9105 | 0.9427 | 0.9693 | 0.9792 | 0.9875 | 0.9850 | 0.9828 |
| seqfish | 0.8342 | 0.8745 | 0.8820 | 0.9040 | 0.9347 | 0.9225 | 0.9408 | 0.9437 | 0.9416 | 0.9366 |
| simfish | 0.8148 | 0.8653 | 0.9053 | 0.9451 | 0.9576 | 0.9718 | 0.9792 | 0.9825 | 0.9787 | 0.9862 |
| deepspot | 0.8622 | 0.8999 | 0.9108 | 0.9421 | 0.9574 | 0.9013 | 0.8605 | 0.7968 | 0.6714 | 0.4447 |
| exseq | 0.8307 | 0.8737 | 0.8861 | 0.9161 | 0.9356 | 0.9780 | 0.9871 | 0.9896 | 0.9867 | 0.9858 |
| dnafish | 0.9214 | 0.9441 | 0.9581 | 0.9562 | 0.9314 | 0.8211 | 0.8002 | 0.7773 | 0.7594 | 0.7239 |
| rca | 0.8485 | 0.8938 | 0.9316 | 0.9604 | 0.9817 | 0.9780 | 0.9826 | 0.9879 | 0.9820 | 0.9853 |
| deepblink | 0.6968 | 0.7023 | 0.6762 | 0.6855 | 0.6861 | 0.7196 | 0.5269 | 0.4379 | 0.4029 | 0.3375 |
| suntag | 0.9286 | 0.9430 | 0.9625 | 0.9779 | 0.9799 | 0.9824 | 0.9806 | 0.9728 | 0.9607 | 0.9363 |

#### uniform, axial spacing 1.5 um

| U-FISH model | no-decon fp=0.1 | no-decon fp=0.2 | no-decon fp=0.3 | no-decon fp=0.4 | no-decon fp=0.5 | decon fp=0.1 | decon fp=0.2 | decon fp=0.3 | decon fp=0.4 | decon fp=0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| merfish | 0.8308 | 0.8656 | 0.8938 | 0.9177 | 0.9355 | 0.9687 | 0.9788 | 0.9812 | 0.9790 | 0.9716 |
| seqfish | 0.8767 | 0.8989 | 0.9129 | 0.9403 | 0.9558 | 0.9406 | 0.9599 | 0.9639 | 0.9591 | 0.9547 |
| simfish | 0.8993 | 0.9318 | 0.9592 | 0.9764 | 0.9791 | 0.9575 | 0.9753 | 0.9780 | 0.9820 | 0.9780 |
| deepspot | 0.9289 | 0.9391 | 0.9514 | 0.9625 | 0.9741 | 0.7946 | 0.7501 | 0.6833 | 0.5679 | 0.3931 |
| exseq | 0.8645 | 0.8850 | 0.9124 | 0.9325 | 0.9475 | 0.9678 | 0.9837 | 0.9799 | 0.9785 | 0.9699 |
| dnafish | 0.9578 | 0.9633 | 0.9456 | 0.9229 | 0.8847 | 0.7179 | 0.6877 | 0.6618 | 0.6375 | 0.6100 |
| rca | 0.9599 | 0.9718 | 0.9792 | 0.9862 | 0.9798 | 0.9765 | 0.9711 | 0.9662 | 0.9603 | 0.9566 |
| deepblink | 0.8509 | 0.8540 | 0.8423 | 0.8305 | 0.8177 | 0.8649 | 0.7398 | 0.6746 | 0.6384 | 0.6057 |
| suntag | 0.9531 | 0.9360 | 0.9203 | 0.9113 | 0.9018 | 0.8988 | 0.8855 | 0.8720 | 0.8539 | 0.8381 |
