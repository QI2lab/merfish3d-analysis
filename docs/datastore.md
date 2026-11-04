# qi2lab DataStore overview

## Philosophy

To efficiently handle 3D MERFISH experiments, `qi2labDataStore` stores images and their spatial relationships in an [OME-NGFF v0.6](https://ngff.openmicroscopy.org/latest/) Scene. Images are accessed through [yaozarrs](https://imaging-formats.github.io/yaozarrs/) and TensorStore. Tabular outputs are stored as [Parquet](https://parquet.apache.org/docs/).

## Important considerations

To create a `qi2labDataStore`, we need to know the following metadata:

- the effective xy pixel size and z step
- the objective numerical aperture
- the immersion media refractive index
- the global stage zyx position at each tile
- the camera orientation with respect to the stage orientation
- the direction of stage motion with respect the camera view
- the bits that were collected in each round
- the acquisition order in each tile (channel,z) or (z,channel)
- the excitation and emission wavelengths for each channel

Physical pixel sizes and voxel spacing are rounded to three decimal places in microns (0.001 µm). Fused image metadata records the spacing of the fused image.

Most of these are straightforward to obtain. The camera orientation and stage direction can be the trickiest. In our experience, one way to figure this out is to load a few tiles in the NDV datastore viewer and compare the image orientation with the recorded stage direction.

Because there are so many different microscopes and microscope acquisition software, we rely on the user to provide the images in the correct orientation such that a positive displacement in the global stage coordinates corresponds to a positive displacement in the image and vice-versa. In the [Zhuang lab examples](examples/zhuang_lab_mouse_brain.md), we show how to determine the camera and stage orientations when the metadata is not available.

## Codebook and Experiment Order files

For iterative multiplexing, we need to know the codebook, which connects genes and codewords, and the experiment order, which connects rounds and bits.

We expect these to be in `.csv` or `.tsv` format. 

For example, a 16-bit codebook `codebook.tsv` should have the following structure:

| codeword | bit01 | bit02 | bit03 | bit04 | bit05 | bit06 | bit07 | bit08 | bit09 | bit10 | bit11 | bit12 | bit13 | bit14 | bit15 | bit16 |
| ---- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| word 1 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| word 2 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| word 3 | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| word 4 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
|  --------  | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| word N | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |

`exp_order` should have N columns. The first column is the round, starting from `1`. The remaining columns are the readout bits in the codebook, in order of acquisition. **Important: we assume that each tile has a fiducial channel. If there is not, this software package will not work for your experiment.**

For a 16-bit codebook, where we acquire the bits in sequential order within rounds and across rounds, the `exp_order.tsv` will look like:

| round | readout 1 | readout 2 |
| ----- | --------- | --------- |
|   1   |     1     |     2     |
|   2   |     3     |     4     |
|   3   |     5     |     6     |
|   4   |     7     |     8     |
|   5   |     9     |     10    |
|   6   |     11    |     12    |
|   7   |     13    |     14    |
|   8   |     15    |     16    |

## General use

Here, we use a hypothetical dataset that only has one round with two bits. We assume the data is already gain, offset, and hot pixel corrected.

Tiles, rounds, and bits are indexed from `0` in the Python API. Datastore IDs are zero-padded strings: tiles start at `tile0000`, rounds at `round001`, and bits at `bit001`.

```python
from pathlib import Path

import numpy as np
import pandas as pd
from tifffile import imread

from merfish3danalysis.qi2labDataStore import qi2labDataStore

# define the datastore directory and create the datastore
root_path = Path(r"/path/to/dataset/")
raw_path = root_path / "raw_data"
datastore = qi2labDataStore(root_path / "qi2labdatastore")

# required metadata
datastore.channels_in_data = ["alexa488", "alexa561", "alexa647"]
datastore.num_rounds = 1
datastore.codebook = pd.read_csv(raw_path / "codebook.csv")
datastore.experiment_order = pd.read_csv(raw_path / "exp_order.csv").to_numpy()
datastore.num_tiles = 1
datastore.microscope_type = "3D"
datastore.tile_overlap = 0.2
datastore.e_per_ADU = 0.51
datastore.na = 1.35
datastore.ri = 1.51
datastore.binning = 1
datastore.noise_map = np.zeros((2048, 2048), dtype=np.uint16)
datastore.channel_psfs = (
    channel_psfs  # one experimental or theoretical 3D PSF per channel
)
datastore.voxel_size_zyx_um = [0.31, 0.098, 0.098]

# Mark calibrations complete
datastore.datastore_state = {"Calibrations": True}

# initialize the tile
tile_idx = 0
datastore.initialize_tile(tile_idx)

# code to read image tile here
# Assume the images are of shape [n_channels, nz, ny, nx]
fiducial_data = imread(raw_path / "tile001" / "image.tif")[0, :]

# save image data for tile = 0, round = 0, fiducial
datastore.save_local_corrected_image(
    fiducial_data,
    tile=0,
    psf_idx=0,
    gain_correction=True,
    hotpixel_correction=True,
    shading_correction=False,
    round=0,
)

# save stage position for tile = 0, round = 0, fiducial
datastore.save_local_stage_position_zyx_um(
    [1000.0, 200.0, 500.0],
    np.eye(4, dtype=np.float32),
    tile=0,
    round=0,
)

# save excitation and emission wavelengths for tile = 0, round = 0, fiducial
# this position is used for any bits linked to this round
datastore.save_local_wavelengths_um(
    (0.488, 0.520),
    tile=0,
    round=0,
)

# code to read image tile here
# Assume the images are of shape [n_channels, nz, ny, nx]
bit001_data = imread(raw_path / "tile001" / "image.tif")[1, :]

# save first readout channel for tile = 0, bit = 0 (bit001)
datastore.save_local_corrected_image(
    bit001_data,
    tile=0,
    psf_idx=1,
    gain_correction=True,
    hotpixel_correction=True,
    shading_correction=False,
    bit=0,
)

# save excitation and emission wavelengths for tile = 0, bit = 0
datastore.save_local_wavelengths_um(
    (0.561, 0.590),
    tile=0,
    bit=0,
)

# code to read image tile here
# Assume the images are of shape [n_channels, nz, ny, nx]
bit002_data = imread(raw_path / "tile001" / "image.tif")[2, :]

# save second readout channel for tile = 0, bit = 1 (bit002)
datastore.save_local_corrected_image(
    bit002_data,
    tile=0,
    psf_idx=2,
    gain_correction=True,
    hotpixel_correction=True,
    shading_correction=False,
    bit=1,
)

# save excitation and emission wavelengths for tile = 0, bit = 1
datastore.save_local_wavelengths_um(
    (0.635, 0.670),
    tile=0,
    bit=1,
)

# Mark corrected data complete
datastore.datastore_state = {"Corrected": True}
```

## SOFIMA deformable registration convention

When deformable local registration is enabled, preprocessing estimates a SOFIMA
residual flow field after affine registration to the first round. The field is
stored as a displacement image in the acquisition Scene.

The array has shape `(3, z, y, x)`, with `X, Y, Z` displacement channels and
`Z, Y, X` spatial axes. Values are relative displacements in reference-image
pixels. Adding the interpolated field to a first-round reference coordinate gives
the coordinate in the affine-initialized moving image.

The associated metadata records the flow-map spacing (`map_stride_zyx_px`),
the first sample's reference coordinate (`map_box_start_xyz_px`), the lattice
extent (`map_box_size_xyz_px`), and the output grid shape
(`reference_shape_zyx_px`). SOFIMA vectors are patch-centered, so the first
sample is normally at half the patch size.

## DataStore structure

The datastore root is a directory containing separate NGFF v0.6 Scenes and
associated product folders. Image revision names are generated by the writer;
use the datastore API to locate a product rather than constructing its path.
A typical experiment contains:

```text
/experiment
├── raw_data/
└── qi2labdatastore/
    ├── tiles.ome.zarr/                  # acquisition Scene, images and transforms
    ├── fiducial_fusion.ome.zarr/        # downsampled 3D fiducial fusion Scene
    ├── fiducial_projection.ome.zarr/    # projection Scene and Cellpose labels
    ├── fused.ome.zarr/                  # optional full-resolution fusion Scene
    ├── camera/
    │   ├── camera_noise.ome.zarr/
    │   └── camera_metadata/camera.json
    ├── shading/
    │   ├── shading_maps.ome.zarr/
    │   └── shading_metadata/
    ├── psfs/
    │   ├── images/psf_000.ome.zarr/
    │   └── psf_metadata/manifest.json
    ├── acquisition/
    │   ├── acquisition_metadata/acquisition.json
    │   ├── experiment_order/experiment_order.parquet
    │   └── codebook/codebook.parquet
    ├── datastore_state/state.json
    ├── ufish/localizations/<tile>/<bit>.parquet
    ├── codewords/
    │   ├── tiles/
    │   ├── global_filtered/
    │   ├── normalization/
    │   └── optimization/
    ├── cell_boundaries/
    │   ├── cell_rois.zip
    │   └── pixel_rois.zip
    ├── fiducial_projection_tiff/        # optional TIFF export
    ├── fused_tiff/                      # optional per-channel TIFF exports
    ├── proseg/3D/                       # optional native Proseg output
    └── baysor/3D/                       # optional native Baysor output
```

The projection Scene is created during fiducial fusion and holds the Cellpose
labels. ImageJ ROIs are stored under `cell_boundaries/` in global and pixel
coordinates. Per-tile and filtered global codewords are stored under
`codewords/`; named decode runs use `runs/<run>/` within those folders.

## Metadata conventions

- Each tile, round, and channel image has a Scene relationship to the common
  world coordinate system. Acquisition rounds are separate images, not time
  points. Later rounds register to their tile's first round; stage, camera, and
  global transforms place the first round in the shared world.
- Native fiducial and readout arrays remain in their acquisition frame. Decoding
  applies the stored local affine, applicable optimizer chromatic correction,
  and SOFIMA field; registration and fusion retain their existing behavior.
- Readout corrected images are expected after conversion. Deconvolved and
  feature-predictor images are produced when their corresponding processing
  stages run. Use `local_image_path` and the datastore readers to access them.
- Calibration arrays are NGFF v0.6 images. PSFs are stored individually under
  `psfs/images/`, allowing different shapes per channel.

## DataStore API

Nearly all parameters are accessible as class properties and all data has helper functions for reading and writing. The full API reference is available at [qi2labDataStore](reference/classes/qi2labDataStore.md).

## Image access and coordinates

The fused fiducial image and its maximum projection share the downsampled
segmentation grid. Cellpose labels use YX axes and the projection grid; ImageJ
ROIs are mapped to world coordinates using the fused image metadata. For this
mapping, the projection is assigned pixel Z=10. An optional TIFF can be exported
for the Cellpose GUI.

Processing commands accept either an experiment directory containing
`qi2labdatastore` or the datastore directory itself.

Image readers accept `return_future=True` for a read future, `False` for a NumPy
array, or `None` for a lazy handle. Image writers complete pixel writes before
updating metadata; `return_future=True` returns the completed write handle.
