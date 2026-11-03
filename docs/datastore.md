# qi2lab DataStore overview

## Philosophy

`qi2labDataStore` stores acquisition images and their spatial relationships in
an [OME-NGFF v0.6](https://ngff.openmicroscopy.org/latest/) Scene. It uses
[yaozarrs](https://imaging-formats.github.io/yaozarrs/) for NGFF models and
metadata, and TensorStore for library array I/O. Multiview-stitcher performs
direct-to-Zarr fusion using its own I/O pathway. Tabular outputs use
[Parquet](https://parquet.apache.org/docs/).

The NDTiff and `ndstorage` conversion behavior is preserved. The application
layout version is separate from the OME-NGFF version.

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

Physical pixel sizes and voxel spacing are rounded to three decimal places in microns (0.001 µm). This applies when loading and saving calibration metadata and throughout registration, fusion, decoding, segmentation, and image export. Legacy spacing values are rounded on read; existing files are updated when written again. Global fusion uses the rounded native voxel spacing, and fused OME metadata records the spacing of the fused image. The shared rounding helpers are in `merfish3danalysis.utils.spacing`.

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

Tiles, rounds, and bits are indexed from `0` in the Python API, but datastore IDs are stored as 1-based, zero-padded strings (`round001`, `bit001`, `tile0000`).

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
# Assume the images are of shape [n_channels,nz,nx,ny]
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
# Assume the images are of shape [n_channels,nz,nx,ny]
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
# Assume the images are of shape [n_channels,nz,nx,ny]
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

When deformable local registration is enabled, preprocessing estimates a
SOFIMA residual flow field after the moving fiducial round has already been
affine-initialized into the first fiducial round frame. The field is stored as an
NGFF displacement image in the acquisition Scene.
SOFIMA's local patch-correlation measurements are integer-pixel vectors. The
preprocessing pipeline refines accepted vectors within a small subpixel search
window, composes a bounded residual pass, and saves the float32 result after
the measurements are relaxed with SOFIMA's 3D elastic mesh. The SOFIMA flow
estimation, residual composition, and subpixel patch scoring are run through
JAX on the GPU.

The saved array and attributes use the following convention:

- Array shape is `(3, z, y, x)`.
- Channel order is `X, Y, Z`.
- Spatial axis order is `Z, Y, X`, matching image arrays.
- Values are relative displacements in reference-image pixels. Adding the
  interpolated field to a round001 reference coordinate gives the coordinate in
  the affine-initialized moving image.
- `map_stride_zyx_px` is the flow-map spacing in reference pixels, ordered
  `Z, Y, X`.
- `map_box_start_xyz_px` is the reference pixel coordinate of the first flow
  sample, ordered `X, Y, Z`. SOFIMA estimates patch-centered vectors, so this
  value is normally half the patch size, not `(0, 0, 0)`.
- `map_box_size_xyz_px` is the sampled flow-lattice extent from
  `map_box_start_xyz_px` through the last stored map sample, ordered `X, Y, Z`.
- `reference_shape_zyx_px` records the output grid used when the field is
  applied after reload.

The datastore round trip is expected to preserve the float32 flow array exactly.
Applying a reloaded SOFIMA field with its saved metadata preserves the warp.

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

The projection Scene is created during fiducial fusion. Cellpose labels are
associated with its projection image in the `labels` group. Global-coordinate
and pixel-coordinate ImageJ ROI ZIPs are stored under `cell_boundaries/`.
Optional Baysor and Proseg results retain their native formats.
Per-tile codewords are Parquet files under `codewords/tiles/`. The filtered
global table is saved as Parquet and compressed CSV under
`codewords/global_filtered/`. Named decode runs use `runs/<run>/` under each
of those folders; their normalization metadata is stored separately under
`codewords/normalization/`.

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
- Extra acquisition attributes and non-image outputs remain alongside their
  corresponding application records. Transform models and Scene relationships
  describe the image coordinate systems and their links.

## DataStore API

Nearly all parameters are accessible as class properties and all data has helper functions for reading and writing. The full API reference is available at [qi2labDataStore](reference/classes/qi2labDataStore.md).

## Image access and coordinates

Datastore images use Zarr v3 with OME-NGFF v0.6 metadata. yaozarrs opens and
validates the models, while TensorStore provides the library's image reads and
writes. Metadata-only shape queries do not open a pixel backend. The datastore
coordinates local metadata updates so workers do not overwrite each other's
Scene changes.

The fused fiducial image and its maximum-projection Scene retain the
downsampled segmentation grid. Cellpose labels have YX axes and share the
projection grid. The datastore retains the original fused 3D spacing, origin,
and affine for ROI conversion. A maximum projection has no unique Z; the
existing ROI calculation uses pixel Z=10 when mapping outlines to world space.
An optional projection TIFF can be written for the Cellpose GUI.

Processing entry points accept either an experiment directory containing
`qi2labdatastore` or the datastore directory itself. Existing stores are identified
by `datastore_state/state.json`; a missing input fails before creating output
directories.
Only conversion commands explicitly resolve a new datastore destination.

Public image readers support three modes: `True` returns a read future, `False`
returns a NumPy array, and `None` returns a lazy backend handle opened through
yaozarrs. Public image writers finish pixel writes before updating associated
metadata; `True` returns that completed write handle, while `False` and `None`
return `None`. The private asynchronous writer can return a pending write handle.

Voxel calibration requires three positive finite values in ZYX microns, including
after rounding to stored precision. Missing optional JSON metadata returns an
empty dictionary; malformed or non-object JSON raises an error. Invalid public
image selections and absent images retain their `None` return convention. Write
failures propagate and do not update completion-state caches.
