# qi2lab DataStore overview

## Philosophy

To efficiently handle 3D MERFISH experiments, `qi2labDataStore` stores image arrays as independent [OME-NGFF v0.5](https://ngff.openmicroscopy.org/latest/) images, read and written through [yaozarrs](https://imaging-formats.github.io/yaozarrs/) using the TensorStore interface. Tabular outputs are stored as [Parquet](https://parquet.apache.org/docs/).

For backward compatibility, the reader can still open legacy direct Zarr arrays and legacy `.zattrs` metadata from older datastore versions.

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

Most of these are straightforward to obtain. The camera orientation and stage direction can be the trickiest. In our experience, one way to figure this out is to load a few tiles of the data in [napari](https://github.com/napari) and explore different orientations of the images and stage direction.

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
from merfish3danalysis.qi2labDataStore import qi2labDataStore
from pathlib import Path

# define the datastore directory and create the datastore
root_path = Path(r"/path/to/dataset/")
datastore = qi2labDataStore(root_path / Path("qi2labdatastore"))

# required metadata
datastore.channels_in_data = ["alexa488","alexa561","alexa647"]
datastore.baysor_path = Path(r"/path/to/Baysor/bin/baysor/bin/./baysor")
datastore.baysor_options = Path(r"/path/to/baysor_options.toml")
datastore.julia_threads = 8
datastore.num_rounds = 1
datastore.codebook = Path(r"/path/to/dataset/raw_data/codebook.csv")
datastore.experiment_order = Path(r"/path/to/dataset/raw_data/exp_order.csv")
datastore.num_tiles = 1
datastore.microscope_type = "3D"
datastore.tile_overlap = 0.2
datastore.e_per_ADU = 0.51
datastore.na = 1.35
datastore.ri = 1.51
datastore.binning = 1
datastore.noise_map = None
datastore.channel_psfs = channel_psfs # either experimental or theoretical PSFs
datastore.voxel_size_zyx_um = [.31,.098,.098]

# Update datastore state that Calibration are created
datastore_state = datastore.datastore_state
datastore_state.update({"Calibrations": True})
datastore.datastore_state = datastore_state

# initialize the tile
tile_idx = 0
datastore.initialize_tile(tile_idx)

# code to read image tile here
# Assume the images are of shape [n_channels,nz,nx,ny]
fiducial_data = imread("/path/to/dataset/raw_data/tile001/image.tif")[0,:]

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
    [1000., 200., 500.], tile=0, round=0
)

# save excitation and emission wavelengths for tile = 0, round = 0, fiducial
# this position is used for any bits linked to this round
datastore.save_local_wavelengths_um(
    (.488, .520),
    tile=0,
    round=0,
)

# code to read image tile here
# Assume the images are of shape [n_channels,nz,nx,ny]
bit001_data = imread("/path/to/dataset/raw_data/tile001/image.tif")[1,:]

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
    (.561, .590),
    tile=0,
    bit=0,
)

# code to read image tile here
# Assume the images are of shape [n_channels,nz,nx,ny]
bit002_data = imread("/path/to/dataset/raw_data/tile001/image.tif")[2,:]

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
    (.635, .670),
    tile=0,
    bit=1,
)

# update datastore state that corrected data is saved 
datastore_state = datastore.datastore_state
datastore_state.update({"Corrected": True})
datastore.datastore_state = datastore_state
```

## DataStore structure

```bash
/experiment
├── raw_data/
│   └── <raw experimental data and metadata>
└── qi2labdatastore/
    ├── datastore_state.json
    ├── calibrations.zarr/
    │   ├── .zattrs
    │   │   ├── <experiment metadata: codebook, exp_order, channels, ...>
    │   │   ├── <voxel_size_zyx_um>
    │   │   └── <psf_manifest>
    │   ├── noise_map/                # OME-NGFF v0.5 image
    │   │   ├── zarr.json
    │   │   └── 0/
    │   ├── shading_maps/             # OME-NGFF v0.5 image
    │   │   ├── zarr.json
    │   │   └── 0/
    │   └── psf_data/
    │       ├── psf_000/              # OME-NGFF v0.5 image
    │       │   ├── zarr.json
    │       │   └── 0/
    │       ├── psf_001/
    │       └── ...
    ├── fiducial/
    │   └── tile0000/
    │       ├── round001.zarr/
    │       │   ├── .zattrs           # legacy compatibility mirror
    │       │   ├── corrected_data/   # OME-NGFF v0.5 image
    │       │   │   ├── zarr.json
    │       │   │   └── 0/
    │       │   ├── registered_decon_data/
    │       │   │   ├── zarr.json
    │       │   │   └── 0/
    │       │   └── opticalflow_xform_px/
    │       │       ├── zarr.json
    │       │       └── 0/
    │       ├── round002.zarr/
    │       └── ...
    ├── readouts/
    │   └── tile0000/
    │       ├── bit001.zarr/
    │       │   ├── .zattrs           # legacy compatibility mirror
    │       │   ├── corrected_data/
    │       │   │   ├── zarr.json
    │       │   │   └── 0/
    │       │   ├── registered_decon_data/
    │       │   │   ├── zarr.json
    │       │   │   └── 0/
    │       │   └── registered_feature_predictor_data/
    │       │       ├── zarr.json
    │       │       └── 0/
    │       ├── bit002.zarr/
    │       └── ...
    ├── feature_predictor_localizations/
    │   └── tile0000/
    │       ├── bit001.parquet
    │       └── ...
    ├── fused/
    │   └── fused.zarr/
    │       ├── fused_fiducial_iso_zyx/
    │       │   ├── zarr.json
    │       │   └── 0/
    │       └── fused_all_channels_zyx/   # optional
    ├── segmentation/
    │   └── cellpose/
    │       ├── cellpose.zarr/
    │       │   └── masks_fiducial_iso_zyx/
    │       │       ├── zarr.json
    │       │       └── 0/
    │       └── imagej_rois/global_coords_rois.zip
    ├── decoded/
    │   ├── tile0000_decoded_features.parquet
    │   └── all_tiles_filtered_decoded_features.parquet
    └── mtx_output/
```

## Metadata conventions

- Each image directory (for example `corrected_data/`, `registered_decon_data/`, `masks_fiducial_iso_zyx/`) is a standalone OME-NGFF v0.5 image.
- In OME metadata, we only write voxel scale (`scale`) and original tile position (`translation`) when available.
- All other datastore metadata is written into `zarr.json -> extra_attributes` for that image (for example `bit_linker`, `round_linker`, `psf_idx`, correction flags, wavelengths, transforms).
- For `opticalflow_xform_px`, the dense 4D displacement field is stored only in the OME-Zarr array (`0/`). OME transforms are identity (`scale=1`, `translation=0`) and metadata only stores lightweight fields such as `block_size` and `block_stride`.
- Legacy `.zattrs` files are still mirrored for compatibility with older readers, but new metadata should be considered authoritative in `extra_attributes`.
- PSFs are stored as one image per channel under `calibrations.zarr/psf_data/psf_XXX/`, which allows different PSF array sizes across channels.

## DataStore API

Nearly all parameters are accessible as class properties and all data has helper functions for reading and writing. The full API reference is available at [qi2labDataStore](reference/classes/qi2labDataStore.md).
