# Current API notes

See the [API reference](reference/index.md) for full signatures and the
[workflow](workflow.md) for the processing commands.

## U-FISH model selection

`DataRegistration` uses the `simfish` U-FISH weights when `ufish_model=None`.
Other aliases include `merfish`, `seqfish`, `deepspot`, and `exseq`. You can
also supply a local `.onnx` or `.pth` path.

```python
from merfish3danalysis.DataRegistration import DataRegistration

registration = DataRegistration(datastore=datastore, ufish_model=None)
```

## RLGC deconvolution and PSFs

```python
from merfish3danalysis.utils.rlgc import chunked_rlgc

deconvolved = chunked_rlgc(image=image_zyx, psf=psf_zyx, gpu_id=0, crop_yx=2048)
```

`crop_yx` sets the lateral tile size; Z is processed as a full stack. RLGC
reduces the tile size if a GPU allocation fails and normalizes the PSF unless
`normalize_psf=False`. Datastore creation generates a 3D PSF for each channel,
including for experiments decoded in 2D.

## Preprocessing

The preprocessing CLI deconvolves fiducial and readout images and enables
SOFIMA registration by default. Use `--no-decon` to disable readout
deconvolution. When using `DataRegistration` directly, `decon_readout=False`
is the default.

To rerun global registration and fusion without repeating local preprocessing:

```bash
uv run qi2lab-preprocess /path/to/experiment --global-registration-only --overwrite
```

To recompute local fiducial registration and SOFIMA fields before global
registration and fusion, while retaining readout outputs:

```bash
uv run qi2lab-preprocess /path/to/experiment --fiducial-registration-only --overwrite
```

## Pixel decoding

Use `--decode-mode auto|2d|3d` to select decoding policy. `auto` follows the
datastore microscope type. The `--normalization-features cells|all` option
selects features used for normalization; `cells` is the default and uses all
features if segmentation is absent. An empty segmentation selects no features.
Rerun optimization when changing this setting; `--reprocess-existing` only
refilters saved decoding.

Chromatic affine estimation during iterative normalization is optional:

```bash
uv run qi2lab-decode /path/to/experiment --estimate-chromatic-affines
```

To exclude known failed codewords from iterative optimization, supply a UTF-8
file containing one codebook `gene_id` per line. Blank lines and lines starting
with `#` are ignored.

```bash
uv run qi2lab-decode /path/to/experiment --optimization-exclusions-file bad_codewords.txt
```

Relative paths are resolved in the datastore directory. Excluded codewords remain
in the codebook and final decoding output. Use `--filter-method blank_fraction|lr`
to select the transcript filter.

## Cellpose segmentation

`qi2lab-segment` defaults to the `cpsam_v2` model and leaves diameter unset.
Use `--min-cell-area-um2` to exclude small outlines from the global Cellpose
boundaries. This is separate from `--min-size`, the minimum mask size in pixels.

```bash
uv run qi2lab-segment /path/to/experiment --min-cell-area-um2 20
```

To change the cutoff without rerunning Cellpose:

```bash
uv run qi2lab-segment /path/to/experiment --outlines-only --min-cell-area-um2 20
```

This recreates `cell_boundaries/cell_rois.zip` from the saved pixel ROIs. Cell
IDs may change when the retained outlines change. Reload the viewer and rerun
decoding if cell assignments or normalization need to reflect the updated
outlines. Use `--roi-workers` to set the number of ROI workers or
`--no-roi-multiprocessing` to use one worker.

## Viewer

```bash
uv run viewer /path/to/experiment
```

See the [viewer guide](viewer.md) for the available views and overlays.

## Tile export

Export corrected fiducial rounds and readout bits, registered images, and local
decoded spots annotated with their global filtering results:

```bash
uv run qi2lab-export-tile /path/to/experiment --tile-id tile0000
```

The default output is `tile_exports/tile0000/` beside the datastore. Registered
images use the first round's local grid and the stored affine and SOFIMA
transforms; nonidentity warps require CUDA. Chromatic correction is applied
during decoding. The CSV contains `passes_global_filter` and columns from the
global table. Its local coordinates overlay the registered images. The local
and global decoding outputs must belong to the same run.

Use `--output-dir` to select an output directory outside the datastore and
`--overwrite` to replace existing exports.
