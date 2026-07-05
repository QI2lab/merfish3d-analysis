# Current API notes

This page summarizes the current processing defaults and recent API behavior.
Use the API reference for full signatures, and use this page for the practical
defaults that affect most pipelines.

## U-FISH model selection

`DataRegistration` uses `simfish` as the default U-FISH model when
`ufish_model=None`.

```python
from merfish3danalysis.DataRegistration import DataRegistration

registration = DataRegistration(
    datastore=datastore,
    decon_readout=True,
    ufish_model=None,  # uses simfish
)
```

Known aliases include `simfish`, `smfish`, `merfish`, `seqfish`, `deepspot`,
and `exseq`. `simfish` and `smfish` resolve to the same packaged U-FISH
weights. A local `.onnx` or `.pth` path can also be supplied.

## RLGC deconvolution

The public RLGC API is now the non-accelerated reference-style implementation:

```python
from merfish3danalysis.utils.rlgc import chunked_rlgc

deconvolved = chunked_rlgc(
    image=image_zyx,
    psf=psf_zyx,
    gpu_id=0,
    crop_yx=2048,
)
```

Current behavior:

- Biggs-Andrews acceleration has been removed from the public API.
- `crop_yx` controls the retained lateral tile size.
- `crop_z` is no longer supported. Z is processed as a full stack.
- If a GPU memory allocation fails, RLGC retries with `crop_yx` reduced by
  `128` pixels until the tile fits or the PSF support is reached.
- `DataRegistration` remembers the successful fallback `crop_yx` inside the
  worker, so later deconvolutions do not repeatedly start from a too-large tile.
- PSFs are normalized to unit sum inside RLGC unless `normalize_psf=False` is
  passed for diagnostics.

## Datastore PSFs

Datastore creation writes a 3D PSF for each channel. The PSF z-size follows the
stored image z-size. This is true even when the datastore is marked as a `2D`
experiment for downstream pixel decoding.

The `2D` or `3D` microscope designation controls downstream decoding policy; it
does not force the deconvolution PSF to be a single plane.

## Preprocessing CLI

The local preprocessing command uses the same API defaults:

```bash
uv run qi2lab-preprocess \
  /path/to/experiment \
  --decon \
  --crop-yx-decon 2048 \
  --ufish-model simfish
```

By default, `qi2lab-preprocess` runs local preprocessing and then global
fiducial registration/fusion. Local registration uses the qi2lab GPU path:
fiducial rounds are first registered laterally on a max-Z projection, then in
XYZ, and optional SOFIMA residual flow fields are estimated after the affine
fiducial alignment. Global registration uses multiview-stitcher registration
and direct OME-Zarr fusion. GPU acceleration is used for the fusion backend;
the multiview-stitcher registration step itself is configured through CPU
parallelism and Dask scheduler options.

To rerun the global registration and fusion stage on an existing datastore
without redoing local preprocessing:

```bash
uv run qi2lab-preprocess \
  /path/to/experiment \
  --num-gpus 2 \
  --global-registration-only \
  --overwrite
```

Useful global-stage diagnostics and performance controls are exposed as CLI
flags:

- `--registration-diagnostics` prints per-step geometry and registration
  diagnostics.
- `--global-registration-parallel-jobs` sets multiview-stitcher pairwise
  registration parallelism.
- `--global-registration-scheduler` sets the Dask scheduler used around global
  registration.
- `--global-fusion-n-batch`, `--global-fusion-n-jobs`,
  `--global-fusion-output-chunksize`, and
  `--global-fusion-overlap-in-pixels` control direct-to-Zarr fusion batching.
- `--sofima-*` flags parameterize SOFIMA residual registration. These are CLI
  parameters, not environment variables.

## Pixel Decoding CLI

Use `--decode-mode auto|2d|3d` to control decoding policy. `auto` follows the
datastore microscope type; explicit modes also select the matching default
minimum-pixel, magnitude, and feature-predictor thresholds.

Chromatic affine estimation is opt-in during iterative normalization:

```bash
uv run qi2lab-decode \
  /path/to/experiment \
  --num-gpus 2 \
  --estimate-chromatic-affines
```

The estimator uses decoded RNA on-bit centroids from valid codewords only.
Blank codewords are excluded before pairing transcripts across wavelengths.
The fitting thresholds, RANSAC settings, and centroid support are exposed as
`--chromatic-*` flags and map directly to
`ChromaticAffineEstimationConfig`.

Transcript filtering now uses either the blank-fraction filter or the logistic
regression filter selected by `--filter-method blank_fraction|lr`. The removed
enriched blank-barcode filtering path and its options are no longer part of the
CLI flow.

## Cellpose Segmentation CLI

`qi2lab-segment` defaults to `cpsam_v2`, matching the current Cellpose-SAM v2
GUI model. The default diameter is `None`, so the command does not force a cell
size unless `--diameter` is explicitly provided. The fused fiducial max
projection is passed to Cellpose without rescaling to 8-bit; Cellpose handles
normalization through the supplied percentile settings.
