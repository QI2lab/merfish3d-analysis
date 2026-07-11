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

The preprocessing command intentionally exposes only high-level controls:

```bash
uv run qi2lab-preprocess \
  /path/to/experiment \
  --num-gpus 2
```

Local preprocessing uses the package defaults: fiducial and readout
deconvolution are enabled, feature prediction is always written for readout
bits, and SOFIMA residual registration is enabled. Fiducial rounds are
registered laterally on a max-Z projection and then in XYZ. SOFIMA flow fields
are estimated after affine fiducial alignment and are accepted only when they
improve the fiducial alignment error compared with affine alone. Readout and
fiducial image arrays remain in their native local storage; decode and viewer
paths apply the selected transform chain when they need aligned data.

Global registration follows the multiview-stitcher registration and fusion
workflow. Stage metadata initializes the tile geometry, CPU registration
refines the global tile transforms, and the fused fiducial OME-Zarr is written
directly to disk with the multiview-stitcher/CuPy fusion backend.

To rerun the global registration and fusion stage on an existing datastore
without redoing local preprocessing:

```bash
uv run qi2lab-preprocess \
  /path/to/experiment \
  --num-gpus 2 \
  --global-registration-only \
  --overwrite
```

Registration-specific tuning is kept in the Python API rather than exposed as
routine CLI flags.

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

## Viewer CLI

Use the `viewer` entry point for read-only datastore inspection:

```bash
uv run viewer /path/to/experiment
```

The controller exposes three view modes:

- `Local native` shows stored local images without alignment claims.
- `Local warped` applies user-selected chromatic, stage/round affine, and
  SOFIMA transform components before display.
- `Global fused` opens the fused Zarr image lazily and overlays selected sparse
  data.

NDV displays image channels. Transcript points and cell boundaries from the
datastore, Proseg, Cellpose, or Baysor are rendered as sparse VisPy overlays so
changing transcript selections does not require rebuilding image arrays.
