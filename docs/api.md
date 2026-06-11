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
stored image z-size, including z-strided datastores. This is true even when the
datastore is marked as a `2D` experiment for downstream pixel decoding.

The `2D` or `3D` microscope designation controls downstream decoding policy; it
does not force the deconvolution PSF to be a single plane.

## Preprocessing CLI

The local preprocessing command uses the same API defaults:

```bash
python -m merfish3danalysis.cli.qi2lab_microscopes.preprocess \
  /path/to/experiment \
  --decon \
  --crop-yx-decon 2048 \
  --ufish-model simfish
```

For z-strided datastores, preprocessing still uses `--zstride-level` to select
the corresponding `qi2labdatastore_zstrideXX` directory.

## Pixel Decoding CLI

For decode-time z-striding, `qi2lab-decode --zstride-level N` reads the normal
`qi2labdatastore` source data and decodes planes `0, N, 2N...`. Values `0` and
`1` keep all planes. Non-default decode runs are written under decode subfolders
such as `decoded/zstride_03_2d/` and
`all_tiles_filtered_decoded_features/zstride_03_2d/`.

Use `--decode-mode auto|2d|3d` to control decoding policy. `auto` follows the
datastore microscope type; explicit modes also select the matching default
minimum-pixel, magnitude, and feature-predictor thresholds.
