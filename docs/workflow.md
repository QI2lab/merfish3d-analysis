# Full Experiment Analysis Workflow

## Initialize datastore

```mermaid
%%{init: { "theme": "default", "themeVariables": { "htmlLabels": true, "curve": "linear", "layout": "elk" } } }%%
flowchart TD
 subgraph s1["Define datastore"]
        n1["Number of tiles"]
        n2["Microscope metadata"]
        n3["Experiment metadata"]
        n4["qi2labdatastore"]
  end
    n1 --> n4
    n2 --> n4
    n3 --> n4

    n1@{ shape: notch-rect}
    n2@{ shape: notch-rect}
    n3@{ shape: notch-rect}
    n4@{ shape: lin-cyl}
```

## Populate datastore

```mermaid
%%{init: { "theme": "default", "themeVariables": { "htmlLabels": true, "curve": "linear", "layout": "elk" } } }%%
flowchart TD
 subgraph s1["Place raw data into datastore"]
        n5["Raw data"]
        n12["Camera correction"]
        n6["Geometric transformation
        (local to global)"]
        n7["Experiment order"]
        n8["Fidicual data"]
        n9["MERFISH data"]
        n10["Global tile positions"]
        n11["qi2labdatastore"]
 end
    n5 --> n12
    n12 --> n6
    n6 --> n7
    n7 --> n8
    n7 --> n9
    n8 --> n11
    n9 --> n11
    n10 --> n11

    n5@{ shape: procs}
    n6@{ shape: notch-rect}
    n7@{ shape: notch-rect}
    n8@{ shape: procs}
    n9@{ shape: procs}
    n10@{ shape: procs}
    n11@{ shape: lin-cyl}
    n12@{ shape: notch-rect}
```
##  Deconvolve, locally register, and spot predict all tiles

```mermaid
%%{init: { "theme": "default", "themeVariables": { "htmlLabels": true, "curve": "linear", "layout": "elk" } } }%%
flowchart TD
 subgraph s2["MERFISH preprocessing"]
        n17["Optional deconvolution"]
        n18["Feature predictor prediction"]
        n19["Save native readout arrays"]
  end
 subgraph s3["Fiducial preprocessing"]
        n14["Optional deconvolution"]
        n15["Affine registration"]
        n16["SOFIMA residual flow field"]
  end
 subgraph s1["Local preprocessing"]
        s2
        s3
        n1["Local tile registrations back to round 1"]
  end
  n13["qi2labdatastore"]
   
    s3 --> n1
    n1 --> s2
    s1 <--> n13
    n14 --> n15
    n15 --> n16
    n17 --> n18
    n18 --> n19
    n1@{ shape: notch-rect}
    n13@{ shape: lin-cyl}
    n14@{ shape: procs}
    n15@{ shape: procs}
    n16@{ shape: procs}
    n17@{ shape: procs}
    n18@{ shape: procs}
    n19@{ shape: procs}
```

The qi2lab preprocessing CLI deconvolves fiducial and readout images by default.
Use `--no-decon` to disable readout deconvolution while retaining fiducial
deconvolution. Fiducial rounds are registered to round 1 using affine
registration and, when enabled, SOFIMA residual flow. The residual is saved only
when it improves alignment. Readout preprocessing saves corrected images,
optional deconvolved images, and feature predictions.

Decoding applies chromatic, affine, and SOFIMA transforms on each tile's round-1
grid. Transcript coordinates are then mapped to world coordinates using the
stage position, camera calibration, voxel spacing, and global registration.

## Global registration and fusion of first fiducial round

```mermaid
%%{init: { "theme": "default", "themeVariables": { "htmlLabels": true, "curve": "linear", "layout": "elk" } } }%%
flowchart TD
 subgraph s1["Global registration"]
    n3["Fiducial data"]
    n4["Multiview-Stitcher"]
 end
 n1["qi2labdatastore"]
 n2["Direct-to-Zarr fused fiducial image"]
 n5["Cellpose"]
 n6["2D cell segmentations"]
 n7["Optimized global tile positions"]
 n8["User optimized parameters"]
   
    s1 <--> n7
    n7 <--> n1
    s1 --> n2
    n3 <--> n4
    n2 --> n5
    n5 --> n6
    n6 --> n1
    n8 --> n5
    n1@{ shape: lin-cyl}
    n2@{ shape: procs}
    n3@{ shape: procs}
    n6@{ shape: procs}
    n7@{ shape: notch-rect}
    n8@{ shape: notch-rect}
```

Global registration uses multiview-stitcher with the stored stage positions as
the starting geometry. Fiducial fusion preserves Z spacing and downsamples each
lateral axis by `round(z_spacing / lateral_spacing, 1)`. The maximum-Z
projection and Cellpose labels are stored in `fiducial_projection.ome.zarr`;
the fused grid is used to transform Cellpose ROIs into global coordinates.
Replacing the projection archives the previous segmentation; run Cellpose again
after fusion. Rerun the global stage with:

```bash
uv run qi2lab-preprocess /path/to/experiment --global-registration-only
```

## Fuse all channels

Once local and global registrations are complete, create one multichannel
OME-NGFF v0.6 image in `qi2labdatastore/fused.ome.zarr`:

```bash
uv run qi2lab-fuseall /path/to/experiment \
  --output-chunk-zyx 32,2048,2048 \
  --fusion-workers 30 \
  --compression blosc-lz4 \
  --compression-level 1
```

Channel 0 is the first-round fiducial; remaining channels are the codebook bits
in numeric order. Fusion applies local and global affine registration; chromatic
correction is applied during decoding, and SOFIMA fields are not used in fusion.
Add `--write-ome-tiffs` to export each channel under `qi2labdatastore/fused_tiff/`.

Fusion uses lossless Blosc-Zstd compression by default. The command also accepts
`blosc-lz4` and `zstd`, with compression levels from 1 to 9. Choose
`--output-chunk-zyx` and `--fusion-workers` for the available CPU and memory;
larger chunks require more memory per worker. Rerunning fusion creates a new
fused image revision and starts from the beginning.

## Pixel decoding


```mermaid
%%{init: { "theme": "default", "themeVariables": { "htmlLabels": true, "curve": "linear", "layout": "elk" } } }%%
flowchart TD
 subgraph s1["Pixel decoding"]
    n2["MERFISH data"]
    n3["Global normalization estimate"]
    n4["Iterative normalization estimate"]
    n5["Pixel decoding"]
    n6["Blank-fraction or LR filtering"]
    n7["Overlap cleanup"]
    n8["Cell assignment"]
    n9["Data prep for resegmentation"]
 end
 n1["qi2labdatastore"]

    s1 <--> n1
    n2 --> n3
    n3 --> n4
    n4 --> n5
    n5 --> n6
    n6 --> n7
    n7 --> n8
    n8 --> n9

    n1@{ shape: lin-cyl}
    n2@{ shape: procs}
    n3@{ shape: procs}
    n4@{ shape: procs}
    n5@{ shape: procs}
    n6@{ shape: procs}
    n7@{ shape: procs}
    n8@{ shape: procs}
    n9@{ shape: procs}
```

Optional chromatic affine estimation during iterative normalization uses decoded
RNA centroids and saves the calibration for subsequent decoding.

## Viewer inspection

Use `viewer` to inspect local and global images with transcript and cell boundary
overlays. Local warped views can display affine and SOFIMA registration.

## 3D segmentation based on decoded RNA

```mermaid
%%{init: { "theme": "default", "themeVariables": { "htmlLabels": true, "curve": "linear", "layout": "elk" } } }%%
flowchart TD
 subgraph s1["3D segmentation"]
    n3["Decoded, cell-assigned RNA"]
    n4["Proseg"]
    n6["User optimized parameters"]
 end
 n1["qi2labdatastore"]
 n2["3D segmentations"]
 n5["Updated RNA assignments"]

    s1 <--> n1
    n3 --> n4
    n6 --> n4
    s1 --> n2
    n2 --> n5
    n5 --> n1
    n1@{ shape: lin-cyl}
    n2@{ shape: procs}
    n3@{ shape: procs}
    n5@{ shape: procs}
    n6@{ shape: notch-rect}
```
