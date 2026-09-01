# Zhuang laboratory mouse motor cortex example

The scripts in `examples/zhuang_lab` process the public mouse motor-cortex
MERFISH dataset from the
[Brain Image Library](https://download.brainimagelibrary.org/cf/1c/cf1c1a431ef8d021/).
Download `additional_files`, `mouse1_sample1_raw`, and
`dataset_metadata.xslx`, retaining the source directory layout.

Each script accepts its data root as a positional command-line argument. No
path is embedded in the scripts.

## Inspect image orientation

Pass the top-level downloaded-data directory:

```bash
uv run python examples/zhuang_lab/00a_test_image_orientation.py \
  /path/to/zhuang-data
```

## Convert the public data

The conversion script expects the sample directory containing
`additional_files` and `mouse1_sample1_raw`. With the layout used by the
downloaded example:

```bash
uv run python examples/zhuang_lab/01_convert_to_qi2lab.py \
  /path/to/zhuang-data/mop/mouse_sample1_raw
```

By default, conversion writes `qi2labdatastore` beneath the top-level
`zhuang-data` directory. The remaining scripts therefore take that top-level
directory:

```bash
uv run python examples/zhuang_lab/02_register_and_deconvolve.py \
  /path/to/zhuang-data

uv run python examples/zhuang_lab/03_cellpose_segmentation.py \
  /path/to/zhuang-data

uv run python examples/zhuang_lab/04_pixel_decode.py \
  /path/to/zhuang-data

uv run python examples/zhuang_lab/05_calculate_f1_score.py \
  /path/to/zhuang-data
```

Cellpose parameters in `03_cellpose_segmentation.py` are example values. Tune
them for the downloaded images before running the segmentation step.

For the optional one-tile decoding and visualization workflow:

```bash
uv run python examples/zhuang_lab/05_one_tile_F1.py \
  /path/to/zhuang-data
```
