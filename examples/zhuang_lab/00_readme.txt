See comments in each file and the Zhuang example documentation for specifics.

The Zhuang data has 1.5 micron axial spacing and MUST use 2D processing for
deconvolution, segmentation, normalization, decoding, and chromatic estimation.
Keep microscope_type="2D" and decode_mode="2d"
(--decode-mode 2d with the CLI). Chromatic estimation must use only the
transcript's own plane and preserve Z. Retain the 1.5 micron Z spacing for
physical coordinates. Global tile registration uses the shared DataRegistration
workflow with ZYX binning (1, 3, 3). The fit estimates offsets in Z as well as YX.
Fiducial fusion and its maximum-Z TIFF use the downsampled segmentation grid.
The saved spacing is used to transform Cellpose pixel ROIs to global coordinates.

Run every script through the project's uv environment and pass its data root as
the positional argument. Provided time estimates are for a single workstation
with an RTX 3090 GPU and standard hard disk. Run time can be decreased by using
multiple GPUs and/or faster hard disks (SSD or NVMe).

Order to run:

uv run python examples/zhuang_lab/00a_test_image_orientation.py /path/to/zhuang-data --n-tiles 2 (minutes)
uv run python examples/zhuang_lab/01_convert_to_qi2lab.py /path/to/zhuang-data/mop/mouse_sample1_raw (1 day)
uv run python examples/zhuang_lab/02_register_and_deconvolve.py /path/to/zhuang-data (~1 week)
uv run python examples/zhuang_lab/03_cellpose_segmentation.py /path/to/zhuang-data (hours)
uv run python examples/zhuang_lab/04_pixel_decode.py /path/to/zhuang-data (~0.5 week)
uv run python examples/zhuang_lab/05_calculate_f1_score.py /path/to/zhuang-data

Optional one-tile workflow:

uv run python examples/zhuang_lab/05_one_tile_F1.py /path/to/zhuang-data
