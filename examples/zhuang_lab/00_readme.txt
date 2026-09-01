See comments in each file and the Zhuang example documentation for specifics.

Run every script through the project's uv environment and pass its data root as
the positional argument. Provided time estimates are for a single workstation
with an RTX 3090 GPU and standard hard disk. Run time can be decreased by using
multiple GPUs and/or faster hard disks (SSD or NVMe).

Order to run:

uv run python examples/zhuang_lab/00a_test_image_orientation.py /path/to/zhuang-data (minutes)
uv run python examples/zhuang_lab/01_convert_to_qi2lab.py /path/to/zhuang-data/mop/mouse_sample1_raw (1 day)
uv run python examples/zhuang_lab/02_register_and_deconvolve.py /path/to/zhuang-data (~1 week)
uv run python examples/zhuang_lab/03_cellpose_segmentation.py /path/to/zhuang-data (hours)
uv run python examples/zhuang_lab/04_pixel_decode.py /path/to/zhuang-data (~0.5 week)
uv run python examples/zhuang_lab/05_calculate_f1_score.py /path/to/zhuang-data

Optional one-tile workflow:

uv run python examples/zhuang_lab/05_one_tile_F1.py /path/to/zhuang-data
