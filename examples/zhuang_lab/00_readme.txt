See comments in each file for specifics. 

Provided time estimates are for a single workstation with a RTX 3090 GPU and standard hard disk. Run time
can be decreased by using multiple GPUs and/or faster hard disks (SSD or NVMe).

Order to run:

00a_test_image_orientation.py (minutes)
01_convert_to_qi2lab.py (1 day)
02_register_and_deconvolve.py (~1 week)
03_cellpose_segmentation.py (hours)
04_pixel_decode.py (~0.5 week)