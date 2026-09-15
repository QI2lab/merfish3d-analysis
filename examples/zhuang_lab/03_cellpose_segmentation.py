"""Run the shared qi2lab Cellpose segmentation workflow for the Zhuang data.

The 1.5 micron axial spacing requires 2D segmentation. The shared API segments
only the fused fiducial maximum-Z projection with ``do_3D=False``. All model,
normalization, mask, and ROI options and defaults are the same as qi2lab-segment.
Run this script with --help to see the shared command-line options.
"""

from merfish3danalysis.cli.qi2lab_microscopes.segment_fiducial import main

if __name__ == "__main__":
    main()
