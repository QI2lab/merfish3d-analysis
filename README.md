# wf-merfish

[![License](https://img.shields.io/pypi/l/wf-merfish.svg?color=green)](https://github.com/dpshepherd/wf-merfish/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/wf-merfish.svg?color=green)](https://pypi.org/project/wf-merfish)
[![Python Version](https://img.shields.io/pypi/pyversions/wf-merfish.svg?color=green)](https://python.org)
[![CI](https://github.com/dpshepherd/wf-merfish/actions/workflows/ci.yml/badge.svg)](https://github.com/dpshepherd/wf-merfish/actions/workflows/ci.yml)

Acquisition and GPU accelerated post-processing for qi2lab MERFISH. This package currently requires Linux due to RAPIDS.AI package availabilty. Windows support is planned as possible

## Installation

Create a python 3.10 environment using your favorite package manager, e.g.
```mamba create -n wf-merfish python=3.10```

Activate the environment and install the GPU dependencies.

For Linux OS:
```
mamba activate wf-merfish
mamba install -c conda-forge -c nvidia -c pytorch -c rapidsai cupy cucim=24.02 pylibraft cuda-version=12.1 cudnn cutensor nccl onnx onnxruntime pytorch torchvision pytorch-cuda=12.1
```

Finally, repository using ```git clone https://github.com/QI2lab/wf-merfish``` and then install using `pip install .` or for interactive editing use `pip install -e .`.

## qi2lab MERFISH zarr layout v0.1
<details>
<summary>file format</summary>
  
- /project_root
  - /calibrations.zarr
    - .zattrs
      - <exp_codebook> (with blank codes)
      - <exp_order> (e.g. round 0 -> codebook bits 0,1)
      - <metadata> (objective NA, channels used, etc...)
    - <camera_noise_map> (used for hotpixel correction)
    - <psf_data> (calculated based on experiment metadata)
  - /polyDT
    - /tile000
      - /round0000.zarr
        - .zattrs
          - <stage_zyx_um> (z,y,x in microns)
          - <wavelengths_um> (excitation, emission)
          - <voxel_size_zyx_um> (z,y,x in microns)
          - <bit_linker> (what bits are linked to this polyDT image)
          - <affine_zyx_um> (affine transform for world coordinate warping)
          - <origin_zyx_um> (origin rigid translation transform for world coordinate warping)
          - <spacing_zyx_um> (spacing scale transform for world coordinate warping)
        - <camera_data> (optional: raw camera data)
        - <corrected_data> (hot pixel, offset, and gain corrected data)
        - <registered_decon_data> (deconvolved data, note this is the same as the corrected data for the first round)
      - /round0001.zarr
        - .zattrs
          - <stage_zyx_um> (z,y,x in microns)
          - <wavelengths> (excitation, emission)
          - <voxel_size_zyx_um> (z,y,x in microns)
          - <bit_linker> (what bits are linked to this polyDT image)
          - <rigid_xyz_px> (rigid registration for round 0 alignment in pixels)
        - <camera_data> (optional: raw camera data)
        - <corrected_data> (hot pixel, offset, and gain corrected data)
        - <of_xyz_3x_downsample> (optical flow field for round 0 alginment in pixels)
        - <registered_decon_data> (deconvolved, then rigid and deformable registration applied to warp back to round 0)
      - /roundNNNN.zarr
    - /tile001
    - ...
    - /tileNNN
  - /readouts
    - /tile0000
      - /bit00.zarr
        - .zattrs
          - <wavelengths_um> (excitation, emission)
          - <voxel_size_zyx_um> (z,y,x in microns)
          - <round_linker> (what polyDT round is bit corresponds to)
        - <camera_data> (optional: raw camera data)
        - <corrected_data> (hot pixel, offset, and gain corrected data)
        - <registered_decon_data> (deconvolved, then rigid and deformable registration applied to warp back to round 0)
        - <registered_ufish_data> (ufish prediction applied to registered_decon_data)
      - /bit01.zarr
      - ...
      - /bitNN.zarr
  - /ufish_localizations
    - /tile0000
      - bit01.parquet
      - bitNN.parquet
    - /tile0001
    - ....
    - /tileNNNN
  - /decoded
    - tile0000_decoded_features.csv
    - ...
    - tileNNNN_decod_features.csv
    - all_tiles_filtered_decoded_features.parquet
  - /fused
    - /fused.zarr
      - Either one of:
        - <fused_iso_all_zyx> (polyDT and readouts deconvolved, registered data fused at isotropic voxel spacing)
        - <fused_iso_polyDT_zyx> (polyDT deconvolved, registered data fused at isotropic voxel spacing)
  - /segmentation
    - cell_centroids.parquet (yx centroid for each cell in cellpose mask prediction)
    - cell_outlines.json (yx polygons for each cell in cellpose mask prediction)
    - /cellpose.zarr
      - <masks_iso_zyx> (cellpose masks generated from maximum projection data choice in fused.zarr)

</details>
      

