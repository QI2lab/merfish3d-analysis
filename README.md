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
          - <stage_zyx_um>
          - <wavelengths_um> (excitation, emission)
          - <voxel_size_zyx_um>
          - <bit_linker>
          - <affine_zyx_um> (affine transform for world coordinate warping)
          - <origin_zyx_um> (origin rigid translation transform for world coordinate warping)
          - <spacing_zyx_um> (spacing scale transform for world coordinate warping)
        - <raw_data>
        - <registered_data> (note this is the same as the raw data for the first round)
      - /round0001.zarr
        - .zattrs
          - <stage_zyx_um>
          - <wavelengths> (excitation, emission)
          - <bit_linker>
          - <rigid_xyz_px> (rigid registration for round 0 alignment)
        - <camera_data>
        - <raw_data>
        - <of_xyz_3x_downsample> (optical flow field for round 0 alginment)
        - <registered_decon_data> (rigid and deformable registration applied to warp back to round 0)
      - /roundNNNN.zarr
    - /tile001
    - ...
    - /tileNNN
  - /readouts
    - /tile0000
      - /bit00.zarr
        - .zattrs
          - <wavelengths_um> (excitation, emission)
          - <voxel_size_zyx_um>
          - <round_linker>
        - <camera_data>
        - <raw_data>
        - <registered_decon_data>
        - <registered_ufish_data>
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
      - <fused_iso_zyx>

</details>
      

