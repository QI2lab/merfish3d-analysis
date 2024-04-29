# wf-merfish

[![License](https://img.shields.io/pypi/l/wf-merfish.svg?color=green)](https://github.com/dpshepherd/wf-merfish/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/wf-merfish.svg?color=green)](https://pypi.org/project/wf-merfish)
[![Python Version](https://img.shields.io/pypi/pyversions/wf-merfish.svg?color=green)](https://python.org)
[![CI](https://github.com/dpshepherd/wf-merfish/actions/workflows/ci.yml/badge.svg)](https://github.com/dpshepherd/wf-merfish/actions/workflows/ci.yml)

Acquisition and GPU accelerated post-processing for qi2lab MERFISH

## Installation

Create a python 3.10 environment using your favorite package manager, e.g.
```mamba create -n wf-merfish python=3.10```

Activate the environment and install the GPU dependencies.

For Linux OS:
```
mamba activate wf-merfish
mamba install -c conda-forge -c nvidia -c pytorch -c rapidsai cupy cucim=24.02 cuda-version=12.1 cudnn cutensor nccl onnx onnxruntime pytorch torchvision pytorch-cuda=12.1
```

For Windows:
```
mamba activate wf-merfish
mamba install -c conda-forge -c nvidia -c pytorch cupy cuda-version=12.1 cudnn cutensor nccl onnx onnxruntime pytorch torchvision pytorch-cuda=12.1
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
          - <world_zyx_um> (rigid registration for world coordinate alignment of all polyDT tiles in first round)
        - <raw_data>
        - <registered_data> (note this is the same as the raw data for the first round)
      - /round0001.zarr
        - .zattrs
          - <stage_zyx_um>
          - <wavelengths> (excitation, emission)
          - <bit_linker>
          - <rigid_xyz_um> (rigid registration for round 0 alignment)
        - <camera_data>
        - <raw_data>
        - <of_xyz_4x_downsample> (optical flow field)
        - <registered_decon_data> (deformable registration applied after rigid for round 0 alignment)
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
      

