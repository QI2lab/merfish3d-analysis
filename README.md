# wf-merfish

[![License](https://img.shields.io/pypi/l/wf-merfish.svg?color=green)](https://github.com/dpshepherd/wf-merfish/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/wf-merfish.svg?color=green)](https://pypi.org/project/wf-merfish)
[![Python Version](https://img.shields.io/pypi/pyversions/wf-merfish.svg?color=green)](https://python.org)
[![CI](https://github.com/dpshepherd/wf-merfish/actions/workflows/ci.yml/badge.svg)](https://github.com/dpshepherd/wf-merfish/actions/workflows/ci.yml)

Acquisition and post-processing for qi2lab widefield MERFISH


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
        - <raw_data>
        - <of_xyz_4x_downsample> (optical flow field)
        - <registered_data> (deformable registration applied after rigid for round 0 alignment)
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
        - <raw_data>
      - /bit01.zarr
      - ...
      - /bitNN.zarr
  - /localizations
    - /tile0000
      - /bit00
        - localization_parameters.json
        - raw_localization_results.parquet
        - registered_localization_results.parquet
      - /bit01
      - ...
      - /bitNN
    - /tile0001
    - ....
    - /tileNNNN
  - /decoded
    - /tile0000
      - decoding_parameters.json
      - tile_coord_decoding_results.parquet
      - world_coord_decoding_results.parquet
    - /tile0001
    - ...
    - /tileNNNN
  - /stitching
    - /polydT
      - tile0000.ome.zarr
      - tile0001.ome.zarr
      - ....
      - tileNNNN.ome.zarr
  - /fused
    - fused_polyDT.ome.zarr

</details>
      
## Installation

Clone this repository and then install using `pip install .` or for interactive editing use `pip install -e .`.

On MacOS, if you have issues building `deeds`, run:
```bash
brew install llvm libomp
echo 'export PATH="/opt/homebrew/Cellar/llvm/17.0.6_1/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```
