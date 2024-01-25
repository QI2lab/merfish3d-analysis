# wf-merfish

[![License](https://img.shields.io/pypi/l/wf-merfish.svg?color=green)](https://github.com/dpshepherd/wf-merfish/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/wf-merfish.svg?color=green)](https://pypi.org/project/wf-merfish)
[![Python Version](https://img.shields.io/pypi/pyversions/wf-merfish.svg?color=green)](https://python.org)
[![CI](https://github.com/dpshepherd/wf-merfish/actions/workflows/ci.yml/badge.svg)](https://github.com/dpshepherd/wf-merfish/actions/workflows/ci.yml)

Acquisition and post-processing for qi2lab widefield MERFISH


qi2lab MERFISH zarr layout
- /project_root
  - /calibrations.zarr
    - .zattrs contains codebook, bit order, and other key metadata
    - <noise_map>
    - <psf_data>
  - /polyDT
    - /tile000
      - /round0000.zarr
        - .zattrs contains recorded XYZ stage position, wavelength, bit linker info
        - <raw_data>
        - <registered_data> (note this is the same as the raw data for the first round)
      - /round0001.zarr
        - .zattrs contains
        - <raw_data>
        - <optical_flow_4x_downsample>
        - <registered_data> (warped back to round 0 coordinate system)
      - /roundNNNN.zarr
    - /tile001
    - ...
    - /tileNNN
  - /readouts
    - /tile0000
      - /bit00.zarr
        - .zattrs contains recorded wavelength and round linker info
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
      - local_decoding_results.parquet
      - world_coord_decoding_results.parquet
    - /tile0001
    - ...
    - /tileNNNN
      
      
      
