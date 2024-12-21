# merfish3d-analysis

_WARNING: alpha software._ We are sharing this early in case it is useful to other groups. Please expect breaking changes. Examples of running the package are found in `/Examples`.

GPU accelerated post-processing for 3D MERFISH data generated using qi2lab microscopes. This package currently **Linux only** due to RAPIDS.AI package availabilty.

## Installation

Create a python 3.10 environment using your favorite package manager, e.g.
```mamba create -n wf-merfish python=3.10```

Activate the environment and install the GPU dependencies.

```
mamba activate wf-merfish
mamba install -c conda-forge -c nvidia -c pytorch -c rapidsai cupy cucim pylibraft cudadecon cuda-version=12.5 cudnn cutensor nccl onnx onnxruntime pytorch torchvision 'pytorch=*=*cuda*'
```

Finally, repository using ```git clone https://github.com/QI2lab/wf-merfish``` and then install using `pip install .` or for interactive editing use `pip install -e .`.

## qi2lab MERFISH zarr layout v0.3
<details>
<summary>file format</summary>
  
- /project_root
  - datastore.json (information on state of datastore and paths to other programs (Baysor, Cellpose, Stardist))
  - /calibrations.zarr (calibration information)
    - .zattrs
      - <exp_codebook> (with blank codes)
      - <exp_order> (e.g. round 0 -> codebook bits 0,1)
      - <metadata> (objective NA, channels used, etc...)
    - <camera_noise_map> (used for hotpixel correction)
    - <psf_data> (calculated based on experiment metadata)
  - /polyDT (raw and processed image data for polyDT label)
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
        - <registered_decon_data> (deconvolved data)
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
  - /readouts (raw and processed image data for readout bits)
    - /tile0000
      - /bit00.zarr
        - .zattrs
          - <wavelengths_um> (excitation, emission)
          - <voxel_size_zyx_um> (z,y,x in microns)
          - <round_linker> (what polyDT round is bit corresponds to)
        - <camera_data> (optional: raw camera data)
        - <corrected_data> (hot pixel, offset, and gain corrected data)
        - <registered_corrected_data> (corrected, then rigid and defromable registration applied to warp back to round 0)
        - <registered_decon_data> (deconvolved, then rigid and deformable registration applied to warp back to round 0)
        - <registered_ufish_data> (ufish prediction applied to registered_decon_data)
      - /bit01.zarr
      - ...
      - /bitNN.zarr
  - /ufish_localizations (ufish spot predictions, can be useful for diagnostics)
    - /tile0000
      - bit01.parquet
      - bitNN.parquet
    - /tile0001
    - ....
    - /tileNNNN
  - /decoded (decoding results)
    - tile0000_decoded_features.parquet
    - ...
    - tileNNNN_decoded_features.parquet
    - all_tiles_filtered_decoded_features.parquet
  - /decoded_optimization (temporary files used during iterative optimizing for normalization factors)
    - tileNNNN_decoded_features.parquet
  - /fused (fused, down-sampled polyDT image)
    - /fused.zarr
      - Either one of:
        - <fused_iso_all_zyx> (polyDT and readouts deconvolved, registered data fused at isotropic voxel spacing)
        - <fused_iso_polyDT_zyx> (polyDT deconvolved, registered data fused at isotropic voxel spacing)
  - /segmentation (segmentation and spots-to-cell assignment results)
    - /cellpose
      - cell_centroids.parquet (yx centroid for each cell in maximum Z projection of cellpose mask prediction)
      - cell_outlines.json (yx polygons for each cell in maximum Z projection of cellpose mask prediction)
      - /cellpose.zarr
        - <masks_iso_zyx> (cellpose masks generated from maximum Z projection data in fused.zarr)
    - /baysor
      - baysor_filtered_genes.parquet (gene ID, global zyx position, cell assignment, and confidence for all tiles)
      - segmentation_polygons.json (yx polygons for each cell determined by Baysor. GeoJSON format.)
      - diagnostic outputs (see Baysor repository for explanations)
  - /mtx_output (mtx formatted output of genes assigned to cells)

</details>
      

