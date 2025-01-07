# qi2lab DataStore overview

## Philosophy

To help efficiently handle the data complexity of 3D MERFISH experiments, we have created a dedicated [zarr]() based datastore, using [Tensorstore]() for efficient reading and writing of image data and [Parquet]() for tabular data. 

## Important considerations



## DataStore structure

```text
/experiment 
├── raw_data/ 
│ ├── <data> (raw experimental data and metadata)
|
├── qi2labdatastore/ 
│ ├── datastore.json (information on the state of the datastore)
│ ├── calibrations.zarr/ (calibration information)
| | ├── .zattrs
| | | ├── <exp_codebook>
| | | ├── <exp_order>
| | | ├── <exp_codebook>
| | ├── camera_noise_map/ (camera noise map array)
| | ├── psf_data/ (psf arrays)
|
│ ├── polyDT/ (raw and processed data for polyDT label)
| | ├── tile0000/
| | | ├── round0000.zarr/
| | | | ├── .zattrs
| | | | | ├── <stage_zyx_um> (global stage position in zyx order; unit: microns)
| | | | | ├── <wavelengths_um> (wavelength in (excitation,emission) order; unit: microns)
| | | | | ├── <voxel_size_zyx_um> (voxel size in zyx order; unit: microns)
| | | | | ├── <bit_linker> (what codebook bits are linked to this fidicual image)
| | | | | ├── <affine_zyx_um> (4x4 affine matrix generated during global registration; unit: microns)
| | | | | ├── <origin_zyx_um> (tile origin generated during global registration; unit: microns)
| | | | | ├── <spacing_zyx_um> (voxel size used during global registration, this must match <voxel_size_zyx_um>; unit: microns)
| | | | ├── camera_data/ (gain and offset corrected data in zyx order)
| | | | ├── corrected_data/ (gain and offset corrected data in zyx order)
| | | | ├── registered_decon_data/ (deconvolved data in zyx order)
| | | ├── round0001.zarr
| | | | ├── .zattrs
| | | | | ├── <stage_zyx_um> (global stage position in zyx order; unit: microns)
| | | | | ├── <wavelengths_um> (wavelength in (excitation,emission) order; unit: microns)
| | | | | ├── <voxel_size_zyx_um> (voxel size in zyx order; unit: microns)
| | | | | ├── <bit_linker> (what codebook bits are linked to this fidicual image)
| | | | | ├── <affine_zyx_um> (4x4 affine matrix generated during global registration; unit: microns)
| | | | | ├── <origin_zyx_um> (tile origin generated during global registration; unit: microns)
| | | | | ├── <spacing_zyx_um> (voxel size used during global registration, this must match <voxel_size_zyx_um>; unit: microns)
| | | | ├── camera_data/ (gain and offset corrected data in zyx order)
| | | | ├── corrected_data/ (gain and offset corrected data in zyx order)
| | | | ├── of_xyz_3x_downsample/ (3x downsampled optical flow field for round 0 alginment in pixels)
| | | | ├── /registered_decon_data/ (deconvolved, registered back to round 0 image data in zyx order)
| | | | ├── ... 
| | | | ├── roundNNNN.zarr/
| | ├── tile0001/
| | ├── tile0002/
| | ├── ...
| | ├── tileNNNN/
|
│ ├── readouts/ (raw and processed data for MERFISH bits)
```





## DataStore API

Nearly all parameters are accessible as class properties and all data has helper functions for reading and writing. The full API reference is available at [datastore](reference/classes/qi2labDataStore.md).