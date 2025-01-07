# Zhuang laboratory mouse brain example

## Overview

The goal of this example is to run `merfish3d-analysis` on an existing 2D MERFISH dataset generated by the [Zhuang laboratory]() at Harvard. They graciously deposisted their nearly raw data on BIL and we can adapt the fully featured functionality of our package to re-process their data. This is a ???-gene MERFISH experiment that has been pre-registered across rounds within each tile.

## Preliminaries

You need to make sure you have a working python environment with `merfish3d-analysis` properly installed, `Baysor` properly installed, and the Zhuang laboratory mouse brain 2D MERFISH dataset downloaded. The dataset is ~??? GB and you will need roughly another ??? GB of temporary space to create the `qi2labDataStore` structure we use to perform tile registration, global registration, segmentation, pixel decoding, filtering, and cell assignment. 

We are only going to analyze one of their mouse brain slices, specifically `mouse1_sample1_raw`.

- [merfish3d-analysis](https://www.github.com/qi2lab/merfish3d-analysis)
- [Baysor](https://github.com/kharchenkolab/Baysor)
- [Raw 2D MERFISH data](https://download.brainimagelibrary.org/cf/1c/cf1c1a431ef8d021/)
    - Download the "additional_files" folder, "mouse1_sample1_raw" folder and "dataset_metadata.xslx".

## Downloading the data

All of the required code to process this data is in the [examples\zhuang_lab](https://github.com/QI2lab/merfish3d-analysis/tree/main/examples/zhuang_lab) folder on `merfish3d-analysis` github repo. After downloading the data from BIL, there should be the following data structure on the disk:

/ 
├── mouse1_sample1_raw/ 
│ ├── data_r0001_tile000_xyz/
│ ├── data_r0001_tile000_xyz.csv
| ...
| ...
│ ├── data_r0009_tile041_xyz/
│ ├── data_r0009_tile041_xyz.csv
│ ├── codebook.csv
│ ├── bit_order.csv
│ ├── hot_pixel_flir.tiff
│ ├── scan_metadata.csv
├── additional_files/ 
├── dataset_metadata.xslx

## Processing non-qi2lab data

Because this data is generated by a different custom microscope design with a unique microscope control package and is already pre-processed, we have to manually write most of the data conversion steps. Please see the [DataStore] page for more information on the key parameters that are requred to create a `qi2labDataStore`.

A key issue with this data is the stage direction and camera orientation are flipped, which can be quite confusing when trying to figure out how the tiles are spatially related to each other.

Another difference in this data is that the spacing between z-planes is 1.5 microns, quite a bit larger than the Nyquist sampling of ~0.3 microns. It does not make sense to perform 3D decoding for this large of a sampling, so each z plane is decoding as independent from the surrounding planes. At the end, we collapse decoded transcripts that show up in adajacent Z planes to the transcript with the largest brightness.

Finally, much of the metadata information we need (refractive index, numerical aperture, wavelengths, camera specifications, etc...) is only available via the publication, [pub](). We have noted in the conversion script where we had to look up these values.

While we have already created the data conversion code for this example, please reach out with questions if the process is not clear. 

## Processing steps

For each of the python files in the `examples\zhuang_lab` directory, you will need to scroll to the bottom and replace the path with the correct path. For example, in `01_convert_to_datastore.py` you'll want to change this section:

```python
if __name__ == "__main__":
    root_path = Path(r"/path/to/download/mouse1_sample1_raw")
    baysor_binary_path = Path(
        r"/path/to/Baysor/bin/baysor/bin/./baysor"
    )
    baysor_options_path = Path(
        r"/path/to/merfish3d-analysis/examples/zhuang_lab/zhuang_mouse.toml"
    )
    julia_threads = 20 # replace with number of threads to use.

    hot_pixel_image_path = None

    convert_data(
        root_path=root_path,
        baysor_binary_path=baysor_binary_path,
        baysor_options_path=baysor_options_path,
        julia_threads=julia_threads,
        hot_pixel_image_path=hot_pixel_image_path
    )
```

For all of the files in `processing_code`, you'll set the `root_path` to `root_path = Path(r"/path/to/download/mouse0001")`. The package automatically places the datastore within that directory.

Once that is done, you can run `01_convert_to_datastore.py` and `02_register_and_deconolve.py` without any interactions. Depending on your computing hardware, you should expect 1-2 hours for `01_convert_to_datastore.py` and a couple days for `02_register_and_deconolve.py`. The second file is compute intensive, because it deconvolves 3 channel z-stacks at each tile over 9 hours, runs [U-FISH](https://github.com/UFISH-Team/U-FISH) for 2 MERFISH (or smFISH) channel at each tile, "fakes" the local registration because the data is already registered, and finally performs global registration.

Once `02_register_and_deconolve.py` is finished, you will need to create the correct cellpose settings. We have found that initially peforming cellpose segmentation on a downsampled and maximum Z projected DAPI and polyDT image is sufficient to seed Baysor for segmentation refinement.

Here, we embed a short screen recoding here on how to optimize cellpose parameters. If the standard "cyto3" model does not work for you data, you may need to retrain the cellpose model and pass that to the code. Please see the API documentation for how to perform that step. Given satisfactory cellpose settings, you fill them in at the bottom of `03_cell_segmentation.py`,

```python
if __name__ == "__main__":
    root_path = Path(r"/path/to/download/raw_data")
    cellpose_parameters = {
        'normalization' : [0.5,95.0],
        'blur_kernel_size' : 2.0,
        'flow_threshold' : 0.4,
        'cellprob_threshold' : 0.0,
        'diameter': 36.57
    }
    run_cellpose(root_path, cellpose_parameters)
```

and then run `03_cell_segmentation.py`. This will only take a few minutes to generate the initial 2D segmentation guess.

Next, you'll run `04_pixeldecode_and_baysor.py` to first optimize the pixel-based decoding parameters on a subset of tiles, then perform pixel-based decoding for all tiles, filter the data to limit false positives, remove overlapping spots in adajacent spatial tiles, and finally re-segment the cell boundaries in 3D using Baysor. This step should again take ~1 day, depending on your hard disk and GPU configuration.

## Ensuring a sucessful run

We have included the proper output of the `merfish3d-analysis` package for this dataset on the BIL servers in the `results` directory. You should be able to compare your obtained results to ours to ensure that there were no computation errors. One common issue we have run into is if the optical flow registration fails due to processor specific instructions (specifically older processors that lack AVX2). Please contact us if you have this issue, we have a workaround during installation that we can provide to you. 