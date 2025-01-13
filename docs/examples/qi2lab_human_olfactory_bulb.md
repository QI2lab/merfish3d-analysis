# qi2lab human olfactory bulb example

## Overview

**Important: The data for this example is still processing on the Brain Image Library (BIL) and is not available yet. The example will be updated once it is posted.**

The goal of this example is to retrieve a 3D MERFISH experiment generate by the qi2lab at ASU from the Brain Image Library and run merfish3d-analysis through quantifying transcripts and assigning transcripts to cells. This is a 119-gene MERFISH and 2-gene smFISH experiment on post-mortem human olfactory bulb tissue.

## Preliminaries

You need to make sure you have a working python enviornment with `merfish3d-analysis` properly installed, `Baysor` properly installed, and our human olfactory bulb 3D MERFISH dataset downloaded. The dataset is ~800 GB and you will need roughly another 800 GB of temporary space to create the `qi2labDataStore` structure we use to perform tile registration, global registration, segmentation, pixel decoding, filtering, and cell assignment.

- [merfish3d-analysis](https://www.github.com/qi2lab/merfish3d-analysis)
- [Baysor](https://github.com/kharchenkolab/Baysor)
- Not yet available! ~~[Raw 3D MERFISH data]()~~

## Downloaded data

All of the required code to process this data is in the BIL download. There should be three top-level directories in the downloaded folder, `raw_data` and `results`. The directory structure is as follows:

```bash
/path/to/download/ 
├── raw_data/ 
  ├── data_r0001_tile000_xyz/
  ├── data_r0001_tile000_xyz.csv
  ...
  ...
  ├── data_r0009_tile041_xyz/
  ├── data_r0009_tile041_xyz.csv
  ├── codebook.csv
  ├── bit_order.csv
  ├── hot_pixel_flir.tiff
  └── scan_metadata.csv
├── results/
  ├── transcripts.parquet
  ├── cell_outlines.zip
  |── mtx/
    ├── barcodes.tsv.gzip
    ├── features.tsv.gzip
    └── matrix.mtx.gzip
```

## Processing steps

For each of the python files in the `processing_code` directory, you will need to scroll to the bottom and replace the path with the correct path. For example, in `01_convert_to_datastore.py` you'll want to change this section:

```python
if __name__ == "__main__":
    root_path = Path(r"/path/to/download/raw_data")
    baysor_binary_path = Path(
        r"/path/to/Baysor/bin/baysor/bin/./baysor"
    )
    baysor_options_path = Path(
        r"/path/to/merfish3d-analysis/examples/human_olfactorybulb/qi2lab_humanOB.toml"
    )
    julia_threads = 20 # replace with number of threads to use.

    hot_pixel_image_path = Path(r"/path/to/download/raw_data/flir_hot_pixel_image.tif")

    convert_data(
        root_path=root_path,
        baysor_binary_path=baysor_binary_path,
        baysor_options_path=baysor_options_path,
        julia_threads=julia_threads,
        hot_pixel_image_path=hot_pixel_image_path
    )
```

For all of the files in `processing_code`, you'll set the `root_path` to `root_path = Path(r"/path/to/download/raw_data")`. The package automatically places the datastore within that directory.

Once that is done, you can run `01_convert_to_datastore.py` and `02_register_and_deconvolve.py` without any interactions. Depending on your computing hardware, you should expect 1-2 hours for `01_convert_to_datastore.py` and a couple days for `02_register_and_deconvolve.py`. The second file is compute intensive, because it deconvolves 3 channel z-stacks at each tile over 9 hours, runs [U-FISH](https://github.com/UFISH-Team/U-FISH) for 2 MERFISH (or smFISH) channel at each tile, performs a rigid, affine, and optical flow field registration across imaging rounds, and globally registers the fidicual channel.

Once `02_register_and_deconvolve.py` is finished, you will need to create the correct cellpose settings. We have found that initially peforming cellpose segmentation on a downsampled and maximum Z projected polyDT image is sufficient to seed Baysor for segmentation refinement.

 If the standard "cyto3" model does not work for you data, you may need to retrain the cellpose model according the cellpose documentation and pass that to the code. Please see the API documentation for how to perform that step. Given satisfactory cellpose settings, you fill them in at the bottom of `03_cell_segmentation.py`,

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

For this dataset, we labeled a number of olfactory receptors. We exclude these genes during Baysor segmentation, because they are not highly informative on cell boundaries. Once Baysor has finished, the filtered transcripts are parsed again to assign the excluded olfactory receptors to the new 3D cell boundaries.

## Ensuring a sucessful run

In progress...