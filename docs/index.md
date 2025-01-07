# Welcome to merfish3d-analysis Documentation

merfish3d-analysis provides GPU-accelerated post-processing for MERFISH data.

## Motivation

Iterative multiplexing experiments, such as MERFISH, typically involve 6D data. These dimensions are `[rounds,tile,channel,z,y,x]` and require significant processing across each dimension to go from raw data to quality controlled transcript 3D localizations and 3D cell outlines.

Additionally, our laboratory, the Quantiative Imaging and Inference Laboratory (qi2lab), specializes in high-throughput 3D microscopy using custom microscopes. This includes purpose built high numerical aperture widefield and oblique plane microscopy platforms. While increased sampling provides more information on the sample, it introduces new challenges due to the increase in data density and more complicated MERFISH decoding inverse problem.

To efficiently perform 3D MERFISH processing, we created this `merfish3d-analysis` package. The goal of the package is to aid researchers in rapidly and robustly turning gigabyte to petabyte level MERFISH data into decoded transcripts using chunked, compressed file formats and GPU-accelerated processing.

## Features

- Datastore optimized for large-scale imaging data.
- Processing capabilities for widefield, standard, and skewed lightsheet data.
- Rigid, affine, and deformable local tile registration.
- Rigid and affine global registration.
- GPU-accelerated image processing and decoding.
- Integrated functionality to leaverag machine learning tools such as Cellpose, Baysor, and U-FISH.

## API reference

For more information, check out the [API Reference](reference/index.md).