# Welcome to merfish3d-analysis Documentation

merfish3d-analysis provides GPU-accelerated post-processing for MERFISH data.

## Motivation

Iterative multiplexing experiments, such as MERFISH, typically involve 6D data. These dimensions are `[rounds,tile,channel,z,y,x]` and require significant processing across each dimension to go from raw data to quality controlled transcript 3D localizations and 3D cell outlines.

Additionally, our laboratory, the Quantiative Imaging and Inference Laboratory (qi2lab), specializes in high-throughput 3D microscopy using custom microscopes. This includes purpose built high numerical aperture widefield and oblique plane microscopy platforms. While increased sampling provides more information on the sample, it introduces new challenges due to the increase in data density and more complicated MERFISH decoding inverse problem.

To efficiently perform 3D MERFISH processing, we created this `merfish3d-analysis` package. The goal of the package is to aid researchers in rapidly and robustly turning gigabyte to petabyte level MERFISH data into decoded transcripts using chunked, compressed file formats and GPU-accelerated processing.

## Features

- Decode both 2D and 3D iterative barcoded experiments that use a codebook. Our focus on 3D MERFISH, but this library can be extended to any iterative imaging and barcoded RNA imaging approach.
- Datastore optimized for large-scale imaging data.
  - Read/Write compressed Zarr v2 using Tensorstore library for performance.
- Processing capabilities for widefield, standard light-sheet, and skewed light-sheet data.
- Rigid, affine, and deformable local tile registration.
  - GPU-accelerated registration estimation combined with ITK for image warping.
- Rigid and affine global registration using [multiview-stitcher](https://multiview-stitcher.github.io/multiview-stitcher/main/)
- GPU-accelerated image processing and decoding.
  - Nearly all image processing functions utilize GPU acceleration through [CuPy](https://cupy.dev/), [CuCIM](https://docs.rapids.ai/api/cucim/stable/), [CuVS](https://docs.rapids.ai/api/cuvs/nightly/), and custom CUDA kernels. All non-GPU accelerated functions are [Numba](https://numba.pydata.org/) accelerated.
  - Larger-than-GPU-memory block computations are handled using [Ryomen](https://ryomen.readthedocs.io/en/latest/), a lightweight solution that avoids many issues with other distribution computing solutions.
- Iterative estimation of background and normalization vectors across codebook bits to remove subjective normalization by user that often leads to non-optimal decoding solutions.
- Integrated functionality to leverage machine learning tools such as [Cellpose](https://cellpose.readthedocs.io/en/latest/), [Baysor](https://kharchenkolab.github.io/Baysor/dev/), and [U-FISH](https://github.com/UFISH-Team/U-FISH).

## Examples

Multiple examples are provided with the library, including [qi2lab data](examples/qi2lab_human_olfactory_bulb.md), [Zhuang laboratory data](examples/zhuang_lab_mouse_brain.md), and [synthetic data](examples/statphysbio_synthetic.md).

## API reference

For more information, check out the [API Reference](reference/index.md).