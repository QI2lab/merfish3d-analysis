Introduction
============

Welcome to the **merfish3d-analysis** project!

This project is designed for GPU accelerated post-processing for 2D or 3D MERFISH data. This package currently Linux only due to RAPIDS.AI package availabilty.

Key Features:
- Datastore optimized for handling large-scale experimental MERFISH data. 
- Capable of processing widefield, standard lightsheet, and skeweed lightsheet (lattice or OPM) data.
- Chunked GPU accelerated image processing, local and global registration, and MERFISH decoding.
- Designed to utilize existing community segmentation tools. 

Quick Start:
-------------
To get started, create a Python 3.10 environment using your favorite package manager, e.g. `mamba create -n merfish3d python=3.10`.
Activate the environment and install the GPU dependencies.

```bash
mamba activate merfish3d
mamba install -c conda-forge -c nvidia -c pytorch -c rapidsai cupy cucim=24.08 pylibraft=24.08 raft-dask=24.08 cudadecon "cuda-version>=12.0,<=12.5" cudnn cutensor nccl onnx onnxruntime pytorch torchvision 'pytorch=*=*cuda*'
```

Then clone the repository and install it.

```bash
git clone http://www.github.com/qi2lab/merfish3d-analysis
cd merfish3d-analysis
pip install .
```

Finally, try the examples provided in the `examples` directory.