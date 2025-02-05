# merfish3d-analysis

_WARNING: alpha software._ We are sharing this early in case it is useful to other groups. Please expect breaking changes. Examples of running the package are found in `/Examples`.

GPU accelerated post-processing for 2D / 3D iterative barcoded FISH data. This package currently **Nvidia only** and **Linux only** due to RAPIDS.AI package availabilty. Documentation is available at [https://qi2lab.github.io/merfish3d-analysis/](https://qi2lab.github.io/merfish3d-analysis/).

## Installation

Create a python 3.10 environment using your favorite package manager, e.g.
```mamba create -n merfish3d python=3.10```

Activate the environment and install the GPU dependencies. This install method assumes an Nvidia GPU capable of running CUDA >= 12.0.

```
mamba activate merfish3d
mamba install -c conda-forge -c nvidia -c pytorch -c rapidsai cupy cucim=24.08 pylibraft=24.08 raft-dask=24.08 cudadecon "cuda-version>=12.0,<=12.5" cudnn cutensor nccl onnx onnxruntime pytorch torchvision 'pytorch=*=*cuda*'
```

Finally, clone the repository using ```git clone https://github.com/QI2lab/merfish3d-analysis``` and install using `pip install .`. For interactive editing use `pip install -e .`.

### (Optional) Baysor installation
If you plan on re-segmenting cells using decoded RNA, please follow the [Baysor installation instructions](https://github.com/kharchenkolab/Baysor?tab=readme-ov-file#installation).

## Documentation

To build the documentation, install using `pip install .[docs]`. Then execute `mkdocs build --clean` and `mkdocs serve`. The documentation is available in your web browser at `http://127.0.0.1:8000/`.
