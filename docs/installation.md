# Install

## Python environment creation

Because `merfish3d-analysis` relies on a number of GPU-only functions, we *strongly* recommend that you create a specific python environment to ensure proper functioning.

Here are instructions using Mamba, a very fast implementation of `conda`.

Create a python 3.10 environment using your favorite package manager, e.g.
```mamba create -n merfish3d python=3.10```

## Installing merfish3d-analysis

Activate the environment and install the GPU dependencies. This install method assumes an Nvidia GPU capable of running CUDA >= 12.0.

```bash
mamba activate merfish3d
mamba install -c conda-forge -c nvidia -c pytorch -c rapidsai cupy cucim=24.08 pylibraft=24.08 raft-dask=24.08 cudadecon "cuda-version>=12.0,<=12.5" cudnn cutensor nccl onnx onnxruntime pytorch torchvision 'pytorch=*=*cuda*'
```

Finally, clone the repository using ```git clone https://github.com/QI2lab/merfish3d-analysis``` and install using `pip install .`. For interactive editing use `pip install -e .`.

To build the documentation, install using `pip install .[docs]`. Then execute `mkdocs build --clean` and `mkdocs serve`. The documentation is available in your web browser at `http://127.0.0.1:8000/`.

## Installing Baysor

Please follow the [Baysor documentation](https://github.com/kharchenkolab/Baysor?tab=readme-ov-file#installation) to install for Linux. Keep track of the installation directory for use with `merfish3d-analysis`.