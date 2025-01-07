# Install

## Python environment creation

Because `merfish3d-analysis` relies on a number of GPU-only functions, we *strongly* recommend that you create a specific python environment to ensure proper functioning.

Here are instructions using Mamba, a very fast implementation of `conda`.

```python
mamba create -n merfish3d python=3.10

mamba install -c conda-forge -c nvidia -c pytorch -c rapidsai cupy cucim=24.08 pylibraft=24.08 raft-dask=24.08 cudadecon "cuda-version>=12.0,<=12.5" cudnn cutensor nccl onnx onnxruntime pytorch torchvision 'pytorch=*=*cuda*'

mamba activate merfish3d
```

## Installing merfish3d-analysis

`merfish3d-analysis` can be installed using pip. Make sure you are in your specific environment first!

```python
mamba activate merfish3d
pip install merfish3d-analysis
```