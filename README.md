# merfish3d-analysis

_WARNING: alpha software._ We are sharing this early in case it is useful to other groups. Please expect breaking changes. Examples of running the package are found in `/Examples`.

GPU accelerated post-processing for 2D / 3D iterative barcoded FISH data. This package currently **Nvidia only** and **Linux only** due to RAPIDS.AI package availabilty. Documentation is available at [https://qi2lab.github.io/merfish3d-analysis/](https://qi2lab.github.io/merfish3d-analysis/).

## Installation

Create a python 3.12 environment using your favorite package manager, e.g.
```
conda create -n merfish3d python=3.12
```

Activate the environment and install the GPU dependencies. This install method assumes an Nvidia GPU capable of running CUDA 12.8.
```
conda activate merfish3d
```

Next, clone the repository in your location of choice and enter the directory using
```
git clone https://github.com/QI2lab/merfish3d-analysis
cd merfish3d-analysis
``` 

and install using 
```
pip install .
```

For interactive editing use 
```
pip install -e .
``` 

Finally, install the `merfish3d-analysis` package using the command 
```
setup-merfish3d
```` 

This will automatically setup the correct CUDA libraries and other packages in the conda environmnent. **Note**: Due to package incompatability, the install script currently creates a second conda/mamba environment called `merfish3d-stitcher`. In this environment, we install the minimal packages requried to read the datastore used by `merfish3d-analysis` and [multiview-stitcher](https://github.com/multiview-stitcher/multiview-stitcher). The reason for this change is that one of the `multiview-stitcher` sub-dependencies (`xarray-dataclass`) now requires `numpy>2.0`, which is incompatible with the scientific computing packages used for `merfish3d-analysis`.

The `merfish3d-stitcher` environment is only used when individual tiles are registered into a global coordinate system. The code automatically invokes this second environment, but it is important to note that the current install strategy does create a new conda/mamba environment beyond what you as the user creates. As soon as the dependency issue is solved, we will remove this work around.

### (Optional) Baysor installation
If you plan on re-segmenting cells using decoded RNA, please follow the [Baysor installation instructions](https://github.com/kharchenkolab/Baysor?tab=readme-ov-file#installation).

## Documentation

To build the documentation, install using `pip install .[docs]`. Then execute `mkdocs build --clean` and `mkdocs serve`. The documentation is available in your web browser at `http://127.0.0.1:8000/`.

