# Install

This project uses a single `uv` environment. Clone the repository in your
location of choice and enter the directory using
```
git clone https://github.com/QI2lab/merfish3d-analysis
cd merfish3d-analysis
``` 

Create and sync the environment with CUDA 12.9 GPU dependencies:
```
uv sync
```

For development tools, include the `dev` group:
```
uv sync --group dev
``` 

All package entry points can then be run through `uv run`, for example
`uv run qi2lab-preprocess /path/to/experiment`.

## (Optional) Installing Baysor

Please follow the [Baysor documentation](https://github.com/kharchenkolab/Baysor?tab=readme-ov-file#installation) to install for Linux. Keep track of the installation directory for use with `merfish3d-analysis`.

# Documentation

To build the documentation, install using `uv sync --group docs`.
Then execute `uv run mkdocs build --clean` or `uv run mkdocs serve`. The
documentation is available in your web browser at `http://127.0.0.1:8000/`.
