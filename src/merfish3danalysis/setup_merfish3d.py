import os
import shlex
import stat
import subprocess
import shutil
import platform
from pathlib import Path
import typer
import sys

app = typer.Typer()
app.pretty_exceptions_enable = False

# Base pip deps for the CURRENT env (optional, keep if you want the current env ready too)
BASE_PIP_DEPS = [
    "numpy==1.26.4",
    "numba",
    "llvmlite",
    "tbb",
    "tqdm",
    "ryomen",
    "tensorstore",
    "nvidia-cuda-runtime-cu12==12.8.*",
    "onnx",
    "onnxruntime-gpu",
    "napari[pyqt6]",
    "cellpose[gui]",
    "ufish @ git+https://github.com/QI2lab/U-FISH.git@main",
    "warpfield @ git+https://github.com/QI2lab/warpfield.git@qi2lab-working",
    "basicpy @ git+https://github.com/QI2lab/BaSiCPy.git@main",
    "zarr>3.0.8",
    "tifffile",
    "numcodecs",
    "cmap",
    "psfmodels",
    "SimpleITK",
    "ndstorage",
    "roifile",
    "tbb",
    "shapely",
    "imbalanced-learn",
    "scikit-learn",
    "rtree",
    "anndata",
    "fastparquet"
]

# CUDA conda pkgs (Linux)
LINUX_CONDA_CUDA_PKGS = [
    "'cuda-version=12.8'",
    "'cuda-toolkit=12.8'",
    "cuda-cudart",
    "cucim",
    "cuvs",
    "cupy",
    "scikit-image",
    "cudnn",
    "cutensor",
    "nccl",
    "pyopengl",
    "pyimagej",
]

# Optional: jax local CUDA in CURRENT env
LINUX_JAX_LIB = [
    "jax[cuda12_local]==0.4.38",
]

# multiview-stitcher env settings
MVSTITCHER_ENV_NAME = "merfish3d_stitcher"
MVSTITCHER_ENV_PY = "3.12"

# These ensure all of your listed imports work in the multiview-stitcher env
MVSTITCHER_ENV_PIP_IMPORTS = [
    "pandas",
    "roifile",
    "shapely",
]

def run(command: str, *, cwd: Path | None = None):
    typer.echo(f"$ {command}")
    subprocess.run(command, shell=True, check=True, cwd=str(cwd) if cwd else None)

def _which(names: list[str]) -> str | None:
    for n in names:
        if shutil.which(n):
            return n
    return None

@app.command()
def setup_cuda():
    """
    Linux-only setup:

    1) Install CUDA packages via conda/mamba (rapidsai + conda-forge + nvidia).
    2) Write a Linux activation hook exporting CUDA_PATH, etc.
    3) (Optional) Prepare CURRENT env: torch/vision cu128, BASE_PIP_DEPS, jax local CUDA.
    4) Create NEW env 'merfish3d_stitcher' (Python 3.12) and pip-install:
         - core deps to satisfy your import list
         - your repo (-e .)
         - multiview-stitcher + ngff-zarr[tensorstore]>=0.16.0
       Then smoke-test imports inside the new env.
    """
    if platform.system() != "Linux":
        typer.echo("Error: Linux only.", err=True)
        raise typer.Exit(1)

    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        typer.echo("Error: activate your conda environment first.", err=True)
        raise typer.Exit(1)

    installer = _which(["mamba", "conda"])
    if not installer:
        typer.echo("Error: neither mamba nor conda found.", err=True)
        raise typer.Exit(1)

    # 1) Core CUDA stack from conda
    channels = "-c rapidsai -c conda-forge -c nvidia"
    run(f"{installer} install -y {channels} {' '.join(LINUX_CONDA_CUDA_PKGS)}")

    # 2) Linux activation hook
    activate_dir = Path(prefix) / "etc" / "conda" / "activate.d"
    activate_dir.mkdir(parents=True, exist_ok=True)
    for script in activate_dir.glob("*"):
        try:
            script.unlink()
        except OSError:
            pass

    sh_hook = activate_dir / "cuda_override.sh"
    env_lib = f"{prefix}/lib"
    linux_cuda_root = f"{prefix}/targets/x86_64-linux"
    sh_hook.write_text(
        f"""#!/usr/bin/env sh
# Point at the conda-installed CUDA toolkit
export CUDA_PATH="{linux_cuda_root}"
export CUDA_HOME="$CUDA_PATH"
export PATH="$CUDA_PATH/bin:$PATH"

# Prepend only the conda toolkit lib & env lib
export LD_LIBRARY_PATH="$CUDA_PATH/lib:{env_lib}${{LD_LIBRARY_PATH:+:${{LD_LIBRARY_PATH}}}}"

# NVRTC must compile with C++17 and ignore deprecated dialect
export NVRTC_OPTIONS="--std=c++17"
export CCCL_IGNORE_DEPRECATED_CPP_DIALECT="1"
""",
        encoding="utf-8",
    )
    sh_hook.chmod(sh_hook.stat().st_mode | stat.S_IEXEC)

    # 3) (Optional) Prep CURRENT env
    run("python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
    run(f"python -m pip install {' '.join(shlex.quote(d) for d in BASE_PIP_DEPS)}")
    run(f"python -m pip install {' '.join(shlex.quote(d) for d in LINUX_JAX_LIB)}")

    # 4) Create NEW env and install what you asked
    run(f"{installer} create -y -n {MVSTITCHER_ENV_NAME} python={MVSTITCHER_ENV_PY} pip")
    repo_dir = Path.cwd()

    # Upgrade build tooling, then install the *imports* deps first
    run(f"{installer} run -n {MVSTITCHER_ENV_NAME} python -m pip install -U pip setuptools wheel")

    # Install multiview-stitcher and minimial deps to use merfish3d-analysis datastore class
    run(
        f"""{installer} run -n {MVSTITCHER_ENV_NAME} python -m pip install \
"multiview-stitcher @ git+https://github.com/multiview-stitcher/multiview-stitcher@main" \
"ngff-zarr[tensorstore]>=0.16.0" """
    )
    run(
        f"{installer} run -n {MVSTITCHER_ENV_NAME} python -m pip install "
        + " ".join(shlex.quote(d) for d in MVSTITCHER_ENV_PIP_IMPORTS)
    )
    run(f"{installer} run -n {MVSTITCHER_ENV_NAME} python -m pip install -e .", cwd=repo_dir)


    typer.echo("\nSetup complete.\n")

def main():
    app()

if __name__ == "__main__":
    main()
