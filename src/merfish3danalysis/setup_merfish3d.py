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

# Base pip deps (pure-Python)
BASE_PIP_DEPS = [
    "numpy",
    "numba",
    "llvmlite",
    "tbb",
    "tqdm",
    "ryomen",
    "tensorstore",
    "cmap",
    "napari[all]",
    "zarr>=3.0.8",
    "psfmodels",
    "tifffile>=2025.6.1",
    "nvidia-cuda-runtime-cu12==12.8.*",
    "onnx",
    "onnxruntime-gpu",
    "torch",
    "torchvision",
    "napari[pyqt6]",
    "napari-ome-zarr",
    "cellpose[gui,distributed]",
    "ufish @ git+https://github.com/QI2lab/U-FISH.git@main",
    #"multiview-stitcher @ git+https://github.com/multiview-stitcher/multiview-stitcher@main",
    "deeds @ git+https://github.com/QI2lab/deeds-registration",
    "basicpy @ git+https://github.com/QI2lab/BaSiCPy.git@main",
    "ome-zarr @ git+https://github.com/ome/ome-zarr-py.git@refs/pull/404/head",
    "opm-processing-v2 @ git+https://github.com/QI2lab/opm-processing-v2",
    "numcodecs",
    "psfmodels",
    "cmap",
    "SimpleITK", 
    "ndstorage",
    "roifile",
    "pyarrow",
    "tbb",
    "shapely",
    "scikit-image",
    "imbalanced-learn", 
    "scikit-learn", 
    "rtree",
]

# CUDA conda pkgs
LINUX_CONDA_CUDA_PKGS = [
    "'cuda-version=12.8'",
    "'cuda-toolkit=12.8'",
    "cuda-cudart",
    "'cucim=25.06'",
    "'cuvs=25.06'"
    "cupy",
    "scikit-image",
    "cudnn",
    "cutensor",
    "nccl"
]

WINDOWS_CONDA_CUDA_PKGS = [
    "'cuda-version=12.8'",
    "'cuda-toolkit=12.8'",
    "cuda-cudart",
    "cupy",
    "scikit-image",
    "cudnn",
    "cutensor",
    "cuda-nvcc",
]

LINUX_JAX_LIB = {
    "jax[cuda12_local]==0.4.38"
}

# Extra cucim Git URL for Windows
WINDOWS_CUCIM_GIT = (
    "git+https://github.com/rapidsai/cucim.git@v25.06.00#egg=cucim&subdirectory=python/cucim"
)

def run(command: str):
    typer.echo(f"$ {command}")
    subprocess.run(command, shell=True, check=True)

@app.command()
def setup_cuda():
    """
    1) Installs CUDA packages via conda (RAPIDS.ai channel), using OS-specific lists.
    2) Writes a single activation hook (either .sh on Linux or .bat on Windows).
    3) Installs all pip deps (on Windows including the extra cucim Git URL) in one call.
    """
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        typer.echo("Error: activate your conda environment first.", err=True)
        raise typer.Exit(1)

    installer = shutil.which("mamba") or shutil.which("conda")
    if not installer:
        typer.echo("Error: neither mamba nor conda found.", err=True)
        raise typer.Exit(1)

    is_windows = platform.system() == "Windows"
    if is_windows:
        run(f"{installer} install -y -c rapidsai -c conda-forge -c nvidia {' '.join(WINDOWS_CONDA_CUDA_PKGS)}")
    else:
        run(f"{installer} install -y -c rapidsai -c conda-forge -c nvidia {' '.join(LINUX_CONDA_CUDA_PKGS)}")

    # Clear existing hooks
    activate_dir = Path(prefix) / "etc" / "conda" / "activate.d"
    activate_dir.mkdir(parents=True, exist_ok=True)
    for script in activate_dir.glob("*"):
        try:
            script.unlink()
        except OSError:
            pass

    if is_windows:
# 1) Write the activation hook + which.bat shim
        bat_hook = activate_dir / "cuda_override.bat"
        content = (
            f"@echo off\n"
            f"REM Point at the conda-installed CUDA toolkit\n"
            f"set \"CUDA_PATH={prefix}\\Library\\bin\"\n"
            f"set \"CUDA_HOME=%CUDA_PATH%\"\n"
            f"REM Prepend CUDA bin and lib to PATH\n"
            f"set \"PATH=%CUDA_PATH%;{prefix}\\Library\\lib;%PATH%\"\n"
            f"REM NVRTC must compile with C++17 and ignore deprecated dialect\n"
            f"set \"NVRTC_OPTIONS=--std=c++17\"\n"
            f"set \"CCCL_IGNORE_DEPRECATED_CPP_DIALECT=1\"\n"
            f"REM which.bat shim for rapids-build-backend\n"
            f"echo @echo off > \"{prefix}\\Scripts\\which.bat\"\n"
            f"echo where %%* >> \"{prefix}\\Scripts\\which.bat\"\n"
        )
        bat_hook.write_text(content, encoding="utf-8")

        # 2) Apply the same env changes right now
        os.environ["CUDA_PATH"] = f"{prefix}\\Library\\bin"
        os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"]
        os.environ["PATH"] = (
            os.environ["CUDA_PATH"] + ";" +
            f"{prefix}\\Library\\lib;" +
            os.environ.get("PATH", "")
        )
        os.environ["NVRTC_OPTIONS"] = "--std=c++17"
        os.environ["CCCL_IGNORE_DEPRECATED_CPP_DIALECT"] = "1"

        system_where = Path(os.environ["WINDIR"]) / "System32" / "where.exe"
        dest_which = Path(prefix) / "Scripts" / "which.exe"

        # only copy if it doesn't already exist
        if not dest_which.exists():
            shutil.copy(system_where, dest_which)

        # Ensure Scripts/ is at the front of PATH for the current process
        scripts_dir = str(Path(prefix) / "Scripts")
        os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH","")
    else:
        # Linux shell hook only
        sh_hook = activate_dir / "cuda_override.sh"
        env_lib = f"{prefix}/lib"
        linux_cuda_root = f"{prefix}/targets/x86_64-linux"
        sh_hook.write_text(f"""#!/usr/bin/env sh
# Point at the conda-installed CUDA toolkit
export CUDA_PATH="{linux_cuda_root}"
export CUDA_HOME="$CUDA_PATH"
export PATH="$CUDA_PATH/bin:$PATH"

# Prepend only the conda toolkit lib & env lib
export LD_LIBRARY_PATH="$CUDA_PATH/lib:{env_lib}${{LD_LIBRARY_PATH:+:${{LD_LIBRARY_PATH}}}}"

# NVRTC must compile with C++17 and ignore deprecated dialect
export NVRTC_OPTIONS="--std=c++17"
export CCCL_IGNORE_DEPRECATED_CPP_DIALECT="1"
""")
        sh_hook.chmod(sh_hook.stat().st_mode | stat.S_IEXEC)

    # Single pip install for all deps
    if is_windows:
        
        subprocess.run(
            [sys.executable, "-m", "pip", "install", *BASE_PIP_DEPS],
            check=True
        )
        
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", WINDOWS_CUCIM_GIT],
                check=True
            )
        except Exception:
            pass
        
        try:
            run("python -m cupyx.tools.install_library --cuda 12.x --library nccl")
        except Exception:
            pass
        
    else:
        pip_deps = BASE_PIP_DEPS.copy()
        deps_str = " ".join(shlex.quote(d) for d in pip_deps)
        run(f"pip install {deps_str}")
        
        linux_dep_str = " ".join(shlex.quote(d) for d in LINUX_JAX_LIB)
        run(f"pip install {linux_dep_str}")
    if is_windows:
        typer.echo(f"\nsetup complete!  Please 'conda deactivate' then 'conda activate env_name' to apply changes.")
    else:
        typer.echo(f"\nsetup complete!  Please 'conda deactivate' then 'conda activate {env_lib}' to apply changes.")


def main():
    app()


if __name__ == "__main__":
    main()