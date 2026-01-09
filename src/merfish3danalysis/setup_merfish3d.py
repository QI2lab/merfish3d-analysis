import os
import shlex
import stat
import subprocess
import shutil
import platform
from pathlib import Path
import typer

app = typer.Typer()
app.pretty_exceptions_enable = False

# Base pip deps for the merfish3d env
BASE_PIP_DEPS = [
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
    "tifffile",
    "numcodecs",
    "cmap",
    "psfmodels",
    "SimpleITK",
    "ndstorage",
    "roifile",
    "imbalanced-learn",
    "scikit-learn",
    "anndata",
]

# CUDA conda pkgs (Linux)
LINUX_CONDA_CUDA_PKGS = [
    "'cuda-version=12.8'",
    "'cuda-toolkit=12.8'",
    "cuda-cudart",
    "cuda-nvrtc",
    "cucim",
    "cuvs",
    "cupy",
    "scikit-image",
    "cudnn",
    "cutensor",
    "nccl",
    "pyopengl",
    "pyimagej",
    "openjdk=11",
    "'numpy=1.26.4'",
    "scipy",
    "shapely",
    "rtree",
    "numba",
    "zarr>3.0.8",
    "pandas",
    "fastparquet",
    "tbb",
]

# Optional: jax local CUDA in CURRENT env
LINUX_JAX_LIB = [
    "jax[cuda12_local]==0.4.38",
]

# multiview-stitcher env settings
MVSTITCHER_ENV_NAME = "merfish3d-stitcher"
MVSTITCHER_ENV_PY = "3.12"

# These ensure all of your listed imports work in the multiview-stitcher env
MVSTITCHER_ENV_PIP_IMPORTS = [
    "pandas",
    "roifile",
    "shapely",
    "fastparquet",
]


def run(command: str, *, cwd: Path | None = None):
    typer.echo(f"$ {command}")
    subprocess.run(command, shell=True, check=True, cwd=str(cwd) if cwd else None)


def _find_installer() -> str:
    """
    Pick the package manager executable path, preferring Conda.
    Order:
      1) Explicit override via MERFISH3D_INSTALLER (absolute path or basename)
      2) CONDA_EXE
      3) MAMBA_EXE / MICROMAMBA_EXE
      4) PATH lookups: conda, then mamba, then micromamba
      5) Common locations relative to active env
    """
    # 1) Explicit user override (optional)
    override = os.environ.get("MERFISH3D_INSTALLER")
    if override:
        p = Path(override)
        if p.exists():
            return str(p)
        exe = shutil.which(override)
        if exe:
            return exe
        raise RuntimeError(f"MERFISH3D_INSTALLER={override!r} not found on disk or PATH")

    # 2) Prefer Conda if available
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).exists():
        return conda_exe

    # 3) Fall back to Mamba/MicroMamba env vars
    for var in ("MAMBA_EXE", "MICROMAMBA_EXE"):
        exe = os.environ.get(var)
        if exe and Path(exe).exists():
            return exe

    # 4) PATH lookups (prefer conda)
    for name in ("conda", "mamba", "micromamba"):
        exe = shutil.which(name)
        if exe:
            return exe

    # 5) Try common locations relative to active env
    candidates: list[Path] = []
    prefix = os.environ.get("CONDA_PREFIX") or os.environ.get("MAMBA_ROOT_PREFIX")
    if prefix:
        p = Path(prefix)
        candidates += [
            p / "bin" / "conda",
            p / "bin" / "mamba",
            p / "bin" / "micromamba",
            p.parent / "condabin" / "conda",
        ]
    for c in candidates:
        if c.exists():
            return str(c)

    raise RuntimeError(
        "Neither conda nor (micro)mamba were found. Ensure CONDA_EXE is set or "
        "'conda' is on PATH; alternatively set MERFISH3D_INSTALLER=conda."
    )


@app.command()
def setup_cuda(    
    headless: bool = typer.Option(
        False,
        "--headless",
        help="Skip GUI dependencies such as napari[pyqt6] and cellpose[gui].",
    )
):
    """
    Linux-only setup:

    1) Install CUDA packages via conda/mamba (rapidsai + conda-forge + nvidia).
    2) Write activation hooks exporting CUDA_PATH and JAVA_HOME for this env.
    3) (Optional) Prep CURRENT env: torch/vision cu128, BASE_PIP_DEPS, jax local CUDA.
    4) Create NEW env 'merfish3d-stitcher' (Python 3.12) and pip-install useful deps.
    """
    if platform.system() != "Linux":
        typer.echo("Error: Linux only.", err=True)
        raise typer.Exit(1)

    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        typer.echo("Error: activate your conda environment first.", err=True)
        raise typer.Exit(1)

    try:
        installer = _find_installer()
        typer.echo(f"Using installer: {installer}")
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # 1) Core CUDA + OpenJDK stack from conda
    channels = "-c rapidsai -c conda-forge -c nvidia"
    run(f"{installer} install -y {channels} {' '.join(LINUX_CONDA_CUDA_PKGS)}")

    # 2) Activation / deactivation hooks
    activate_dir = Path(prefix) / "etc" / "conda" / "activate.d"
    deactivate_dir = Path(prefix) / "etc" / "conda" / "deactivate.d"
    activate_dir.mkdir(parents=True, exist_ok=True)
    deactivate_dir.mkdir(parents=True, exist_ok=True)

    # CUDA hook (kept; no longer deletes other hooks)
    sh_cuda = activate_dir / "cuda_override.sh"
    env_lib = f"{prefix}/lib"
    linux_cuda_root = f"{prefix}/targets/x86_64-linux"
    sh_cuda.write_text(
        f"""#!/usr/bin/env sh
# Conda-installed CUDA toolkit
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
    sh_cuda.chmod(sh_cuda.stat().st_mode | stat.S_IEXEC)

    # NEW: Java hook — ensure JPype/pyimagej can see libjvm.so
    sh_java = activate_dir / "java_openjdk.sh"
    sh_java.write_text(
        """#!/usr/bin/env sh
# Point JAVA_HOME at the env's OpenJDK so JPype/scyjava can find libjvm.so
export JAVA_HOME="${CONDA_PREFIX}"
export PATH="$JAVA_HOME/bin:$PATH"

# Helpful for debugging: remember the resolved libjvm path
if [ -f "$JAVA_HOME/lib/server/libjvm.so" ]; then
  export _CONDA_JAVA_LIBJVM="$JAVA_HOME/lib/server/libjvm.so"
else
  # Fallback for nonstandard layouts
  _ALT_JVM="$(command -v find >/dev/null 2>&1 && find "$JAVA_HOME" -type f -name libjvm.so 2>/dev/null | head -n1)"
  [ -n "$_ALT_JVM" ] && export _CONDA_JAVA_LIBJVM="$_ALT_JVM"
fi
""",
        encoding="utf-8",
    )
    sh_java.chmod(sh_java.stat().st_mode | stat.S_IEXEC)

    # Deactivation: clean up JAVA_HOME (leave PATH/LD_LIBRARY_PATH alone to avoid tricky string surgery)
    sh_java_deact = deactivate_dir / "java_openjdk.sh"
    sh_java_deact.write_text(
        """#!/usr/bin/env sh
unset JAVA_HOME
unset _CONDA_JAVA_LIBJVM
""",
        encoding="utf-8",
    )
    sh_java_deact.chmod(sh_java_deact.stat().st_mode | stat.S_IEXEC)

    # 3) (Optional) Prep CURRENT env
    run("python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
    if headless:
        pip_deps = [
            d
            for d in BASE_PIP_DEPS
            if not d.startswith("napari[") and not d.startswith("cellpose[")
        ]
    else:
        pip_deps = BASE_PIP_DEPS
    run(f"python -m pip install {' '.join(shlex.quote(d) for d in pip_deps)}")
    run(f"python -m pip install {' '.join(shlex.quote(d) for d in LINUX_JAX_LIB)}")

    # 4) Create NEW env and install what you asked
    run(f"{installer} create -y -n {MVSTITCHER_ENV_NAME} python={MVSTITCHER_ENV_PY} pip")
    repo_dir = Path.cwd()

    # Upgrade build tooling, then install the *imports* deps first
    run(f"{installer} run -n {MVSTITCHER_ENV_NAME} python -m pip install -U pip setuptools wheel")

    # Install multiview-stitcher and minimal deps to use merfish3d-analysis datastore class
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
