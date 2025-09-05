"""
Colab setup for merfish3d (pip-only, no conda/mamba).

- Uses Colab's system CUDA (/usr/local/cuda)
- Installs GPU wheels: CuPy (CUDA12x), cuCIM, cuVS
- Installs BASE_PIP_DEPS but removes GUI extras; installs plain 'cellpose'
- Installs JAX EXACTLY pinned: jax[cuda12_local]==0.4.38
- Installs multiview-stitcher + ngff-zarr[tensorstore] + current repo (-e .)
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

BASE_PIP_DEPS = [
    "tqdm",
    "ryomen",
    "tensorstore",
    "nvidia-cuda-runtime-cu12==12.8.*",
    "onnx",
    "onnxruntime-gpu",
    "cellpose",
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
    "pandas",
    "roifile",
    "shapely",
    "fastparquet"
]

LINUX_JAX_LIB = [
    "jax[cuda12_local]==0.4.38",
]

def run(cmd, cwd=None):
    print("$ {}".format(cmd), flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=str(cwd) if cwd else None)

def ensure_cuda_env():
    cuda_root = Path("/usr/local/cuda")
    if cuda_root.exists():
        os.environ["CUDA_HOME"] = str(cuda_root)
        os.environ["CUDA_PATH"] = str(cuda_root)
        os.environ["PATH"] = "{}:{}".format(f"{cuda_root}/bin", os.environ.get("PATH", ""))
        ld = f"{cuda_root}/lib64"
        if "LD_LIBRARY_PATH" in os.environ:
            os.environ["LD_LIBRARY_PATH"] = "{}:{}".format(ld, os.environ["LD_LIBRARY_PATH"])
        else:
            os.environ["LD_LIBRARY_PATH"] = ld
        print("Using CUDA at {}".format(cuda_root))
    else:
        print("Warning: /usr/local/cuda not found; continuing without CUDA env hints.", file=sys.stderr)

def build_colab_base_deps():
    deps = []
    for d in BASE_PIP_DEPS:
        deps.append(d)
    return deps

def setup_colab():
    run("python -m pip install -U pip setuptools wheel")
    ensure_cuda_env()
    run("python -m pip install cupy-cuda12x")                       # CuPy for CUDA 12.x
    run("python -m pip install cucim")                              # cuCIM
    run("python -m pip install cuvs-cu12 --extra-index-url=https://pypi.nvidia.com")  # cuVS
    try:
        run(
            "python -m pip install --no-warn-script-location "
            "nvidia-cuda-runtime-cu12 "
            "nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 "
            "nvidia-cusparse-cu12 nvidia-cusolver-cu12 "
            "nvidia-cuda-nvrtc-cu12 nvidia-nvjitlink-cu12 "
            "nvidia-cudnn-cu12 nvidia-nccl-cu12 nvidia-cutensor-cu12"
        )
    except subprocess.CalledProcessError:
        print("Skipping optional NVIDIA CUDA component wheels.", file=sys.stderr)

    colab_deps = build_colab_base_deps()
    run("python -m pip install " + " ".join(shlex.quote(d) for d in colab_deps))
    run("python -m pip install " + " ".join(shlex.quote(d) for d in LINUX_JAX_LIB))

    print("\nColab setup complete.")

def main():
    setup_colab()

if __name__ == "__main__":
    main()