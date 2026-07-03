"""
Perform pixel-based decoding for qi2lab widefield MERFISH data using GPU acceleration.

This module leverages GPU acceleration to decode pixel-based widefield
MERFISH datasets efficiently.

History:
---------
- **2026/04**:
    - Replace pixel-threshold decoding with the exact two-threshold MERFISH caller.
    - Add blank-fraction transcript filtering as the default downstream filter.
    - Remove one-bit codewords from the MERFISH decode path.
- **2025/07**:
    - Refactor for multiple GPU support.
    - Switch to cuvs for distance calculations.
- **2024/12**: Refactor repo structure.
- **2024/03**: Reworked GPU logic to reduce out-of-memory crashes.
- **2024/01**: Updated for qi2lab MERFISH file format v1.0.
"""

import multiprocessing as mp

mp.set_start_method("spawn", force=True)

import ctypes
import gc
import operator
import os
import shutil
import sys
import tempfile
import warnings
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from random import sample
from typing import Literal

_CUDA_LIBRARY_HANDLES: list[ctypes.CDLL] = []


def preload_cuda_libraries() -> None:
    """Preload CUDA libs from NVIDIA pip wheels so GPU libraries can resolve them."""

    if _CUDA_LIBRARY_HANDLES:
        return

    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    base = Path(sys.prefix) / "lib" / pyver / "site-packages" / "nvidia"

    package_libraries = [
        ("cuda_runtime", ["libcudart.so.12"]),
        ("cuda_nvrtc", ["libnvrtc.so.12"]),
        ("cuda_cupti", ["libcupti.so.12"]),
        ("cufft", ["libcufft.so.11"]),
        ("cublas", ["libcublasLt.so.12", "libcublas.so.12"]),
        ("curand", ["libcurand.so.10"]),
        ("cusparse", ["libcusparse.so.12"]),
        ("cusolver", ["libcusolver.so.11", "libcusolverMg.so.11"]),
        ("nvjitlink", ["libnvJitLink.so.12"]),
    ]

    for package_dir, names in package_libraries:
        directory = base / package_dir / "lib"
        for name in names:
            lib_path = directory / name
            if not lib_path.exists():
                continue
            try:
                _CUDA_LIBRARY_HANDLES.append(
                    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                )
            except OSError:
                pass


preload_cuda_libraries()

import itertools

import cupy as cp
import numpy as np
import pandas as pd
import rtree
from cucim.skimage.measure import label
from cucim.skimage.measure import regionprops_table as gpu_regionprops_table
from cucim.skimage.morphology import remove_small_objects
from cupyx.scipy.ndimage import gaussian_filter
from cuvs.distance import pairwise_distance
from roifile import roiread
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon
from skimage.measure import regionprops_table
from tqdm.auto import tqdm, trange

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.decode_warping import warp_bit_image_to_reference

DEFAULT_DECODE_LOWPASS_SIGMA = (3.0, 1.0, 1.0)
DEFAULT_DECODE_MAGNITUDE_THRESHOLD = (1.5, 10.0)
DEFAULT_2D_MINIMUM_PIXELS = 7.0
DEFAULT_3D_MINIMUM_PIXELS = 16.0
DEFAULT_ZSTRIDE_3D_MINIMUM_PIXELS = 10.0

# filter warning from skimage
warnings.filterwarnings(
    "ignore",
    message="Only one label was provided to `remove_small_objects`. Did you mean to use a boolean array?",
)

# GPU helper functions


def _start_gpu_worker_process(
    *,
    target: object,
    args: tuple,
    physical_gpu_id: int,
) -> mp.Process:
    """
    Start one worker with only its assigned physical GPU visible.

    Parameters
    ----------
    target : object
        Worker function passed to :class:`multiprocessing.Process`.
    args : tuple
        Worker arguments. These should use local GPU index 0 because the child
        process sees only ``physical_gpu_id``.
    physical_gpu_id : int
        Physical GPU index to expose to the child process.

    Returns
    -------
    multiprocessing.Process
        Started worker process.
    """

    previous_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(physical_gpu_id))
    try:
        process = mp.Process(target=target, args=args)
        process.start()
    finally:
        if previous_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible_devices
    return process


def _join_gpu_workers(processes: Sequence[mp.Process], label: str) -> None:
    """
    Join GPU workers and raise if any worker failed.

    Parameters
    ----------
    processes : Sequence[multiprocessing.Process]
        Worker processes to join.
    label : str
        Human-readable worker group name used in the error message.

    Returns
    -------
    None
        Raises RuntimeError when at least one worker exits nonzero.
    """

    errors = []
    for process in processes:
        process.join()
        if process.exitcode not in (0, None):
            errors.append(
                f"{label} worker pid={process.pid} failed with "
                f"exitcode={process.exitcode}."
            )
    if errors:
        raise RuntimeError(f"{label} failed:\n" + "\n".join(errors))


def decode_tiles_worker(
    datastore_path: Path,
    tile_indices: Sequence[int],
    gpu_id: int,
    merfish_bits: int,
    verbose: int,
    zstride_level: int,
    decode_mode: Literal["auto", "2d", "3d"],
    lowpass_sigma: Sequence[float],
    magnitude_threshold: Sequence[float],
    minimum_pixels: float,
    feature_predictor_threshold: float,
    normalization_method: Literal["iterative", "global", "none"],
) -> None:
    """
    Worker that runs decode_one_tile on a subset of tiles under one GPU.

    Parameters
    ----------
    datastore_path : Path
        Function argument.
    tile_indices : Sequence[int]
        Function argument.
    gpu_id : int
        Function argument.
    merfish_bits : int
        Function argument.
    verbose : int
        Function argument.
    zstride_level : int
        Decode-time z stride.
    decode_mode : Literal['auto', '2d', '3d']
        Decode connected-component/filtering mode.
    lowpass_sigma : Sequence[float]
        Function argument.
    magnitude_threshold : Sequence[float]
        Function argument.
    minimum_pixels : float
        Function argument.
    feature_predictor_threshold : float
        Function argument.
    normalization_method : Literal['iterative', 'global', 'none']
        Function argument.

    Returns
    -------
    None
        Function result.
    """
    preload_cuda_libraries()

    import cupy as cp
    import torch

    torch.cuda.set_device(gpu_id)
    cp.cuda.Device(gpu_id).use()
    cp.cuda.Stream.null.synchronize()

    local_datastore = qi2labDataStore(datastore_path, validate=False)
    local_decoder = PixelDecoder(
        datastore=local_datastore,
        use_mask=False,
        merfish_bits=merfish_bits,
        num_gpus=1,
        verbose=0,
        zstride_level=zstride_level,
        decode_mode=decode_mode,
    )

    local_decoder._load_global_normalization_vectors(
        gpu_id=gpu_id,
        lowpass_sigma=lowpass_sigma,
    )
    local_decoder._load_iterative_normalization_vectors(gpu_id=gpu_id)
    local_decoder._optimize_normalization_weights = False

    for tile_tracker, tile_idx in enumerate(tile_indices):
        if verbose >= 1:
            print(
                time_stamp(),
                f"GPU {gpu_id}: starting tile {tile_tracker + 1} of {len(tile_indices)} (tile index: {tile_idx}).",
                flush=True,
            )
        local_decoder.decode_one_tile(
            tile_idx=tile_idx,
            display_results=False,
            return_results=False,
            lowpass_sigma=lowpass_sigma,
            magnitude_threshold=magnitude_threshold,
            minimum_pixels=minimum_pixels,
            feature_predictor_threshold=feature_predictor_threshold,
            normalization_method=normalization_method,
            gpu_id=gpu_id,
        )

        local_decoder._save_barcodes()
        local_decoder._cleanup()
        if verbose >= 1:
            print(
                time_stamp(),
                f"GPU {gpu_id}: decoded and saved tile {tile_tracker + 1} of {len(tile_indices)} (tile index: {tile_idx}).",
                flush=True,
            )

    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _optimize_norm_worker(
    datastore_path: Path,
    tile_indices: Sequence[int],
    gpu_id: int,
    merfish_bits: int,
    zstride_level: int,
    decode_mode: Literal["auto", "2d", "3d"],
    temp_dir: Path,
    iteration: int,
    lowpass_sigma: Sequence[float],
    magnitude_threshold: Sequence[float],
    minimum_pixels: float,
    feature_predictor_threshold: float,
    collect_chromatic_centroids: bool,
) -> None:
    """
    Worker that runs one iteration of normalization-by-decoding on a GPU.

    Parameters
    ----------
    datastore_path : Path
        Function argument.
    tile_indices : Sequence[int]
        Function argument.
    gpu_id : int
        Function argument.
    merfish_bits : int
        Function argument.
    zstride_level : int
        Decode-time z stride.
    decode_mode : Literal['auto', '2d', '3d']
        Decode connected-component/filtering mode.
    temp_dir : Path
        Function argument.
    iteration : int
        Function argument.
    lowpass_sigma : Sequence[float]
        Function argument.
    magnitude_threshold : Sequence[float]
        Function argument.
    minimum_pixels : float
        Function argument.
    feature_predictor_threshold : float
        Function argument.
    collect_chromatic_centroids : bool
        If True, collect per-on-bit centroid features needed for chromatic
        affine estimation.

    Returns
    -------
    None
        Function result.
    """
    preload_cuda_libraries()

    import cupy as cp
    import torch

    torch.cuda.set_device(gpu_id)
    cp.cuda.Device(gpu_id).use()
    cp.cuda.Stream.null.synchronize()

    local_datastore = qi2labDataStore(datastore_path, validate=False)
    local_decoder = PixelDecoder(
        datastore=local_datastore,
        use_mask=False,
        merfish_bits=merfish_bits,
        num_gpus=1,
        verbose=0,
        zstride_level=zstride_level,
        decode_mode=decode_mode,
    )

    local_decoder._load_global_normalization_vectors(
        gpu_id=gpu_id,
        lowpass_sigma=lowpass_sigma,
    )
    local_decoder._optimize_normalization_weights = True
    local_decoder._collect_chromatic_centroids = bool(collect_chromatic_centroids)
    local_decoder._temp_dir = temp_dir

    # Seed the first iteration from global normalization, then refine iteratively.
    use_norm = iteration > 0
    for tile_idx in tile_indices:
        local_decoder.decode_one_tile(
            tile_idx=tile_idx,
            display_results=False,
            return_results=False,
            lowpass_sigma=lowpass_sigma,
            magnitude_threshold=magnitude_threshold,
            minimum_pixels=minimum_pixels,
            feature_predictor_threshold=feature_predictor_threshold,
            normalization_method="iterative" if use_norm else "global",
            gpu_id=gpu_id,
        )
        local_decoder._save_barcodes()

    local_decoder._optimize_normalization_weights = False

    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


class PixelDecoder:
    """
    Retrieve and process one tile from qi2lab 3D widefield zarr structure.
    Normalize the MERFISH codebook and image data, perform
    plane-by-plane voxel decoding with the exact two-threshold caller,
    extract transcript features, and save decoded transcripts to disk.

    Parameters
    ----------
    datastore: qi2labDataStore
        qi2labDataStore object
    merfish_bits: int, default 16
        number of merfish bits. Assumes that in codebook, MERFISH rounds are [0,merfish_bits].
    num_gpus: int, default 1
        number of GPUs to use for decoding. If > 1, will split decoding across GPUs.
    verbose: int, default 1
        control verbosity. 0 - no output, 1 - tqdm bars, 2 - diagnostic outputs
    use_mask: bool, default False
        use mask stored in fiducial directory
    z_range: Sequence[int], default None
        z range to analyze. In integer indices from [0,N] where N is number of
        z planes.
    zstride_level : int, default 0
        Decode-time z stride. Values 0 and 1 keep all planes; values >= 2 keep
        planes 0, N, 2N...
    decode_mode : {'auto', '2d', '3d'}, default 'auto'
        Connected-component and filtering mode. ``auto`` follows the datastore
        microscope type.
    estimate_chromatic_affines : bool, default False
        If True, estimate chromatic affine transforms during iterative
        normalization. If False, use existing datastore calibration metadata
        with identity fallback.
    """

    def __init__(
        self,
        datastore: qi2labDataStore,
        merfish_bits: int = 16,
        num_gpus: int = 1,
        verbose: int = 1,
        use_mask: bool | None = False,
        z_range: Sequence[int] | None = None,
        zstride_level: int = 0,
        decode_mode: Literal["auto", "2d", "3d"] = "auto",
        estimate_chromatic_affines: bool = False,
    ) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        datastore : qi2labDataStore
            Function argument.
        merfish_bits : int
            Function argument.
        num_gpus : int
            Function argument.
        verbose : int
            Function argument.
        use_mask : bool | None
            Function argument.
        z_range : Sequence[int] | None
            Function argument.
        zstride_level : int
            Function argument.
        decode_mode : Literal['auto', '2d', '3d']
            Function argument.
        estimate_chromatic_affines : bool
            If True, iterative normalization estimates chromatic affine
            transforms from decoded on-bit centroids. If False, decoding uses
            existing datastore chromatic calibration metadata with identity
            fallback.
        """
        self._datastore_path = Path(datastore._datastore_path)
        self._datastore = datastore
        self._num_gpus = num_gpus
        self._verbose = verbose
        self._barcodes_filtered = False

        self._n_merfish_bits = merfish_bits

        if zstride_level < 0:
            raise ValueError("zstride_level must be greater than or equal to 0.")
        if decode_mode not in {"auto", "2d", "3d"}:
            raise ValueError("decode_mode must be one of 'auto', '2d', or '3d'.")

        self._decode_mode = decode_mode
        self._estimate_chromatic_affines = bool(estimate_chromatic_affines)
        self._zstride_level = int(zstride_level)
        self._zstride = max(1, int(zstride_level))
        if decode_mode == "auto":
            effective_decode_mode = (
                "2d" if self._datastore.microscope_type == "2D" else "3d"
            )
        else:
            effective_decode_mode = decode_mode
        self._effective_decode_mode = effective_decode_mode

        if effective_decode_mode == "2d":
            self._is_3D = False
        else:
            self._is_3D = True
        self._decode_run_key = (
            None
            if self._zstride <= 1
            else f"zstride_{self._zstride:02d}_{effective_decode_mode}"
        )
        if z_range is None:
            self._z_crop = False
            self._z_range = [0, None]
        else:
            self._z_crop = True
            self._z_range = [z_range[0], z_range[1]]
        self._z_slice = slice(self._z_range[0], self._z_range[1], self._zstride)

        self._load_codebook()
        self._decoding_matrix_no_errors = self._normalize_codebook(include_errors=False)
        self._decoding_matrix = self._decoding_matrix_no_errors.copy()
        self._barcode_count = self._decoding_matrix.shape[0]
        self._bit_count = self._decoding_matrix.shape[1]

        if use_mask:
            self._load_mask()  # TO DO: implement
        else:
            self._mask_image = None

        self._codebook_style = 1
        self._optimize_normalization_weights = False
        self._collect_chromatic_centroids = False
        self._global_normalization_loaded = False
        self._iterative_normalization_loaded = False
        self._blank_fraction_filter_results: dict[str, object] | None = None

    def _load_codebook(self) -> None:
        """
        Load the MERFISH codebook and derive caller geometry from it.

        Returns
        -------
        None
            Function result.
        """

        self._df_codebook = self._datastore.codebook.copy()
        self._df_codebook.fillna(0, inplace=True)
        bit_columns = self._df_codebook.columns[1 : self._n_merfish_bits + 1]
        on_counts = (
            self._df_codebook.loc[:, bit_columns]
            .to_numpy(dtype=np.int8, copy=False)
            .sum(axis=1)
        )
        self._df_codebook = self._df_codebook.loc[on_counts != 1].reset_index(drop=True)

        self._codebook_matrix = self._df_codebook.loc[:, bit_columns].to_numpy(
            dtype=int, copy=False
        )
        on_bit_count = int(np.median(on_counts[on_counts != 1]))
        self._pixel_assignment_threshold = float(
            np.sqrt(
                2.0
                - 2.0
                * ((on_bit_count - 2.0) / np.sqrt(on_bit_count * (on_bit_count - 2.0)))
            )
        )
        self._transcript_distance_threshold = float(
            np.sqrt(
                2.0
                - 2.0 * (on_bit_count / np.sqrt(on_bit_count * (on_bit_count + 2.0)))
            )
        )

        self._blank_count = int(
            self._df_codebook["gene_id"]
            .astype("string")
            .str.lower()
            .str.startswith("blank", na=False)
            .sum()
        )
        self._gene_ids = self._df_codebook.iloc[:, 0].tolist()

    def _normalize_codebook(
        self, gpu_id: int = 0, include_errors: bool = False
    ) -> np.ndarray:
        """Normalize each codeword by L2 norm.

        Parameters
        ----------
        gpu_id: int, default = 0
            GPU identifier
        include_errors : bool, default False
            Include single-bit errors as unique barcodes in the decoding matrix.

        Returns
        -------
        all_barcodes: np.ndarray
            normalized codebook
        """

        with cp.cuda.Device(gpu_id):
            self._barcode_set = cp.asarray(
                self._codebook_matrix[:, 0 : self._n_merfish_bits]
            )
            magnitudes = cp.linalg.norm(self._barcode_set, axis=1, keepdims=True)
            magnitudes[magnitudes == 0] = 1

            if not include_errors:
                # Normalize directly using broadcasting
                normalized_barcodes = self._barcode_set / magnitudes
                return cp.asnumpy(normalized_barcodes)
            else:
                # Pre-compute the normalized barcodes
                normalized_barcodes = self._barcode_set / magnitudes

                # Initialize an empty list to hold all barcodes with single errors
                barcodes_with_single_errors = [normalized_barcodes]

                # Generate single-bit errors
                for bit_index in range(self._barcode_set.shape[1]):
                    flipped_barcodes = self._barcode_set.copy()
                    flipped_barcodes[:, bit_index] = 1 - flipped_barcodes[:, bit_index]
                    flipped_magnitudes = cp.sqrt(cp.sum(flipped_barcodes**2, axis=1))
                    flipped_magnitudes = cp.where(
                        flipped_magnitudes == 0, 1, flipped_magnitudes
                    )
                    normalized_flipped = flipped_barcodes / flipped_magnitudes
                    barcodes_with_single_errors.append(normalized_flipped)

                # Stack all barcodes (original normalized + with single errors)
                all_barcodes = cp.vstack(barcodes_with_single_errors)

                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

                return cp.asnumpy(all_barcodes)

    def _load_global_normalization_vectors(
        self,
        gpu_id: int = 0,
        recalculate: bool = False,
        tile_indices: Sequence[int] | None = None,
        lowpass_sigma: Sequence[float] | None = DEFAULT_DECODE_LOWPASS_SIGMA,
    ) -> None:
        """Load or calculate global normalization and background vectors.

        Parameters
        ----------
        gpu_id: int, default = 0
            GPU identifier
        recalculate : bool, default False
            Recompute global normalization/background vectors instead of
            reusing cached datastore values.
        tile_indices : Sequence[int], optional
            Explicit tile indices to use when recalculating.
        lowpass_sigma : Sequence[float], default = (3, 1, 1)
            Lowpass sigma applied to ``data * prediction`` before estimating
            the global background and foreground normalization vectors.
        """
        with cp.cuda.Device(gpu_id):
            normalization_vector, background_vector = (
                self._datastore.load_decode_normalization_vectors(
                    self._decode_run_key, "global"
                )
            )
            if (
                not recalculate
                and normalization_vector is not None
                and background_vector is not None
            ):
                self._global_normalization_vector = cp.asarray(normalization_vector)
                self._global_background_vector = cp.asarray(background_vector)
                self._global_normalization_loaded = True
            else:
                self._global_normalization_vectors(
                    gpu_id=gpu_id,
                    tile_indices=tile_indices,
                    lowpass_sigma=lowpass_sigma,
                )

            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def _global_normalization_vectors(
        self,
        low_percentile_cut: float = 10.0,
        high_percentile_cut: float = 90.0,
        hot_pixel_threshold: int = 50000,
        gpu_id: int = 0,
        tile_indices: Sequence[int] | None = None,
        lowpass_sigma: Sequence[float] | None = DEFAULT_DECODE_LOWPASS_SIGMA,
    ) -> None:
        """Calculate global normalization and background vectors.

        Parameters
        ----------
        low_percentile_cut : float, default 10.0
            Lower percentile cut for background estimation.
        high_percentile_cut : float, default 90.0
            Upper percentile cut for normalization estimation.
        hot_pixel_threshold : int, default 50000
            Threshold for hot pixel removal.
        gpu_id: int, default = 0
            GPU identifier
        tile_indices : Sequence[int], optional
            Explicit tile indices to use. If omitted, up to five random tiles
            are sampled from the datastore.
        lowpass_sigma : Sequence[float], default = (3, 1, 1)
            Lowpass sigma applied to ``data * prediction`` before estimating
            background and foreground normalization vectors.
        """

        with cp.cuda.Device(gpu_id):
            effective_lowpass_sigma = self._effective_lowpass_sigma(lowpass_sigma)
            if tile_indices is not None:
                random_tiles = [
                    self._datastore.tile_ids[tile_idx] for tile_idx in tile_indices
                ]
            elif len(self._datastore.tile_ids) > 5:
                random_tiles = sample(self._datastore.tile_ids, 5)
            else:
                random_tiles = self._datastore.tile_ids

            normalization_vector = cp.ones(
                len(self._datastore.bit_ids), dtype=cp.float32
            )
            background_vector = cp.zeros(len(self._datastore.bit_ids), dtype=cp.float32)

            if self._verbose >= 1:
                print("calculate normalizations")
                iterable_bits = enumerate(
                    tqdm(self._datastore.bit_ids, desc="bit", leave=False)
                )
            else:
                iterable_bits = enumerate(self._datastore.bit_ids)

            for bit_idx, bit_id in iterable_bits:
                all_images = []

                if self._verbose >= 1:
                    iterable_tiles = tqdm(
                        random_tiles, desc="loading tiles", leave=False
                    )
                else:
                    iterable_tiles = random_tiles

                for tile_id in iterable_tiles:
                    decon_image = self._datastore.load_local_registered_image(
                        tile=tile_id, bit=bit_id, return_future=False
                    )
                    feature_predictor_image = (
                        self._datastore.load_local_feature_predictor_image(
                            tile=tile_id, bit=bit_id, return_future=False
                        )
                    )
                    _ex_wvl, em_wvl = self._datastore.load_local_wavelengths_um(
                        tile=tile_id,
                        bit=bit_id,
                    )

                    current_image = np.asarray(
                        decon_image, dtype=np.float32
                    ) * np.asarray(feature_predictor_image, dtype=np.float32)
                    current_image = warp_bit_image_to_reference(
                        current_image,
                        datastore=self._datastore,
                        tile=tile_id,
                        bit_id=bit_id,
                        emission_wavelength_um=em_wvl,
                        gpu_id=gpu_id,
                    )
                    current_image = cp.asarray(current_image, dtype=cp.float32)
                    current_image[current_image > hot_pixel_threshold] = cp.median(
                        current_image[current_image.shape[0] // 2, :, :]
                    ).astype(cp.float32)
                    current_image = current_image[self._z_slice, :, :]
                    current_image = self._lowpass_image(
                        current_image,
                        sigma=effective_lowpass_sigma,
                    )
                    all_images.append(cp.asnumpy(current_image).astype(np.float32))
                    del current_image
                    cp.get_default_memory_pool().free_all_blocks()
                    gc.collect()

                all_images = np.array(all_images)

                if self._verbose >= 1:
                    iterable_tiles = enumerate(
                        tqdm(random_tiles, desc="background est.", leave=False)
                    )
                else:
                    iterable_tiles = enumerate(random_tiles)

                low_pixels = []
                for tile_idx, _ in iterable_tiles:
                    current_image = cp.asarray(
                        all_images[tile_idx, :], dtype=cp.float32
                    )
                    low_cutoff = cp.percentile(current_image, low_percentile_cut)
                    low_pixels.append(
                        current_image[current_image < low_cutoff]
                        .flatten()
                        .astype(cp.float32)
                    )
                    del current_image
                    cp.get_default_memory_pool().free_all_blocks()
                    gc.collect()

                low_pixels = cp.concatenate(low_pixels, axis=0)
                if low_pixels.shape[0] > 0:
                    background_vector[bit_idx] = cp.median(low_pixels)
                else:
                    background_vector[bit_idx] = 0

                del low_pixels
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()

                if self._verbose >= 1:
                    iterable_tiles = enumerate(
                        tqdm(random_tiles, desc="normalization est.", leave=False)
                    )
                else:
                    iterable_tiles = enumerate(random_tiles)

                high_pixels = []
                for tile_idx, _ in iterable_tiles:
                    current_image = (
                        cp.asarray(all_images[tile_idx, :], dtype=cp.float32)
                        - background_vector[bit_idx]
                    )
                    current_image[current_image < 0] = 0
                    high_cutoff = cp.percentile(current_image, high_percentile_cut)
                    high_pixels.append(
                        current_image[current_image > high_cutoff]
                        .flatten()
                        .astype(cp.float32)
                    )

                    del current_image
                    cp.get_default_memory_pool().free_all_blocks()
                    gc.collect()

                high_pixels = cp.concatenate(high_pixels, axis=0)
                if high_pixels.shape[0] > 0:
                    normalization_vector[bit_idx] = cp.median(high_pixels)
                else:
                    normalization_vector[bit_idx] = 1

                del high_pixels
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()

            self._datastore.save_decode_normalization_vectors(
                self._decode_run_key,
                "global",
                cp.asnumpy(normalization_vector).astype(np.float32),
                cp.asnumpy(background_vector).astype(np.float32),
                zstride_level=self._zstride,
                decode_mode=self._effective_decode_mode,
            )

            self._global_background_vector = background_vector
            self._global_normalization_vector = normalization_vector
            self._global_normalization_loaded = True

            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def _load_iterative_normalization_vectors(self, gpu_id: int = 0) -> None:
        """Load or calculate iterative normalization and background vectors.

        Parameters
        ----------
        gpu_id: int, default = 0
            GPU identifier
        """
        with cp.cuda.Device(gpu_id):
            normalization_vector, background_vector = (
                self._datastore.load_decode_normalization_vectors(
                    self._decode_run_key, "iterative"
                )
            )

            if normalization_vector is not None and background_vector is not None:
                background_vector = np.nan_to_num(background_vector, 0.0)
                normalization_vector = np.nan_to_num(normalization_vector, 1.0)
                self._iterative_normalization_vector = cp.asarray(normalization_vector)
                self._iterative_background_vector = cp.asarray(background_vector)
                self._iterative_normalization_loaded = True
            else:
                self._iterative_normalization_vectors(gpu_id=gpu_id)

            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def _iterative_normalization_vectors(self, gpu_id: int = 0) -> None:
        """Calculate iterative normalization and background vectors.

        Parameters
        ----------
        gpu_id: int, default = 0
            GPU identifier
        """
        with cp.cuda.Device(gpu_id):
            keep = ~(
                self._df_barcodes_loaded["gene_id"]
                .astype("string")
                .str.lower()
                .str.startswith("blank", na=False)
            )

            df_barcodes_loaded_no_blanks = self._df_barcodes_loaded[keep]

            if (
                self._iterative_background_vector is None
                and self._iterative_normalization_vector is None
            ):
                old_iterative_background_vector = np.round(
                    cp.asnumpy(
                        self._global_background_vector[0 : self._n_merfish_bits]
                    ),
                    1,
                )
                old_iterative_normalization_vector = np.round(
                    cp.asnumpy(
                        self._global_normalization_vector[0 : self._n_merfish_bits]
                    ),
                    1,
                )
            else:
                old_iterative_background_vector = np.asarray(
                    cp.asnumpy(self._iterative_background_vector)
                )
                old_iterative_normalization_vector = np.asarray(
                    cp.asnumpy(self._iterative_normalization_vector)
                )

            bit_columns = [
                col
                for col in df_barcodes_loaded_no_blanks.columns
                if col.startswith("bit") and col.endswith("_mean_intensity")
            ]

            if df_barcodes_loaded_no_blanks.empty or not bit_columns:
                self._datastore.save_decode_normalization_vectors(
                    self._decode_run_key,
                    "iterative",
                    old_iterative_normalization_vector.astype(np.float32),
                    old_iterative_background_vector.astype(np.float32),
                    zstride_level=self._zstride,
                    decode_mode=self._effective_decode_mode,
                )
                return

            barcode_intensities = []
            barcode_background = []
            for _index, row in df_barcodes_loaded_no_blanks.iterrows():
                selected_columns = [
                    f"bit{int(row['on_bit_1']):02d}_mean_intensity",
                    f"bit{int(row['on_bit_2']):02d}_mean_intensity",
                    f"bit{int(row['on_bit_3']):02d}_mean_intensity",
                    f"bit{int(row['on_bit_4']):02d}_mean_intensity",
                ]

                selected_dict = {
                    col: (row[col] if col in selected_columns else None)
                    for col in bit_columns
                }
                not_selected_dict = {
                    col: (row[col] if col not in selected_columns else None)
                    for col in bit_columns
                }

                barcode_intensities.append(selected_dict)
                barcode_background.append(not_selected_dict)

            df_barcode_intensities = pd.DataFrame(barcode_intensities)
            df_barcode_background = pd.DataFrame(barcode_background)

            df_barcode_intensities = df_barcode_intensities.reindex(
                sorted(df_barcode_intensities.columns), axis=1
            )
            df_barcode_background = df_barcode_background.reindex(
                sorted(df_barcode_background.columns), axis=1
            )

            barcode_based_normalization_vector = np.round(
                df_barcode_intensities.median(skipna=True).to_numpy(
                    dtype=np.float32, copy=True
                ),
                1,
            )
            barcode_based_background_vector = np.round(
                df_barcode_background.median(skipna=True).to_numpy(
                    dtype=np.float32, copy=True
                ),
                1,
            )

            barcode_based_normalization_vector = np.nan_to_num(
                barcode_based_normalization_vector, 1.0
            )
            barcode_based_normalization_vector = np.where(
                barcode_based_normalization_vector == 0.0,
                1.0,
                barcode_based_normalization_vector,
            )
            barcode_based_background_vector = np.nan_to_num(
                barcode_based_background_vector, 0.0
            )

            diff_iterative_background_vector = np.round(
                np.abs(
                    barcode_based_background_vector - old_iterative_background_vector
                ),
                1,
            )
            diff_iterative_normalization_vector = np.round(
                np.abs(
                    barcode_based_normalization_vector
                    - old_iterative_normalization_vector
                ),
                1,
            )
            self._datastore.save_decode_normalization_vectors(
                self._decode_run_key,
                "iterative",
                barcode_based_normalization_vector.astype(np.float32),
                barcode_based_background_vector.astype(np.float32),
                zstride_level=self._zstride,
                decode_mode=self._effective_decode_mode,
            )

            if self._verbose > 1:
                print(time_stamp(), "Normalizations updated.")
                print("---")
                print(f"Background delta: {diff_iterative_background_vector}")
                print(f"Background estimate: {barcode_based_background_vector}")
                print("---")
                print(f"Foreground delta: {diff_iterative_normalization_vector}")
                print(f"Foreground estimate: {barcode_based_normalization_vector}")
                print("---")
                print(f"Num. barcodes: {len(df_barcodes_loaded_no_blanks)}")
                print("---")

            self._iterative_normalization_vector = barcode_based_normalization_vector
            self._iterative_background_vector = barcode_based_background_vector
            self._datastore.save_decode_normalization_vectors(
                self._decode_run_key,
                "iterative",
                barcode_based_normalization_vector,
                barcode_based_background_vector,
                zstride_level=self._zstride,
                decode_mode=self._effective_decode_mode,
            )

            self._iterative_normalization_loaded = True

            del df_barcodes_loaded_no_blanks
            gc.collect()
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def _estimate_chromatic_affines_from_barcodes(
        self,
        min_pairs: int = 20,
    ) -> None:
        """
        Estimate 3D chromatic affine matrices from decoded RNA on-bit centroids.

        Parameters
        ----------
        min_pairs : int, default=20
            Minimum number of paired on-bit centroid measurements required to
            estimate a wavelength-pair affine.

        Returns
        -------
        None
            Chromatic affine metadata are saved into the datastore calibration
            sidecar.
        """

        if self._df_barcodes_loaded.empty:
            return
        if "gene_id" not in self._df_barcodes_loaded.columns:
            return

        keep_coding_transcripts = ~(
            self._df_barcodes_loaded["gene_id"]
            .astype("string")
            .str.lower()
            .str.startswith("blank", na=False)
        )
        barcode_table = self._df_barcodes_loaded.loc[
            keep_coding_transcripts
            & self._df_barcodes_loaded["gene_id"].notna()
            & self._df_barcodes_loaded["gene_id"].astype(str).str.strip().ne("")
        ].reset_index(drop=True)
        if barcode_table.empty:
            return

        bit_wavelengths = {}
        bit_ids = self._datastore.bit_ids[0 : self._n_merfish_bits]
        reference_tile = self._datastore.tile_ids[0]
        for bit_index, bit_id in enumerate(bit_ids, start=1):
            _ex_wvl, em_wvl = self._datastore.load_local_wavelengths_um(
                tile=reference_tile,
                bit=bit_id,
            )
            bit_wavelengths[bit_index] = float(em_wvl)

        unique_wavelengths = sorted(set(bit_wavelengths.values()))
        reference_wavelength = unique_wavelengths[0]
        spacing = np.asarray(self._datastore.voxel_size_zyx_um, dtype=np.float32)
        wavelength_to_index = {
            wavelength: index for index, wavelength in enumerate(unique_wavelengths)
        }
        previous_chromatic_affines = {}
        previous_calibration = self._datastore.load_chromatic_affine_transforms_zyx_um()
        if isinstance(previous_calibration, dict):
            for channel in previous_calibration.get("channels", {}).values():
                if not isinstance(channel, dict):
                    continue
                wavelength = channel.get("wavelength_um")
                affine = channel.get("affine_zyx_um")
                if wavelength is None or affine is None:
                    continue
                previous_chromatic_affines[float(wavelength)] = np.asarray(
                    affine,
                    dtype=np.float32,
                )
        pair_points: dict[
            tuple[float, float], tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {
            (source_wavelength, target_wavelength): (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
            for source_wavelength in unique_wavelengths
            for target_wavelength in unique_wavelengths
            if not np.isclose(source_wavelength, target_wavelength)
        }

        on_bit_columns = ["on_bit_1", "on_bit_2", "on_bit_3", "on_bit_4"]
        if not all(column in barcode_table.columns for column in on_bit_columns):
            return
        on_bits = barcode_table[on_bit_columns].to_numpy(dtype=np.int16)
        num_barcodes = len(barcode_table)
        centers_by_wavelength_um = {}
        weights_by_wavelength = {}
        valid_by_wavelength = {}

        for wavelength in unique_wavelengths:
            weighted_sum = np.zeros((num_barcodes, 3), dtype=np.float64)
            weight_sum = np.zeros(num_barcodes, dtype=np.float64)
            for bit, bit_wavelength in bit_wavelengths.items():
                if not np.isclose(bit_wavelength, wavelength):
                    continue
                center_cols = [
                    f"bit{bit:02d}_center_z",
                    f"bit{bit:02d}_center_y",
                    f"bit{bit:02d}_center_x",
                ]
                if not all(col in barcode_table.columns for col in center_cols):
                    continue
                centers = barcode_table[center_cols].to_numpy(dtype=np.float64)
                intensity_col = f"bit{bit:02d}_intensity_sum"
                if intensity_col in barcode_table.columns:
                    weights = barcode_table[intensity_col].to_numpy(dtype=np.float64)
                else:
                    weights = np.ones(num_barcodes, dtype=np.float64)
                bit_is_on = np.any(on_bits == int(bit), axis=1)
                valid = bit_is_on & np.all(np.isfinite(centers), axis=1)
                valid &= np.isfinite(weights) & (weights > 0)
                if not np.any(valid):
                    continue
                weighted_sum[valid] += centers[valid] * weights[valid, np.newaxis]
                weight_sum[valid] += weights[valid]

            valid_wavelength = weight_sum > 0
            centers_um = np.full((num_barcodes, 3), np.nan, dtype=np.float32)
            centers_um[valid_wavelength] = (
                weighted_sum[valid_wavelength]
                / weight_sum[valid_wavelength, np.newaxis]
                * spacing
            ).astype(np.float32)
            centers_by_wavelength_um[wavelength] = centers_um
            weights_by_wavelength[wavelength] = weight_sum.astype(np.float32)
            valid_by_wavelength[wavelength] = valid_wavelength

        valid_wavelength_count = np.zeros(num_barcodes, dtype=np.int16)
        for valid in valid_by_wavelength.values():
            valid_wavelength_count += valid.astype(np.int16)
        contributing_transcripts = int(np.sum(valid_wavelength_count >= 2))

        for source_wavelength in unique_wavelengths:
            for target_wavelength in unique_wavelengths:
                if np.isclose(source_wavelength, target_wavelength):
                    continue
                valid_pair = (
                    valid_by_wavelength[source_wavelength]
                    & valid_by_wavelength[target_wavelength]
                )
                pair_weights = np.sqrt(
                    weights_by_wavelength[source_wavelength][valid_pair]
                    * weights_by_wavelength[target_wavelength][valid_pair]
                ).astype(np.float32)
                finite_pair = np.isfinite(pair_weights) & (pair_weights > 0)
                source_points = centers_by_wavelength_um[source_wavelength][valid_pair][
                    finite_pair
                ]
                target_points = centers_by_wavelength_um[target_wavelength][valid_pair][
                    finite_pair
                ]
                pair_weights = pair_weights[finite_pair]
                if pair_weights.size >= 2 * int(min_pairs):
                    min_weight = np.percentile(pair_weights, 25)
                    strong = pair_weights >= min_weight
                    if int(np.sum(strong)) >= int(min_pairs):
                        source_points = source_points[strong]
                        target_points = target_points[strong]
                        pair_weights = pair_weights[strong]
                pair_points[(source_wavelength, target_wavelength)] = (
                    source_points,
                    target_points,
                    pair_weights,
                )

        edge_affines = {}
        edge_diagnostics = {}
        for wavelength_pair, points in pair_points.items():
            source_points, target_points, pair_weights = points
            if source_points.shape[0] < int(min_pairs):
                continue
            affine, diagnostics = self._fit_affine_zyx_um(
                source_points,
                target_points,
                weights=pair_weights,
                min_pairs=min_pairs,
            )
            diagnostics["candidate_pairs"] = int(source_points.shape[0])
            edge_diagnostics[wavelength_pair] = diagnostics
            if affine is not None:
                edge_affines[wavelength_pair] = affine

        adjacency = {wavelength: [] for wavelength in unique_wavelengths}
        for source_wavelength, target_wavelength in edge_affines:
            adjacency[source_wavelength].append(target_wavelength)

        affines_by_wavelength = {
            reference_wavelength: np.eye(4, dtype=np.float32),
        }
        status_by_wavelength = {reference_wavelength: "identity_reference"}
        diagnostics_by_wavelength = {
            wavelength: {
                "paired_transcripts": contributing_transcripts,
                "pair_constraints": 0,
                "path_wavelengths_um": [],
            }
            for wavelength in unique_wavelengths
        }

        for wavelength in unique_wavelengths:
            if np.isclose(wavelength, reference_wavelength):
                continue

            queue = [(wavelength, [wavelength], np.eye(4, dtype=np.float32))]
            visited = {wavelength}
            while queue:
                current_wavelength, path, composed_affine = queue.pop(0)
                if np.isclose(current_wavelength, reference_wavelength):
                    affines_by_wavelength[wavelength] = composed_affine
                    status_by_wavelength[wavelength] = "affine_estimated"
                    pair_count = 0
                    path_diagnostics = []
                    for source_wavelength, target_wavelength in itertools.pairwise(
                        path
                    ):
                        pair_count += pair_points[
                            (source_wavelength, target_wavelength)
                        ][0].shape[0]
                        path_diagnostics.append(
                            {
                                "source_wavelength_um": float(source_wavelength),
                                "target_wavelength_um": float(target_wavelength),
                                "fit": edge_diagnostics[
                                    (source_wavelength, target_wavelength)
                                ],
                            }
                        )
                    diagnostics_by_wavelength[wavelength] = {
                        "paired_transcripts": contributing_transcripts,
                        "pair_constraints": int(pair_count),
                        "path_wavelengths_um": [float(v) for v in path],
                        "path_fits": path_diagnostics,
                    }
                    break
                for neighbor in adjacency[current_wavelength]:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    edge_affine = edge_affines[(current_wavelength, neighbor)]
                    queue.append(
                        (
                            neighbor,
                            [*path, neighbor],
                            edge_affine @ composed_affine,
                        )
                    )

            if wavelength not in affines_by_wavelength:
                affines_by_wavelength[wavelength] = np.eye(4, dtype=np.float32)
                candidate_pairs = sum(
                    pair_points[(wavelength, other_wavelength)][0].shape[0]
                    for other_wavelength in unique_wavelengths
                    if not np.isclose(wavelength, other_wavelength)
                )
                status_by_wavelength[wavelength] = (
                    "identity_fallback_unconnected"
                    if candidate_pairs >= int(min_pairs)
                    else "identity_fallback_too_few_pairs"
                )
                diagnostics_by_wavelength[wavelength] = {
                    "paired_transcripts": contributing_transcripts,
                    "pair_constraints": int(candidate_pairs),
                    "path_wavelengths_um": [],
                }

        channels = {}
        for wavelength in unique_wavelengths:
            residual_affine = affines_by_wavelength[wavelength].astype(
                np.float32,
                copy=False,
            )
            previous_affine = previous_chromatic_affines.get(
                wavelength,
                np.eye(4, dtype=np.float32),
            )
            cumulative_affine = residual_affine @ previous_affine
            if np.isclose(wavelength, reference_wavelength):
                cumulative_affine = np.eye(4, dtype=np.float32)
            elif not np.allclose(previous_affine, np.eye(4, dtype=np.float32)):
                z_limit_um = 3.5 * float(spacing[0])
                lateral_scale = np.diag(cumulative_affine[1:3, 1:3])
                lateral_shear = cumulative_affine[1:3, 1:3] - np.diag(lateral_scale)
                plausible_cumulative = (
                    abs(float(cumulative_affine[0, 3])) <= z_limit_um
                    and np.all(lateral_scale >= 0.85)
                    and np.all(lateral_scale <= 1.05)
                    and float(np.max(np.abs(lateral_shear))) <= 0.08
                )
                if not plausible_cumulative:
                    cumulative_affine = previous_affine.astype(
                        np.float32,
                        copy=True,
                    )
                    diagnostics_by_wavelength[wavelength][
                        "rejected_residual_status"
                    ] = "implausible_cumulative_affine"
            key = f"wavelength_{wavelength:.6f}"
            channels[key] = {
                "channel_index": wavelength_to_index[wavelength],
                "channel_name": key,
                "wavelength_um": float(wavelength),
                "reference_channel": bool(np.isclose(wavelength, reference_wavelength)),
                "affine_zyx_um": cumulative_affine.tolist(),
                "diagnostics": diagnostics_by_wavelength[wavelength],
                "status": status_by_wavelength.get(
                    wavelength, "identity_fallback_unconnected"
                ),
            }

        self._datastore.save_chromatic_affine_transforms_zyx_um(
            {
                "reference_wavelength_um": float(reference_wavelength),
                "voxel_size_zyx_um": [float(v) for v in spacing],
                "estimator": (
                    "decoded_rna_on_bit_weighted_centroid_z_translation_yx_affine_graph"
                ),
                "pair_constraints": int(
                    sum(points[0].shape[0] for points in pair_points.values())
                ),
                "contributing_transcripts": contributing_transcripts,
                "channels": channels,
            }
        )

    def _save_identity_chromatic_affines(self) -> None:
        """
        Save identity chromatic affine transforms for the current bit wavelengths.

        Returns
        -------
        None
            Identity channel transforms are written to datastore metadata.
        """

        bit_ids = self._datastore.bit_ids[0 : self._n_merfish_bits]
        reference_tile = self._datastore.tile_ids[0]
        wavelengths = []
        for bit_id in bit_ids:
            _ex_wvl, em_wvl = self._datastore.load_local_wavelengths_um(
                tile=reference_tile,
                bit=bit_id,
            )
            wavelengths.append(float(em_wvl))
        unique_wavelengths = sorted(set(wavelengths))
        reference_wavelength = unique_wavelengths[0]
        channels = {}
        for index, wavelength in enumerate(unique_wavelengths):
            key = f"wavelength_{wavelength:.6f}"
            channels[key] = {
                "channel_index": index,
                "channel_name": key,
                "wavelength_um": float(wavelength),
                "reference_channel": bool(np.isclose(wavelength, reference_wavelength)),
                "affine_zyx_um": np.eye(4, dtype=np.float32).tolist(),
                "diagnostics": {
                    "paired_transcripts": 0,
                    "pair_constraints": 0,
                    "path_wavelengths_um": [],
                },
                "status": "identity_reference"
                if np.isclose(wavelength, reference_wavelength)
                else "identity_initialization",
            }
        self._datastore.save_chromatic_affine_transforms_zyx_um(
            {
                "reference_wavelength_um": float(reference_wavelength),
                "voxel_size_zyx_um": [
                    float(v) for v in self._datastore.voxel_size_zyx_um
                ],
                "estimator": "identity_initialization_for_iterative_decoding",
                "pair_constraints": 0,
                "contributing_transcripts": 0,
                "channels": channels,
            }
        )

    def _load_bit_data(
        self,
        feature_predictor_threshold: float | None = 0.1,
        gpu_id: int = 0,
    ) -> None:
        """Load prediction-weighted readout data for all bits in the tile.

        Parameters
        ----------
        feature_predictor_threshold : float, default 0.1
            Legacy argument kept for API compatibility. The loaded image is
            weighted by the feature-predictor image rather than thresholded.
        gpu_id : int, default=0
            CUDA device ID used for decode-time warping.
        """

        if self._verbose > 1:
            print("load raw data")
            iterable_bits = tqdm(
                self._datastore.bit_ids[0 : self._n_merfish_bits],
                desc="bit",
                leave=False,
            )
        elif self._verbose >= 1:
            iterable_bits = tqdm(
                self._datastore.bit_ids[0 : self._n_merfish_bits],
                desc="loading",
                leave=False,
            )
        else:
            iterable_bits = self._datastore.bit_ids[0 : self._n_merfish_bits]

        images = []
        self._em_wvl = []
        for bit_id in iterable_bits:
            decon_image = self._datastore.load_local_registered_image(
                tile=self._tile_idx,
                bit=bit_id,
            )
            feature_predictor_image = (
                self._datastore.load_local_feature_predictor_image(
                    tile=self._tile_idx,
                    bit=bit_id,
                )
            )

            feature_predictor_array = feature_predictor_image.result()
            decon_array = decon_image.result()
            _ex_wvl, em_wvl = self._datastore.load_local_wavelengths_um(
                tile=self._tile_idx,
                bit=bit_id,
            )
            prediction_weighted = np.asarray(
                decon_array, dtype=np.float32
            ) * np.asarray(feature_predictor_array, dtype=np.float32)
            registered_data = warp_bit_image_to_reference(
                prediction_weighted,
                datastore=self._datastore,
                tile=self._tile_idx,
                bit_id=bit_id,
                emission_wavelength_um=em_wvl,
                gpu_id=gpu_id,
            )
            images.append(registered_data[self._z_slice, :, :])
            del feature_predictor_array, decon_array
            self._em_wvl.append(em_wvl)

        self._image_data = np.stack(images, axis=0)
        if self._decode_mode == "3d" and self._image_data.shape[1] < 2:
            raise ValueError(
                "decode_mode='3d' requires at least two z planes after applying "
                "z_range and zstride_level."
            )
        voxel_size_zyx_um = self._datastore.voxel_size_zyx_um
        self._pixel_size = voxel_size_zyx_um[1]
        self._axial_step = voxel_size_zyx_um[0]

        stage_metadata = self._datastore.load_local_stage_position_zyx_um(
            tile=self._tile_idx, round=0
        )
        stage_origin = None
        camera_to_stage_affine = np.eye(4, dtype=np.float32)
        if stage_metadata is not None:
            stage_origin, camera_to_stage_affine = stage_metadata
            stage_origin = np.asarray(stage_origin, dtype=np.float32)
            camera_to_stage_affine = np.asarray(
                camera_to_stage_affine, dtype=np.float32
            )

        affine, origin, spacing = self._datastore.load_global_coord_xforms_um(
            tile=self._tile_idx
        )
        if affine is None or origin is None or spacing is None:
            if self._is_3D:
                affine = np.eye(4)
                origin = (
                    stage_origin
                    if stage_origin is not None
                    else np.zeros(3, dtype=np.float32)
                )
                spacing = self._datastore.voxel_size_zyx_um
            else:
                affine = np.eye(4)
                if stage_origin is None:
                    origin = np.zeros(3, dtype=np.float32)
                elif stage_origin.size == 2:
                    origin = np.asarray(
                        [0, stage_origin[0], stage_origin[1]], dtype=np.float32
                    )
                else:
                    origin = stage_origin
                spacing = self._datastore.voxel_size_zyx_um

        self._affine = np.asarray(affine, dtype=np.float32)
        self._origin = np.asarray(origin, dtype=np.float32)
        self._spacing = np.asarray(spacing, dtype=np.float32)
        self._camera_to_stage_affine = camera_to_stage_affine

        del images
        gc.collect()

    def _lowpass_image(
        self,
        image: cp.ndarray,
        sigma: Sequence[float] | None,
    ) -> cp.ndarray:
        """
        Apply lowpass filtering to one prediction-weighted bit image.

        Parameters
        ----------
        image : cupy.ndarray
            Prediction-weighted image in Z, Y, X order.
        sigma : Sequence[float] or None
            Lowpass sigma in the sampled image coordinates. If None or any
            element is zero, the image is returned unchanged.

        Returns
        -------
        cupy.ndarray
            Filtered image. No post-filter intensity rescaling is applied.
        """

        if sigma is None or np.any(np.asarray(sigma, dtype=float) == 0):
            return image
        if self._is_3D:
            return gaussian_filter(image, sigma=sigma)

        filtered = cp.empty_like(image)
        for z_idx in range(image.shape[0]):
            filtered[z_idx, :, :] = gaussian_filter(
                image[z_idx, :, :],
                sigma=(sigma[1], sigma[2]),
            )
        return filtered

    def _lp_filter(
        self,
        gpu_id: int = 0,
        sigma: Sequence[float] = DEFAULT_DECODE_LOWPASS_SIGMA,
    ) -> None:
        """Apply low-pass filter to the raw data.

        Parameters
        ----------
        gpu_id: int, default = 0
            GPU identifier
        sigma : Sequence[int, int, int], default [3,1,1]
            Sigma values for Gaussian filter.
        """

        with cp.cuda.Device(gpu_id):
            self._image_data_lp = self._image_data.copy()

            if self._verbose > 1:
                print("lowpass filter")
                iterable_lp = tqdm(
                    range(self._image_data_lp.shape[0]), desc="bit", leave=False
                )
            elif self._verbose >= 1:
                iterable_lp = tqdm(
                    range(self._image_data_lp.shape[0]), desc="lowpass", leave=False
                )
            else:
                iterable_lp = range(self._image_data_lp.shape[0])

            for i in iterable_lp:
                image_data_cp = cp.asarray(self._image_data[i, :], dtype=cp.float32)
                filtered_cp = self._lowpass_image(image_data_cp, sigma=sigma)
                self._image_data_lp[i, :, :, :] = cp.asnumpy(filtered_cp)
                del filtered_cp

            self._filter_type = "lp"

            del image_data_cp
            del self._image_data
            gc.collect()
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def _effective_lowpass_sigma(
        self, sigma: Sequence[float] | None
    ) -> tuple[float, float, float] | None:
        """
        Resolve lowpass sigma in the sampled decoding volume.

        Public decode arguments are expressed in source-image voxel units. For
        strided 3D decoding, the sampled z axis has fewer planes, so the z sigma
        must be divided by the stride to preserve the same physical smoothing.
        """

        if sigma is None:
            return None
        sigma_zyx = tuple(float(v) for v in sigma)
        if len(sigma_zyx) != 3:
            raise ValueError("lowpass_sigma must contain three values: z, y, x.")
        if self._is_3D and self._zstride > 1:
            sigma_zyx = (sigma_zyx[0] / float(self._zstride), *sigma_zyx[1:])
        return sigma_zyx

    def _default_minimum_pixels(self) -> float:
        """Return the production connected-component size threshold."""

        if not self._is_3D:
            return DEFAULT_2D_MINIMUM_PIXELS
        if self._zstride > 1:
            return DEFAULT_ZSTRIDE_3D_MINIMUM_PIXELS
        return DEFAULT_3D_MINIMUM_PIXELS

    @staticmethod
    def _fit_affine_zyx_um(
        source_zyx_um: np.ndarray,
        target_zyx_um: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        min_pairs: int,
        residual_threshold_um: float = 0.35,
        max_iterations: int = 6,
        scale_regularization: float = 0.0,
    ) -> tuple[np.ndarray | None, dict[str, float | int | str | list[float]]]:
        """
        Fit a robust chromatic affine from decoded transcript centroids.

        The iterative chromatic estimator uses decoded RNA centroids, not bead
        calibration volumes. These centroids span the lateral field of view
        well, but usually span only a thin axial range. A full 3D affine is
        therefore poorly conditioned and can turn centroid noise into axial
        scale or shear. This fit estimates the chromatic model supported by
        decoded RNA features: one shared radial Y/X scale, Y/X translations,
        and a Z translation, returned as a 4x4 Z, Y, X physical transform.

        Parameters
        ----------
        source_zyx_um : numpy.ndarray
            Source points in physical Z, Y, X microns.
        target_zyx_um : numpy.ndarray
            Target points in physical Z, Y, X microns.
        weights : numpy.ndarray or None, optional
            Nonnegative per-pair confidence weights. Larger values increase the
            influence of bright, well-supported transcript centroids.
        min_pairs : int
            Minimum number of retained point correspondences.
        residual_threshold_um : float, default=0.35
            Residual threshold used for iterative outlier rejection.
        max_iterations : int, default=6
            Number of robust refit iterations.
        scale_regularization : float, default=0.0
            Penalty against lateral scale changes away from identity. The
            default leaves radial scale unconstrained.

        Returns
        -------
        tuple[numpy.ndarray or None, dict]
            Constrained affine mapping source coordinates to target coordinates
            and fit diagnostics. The affine is None when the point set is not
            sufficient for the lateral affine model.
        """

        source = np.asarray(source_zyx_um, dtype=np.float64)
        target = np.asarray(target_zyx_um, dtype=np.float64)
        diagnostics: dict[str, float | int | str | list[float]] = {
            "input_pairs": int(source.shape[0]),
            "used_pairs": 0,
            "median_residual_um": np.nan,
            "p95_residual_um": np.nan,
            "source_extent_zyx_um": [0.0, 0.0, 0.0],
            "model": "z_translation_yx_radial_scale",
            "status": "insufficient_pairs",
        }
        if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
            diagnostics["status"] = "invalid_point_shape"
            return None, diagnostics
        if source.shape[0] < max(3, int(min_pairs)):
            return None, diagnostics
        if weights is None:
            weights_arr = np.ones(source.shape[0], dtype=np.float64)
        else:
            weights_arr = np.asarray(weights, dtype=np.float64)
            if weights_arr.shape != (source.shape[0],):
                diagnostics["status"] = "invalid_weight_shape"
                return None, diagnostics
            weights_arr = np.nan_to_num(weights_arr, nan=0.0, posinf=0.0, neginf=0.0)
            weights_arr = np.maximum(weights_arr, 0.0)
            if not np.any(weights_arr > 0):
                diagnostics["status"] = "invalid_weights"
                return None, diagnostics
            weights_arr = weights_arr / np.median(weights_arr[weights_arr > 0])

        source_extent = np.ptp(source, axis=0)
        diagnostics["source_extent_zyx_um"] = [float(v) for v in source_extent]
        if np.linalg.matrix_rank(source[:, 1:3] - np.mean(source[:, 1:3], axis=0)) < 2:
            diagnostics["status"] = "insufficient_lateral_spatial_rank"
            return None, diagnostics

        def solve_yx_radial_scale(
            source_yx: np.ndarray,
            target_yx: np.ndarray,
            fit_weights: np.ndarray,
        ) -> tuple[float, float, float]:
            design_y = np.column_stack(
                [
                    source_yx[:, 0],
                    np.ones(source_yx.shape[0], dtype=np.float64),
                    np.zeros(source_yx.shape[0], dtype=np.float64),
                ]
            )
            design_x = np.column_stack(
                [
                    source_yx[:, 1],
                    np.zeros(source_yx.shape[0], dtype=np.float64),
                    np.ones(source_yx.shape[0], dtype=np.float64),
                ]
            )
            design = np.vstack([design_y, design_x])
            target_values = np.concatenate([target_yx[:, 0], target_yx[:, 1]])
            sqrt_weights = np.sqrt(np.maximum(fit_weights, 1e-12))
            stacked_weights = np.concatenate([sqrt_weights, sqrt_weights])
            weighted_design = design * stacked_weights[:, np.newaxis]
            weighted_target = target_values * stacked_weights

            if scale_regularization > 0:
                penalty = np.sqrt(float(scale_regularization))
                weighted_design = np.vstack(
                    [weighted_design, np.asarray([[penalty, 0.0, 0.0]])]
                )
                weighted_target = np.concatenate(
                    [weighted_target, np.asarray([penalty])]
                )

            solution, *_ = np.linalg.lstsq(
                weighted_design,
                weighted_target,
                rcond=None,
            )
            return float(solution[0]), float(solution[1]), float(solution[2])

        def robust_weighted_z_translation(
            z_offsets: np.ndarray,
            fit_weights: np.ndarray,
        ) -> float:
            finite = np.isfinite(z_offsets) & np.isfinite(fit_weights)
            finite &= fit_weights > 0
            if not np.any(finite):
                return 0.0
            offsets = z_offsets[finite]
            local_weights = fit_weights[finite]
            center = float(np.median(offsets))
            spread = float(np.median(np.abs(offsets - center)))
            if spread > 0:
                keep_offsets = np.abs(offsets - center) <= 3.0 * 1.4826 * spread
                if np.any(keep_offsets):
                    offsets = offsets[keep_offsets]
                    local_weights = local_weights[keep_offsets]
            return float(np.average(offsets, weights=local_weights))

        rng = np.random.default_rng(1729)
        keep = np.ones(source.shape[0], dtype=bool)
        best_keep = None
        best_score = -1
        best_weighted_score = -1.0
        best_median_residual = np.inf
        max_ransac_iterations = min(512, max(64, source.shape[0]))
        sample_probability = weights_arr / np.sum(weights_arr)
        for _iteration in range(max_ransac_iterations):
            sample_indices = rng.choice(
                source.shape[0],
                size=3,
                replace=False,
                p=sample_probability,
            )
            sample_source = source[sample_indices]
            if (
                np.linalg.matrix_rank(
                    sample_source[:, 1:3] - np.mean(sample_source[:, 1:3], axis=0)
                )
                < 2
            ):
                continue

            sample_scale, sample_y_translation, sample_x_translation = (
                solve_yx_radial_scale(
                    sample_source[:, 1:3],
                    target[sample_indices, 1:3],
                    weights_arr[sample_indices],
                )
            )
            sample_affine = np.eye(4, dtype=np.float64)
            sample_affine[0, 3] = robust_weighted_z_translation(
                target[sample_indices, 0] - sample_source[:, 0],
                weights_arr[sample_indices],
            )
            sample_affine[1, 1] = sample_scale
            sample_affine[1, 3] = sample_y_translation
            sample_affine[2, 2] = sample_scale
            sample_affine[2, 3] = sample_x_translation

            predicted = (
                np.concatenate(
                    [source, np.ones((source.shape[0], 1), dtype=np.float64)],
                    axis=1,
                )
                @ sample_affine.T
            )[:, :3]
            residuals = np.linalg.norm(predicted - target, axis=1)
            sample_keep = residuals <= float(residual_threshold_um)
            sample_score = int(np.sum(sample_keep))
            if sample_score < max(3, int(min_pairs)):
                continue
            sample_weighted_score = float(np.sum(weights_arr[sample_keep]))
            sample_median_residual = float(np.median(residuals[sample_keep]))
            if (
                sample_score > best_score
                or (
                    sample_score == best_score
                    and sample_weighted_score > best_weighted_score
                )
                or (
                    sample_score == best_score
                    and np.isclose(sample_weighted_score, best_weighted_score)
                    and sample_median_residual < best_median_residual
                )
            ):
                best_keep = sample_keep
                best_score = sample_score
                best_weighted_score = sample_weighted_score
                best_median_residual = sample_median_residual

        if best_keep is not None:
            keep = best_keep
        affine = np.eye(4, dtype=np.float64)
        for _iteration in range(max(1, int(max_iterations))):
            scale, y_translation, x_translation = solve_yx_radial_scale(
                source[keep, 1:3],
                target[keep, 1:3],
                weights_arr[keep],
            )
            z_offsets = target[keep, 0] - source[keep, 0]
            z_translation = robust_weighted_z_translation(z_offsets, weights_arr[keep])
            affine = np.eye(4, dtype=np.float64)
            affine[0, 3] = z_translation
            affine[1, 1] = scale
            affine[1, 3] = y_translation
            affine[2, 2] = scale
            affine[2, 3] = x_translation

            predicted = (
                np.concatenate(
                    [source, np.ones((source.shape[0], 1), dtype=np.float64)],
                    axis=1,
                )
                @ affine.T
            )[:, :3]
            residuals = np.linalg.norm(predicted - target, axis=1)
            next_keep = residuals <= float(residual_threshold_um)
            if np.sum(next_keep) < max(3, int(min_pairs)):
                break
            if np.array_equal(next_keep, keep):
                keep = next_keep
                break
            keep = next_keep

        predicted = (
            np.concatenate(
                [source, np.ones((source.shape[0], 1), dtype=np.float64)],
                axis=1,
            )
            @ affine.T
        )[:, :3]
        residuals = np.linalg.norm(predicted - target, axis=1)
        kept_residuals = residuals[keep]
        if kept_residuals.size < max(3, int(min_pairs)):
            diagnostics["status"] = "too_few_inliers"
            diagnostics["used_pairs"] = int(kept_residuals.size)
            return None, diagnostics

        diagnostics.update(
            {
                "used_pairs": int(kept_residuals.size),
                "median_residual_um": float(np.median(kept_residuals)),
                "p95_residual_um": float(np.percentile(kept_residuals, 95)),
                "status": "ok",
            }
        )
        return affine.astype(np.float32), diagnostics

    @staticmethod
    def _scale_pixel_traces(
        pixel_traces: np.ndarray | cp.ndarray,
        background_vector: np.ndarray | cp.ndarray,
        normalization_vector: np.ndarray | cp.ndarray,
        merfish_bits: int = 16,
        gpu_id: int = 0,
    ) -> cp.ndarray:
        """Scale pixel traces using background and normalization vectors.

        Parameters
        ----------
        pixel_traces : Union[np.ndarray, cp.ndarray]
            Pixel traces to scale.
        background_vector : Union[np.ndarray, cp.ndarray]
            Background vector.
        normalization_vector : Union[np.ndarray, cp.ndarray]
            Normalization vector.
        merfish_bits : int, default = 16
            Number of MERFISH bits. Default 16. Assume MERFISH bits are [0, merfish_bits].
        gpu_id: int, default = 0
            GPU identifier

        Returns
        -------
        scaled_traces : cp.ndarray
            Scaled pixel traces.
        """

        with cp.cuda.Device(gpu_id):
            if isinstance(pixel_traces, np.ndarray):
                pixel_traces = cp.asarray(pixel_traces, dtype=cp.float32)
            if isinstance(background_vector, np.ndarray):
                background_vector = cp.asarray(background_vector, dtype=cp.float32)
            if isinstance(normalization_vector, np.ndarray):
                normalization_vector = cp.asarray(
                    normalization_vector, dtype=cp.float32
                )

            background_vector = background_vector[0:merfish_bits]
            normalization_vector = normalization_vector[0:merfish_bits]

            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

            return (
                pixel_traces - background_vector[:, cp.newaxis]
            ) / normalization_vector[:, cp.newaxis]

    @staticmethod
    def _clip_pixel_traces(
        pixel_traces: np.ndarray | cp.ndarray,
        clip_lower: float = 0.0,
        clip_upper: float = 1.0,
        gpu_id: int = 0,
    ) -> cp.ndarray:
        """Clip pixel traces to a range.

        Parameters
        ----------
        pixel_traces : Union[np.ndarray, cp.ndarray]
            Pixel traces to clip.
        clip_lower : float, default 0.0
            clip lower bound.
        clip_upper : float, default 1.0
            clip upper bound.
        gpu_id: int, default = 0
            GPU identifier

        Returns
        -------
        clipped_traces : cp.ndarray
            Clipped pixel traces.
        """
        with cp.cuda.Device(gpu_id):
            clipped = cp.clip(pixel_traces, clip_lower, clip_upper, pixel_traces)
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            return clipped

    @staticmethod
    def _normalize_pixel_traces(
        pixel_traces: np.ndarray | cp.ndarray, gpu_id: int = 0
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Normalize pixel traces by L2 norm.

        Parameters
        ----------
        pixel_traces : Union[np.ndarray, cp.ndarray]
            Pixel traces to normalize.
        gpu_id: int, default = 0
            GPU identifier

        Returns
        -------
        normalized_traces : cp.ndarray
            Normalized pixel traces.
        norms : cp.ndarray
            L2 norms of pixel traces.
        """

        with cp.cuda.Device(gpu_id):
            if isinstance(pixel_traces, np.ndarray):
                pixel_traces = cp.asarray(pixel_traces, dtype=cp.float32)

            norms = cp.linalg.norm(pixel_traces, axis=0)
            norms = cp.where(norms == 0, np.inf, norms)
            normalized_traces = pixel_traces / norms
            norms = cp.where(norms == np.inf, -1, norms)

            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

            return normalized_traces, norms

    @staticmethod
    def _calculate_distances(
        pixel_traces: np.ndarray | cp.ndarray,
        codebook_matrix: np.ndarray | cp.ndarray,
        gpu_id: int = 0,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Calculate distances between pixel traces and codebook matrix.

        Parameters
        ----------
        pixel_traces : Union[np.ndarray, cp.ndarray]
            Pixel traces.
        codebook_matrix : Union[np.ndarray, cp.ndarray]
            Codebook matrix.
        gpu_id: int, default = 0
            GPU identifier

        Returns
        -------
        min_distances : cp.ndarray
            Minimum distances.
        min_indices : cp.ndarray
            Minimum indices.
        """

        with cp.cuda.Device(gpu_id):
            if isinstance(pixel_traces, np.ndarray):
                pixel_traces = cp.asarray(pixel_traces, dtype=cp.float32)
            if isinstance(codebook_matrix, np.ndarray):
                codebook_matrix = cp.asarray(codebook_matrix, dtype=cp.float32)

            distances = cp.ascontiguousarray(
                cp.zeros(
                    (pixel_traces.shape[1], codebook_matrix.shape[0]), dtype=cp.float32
                )
            )
            pairwise_distance(
                cp.ascontiguousarray(pixel_traces.T),
                cp.ascontiguousarray(codebook_matrix),
                metric="euclidean",
                out=distances,
            )

            min_indices = cp.argmin(distances, axis=1)
            min_distances = cp.min(distances, axis=1)

            del pixel_traces, codebook_matrix
            gc.collect()
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

            return min_distances, min_indices

    def _decode_pixels(
        self,
        magnitude_threshold: Sequence[float] = (1.1, 2.0),
        gpu_id: int = 0,
    ) -> None:
        """Decode pixels using the decoding matrix.

        Parameters
        ----------
        magnitude_threshold : Sequence[float], default (1.1, 2.0).
            Magnitude threshold for decoding.
        gpu_id: int, default = 0
            GPU identifier
        """

        with cp.cuda.Device(gpu_id):
            if self._filter_type == "lp":
                original_shape = self._image_data_lp.shape
                self._decoded_image = np.zeros((original_shape[1:]), dtype=np.int16)
                self._magnitude_image = np.zeros((original_shape[1:]), dtype=np.float16)
                self._scaled_pixel_images = np.zeros((original_shape), dtype=np.float16)
                self._distance_image = np.zeros((original_shape[1:]), dtype=np.float16)
            else:
                original_shape = self._image_data.shape
                self._decoded_image = np.zeros((original_shape[1:]), dtype=np.int16)
                self._magnitude_image = np.zeros((original_shape[1:]), dtype=np.float16)
                self._scaled_pixel_images = np.zeros((original_shape), dtype=np.float16)
                self._distance_image = np.zeros((original_shape[1:]), dtype=np.float16)

            if self._verbose > 1:
                print("decode pixels")
                iterable_z = tqdm(range(original_shape[1]), desc="z", leave=False)
            elif self._verbose >= 1:
                iterable_z = tqdm(
                    range(original_shape[1]), desc="decoding", leave=False
                )
            else:
                iterable_z = range(original_shape[1])

            for z_idx in iterable_z:
                if self._filter_type == "lp":
                    z_plane_shape = self._image_data_lp[:, z_idx, :].shape
                    scaled_pixel_traces = (
                        cp.asarray(self._image_data_lp[:, z_idx, :])
                        .reshape(self._n_merfish_bits, -1)
                        .astype(cp.float32)
                    )
                else:
                    z_plane_shape = self._image_data[:, z_idx, :].shape
                    scaled_pixel_traces = (
                        cp.asarray(self._image_data[:, z_idx, :])
                        .reshape(self._n_merfish_bits, -1)
                        .astype(cp.float32)
                    )

                if self._iterative_normalization_loaded:
                    scaled_pixel_traces = self._scale_pixel_traces(
                        scaled_pixel_traces,
                        self._iterative_background_vector,
                        self._iterative_normalization_vector,
                        self._n_merfish_bits,
                        gpu_id=gpu_id,
                    )
                elif self._global_normalization_loaded:
                    scaled_pixel_traces = self._scale_pixel_traces(
                        scaled_pixel_traces,
                        self._global_background_vector,
                        self._global_normalization_vector,
                        self._n_merfish_bits,
                        gpu_id=gpu_id,
                    )

                scaled_pixel_traces = self._clip_pixel_traces(
                    scaled_pixel_traces, gpu_id=gpu_id
                )
                normalized_pixel_traces, pixel_magnitude_trace = (
                    self._normalize_pixel_traces(scaled_pixel_traces, gpu_id=gpu_id)
                )
                distance_trace, codebook_index_trace = self._calculate_distances(
                    normalized_pixel_traces, self._decoding_matrix, gpu_id=gpu_id
                )

                del normalized_pixel_traces
                gc.collect()
                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

                decoded_trace = cp.full(distance_trace.shape[0], -1, dtype=cp.int16)
                mask_trace = distance_trace <= self._pixel_assignment_threshold
                decoded_trace[mask_trace] = codebook_index_trace[mask_trace]
                decoded_trace[pixel_magnitude_trace < magnitude_threshold[0]] = -1
                decoded_trace[pixel_magnitude_trace > magnitude_threshold[1]] = -1

                self._decoded_image[z_idx, :] = cp.asnumpy(
                    cp.reshape(cp.round(decoded_trace, 5), z_plane_shape[1:])
                )
                self._magnitude_image[z_idx, :] = cp.asnumpy(
                    cp.reshape(cp.round(pixel_magnitude_trace, 5), z_plane_shape[1:])
                )
                self._scaled_pixel_images[:, z_idx, :] = cp.asnumpy(
                    cp.reshape(cp.round(scaled_pixel_traces, 5), z_plane_shape)
                )
                self._distance_image[z_idx, :] = cp.asnumpy(
                    cp.reshape(cp.round(distance_trace, 5), z_plane_shape[1:])
                )

                del (
                    decoded_trace,
                    pixel_magnitude_trace,
                    scaled_pixel_traces,
                    distance_trace,
                )
                gc.collect()
                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

    @staticmethod
    def _warp_pixel(
        pixel_space_point: np.ndarray,
        spacing: np.ndarray,
        origin: np.ndarray,
        affine: np.ndarray,
        camera_to_stage_affine: np.ndarray | None = None,
    ) -> np.ndarray:
        """Warp pixel space point to physical space point.

        Parameters
        ----------
        pixel_space_point : np.ndarray
            Pixel space point.
        spacing : np.ndarray
            Spacing.
        origin : np.ndarray
            Origin.
        affine : np.ndarray
            Affine transformation matrix.
        camera_to_stage_affine : np.ndarray | None, optional
            Camera-to-stage affine transform from datastore stage metadata.

        Returns
        -------
        registered_space_point : np.ndarray
            Registered space point.
        """

        physical_space_point = pixel_space_point * spacing + origin
        if camera_to_stage_affine is not None:
            physical_space_point = (
                np.asarray(camera_to_stage_affine)
                @ np.array([*list(physical_space_point), 1])
            )[:-1]
        registered_space_point = (
            np.array(affine) @ np.array([*list(physical_space_point), 1])
        )[:-1]

        return registered_space_point

    def _decoded_z_to_source_z(self, decoded_z: pd.Series | np.ndarray) -> pd.Series:
        """
        Map z coordinates from the strided decoding volume back to source planes.

        Parameters
        ----------
        decoded_z : pandas.Series or numpy.ndarray
            Z coordinate in the decoded image.

        Returns
        -------
        pandas.Series
            Z coordinate in the source image.
        """

        return float(self._z_range[0]) + decoded_z * float(self._zstride)

    def _add_on_bit_weighted_centroids(
        self,
        df_barcode: pd.DataFrame,
        codewords_label_image: cp.ndarray,
        intensity_image: cp.ndarray,
        on_bits_1based: np.ndarray,
    ) -> pd.DataFrame:
        """
        Add per-on-bit intensity-weighted centroids for chromatic estimation.

        Parameters
        ----------
        df_barcode : pandas.DataFrame
            Decoded barcode table with ``label`` and centroid columns.
        codewords_label_image : cupy.ndarray
            Connected-component label image in decoded Z, Y, X coordinates.
        intensity_image : cupy.ndarray
            Intensity image in Z, Y, X, bit order.
        on_bits_1based : numpy.ndarray
            On-bit indices for each barcode row, 1-based.

        Returns
        -------
        pandas.DataFrame
            Barcode table with sparse per-bit center and intensity-support
            columns.
        """

        if df_barcode.empty:
            return df_barcode

        extra_columns = {
            f"bit{bit_idx:02d}_{suffix}": np.full(len(df_barcode), np.nan)
            for bit_idx in range(1, self._n_merfish_bits + 1)
            for suffix in (
                "center_z",
                "center_y",
                "center_x",
                "intensity_sum",
                "intensity_peak",
                "voxel_count",
            )
        }
        extra_df = pd.DataFrame(extra_columns, index=df_barcode.index)
        region_labels = df_barcode["label"].to_numpy(dtype=np.int64)
        fallback_centers = df_barcode[["z", "y", "x"]].to_numpy(dtype=np.float64)
        labels_cp = codewords_label_image.astype(cp.int32, copy=False)
        labels_flat = labels_cp.ravel()
        max_label = int(cp.max(labels_cp).get()) if labels_cp.size else 0
        if max_label < 1:
            return pd.concat([df_barcode, extra_df], axis=1)
        label_lookup = cp.asarray(region_labels, dtype=cp.int32)
        label_lookup = cp.clip(label_lookup, 0, max_label)
        minlength = max_label + 1
        area_by_label = cp.bincount(labels_flat, minlength=minlength).astype(
            cp.float32,
            copy=False,
        )
        z_coords = cp.arange(labels_cp.shape[0], dtype=cp.float32)[:, None, None]
        y_coords = cp.arange(labels_cp.shape[1], dtype=cp.float32)[None, :, None]
        x_coords = cp.arange(labels_cp.shape[2], dtype=cp.float32)[None, None, :]

        for bit_idx in range(1, self._n_merfish_bits + 1):
            active_rows = np.flatnonzero(np.any(on_bits_1based == bit_idx, axis=1))
            if active_rows.size == 0:
                continue

            bit_image = cp.maximum(
                intensity_image[..., bit_idx - 1],
                cp.float32(0),
            )
            weights_flat = bit_image.ravel()
            weight_by_label = cp.bincount(
                labels_flat,
                weights=weights_flat,
                minlength=minlength,
            )
            z_sum_by_label = cp.bincount(
                labels_flat,
                weights=(bit_image * z_coords).ravel(),
                minlength=minlength,
            )
            y_sum_by_label = cp.bincount(
                labels_flat,
                weights=(bit_image * y_coords).ravel(),
                minlength=minlength,
            )
            x_sum_by_label = cp.bincount(
                labels_flat,
                weights=(bit_image * x_coords).ravel(),
                minlength=minlength,
            )
            peak_by_label = cp.zeros(minlength, dtype=cp.float32)
            cp.maximum.at(peak_by_label, labels_flat, weights_flat)

            active_labels = label_lookup[active_rows]
            weight_sum_cp = weight_by_label[active_labels]
            centers_cp = cp.column_stack(
                (
                    z_sum_by_label[active_labels]
                    / cp.maximum(weight_sum_cp, cp.float32(1e-6)),
                    y_sum_by_label[active_labels]
                    / cp.maximum(weight_sum_cp, cp.float32(1e-6)),
                    x_sum_by_label[active_labels]
                    / cp.maximum(weight_sum_cp, cp.float32(1e-6)),
                )
            )
            centers = cp.asnumpy(centers_cp).astype(np.float64, copy=False)
            fallback = fallback_centers[active_rows]
            weight_sum = cp.asnumpy(weight_sum_cp).astype(np.float64, copy=False)
            invalid_centers = (~np.all(np.isfinite(centers), axis=1)) | (
                weight_sum <= 0
            )
            centers[invalid_centers] = fallback[invalid_centers]

            if self._z_crop or self._zstride != 1:
                centers[:, 0] = self._decoded_z_to_source_z(centers[:, 0])

            area = cp.asnumpy(area_by_label[active_labels]).astype(
                np.float64,
                copy=False,
            )
            intensity_peak = cp.asnumpy(peak_by_label[active_labels]).astype(
                np.float64,
                copy=False,
            )
            missing = (~np.isfinite(weight_sum)) | (weight_sum <= 0)
            area[missing] = 0.0
            intensity_peak[~np.isfinite(intensity_peak)] = 0.0
            weight_sum[missing] = 0.0

            extra_df.loc[active_rows, f"bit{bit_idx:02d}_center_z"] = centers[:, 0]
            extra_df.loc[active_rows, f"bit{bit_idx:02d}_center_y"] = centers[:, 1]
            extra_df.loc[active_rows, f"bit{bit_idx:02d}_center_x"] = centers[:, 2]
            extra_df.loc[active_rows, f"bit{bit_idx:02d}_intensity_sum"] = weight_sum
            extra_df.loc[active_rows, f"bit{bit_idx:02d}_intensity_peak"] = (
                intensity_peak
            )
            extra_df.loc[active_rows, f"bit{bit_idx:02d}_voxel_count"] = area

        return pd.concat([df_barcode, extra_df], axis=1)

    def _extract_barcodes(
        self, minimum_pixels: int = 3, maximum_pixels: int = 500, gpu_id: int = 0
    ) -> None:
        """Extract connected-component transcripts from the decoded image.

        Parameters
        ----------
        minimum_pixels : int, default 3
            Minimum number of pixels for a barcode.
        maximum_pixels : int, default 500
            Maximum number of pixels for a barcode.
        gpu_id: int, default = 0
            GPU identifier

        Notes
        -----
        After connected-component extraction, each transcript is annotated with
        ``distance_min`` from the voxelwise distance image and filtered locally
        against the exact transcript-distance threshold before saving.
        """

        self._df_barcodes = pd.DataFrame()

        with cp.cuda.Device(gpu_id):
            if self._verbose > 1:
                print("extract barcodes")

            decoded_image_cp = cp.asarray(self._decoded_image, dtype=cp.int16)
            if self._optimize_normalization_weights:
                if self._filter_type == "lp":
                    intensity_image = self._image_data_lp.transpose(1, 2, 3, 0)
                else:
                    intensity_image = self._image_data.transpose(1, 2, 3, 0)
            else:
                intensity_image = self._scaled_pixel_images.transpose(1, 2, 3, 0)

            if self._verbose > 1:
                print("")
                print("label image")
            if self._is_3D:
                # create label image for all valid codewords at once
                codewords_label_image_cp = label(
                    decoded_image_cp, background=-1, connectivity=3
                ).astype(cp.int32, copy=False)
            else:
                z = decoded_image_cp.shape[0]
                codewords_label_image_cp = cp.zeros(
                    decoded_image_cp.shape, dtype=cp.int32
                )

                offset = 0
                for zi in range(z):
                    lab2d, n = label(
                        decoded_image_cp[zi],
                        background=-1,
                        connectivity=2,
                        return_num=True,
                    )
                    lab2d = lab2d.astype(cp.int32, copy=False)

                    # make globally unique, keep background as 0
                    if n:
                        codewords_label_image_cp[zi] = cp.where(
                            lab2d != 0, lab2d + offset, 0
                        )
                        offset += int(n)

            if self._verbose > 1:
                print("remove large")
            pixel_counts = cp.bincount(codewords_label_image_cp.ravel())
            large_labels = cp.where(pixel_counts > maximum_pixels)[0]
            large_labels = large_labels[large_labels != 0]  # exclude background

            if large_labels.size:
                codewords_label_image_cp[
                    cp.isin(codewords_label_image_cp, large_labels)
                ] = 0

            if self._verbose > 1:
                print("remove small")
            codewords_label_image_cp = remove_small_objects(
                codewords_label_image_cp, max_size=max(int(minimum_pixels) - 1, 0)
            )

            props_distance = gpu_regionprops_table(
                codewords_label_image_cp,
                intensity_image=cp.asarray(self._distance_image, dtype=cp.float32),
                properties=["label", "intensity_min", "area"],
            )
            df_distance = pd.DataFrame(
                {
                    key: (
                        cp.asnumpy(value)
                        if isinstance(value, cp.ndarray)
                        else np.asarray(value)
                    )
                    for key, value in props_distance.items()
                }
            )
            if not df_distance.empty:
                df_distance = df_distance[df_distance["area"] > 0].copy()
                df_distance = df_distance.sort_values("label").reset_index(drop=True)
                df_distance = df_distance.rename(
                    columns={"intensity_min": "distance_min"}
                )

            # Move labels to CPU for the existing regionprops path. Keep the
            # GPU labels alive for the optional iterative chromatic estimates.
            codewords_label_image = cp.asnumpy(codewords_label_image_cp)

            del decoded_image_cp
            gc.collect()
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()

            if self._verbose > 1:
                print("regionprops table")

            props = regionprops_table(
                codewords_label_image,
                intensity_image=intensity_image,
                properties=[
                    "label",
                    "area",
                    "centroid",
                    "intensity_mean",
                    "inertia_tensor_eigvals",
                ],
            )
            df_barcode = pd.DataFrame(props)

            props_magnitude = regionprops_table(
                codewords_label_image,
                intensity_image=self._magnitude_image,
                properties=["label", "intensity_mean"],
            )
            df_magnitude = pd.DataFrame(props_magnitude)

            if not df_distance.empty:
                df_barcode = df_barcode.merge(
                    df_distance[["label", "distance_min"]],
                    on="label",
                    how="left",
                )

            if not df_magnitude.empty:
                df_magnitude = df_magnitude.rename(
                    columns={"intensity_mean": "magnitude_mean"}
                )
                df_barcode = df_barcode.merge(
                    df_magnitude[["label", "magnitude_mean"]],
                    on="label",
                    how="left",
                )

            df_barcode = df_barcode[df_barcode["area"] > 0.1].reset_index(drop=True)
            if "distance_min" not in df_barcode.columns:
                df_barcode["distance_min"] = np.nan

            labels_flat = codewords_label_image.ravel()
            decoded_flat = self._decoded_image.ravel()

            max_label = int(labels_flat.max())
            label_to_id = np.full(max_label + 1, -1, dtype=np.int16)

            order = np.argsort(labels_flat, kind="mergesort")
            labels_sorted = labels_flat[order]
            decoded_sorted = decoded_flat[order]

            uniq_labels, first_idx = np.unique(labels_sorted, return_index=True)
            label_to_id[uniq_labels] = decoded_sorted[first_idx]
            label_to_id[0] = -1

            region_labels = df_barcode["label"].to_numpy(dtype=np.int64)
            decoded_ids = label_to_id[region_labels].astype(np.int32, copy=False)
            df_barcode["decoded_id"] = decoded_ids
            df_barcode = df_barcode[df_barcode["decoded_id"] >= 0].reset_index(
                drop=True
            )

            # barcode_id is 1-based, matches old code
            df_barcode["barcode_id"] = df_barcode["decoded_id"].astype(np.int32) + 1
            df_barcode["gene_id"] = [
                self._gene_ids[x]
                for x in df_barcode["decoded_id"].to_numpy(dtype=np.int32)
            ]
            df_barcode["tile_idx"] = self._tile_idx

            # Precompute on-bits per codeword once (vectorized)
            # Assumes codebook_matrix is shape (n_codewords, n_bits), bool or 0/1
            codebook_bool = self._codebook_matrix.astype(bool, copy=False)
            _n_codewords, _n_bits = codebook_bool.shape

            # Exactly 4 on-bits per codeword after codebook validation.
            on_bits_0based = np.argsort(~codebook_bool, axis=1)[:, :4].astype(np.int32)
            on_bits_1based = on_bits_0based + 1
            sel = df_barcode["decoded_id"].to_numpy(dtype=np.int32)
            on_sel = on_bits_1based[sel]  # shape (n_regions, 4)

            df_barcode["on_bit_1"] = on_sel[:, 0]
            df_barcode["on_bit_2"] = on_sel[:, 1]
            df_barcode["on_bit_3"] = on_sel[:, 2]
            df_barcode["on_bit_4"] = on_sel[:, 3]
            n_on = 4

            df_barcode = df_barcode.rename(
                columns={"centroid-0": "z", "centroid-1": "y", "centroid-2": "x"}
            )
            if (
                self._optimize_normalization_weights
                and self._collect_chromatic_centroids
            ):
                df_barcode = self._add_on_bit_weighted_centroids(
                    df_barcode,
                    codewords_label_image_cp,
                    cp.asarray(intensity_image, dtype=cp.float32),
                    on_sel,
                )

            if self._z_crop or self._zstride != 1:
                df_barcode["z"] = self._decoded_z_to_source_z(df_barcode["z"])

            df_barcode["tile_z"] = np.round(df_barcode["z"], 0).astype(int)
            df_barcode["tile_y"] = np.round(df_barcode["y"], 0).astype(int)
            df_barcode["tile_x"] = np.round(df_barcode["x"], 0).astype(int)

            pts = df_barcode[["z", "y", "x"]].to_numpy()
            for pt_idx in range(pts.shape[0]):
                pts[pt_idx, :] = self._warp_pixel(
                    pts[pt_idx, :].copy(),
                    self._spacing,
                    self._origin,
                    self._affine,
                    self._camera_to_stage_affine,
                )

            df_barcode["global_z"] = np.round(pts[:, 0], 2)
            df_barcode["global_y"] = np.round(pts[:, 1], 2)
            df_barcode["global_x"] = np.round(pts[:, 2], 2)

            for i in range(1, self._n_merfish_bits + 1):
                src = f"intensity_mean-{i - 1}"
                dst = f"bit{i:02d}_mean_intensity"
                if src in df_barcode.columns:
                    df_barcode = df_barcode.rename(columns={src: dst})

            # Build a dense matrix of bit means: shape (n_regions, n_bits)
            bit_cols = [
                f"bit{i:02d}_mean_intensity" for i in range(1, self._n_merfish_bits + 1)
            ]
            bit_means = df_barcode[bit_cols].to_numpy(dtype=np.float64, copy=False)

            total_sum = bit_means.sum(axis=1)

            on0 = (
                df_barcode[["on_bit_1", "on_bit_2", "on_bit_3", "on_bit_4"]].to_numpy(
                    dtype=np.int32
                )
                - 1
            )  # shape (n_regions, 4)
            signal_vals = np.take_along_axis(bit_means, on0, axis=1)  # (n_regions, 4)
            signal_sum = signal_vals.sum(axis=1)
            denom = float(self._n_merfish_bits - 4)

            df_barcode["signal_mean"] = signal_sum / float(n_on)
            df_barcode["bkd_mean"] = (total_sum - signal_sum) / denom
            df_barcode["s-b_mean"] = df_barcode["signal_mean"] - df_barcode["bkd_mean"]
            df_barcode = df_barcode.drop(columns=["label", "decoded_id"])
            df_barcode = df_barcode[
                df_barcode["distance_min"] <= self._transcript_distance_threshold
            ].reset_index(drop=True)

            if self._df_barcodes.empty:
                self._df_barcodes = df_barcode.copy()
            else:
                if not df_barcode.empty:
                    self._df_barcodes = pd.concat(
                        [self._df_barcodes, df_barcode], ignore_index=True
                    )

            # Cleanup
            del (
                codewords_label_image,
                codewords_label_image_cp,
                props,
                props_distance,
                props_magnitude,
                df_distance,
                df_magnitude,
                df_barcode,
            )
            gc.collect()
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def _save_barcodes(self) -> None:
        """
        Save barcodes to datastore.

        Returns
        -------
        None
            Function result.
        """

        if self._verbose > 1:
            print("save barcodes")

        if self._optimize_normalization_weights:
            decoded_dir_path = self._temp_dir
            decoded_dir_path.mkdir(parents=True, exist_ok=True)
            temp_decoded_path = decoded_dir_path / Path(
                "tile" + str(self._tile_idx).zfill(3) + "_temp_decoded.parquet"
            )
            self._df_barcodes.to_parquet(temp_decoded_path)
        else:
            if not (self._barcodes_filtered):
                self._datastore.save_local_decoded_spots(
                    self._df_barcodes,
                    tile=self._tile_idx,
                    decode_run_key=self._decode_run_key,
                )
            else:
                self._datastore.save_global_filtered_decoded_spots(
                    self._df_filtered_barcodes,
                    decode_run_key=self._decode_run_key,
                )

    @property
    def decoded_barcodes(self) -> pd.DataFrame:
        """
        Decoded barcodes from the most recent ``decode_one_tile`` call.

        Returns
        -------
        pd.DataFrame
            Function result.
        """

        if not hasattr(self, "_df_barcodes"):
            return pd.DataFrame()
        return self._df_barcodes.copy()

    @property
    def decoded_image(self) -> np.ndarray:
        """
        Decoded pixel-label image from the most recent ``decode_one_tile`` call.

        Returns
        -------
        np.ndarray
            Function result.
        """

        if not hasattr(self, "_decoded_image"):
            return np.empty((0,), dtype=np.int16)
        return self._decoded_image.copy()

    def save_decoded_barcodes(self) -> None:
        """
        Save decoded barcodes from the most recent decoding/filtering step.

        Returns
        -------
        None
            Function result.
        """

        self._save_barcodes()

    def _prepare_normalization_state(
        self,
        normalization_method: Literal["iterative", "global", "none"] | None,
        use_normalization: bool | None,
        gpu_id: int = 0,
        lowpass_sigma: Sequence[float] | None = DEFAULT_DECODE_LOWPASS_SIGMA,
    ) -> None:
        """
        Select and load the normalization state used by pixel decoding.

        Parameters
        ----------
        normalization_method : Literal['iterative', 'global', 'none'] | None
            Function argument.
        use_normalization : bool | None
            Function argument.
        gpu_id : int
            Function argument.
        lowpass_sigma : Sequence[float], default = (3, 1, 1)
            Lowpass sigma used when global normalization needs to be
            recalculated, keeping normalization preprocessing aligned with
            decoding preprocessing.

        Returns
        -------
        None
            Function result.
        """

        if normalization_method is None:
            normalization_method = "iterative" if use_normalization else "none"

        if normalization_method == "iterative":
            self._load_iterative_normalization_vectors(gpu_id=gpu_id)
        elif normalization_method == "global":
            self._iterative_normalization_loaded = False
            self._load_global_normalization_vectors(
                gpu_id=gpu_id,
                lowpass_sigma=lowpass_sigma,
            )
        elif normalization_method == "none":
            self._iterative_normalization_loaded = False
            self._global_normalization_loaded = False
        elif normalization_method is not None:
            raise ValueError(
                "normalization_method must be one of 'iterative', 'global', "
                f"'none', or None. Got {normalization_method!r}."
            )

    def _load_all_barcodes(self) -> None:
        """
        Load all barcodes from datastore.

        Returns
        -------
        None
            Function result.
        """

        if self._optimize_normalization_weights:
            decoded_dir_path = self._temp_dir

            tile_files = decoded_dir_path.glob("*.parquet")
            tile_files = sorted(tile_files, key=operator.attrgetter("name"))

            if self._verbose >= 1:
                iterable_files = tqdm(tile_files, desc="tile", leave=False)
            else:
                iterable_files = tile_files

            tile_data = [
                pd.read_parquet(parquet_file) for parquet_file in iterable_files
            ]
            self._df_barcodes_loaded = pd.concat(tile_data)
        elif self._load_tile_decoding:
            tile_data = []
            for tile_id in self._datastore.tile_ids:
                tile_data.append(
                    self._datastore.load_local_decoded_spots(
                        tile_id,
                        decode_run_key=self._decode_run_key,
                    )
                )
            self._df_barcodes_loaded = pd.concat(tile_data)
        else:
            self._df_filtered_barcodes = (
                self._datastore.load_global_filtered_decoded_spots(
                    decode_run_key=self._decode_run_key,
                )
            )
            self._df_barcodes_loaded = self._df_filtered_barcodes.copy()
            self._barcodes_filtered = True

        if self._df_barcodes_loaded.empty:
            if "gene_id" not in self._df_barcodes_loaded.columns:
                self._df_barcodes_loaded["gene_id"] = pd.Series(dtype="string")
            for column in ("distance_min", "magnitude_mean", "area"):
                if column not in self._df_barcodes_loaded.columns:
                    self._df_barcodes_loaded[column] = pd.Series(dtype=np.float32)

        self._df_barcodes_loaded = self._df_barcodes_loaded[
            self._df_barcodes_loaded["gene_id"].notna()
            & self._df_barcodes_loaded["gene_id"].astype(str).str.strip().ne("")
        ]
        if "distance_min" not in self._df_barcodes_loaded.columns:
            raise ValueError(
                "Decoded transcripts are missing 'distance_min'. "
                "Re-decode local transcript parquet files with the exact two-threshold caller."
            )

    def _filter_all_barcodes_blank_fraction(
        self,
        target_gross_misid_rate: float = 0.05,
        intensity_bins: Sequence[float] | None = None,
        voxel_number_bins: Sequence[float] | None = None,
        vector_distance_bins: Sequence[float] | None = None,
    ) -> None:
        """
        Filter transcripts using blank-fraction histograms in feature space.

        Parameters
        ----------
        target_gross_misid_rate : float
            Function argument.
        intensity_bins : Sequence[float] | None
            Function argument.
        voxel_number_bins : Sequence[float] | None
            Function argument.
        vector_distance_bins : Sequence[float] | None
            Function argument.

        Returns
        -------
        None
            Function result.
        """

        required_columns = {"gene_id", "magnitude_mean", "area", "distance_min"}
        missing = sorted(required_columns.difference(self._df_barcodes_loaded.columns))
        if missing:
            raise ValueError(
                "Blank-fraction filtering requires columns: "
                + ", ".join(missing)
                + ". Re-decode transcripts with the exact caller."
            )

        annotated = self._df_barcodes_loaded.copy()
        annotated["voxel_intensity"] = annotated["magnitude_mean"].to_numpy(
            dtype=float, copy=False
        )
        annotated["voxel_number"] = annotated["area"].to_numpy(dtype=float, copy=False)
        annotated["vector_distance"] = annotated["distance_min"].to_numpy(
            dtype=float, copy=False
        )
        annotated["is_blank"] = (
            annotated["gene_id"]
            .astype("string")
            .str.lower()
            .str.startswith("blank", na=False)
            .to_numpy(dtype=bool, copy=False)
        )
        annotated["blank_fraction_bin"] = -1
        annotated["blank_fraction"] = np.nan
        annotated["blank_fraction_keep"] = False

        diagnostics: dict[str, object] = {
            "target_gross_misid_rate": float(target_gross_misid_rate),
            "chosen_threshold": np.nan,
            "achieved_gross_misid_rate": np.inf,
            "target_reached": False,
            "all_histogram": np.zeros((0, 0, 0), dtype=np.int64),
            "blank_histogram": np.zeros((0, 0, 0), dtype=np.int64),
            "blank_fraction_histogram": np.zeros((0, 0, 0), dtype=float),
            "intensity_bins": np.array([], dtype=float),
            "voxel_number_bins": np.array([], dtype=float),
            "vector_distance_bins": np.array([], dtype=float),
            "threshold_sweep": pd.DataFrame(
                columns=["threshold", "gross_misid_rate", "kept_transcripts"]
            ),
        }

        if annotated.empty:
            diagnostics["reason"] = "no_transcripts"
        else:
            voxel_intensity_values = cp.asarray(
                annotated["voxel_intensity"].to_numpy(dtype=np.float32, copy=False)
            )
            voxel_number_values = cp.asarray(
                annotated["voxel_number"].to_numpy(dtype=np.float32, copy=False)
            )
            vector_distance_values = cp.asarray(
                annotated["vector_distance"].to_numpy(dtype=np.float32, copy=False)
            )
            is_blank_values = cp.asarray(
                annotated["is_blank"].to_numpy(dtype=bool, copy=False)
            )

            features = cp.column_stack(
                (voxel_intensity_values, voxel_number_values, vector_distance_values)
            )
            valid_features = cp.asnumpy(cp.all(cp.isfinite(features), axis=1)).astype(
                bool, copy=False
            )

            if not np.any(valid_features):
                diagnostics["reason"] = "no_valid_features"
            else:
                valid = annotated.loc[valid_features]
                if self._blank_count <= 0:
                    annotated.loc[valid_features, "blank_fraction_keep"] = True
                    diagnostics["reason"] = "no_blank_barcodes"
                elif not np.any(valid["is_blank"].to_numpy(dtype=bool, copy=False)):
                    annotated.loc[valid_features, "blank_fraction_keep"] = True
                    diagnostics["reason"] = "no_blank_transcripts"
                else:
                    intensity_values = valid["voxel_intensity"].to_numpy(
                        dtype=float, copy=False
                    )
                    voxel_count_values = valid["voxel_number"].to_numpy(
                        dtype=float, copy=False
                    )
                    distance_values = valid["vector_distance"].to_numpy(
                        dtype=float, copy=False
                    )

                    if intensity_bins is not None:
                        intensity_edges = np.unique(
                            np.asarray(intensity_bins, dtype=float)
                        )
                        intensity_edges = intensity_edges[np.isfinite(intensity_edges)]
                        if intensity_edges.size < 2:
                            raise ValueError(
                                "Explicit histogram edges must contain at least two finite values."
                            )
                        intensity_edges[-1] = np.nextafter(intensity_edges[-1], np.inf)
                    else:
                        intensity_edges = np.unique(
                            np.quantile(intensity_values, np.linspace(0.0, 1.0, 11))
                        )
                        intensity_edges = intensity_edges[np.isfinite(intensity_edges)]
                        if intensity_edges.size < 2:
                            center = float(np.mean(intensity_values))
                            intensity_edges = np.array(
                                [center - 0.5, center + 0.5], dtype=float
                            )
                        elif np.allclose(intensity_edges[0], intensity_edges[-1]):
                            center = float(intensity_edges[0])
                            intensity_edges = np.array(
                                [center - 0.5, center + 0.5], dtype=float
                            )
                        intensity_edges[0] = min(
                            intensity_edges[0], float(np.min(intensity_values))
                        )
                        intensity_edges[-1] = max(
                            intensity_edges[-1], float(np.max(intensity_values))
                        )
                        intensity_edges[-1] = np.nextafter(intensity_edges[-1], np.inf)

                    if voxel_number_bins is not None:
                        voxel_number_edges = np.unique(
                            np.asarray(voxel_number_bins, dtype=float)
                        )
                        voxel_number_edges = voxel_number_edges[
                            np.isfinite(voxel_number_edges)
                        ]
                        if voxel_number_edges.size < 2:
                            raise ValueError(
                                "Explicit histogram edges must contain at least two finite values."
                            )
                        voxel_number_edges[-1] = np.nextafter(
                            voxel_number_edges[-1], np.inf
                        )
                    else:
                        min_value = int(np.floor(np.min(voxel_count_values)))
                        max_value = int(np.ceil(np.max(voxel_count_values)))
                        if max_value - min_value + 1 <= 10:
                            voxel_number_edges = np.arange(
                                min_value - 0.5, max_value + 1.5, 1.0
                            )
                        else:
                            quantiles = np.quantile(
                                voxel_count_values, np.linspace(0.0, 1.0, 11)
                            )
                            quantile_edges = np.unique(
                                np.floor(quantiles).astype(float)
                            )
                            if quantile_edges.size == 0:
                                quantile_edges = np.array(
                                    [float(min_value), float(max_value + 1)]
                                )
                            if quantile_edges[0] > min_value:
                                quantile_edges = np.insert(
                                    quantile_edges, 0, float(min_value)
                                )
                            if quantile_edges[-1] <= max_value:
                                quantile_edges = np.append(
                                    quantile_edges, float(max_value + 1)
                                )
                            voxel_number_edges = quantile_edges - 0.5
                        voxel_number_edges = np.unique(
                            np.asarray(voxel_number_edges, dtype=float)
                        )
                        voxel_number_edges = voxel_number_edges[
                            np.isfinite(voxel_number_edges)
                        ]
                        if voxel_number_edges.size < 2:
                            center = float(np.mean(voxel_count_values))
                            voxel_number_edges = np.array(
                                [center - 0.5, center + 0.5], dtype=float
                            )
                        elif np.allclose(voxel_number_edges[0], voxel_number_edges[-1]):
                            center = float(voxel_number_edges[0])
                            voxel_number_edges = np.array(
                                [center - 0.5, center + 0.5], dtype=float
                            )
                        voxel_number_edges[0] = min(
                            voxel_number_edges[0], float(np.min(voxel_count_values))
                        )
                        voxel_number_edges[-1] = max(
                            voxel_number_edges[-1], float(np.max(voxel_count_values))
                        )
                        voxel_number_edges[-1] = np.nextafter(
                            voxel_number_edges[-1], np.inf
                        )

                    if vector_distance_bins is not None:
                        vector_distance_edges = np.unique(
                            np.asarray(vector_distance_bins, dtype=float)
                        )
                        vector_distance_edges = vector_distance_edges[
                            np.isfinite(vector_distance_edges)
                        ]
                        if vector_distance_edges.size < 2:
                            raise ValueError(
                                "Explicit histogram edges must contain at least two finite values."
                            )
                        vector_distance_edges[-1] = np.nextafter(
                            vector_distance_edges[-1], np.inf
                        )
                    else:
                        vector_distance_edges = np.linspace(
                            float(np.min(distance_values)),
                            float(np.max(distance_values)),
                            11,
                        )
                        vector_distance_edges = np.unique(
                            np.asarray(vector_distance_edges, dtype=float)
                        )
                        vector_distance_edges = vector_distance_edges[
                            np.isfinite(vector_distance_edges)
                        ]
                        if vector_distance_edges.size < 2:
                            center = float(np.mean(distance_values))
                            vector_distance_edges = np.array(
                                [center - 0.5, center + 0.5], dtype=float
                            )
                        elif np.allclose(
                            vector_distance_edges[0], vector_distance_edges[-1]
                        ):
                            center = float(vector_distance_edges[0])
                            vector_distance_edges = np.array(
                                [center - 0.5, center + 0.5], dtype=float
                            )
                        vector_distance_edges[0] = min(
                            vector_distance_edges[0], float(np.min(distance_values))
                        )
                        vector_distance_edges[-1] = max(
                            vector_distance_edges[-1], float(np.max(distance_values))
                        )
                        vector_distance_edges[-1] = np.nextafter(
                            vector_distance_edges[-1], np.inf
                        )

                    diagnostics["intensity_bins"] = intensity_edges
                    diagnostics["voxel_number_bins"] = voxel_number_edges
                    diagnostics["vector_distance_bins"] = vector_distance_edges

                    intensity_edges_cp = cp.asarray(intensity_edges, dtype=cp.float32)
                    voxel_number_edges_cp = cp.asarray(
                        voxel_number_edges, dtype=cp.float32
                    )
                    vector_distance_edges_cp = cp.asarray(
                        vector_distance_edges, dtype=cp.float32
                    )

                    bin_indices = cp.column_stack(
                        (
                            cp.searchsorted(
                                intensity_edges_cp,
                                voxel_intensity_values,
                                side="right",
                            )
                            - 1,
                            cp.searchsorted(
                                voxel_number_edges_cp,
                                voxel_number_values,
                                side="right",
                            )
                            - 1,
                            cp.searchsorted(
                                vector_distance_edges_cp,
                                vector_distance_values,
                                side="right",
                            )
                            - 1,
                        )
                    )

                    in_range = valid_features & cp.asnumpy(
                        cp.all(
                            cp.column_stack(
                                (
                                    cp.isfinite(voxel_intensity_values)
                                    & (bin_indices[:, 0] >= 0)
                                    & (bin_indices[:, 0] < len(intensity_edges_cp) - 1),
                                    cp.isfinite(voxel_number_values)
                                    & (bin_indices[:, 1] >= 0)
                                    & (
                                        bin_indices[:, 1]
                                        < len(voxel_number_edges_cp) - 1
                                    ),
                                    cp.isfinite(vector_distance_values)
                                    & (bin_indices[:, 2] >= 0)
                                    & (
                                        bin_indices[:, 2]
                                        < len(vector_distance_edges_cp) - 1
                                    ),
                                )
                            ),
                            axis=1,
                        )
                    ).astype(bool, copy=False)

                    if not np.any(in_range):
                        diagnostics["reason"] = "no_transcripts_in_histogram_range"
                    else:
                        histogram_shape = (
                            len(intensity_edges) - 1,
                            len(voxel_number_edges) - 1,
                            len(vector_distance_edges) - 1,
                        )
                        all_histogram = cp.zeros(histogram_shape, dtype=cp.int32)
                        blank_histogram = cp.zeros(histogram_shape, dtype=cp.int32)

                        in_range_indices = bin_indices[in_range]
                        cp.add.at(
                            all_histogram,
                            tuple(in_range_indices[:, axis] for axis in range(3)),
                            1,
                        )

                        blank_in_range = in_range & cp.asnumpy(is_blank_values).astype(
                            bool, copy=False
                        )
                        blank_indices = bin_indices[blank_in_range]
                        if blank_indices.size:
                            cp.add.at(
                                blank_histogram,
                                tuple(blank_indices[:, axis] for axis in range(3)),
                                1,
                            )

                        blank_fraction_histogram = cp.full(
                            histogram_shape, cp.nan, dtype=cp.float32
                        )
                        nonempty = all_histogram > 0
                        blank_fraction_histogram[nonempty] = (
                            blank_histogram[nonempty] / all_histogram[nonempty]
                        )

                        in_range_flat_bins = cp.ravel_multi_index(
                            tuple(in_range_indices[:, axis] for axis in range(3)),
                            dims=histogram_shape,
                        )
                        flat_bins = np.full(len(annotated), -1, dtype=np.int64)
                        flat_bins[in_range] = cp.asnumpy(in_range_flat_bins)
                        annotated["blank_fraction_bin"] = flat_bins
                        annotated.loc[in_range, "blank_fraction"] = cp.asnumpy(
                            blank_fraction_histogram.ravel()[in_range_flat_bins]
                        )

                        thresholds = np.unique(
                            cp.asnumpy(blank_fraction_histogram[nonempty])
                        )
                        sweep_rows: list[dict[str, float | int]] = []
                        chosen_threshold = np.nan
                        achieved_rate = np.inf
                        chosen_keep = np.zeros(len(annotated), dtype=bool)
                        target_reached = False

                        blank_fraction_values = annotated["blank_fraction"].to_numpy(
                            dtype=float, copy=False
                        )
                        is_blank_numpy = cp.asnumpy(is_blank_values).astype(
                            bool, copy=False
                        )
                        for threshold in thresholds:
                            keep_mask = in_range & (
                                blank_fraction_values <= float(threshold)
                            )
                            if self._blank_count <= 0 or self._barcode_count <= 0:
                                gross_misid = np.inf
                            else:
                                keep = np.asarray(keep_mask, dtype=bool)
                                if not np.any(keep):
                                    gross_misid = np.inf
                                else:
                                    blank_kept = np.count_nonzero(keep & is_blank_numpy)
                                    total_kept = np.count_nonzero(keep)
                                    gross_misid = (
                                        blank_kept / float(self._blank_count)
                                    ) / (total_kept / float(self._barcode_count))
                            sweep_rows.append(
                                {
                                    "threshold": float(threshold),
                                    "gross_misid_rate": float(gross_misid),
                                    "kept_transcripts": int(
                                        np.count_nonzero(keep_mask)
                                    ),
                                }
                            )

                            if gross_misid <= target_gross_misid_rate:
                                chosen_threshold = float(threshold)
                                achieved_rate = float(gross_misid)
                                chosen_keep = keep_mask.copy()
                                target_reached = True

                        if not sweep_rows:
                            diagnostics["reason"] = "no_nonempty_histogram_bins"
                        else:
                            sweep_df = pd.DataFrame(sweep_rows)
                            if not target_reached:
                                best_idx = int(sweep_df["gross_misid_rate"].argmin())
                                chosen_threshold = float(
                                    sweep_df.loc[best_idx, "threshold"]
                                )
                                achieved_rate = float(
                                    sweep_df.loc[best_idx, "gross_misid_rate"]
                                )
                                chosen_keep = in_range & (
                                    blank_fraction_values <= chosen_threshold
                                )

                            annotated["blank_fraction_keep"] = chosen_keep
                            diagnostics.update(
                                {
                                    "chosen_threshold": chosen_threshold,
                                    "achieved_gross_misid_rate": achieved_rate,
                                    "target_reached": target_reached,
                                    "all_histogram": cp.asnumpy(all_histogram),
                                    "blank_histogram": cp.asnumpy(blank_histogram),
                                    "blank_fraction_histogram": cp.asnumpy(
                                        blank_fraction_histogram
                                    ),
                                    "threshold_sweep": sweep_df,
                                }
                            )

        if self._verbose > 1:
            print("blank fraction filter diagnostics:")
            for key, value in diagnostics.items():
                if isinstance(value, (float, int, str, bool)):
                    print(f"{key}: {value}")
                else:
                    print(
                        f"{key}: {type(value)} with shape {getattr(value, 'shape', 'N/A')}"
                    )
        self._blank_fraction_filter_results = diagnostics
        self._df_filtered_barcodes = annotated[annotated["blank_fraction_keep"]].copy()
        self._df_filtered_barcodes["cell_id"] = -1
        self._barcodes_filtered = True

    @staticmethod
    def _calculate_lr_fdr(
        df: pd.DataFrame,
        threshold: float,
        blank_count: int,
        barcode_count: int,
        verbose: bool = False,
    ) -> float:
        """Calculate LR-filter false discovery rate.

        (# noncoding found ) / (# noncoding in codebook) / (# coding found) / (# coding in codebook)

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing decoded spots.
        threshold : float
            Threshold for predicted probability.
        blank_count : int
            Number of blank barcodes.
        barcode_count : int
            Number of barcodes.
        verbose : bool = False
            Verbose output. Default False.

        Returns
        -------
        lr_fdr : float
            False discovery rate for the LR filter.
        """

        blank_mask = (
            df["gene_id"].astype("string").str.lower().str.startswith("blank", na=False)
        )

        if threshold >= 0:
            df["prediction"] = df["predicted_probability"] > threshold

            predicted_positive = df["predicted_probability"] > threshold
            coding = df[(~blank_mask) & predicted_positive].shape[0]
            noncoding = df[blank_mask & predicted_positive].shape[0]
        else:
            coding = df[~blank_mask].shape[0]
            noncoding = df[blank_mask].shape[0]

        if coding > 0:
            lr_fdr = (noncoding / blank_count) / (
                coding / (barcode_count - blank_count)
            )
        else:
            lr_fdr = np.inf

        if verbose > 1:
            print(f"threshold: {threshold}")
            print(f"coding: {coding}")
            print(f"noncoding: {noncoding}")
            print(f"lr_fdr: {lr_fdr}")

        return lr_fdr

    def _filter_all_barcodes_LR(self, lr_fdr_target: float = 0.05) -> None:
        """Filter barcodes using a classifier and LR FDR target.

        Uses a logistic regression classifier to predict whether a barcode is a blank or not.

        Parameters
        ----------
        lr_fdr_target : float, default 0.05
            False discovery rate target for LR filtering.
        """

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        df_barcodes_to_filter = self._df_barcodes_loaded.copy()
        df_barcodes_to_filter["X"] = ~(
            df_barcodes_to_filter["gene_id"]
            .astype("string")
            .str.lower()
            .str.startswith("blank", na=False)
            .to_numpy(dtype=bool, copy=False)
        )

        if self._is_3D:
            columns = [
                "X",
                "area",
                "signal_mean",
                "s-b_mean",
                "distance_min",
                "magnitude_mean",
                "inertia_tensor_eigvals-0",
                "inertia_tensor_eigvals-1",
                "inertia_tensor_eigvals-2",
            ]
        else:
            columns = [
                "X",
                "area",
                "signal_mean",
                "s-b_mean",
                "distance_min",
                "magnitude_mean",
                "inertia_tensor_eigvals-0",
                "inertia_tensor_eigvals-1",
            ]
        df_true = df_barcodes_to_filter[df_barcodes_to_filter["X"]][columns]
        df_false = df_barcodes_to_filter[~df_barcodes_to_filter["X"]][columns]
        if self._verbose > 1:
            print(f"Number of blanks: {len(df_false)}")
        if len(df_false) > 1:
            df_true_sampled = df_true.sample(n=len(df_false), random_state=42)
            df_combined = pd.concat([df_true_sampled, df_false])
            x = df_combined.drop("X", axis=1)
            y = df_combined["X"]
            X_train, X_test, y_train, y_test = train_test_split(
                x, y, test_size=0.1, random_state=42
            )

            if self._verbose > 1:
                print("generating synthetic samples for class balance")
            # SMOTE(random_state=42)
            # X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            X_train_resampled = X_train.copy()
            y_train_resampled = y_train.copy()

            if self._verbose > 1:
                print("scaling features")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_resampled)
            X_test_scaled = scaler.transform(X_test)

            if self._verbose > 1:
                print("training classifier")
            logistic = LogisticRegression(solver="liblinear", random_state=42)
            logistic.fit(X_train_scaled, y_train_resampled)
            predictions = logistic.predict(X_test_scaled)

            if self._verbose > 1:
                print(classification_report(y_test, predictions))

            if self._verbose > 1:
                print("predicting on full data")

            full_data_scaled = scaler.transform(df_barcodes_to_filter[columns[1:]])
            df_barcodes_to_filter["predicted_probability"] = logistic.predict_proba(
                full_data_scaled
            )[:, 1]

            if self._verbose > 1:
                print("filtering blanks")

            coarse_threshold = 0
            for threshold in np.arange(0, 1, 0.1):
                lr_fdr = self._calculate_lr_fdr(
                    df_barcodes_to_filter,
                    threshold,
                    self._blank_count,
                    self._barcode_count,
                    self._verbose,
                )
                if lr_fdr <= lr_fdr_target:
                    coarse_threshold = threshold
                    break

            fine_threshold = coarse_threshold
            for threshold in np.arange(
                coarse_threshold - 0.1, coarse_threshold + 0.1, 0.01
            ):
                lr_fdr = self._calculate_lr_fdr(
                    df_barcodes_to_filter,
                    threshold,
                    self._blank_count,
                    self._barcode_count,
                    self._verbose,
                )
                if lr_fdr <= lr_fdr_target:
                    fine_threshold = threshold
                    break

            df_above_threshold = df_barcodes_to_filter[
                df_barcodes_to_filter["predicted_probability"] > fine_threshold
            ]
            self._df_filtered_barcodes = df_above_threshold.copy()
            self._df_filtered_barcodes["cell_id"] = -1
            self._barcodes_filtered = True

            if self._verbose > 1:
                print(f"lr_fdr : {lr_fdr}")
                print(f"retained barcodes: {len(self._df_filtered_barcodes)}")

            del df_above_threshold, full_data_scaled
            del (
                logistic,
                predictions,
                X_train,
                X_test,
                y_test,
                y_train,
                X_train_scaled,
                X_test_scaled,
            )
            del df_true, df_false, df_true_sampled, df_combined
            gc.collect()
        else:
            self._df_filtered_barcodes = self._df_barcodes_loaded.copy()
            self._df_filtered_barcodes["cell_id"] = -1
            self._barcodes_filtered = True
            if self._verbose >= 1:
                print("Insufficient Blank barcodes called for filtering.")

    @staticmethod
    def _roi_to_shapely(roi):  # noqa
        """
        Roi to shapely.

        Parameters
        ----------
        roi : Any
            Function argument.

        Returns
        -------
        Any
            Function result.
        """
        return Polygon(roi.subpixel_coordinates[:, ::-1])

    def _assign_cells(self) -> None:
        """
        Assign cells to barcodes using Cellpose ROIs.

        Returns
        -------
        None
            Function result.
        """

        cellpose_roi_path = (
            self._datastore._datastore_path
            / Path("segmentation")
            / Path("cellpose")
            / Path("imagej_rois")
            / Path("global_coords_rois.zip")
        )

        try:
            rois = roiread(cellpose_roi_path)
        except (OSError, FileNotFoundError, ValueError) as e:
            print(f"Failed to read ROIs: {e}")
            return

        shapely_polygons = []
        for roi in rois:
            shapely_polygon = self._roi_to_shapely(roi)
            if shapely_polygon:
                shapely_polygons.append(shapely_polygon)

        rtree_index = rtree.index.Index()
        for polygon_idx, polygon in enumerate(shapely_polygons):
            try:
                rtree_index.insert(polygon_idx, polygon.bounds)
            except rtree.RTreeError as e:
                print(f"Failed to insert polygon into R-tree: {e}")

        def check_point(row: pd.Series):  # noqa
            """Check if point is within a polygon.

            Parameters
            ----------
            row : pd.Series
                Row containing global coordinates.

            Returns
            -------
            cell_id : int
                Cell ID. Returns 0 if not found.
            """
            point = Point(row["global_y"], row["global_x"])

            candidate_ids = list(rtree_index.intersection(point.bounds))
            for candidate_id in candidate_ids:
                if shapely_polygons[candidate_id].contains(point):
                    return candidate_id + 1
            return 0

        self._df_filtered_barcodes["cell_id"] = self._df_filtered_barcodes.apply(
            check_point, axis=1
        )

    def _remove_duplicates_in_tile_overlap(self, radius: float = 0.75) -> None:
        """Remove duplicates in tile overlap.

        Parameters
        ----------
        radius : float, default 0.75
            3D radius, in microns, for duplicate removal.
        """

        self._df_filtered_barcodes.reset_index(drop=True, inplace=True)

        coords = self._df_filtered_barcodes[["global_z", "global_y", "global_x"]].values
        tile_idxs = self._df_filtered_barcodes["tile_idx"].values
        distance_min = self._df_filtered_barcodes["distance_min"].to_numpy(
            dtype=float, copy=False
        )

        tree = cKDTree(coords)
        pairs = tree.query_pairs(radius)

        rows_to_drop = set()
        distances = []
        for i, j in pairs:
            if tile_idxs[i] != tile_idxs[j]:
                if (distance_min[i], i) <= (distance_min[j], j):
                    rows_to_drop.add(j)
                    distances.append(distance_min[j])
                else:
                    rows_to_drop.add(i)
                    distances.append(distance_min[i])

        self._df_filtered_barcodes.drop(rows_to_drop, inplace=True)
        self._df_filtered_barcodes.reset_index(drop=True, inplace=True)

        avg_distance = np.mean(distances) if distances else 0
        dropped_count = len(rows_to_drop)

        if self._verbose > 1:
            print(
                "Average distance_min of dropped points (overlap): " + str(avg_distance)
            )
            print("Dropped points: " + str(dropped_count))

    def _remove_duplicates_within_tile(
        self,
        radius_xy: float = 0.1,
        radius_z: float = 0.50,
    ) -> None:
        """Collapse cross-plane near-duplicate detections within each tile.

        Two rows are considered neighbors if and only if:
        1) They belong to the same tile (``tile_idx``),
        2) Their XY separation is within ``radius_xy`` (microns),
        3) They are in different Z planes and their absolute Z separation is
           within ``radius_z`` (microns), and
        4) Their identity matches (``gene_id`` is equal).

        For each connected component (cluster) under this neighbor relation,
        keep exactly one row: the one with the smallest ``distance_min``.
        Ties are broken deterministically by original row index
        (lower index wins).

        Parameters
        ----------
        radius_xy : float, default 0.1
            Neighborhood radius in the XY plane, in microns.
        radius_z : float, default 0.50
            Neighborhood half-extent along Z, in microns.

        Modifies
        --------
        self._df_filtered_barcodes : pandas.DataFrame
            Drops non-winning rows per cluster; resets index at the end.

        Notes
        -----
        Expected columns: ``global_z``, ``global_y``, ``global_x``,
        ``tile_idx``, ``gene_id``, ``distance_min``.
        """
        df = getattr(self, "_df_filtered_barcodes", None)
        filtered = df is not None
        if df is None:
            df = getattr(self, "_df_barcodes_loaded", None)

        if df is None or df.empty or len(df) < 2:
            return

        # Stable order & deterministic tie-breaks
        df.reset_index(drop=True, inplace=True)

        coords = df[["global_z", "global_y", "global_x"]].to_numpy(
            dtype=float, copy=False
        )
        tiles = df["tile_idx"].to_numpy()
        genes = df[
            "gene_id"
        ].to_numpy()  # dtype can be int/str/object; equality works elementwise
        dmin = df["distance_min"].to_numpy(dtype=float, copy=False)

        rows_to_drop: set[int] = set()

        # Union-Find (Disjoint Set)
        def uf_find(parent: np.ndarray, x: int) -> int:
            """
            Uf find.

            Parameters
            ----------
            parent : np.ndarray
                Function argument.
            x : int
                Function argument.

            Returns
            -------
            int
                Function result.
            """
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def uf_union(parent: np.ndarray, rank: np.ndarray, a: int, b: int) -> None:
            """
            Uf union.

            Parameters
            ----------
            parent : np.ndarray
                Function argument.
            rank : np.ndarray
                Function argument.
            a : int
                Function argument.
            b : int
                Function argument.

            Returns
            -------
            None
                Function result.
            """
            ra, rb = uf_find(parent, a), uf_find(parent, b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        # Process each tile independently
        for t in np.unique(tiles):
            local_idx = np.flatnonzero(tiles == t)
            if local_idx.size < 2:
                continue

            sub = coords[local_idx]
            z_local = sub[:, 0]
            xy_local = sub[:, 1:3]  # (Y, X)
            genes_local = genes[local_idx]

            # 1) XY-near candidate pairs
            tree = cKDTree(xy_local)
            pairs_local = tree.query_pairs(r=radius_xy)
            if not pairs_local:
                continue

            # 2) Filter by cross-plane Z window *and same gene_id*
            filtered_pairs = [
                (i, j)
                for (i, j) in pairs_local
                if (0.0 < abs(z_local[i] - z_local[j]) <= radius_z)
                and (genes_local[i] == genes_local[j])
            ]
            if not filtered_pairs:
                continue

            # 3) Union-Find over local nodes
            n = local_idx.size
            parent = np.arange(n)
            rank = np.zeros(n, dtype=np.int8)
            for i_loc, j_loc in filtered_pairs:
                uf_union(parent, rank, i_loc, j_loc)

            # 4) Gather components
            comps: dict[int, list[int]] = {}
            for i_loc in range(n):
                r = uf_find(parent, i_loc)
                comps.setdefault(r, []).append(i_loc)

            # 5) Keep exactly one best per multi-member component
            for members in comps.values():
                if len(members) < 2:
                    continue
                glob_members = local_idx[np.asarray(members)]
                # Lexicographic: primary key is distance_min, then row index.
                best_global = glob_members[
                    np.lexsort((glob_members, dmin[glob_members]))
                ][0]
                for g in glob_members:
                    if g != best_global:
                        rows_to_drop.add(g)

        if rows_to_drop:
            df.drop(index=list(rows_to_drop), inplace=True)
            df.reset_index(drop=True, inplace=True)

        if getattr(self, "_verbose", 0) > 1:
            dropped = (
                dmin[list(rows_to_drop)] if rows_to_drop else np.array([], dtype=float)
            )
            avg = float(dropped.mean()) if dropped.size else 0.0
            print(
                "Average distance_min of dropped points (within-tile, same gene, clusters): "
                + str(avg)
            )
            print("Dropped points: " + str(len(rows_to_drop)))

        if filtered:
            del self._df_filtered_barcodes
            self._df_filtered_barcodes = df.copy()
        else:
            del self._df_barcodes_loaded
            self._df_barcodes_loaded = df.copy()

    def _display_results(self) -> None:
        """
        Display results using Napari.

        Returns
        -------
        None
            Function result.
        """

        import napari
        from qtpy.QtWidgets import QApplication

        def on_close_callback() -> None:
            """
            On close callback.

            Returns
            -------
            None
                Function result.
            """
            viewer.layers.clear()
            gc.collect()

        viewer = napari.Viewer()
        app = QApplication.instance()

        app.lastWindowClosed.connect(on_close_callback)

        viewer.add_image(
            self._image_data_lp,
            scale=[self._axial_step, self._pixel_size, self._pixel_size],
            name="image",
        )

        viewer.add_image(
            self._scaled_pixel_images,
            scale=[self._axial_step, self._pixel_size, self._pixel_size],
            name="scaled pixels",
        )

        viewer.add_image(
            self._decoded_image,
            scale=[self._axial_step, self._pixel_size, self._pixel_size],  # yes.
            name="decoded",
        )

        viewer.add_image(
            self._magnitude_image,
            scale=[self._axial_step, self._pixel_size, self._pixel_size],
            name="magnitude",
        )

        viewer.add_image(
            self._distance_image,
            scale=[self._axial_step, self._pixel_size, self._pixel_size],
            name="distance",
        )

        napari.run()

    def _cleanup(self) -> None:
        """
        Cleanup memory.

        Returns
        -------
        None
            Function result.
        """
        for gpu_id in range(self._num_gpus):
            cp.cuda.Device(gpu_id).use()
            cp.cuda.Device(gpu_id).synchronize()
            try:
                if self._filter_type == "lp":
                    del self._image_data_lp
                else:
                    del self._image_data
            except AttributeError:
                pass

            try:
                del (
                    self._scaled_pixel_images,
                    self._decoded_image,
                    self._distance_image,
                    self._magnitude_image,
                )
            except AttributeError:
                pass

            try:
                del self._df_barcodes
            except AttributeError:
                pass
            if self._barcodes_filtered:
                try:
                    del self._df_filtered_barcodes
                except AttributeError:
                    pass

            gc.collect()
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def decode_one_tile(
        self,
        tile_idx: int = 0,
        gpu_id: int = 0,
        display_results: bool = False,
        return_results: bool = False,
        lowpass_sigma: Sequence[float] | None = DEFAULT_DECODE_LOWPASS_SIGMA,
        magnitude_threshold: Sequence[float] | None = None,
        minimum_pixels: float | None = None,
        use_normalization: bool | None = True,
        normalization_method: Literal["iterative", "global", "none"] | None = None,
        feature_predictor_threshold: float | None = 0.1,
    ) -> tuple[np.ndarray, ...] | None:
        """Decode one tile.

        Helper function to decode one tile. Can also display results in napari or return results as np.ndarray.

        Parameters
        ----------
        tile_idx : int, default 0
            Tile index.
        gpu_id : int, default 0
            GPU ID to use for decoding.
        display_results : bool, default False
            Display results in napari.
        return_results : bool, default False
            Return results as np.ndarray
        lowpass_sigma : Sequence[float], default (3, 1, 1)
            Lowpass sigma.
        magnitude_threshold: Sequence[float], optional
            L2-norm threshold
        minimum_pixels : float, optional
            Minimum number of pixels for a barcode.
        use_normalization : bool, default True
            Use iterative normalization when ``normalization_method`` is not set.
            This legacy argument is kept for compatibility.
        normalization_method : {"iterative", "global", "none"}, optional
            Normalization source for pixel traces. ``iterative`` uses cached or
            calculated iterative normalization, ``global`` uses global
            normalization, and ``none`` decodes unnormalized traces.
        feature_predictor_threshold : float, default 0.1
            Legacy argument retained for compatibility. Decode input is weighted
            by the feature-predictor image rather than thresholded.

        Returns
        -------
        tuple[np.ndarray,...]
            If return_results is True, returns a tuple of np.ndarray containing the following:
            1. Image data (filtered or unfiltered).
            2. Scaled pixel images.
            3. Magnitude image.
            4. Distance image.
            5. Decoded image.
        """

        with cp.cuda.Device(gpu_id):
            if magnitude_threshold is None:
                magnitude_threshold = DEFAULT_DECODE_MAGNITUDE_THRESHOLD
            if minimum_pixels is None:
                minimum_pixels = self._default_minimum_pixels()
            self._prepare_normalization_state(
                normalization_method=normalization_method,
                use_normalization=use_normalization,
                gpu_id=gpu_id,
                lowpass_sigma=lowpass_sigma,
            )

            self._tile_idx = tile_idx
            self._load_bit_data(
                feature_predictor_threshold=feature_predictor_threshold,
                gpu_id=gpu_id,
            )
            self._filter_type = "raw"
            effective_lowpass_sigma = self._effective_lowpass_sigma(lowpass_sigma)
            if effective_lowpass_sigma is not None and not np.any(
                np.asarray(effective_lowpass_sigma, dtype=float) == 0
            ):
                self._lp_filter(
                    sigma=effective_lowpass_sigma,
                    gpu_id=gpu_id,
                )
            self._decode_pixels(
                magnitude_threshold=magnitude_threshold,
                gpu_id=gpu_id,
            )
            self._extract_barcodes(minimum_pixels=minimum_pixels, gpu_id=gpu_id)

            if display_results:
                if not (self._df_barcodes.empty):
                    print(f"Number of extracted barcodes: {len(self._df_barcodes)}")
                else:
                    print("No barcodes extracted.")
                self._display_results()
            if return_results:
                if self._filter_type == "lp":
                    return (
                        self._image_data_lp,
                        self._scaled_pixel_images,
                        self._magnitude_image,
                        self._distance_image,
                        self._decoded_image,
                    )
                else:
                    return (
                        self._image_data,
                        self._scaled_pixel_images,
                        self._magnitude_image,
                        self._distance_image,
                        self._decoded_image,
                    )

    def optimize_normalization_by_decoding(
        self,
        n_random_tiles: int = 5,
        n_iterations: int = 10,
        minimum_pixels: float | None = None,
        feature_predictor_threshold: float | None = 0.1,
        lowpass_sigma: Sequence[float] | None = DEFAULT_DECODE_LOWPASS_SIGMA,
        magnitude_threshold: Sequence[float] | None = None,
        tile_indices: Sequence[int] | None = None,
        estimate_chromatic_affines: bool | None = None,
    ) -> None:
        """Iteratively refine normalization vectors using exact-called transcripts.

        Parameters
        ----------
        n_random_tiles : int, default 5
            Number of random tiles.
        n_iterations : int, default 10
            Number of iterations.
        minimum_pixels : float, optional
            Minimum number of pixels for a barcode.
        feature_predictor_threshold : float, default = 0.1
            feature_predictor threshold.
        lowpass_sigma : Sequence[float], default = (3, 1, 1)
            Lowpass sigma.
        magnitude_threshold: Sequence[float], optional
            L2-norm threshold
        tile_indices : Sequence[int], optional
            Explicit tile indices to use for normalization. If omitted, a random
            subset of ``n_random_tiles`` is used.
        estimate_chromatic_affines : bool or None, optional
            If True, estimate chromatic affine transforms after each iterative
            decoding round. If None, use the instance setting supplied at
            construction.
        """
        if self._num_gpus < 1:
            raise RuntimeError("No GPUs allocated.")
        if magnitude_threshold is None:
            magnitude_threshold = DEFAULT_DECODE_MAGNITUDE_THRESHOLD
        if minimum_pixels is None:
            minimum_pixels = self._default_minimum_pixels()
        all_tiles = list(range(len(self._datastore.tile_ids)))

        # preload global normalization once
        self._iterative_background_vector = None
        self._iterative_normalization_vector = None
        self._global_background_vector = None
        self._optimize_normalization_weights = True
        run_chromatic_estimation = (
            self._estimate_chromatic_affines
            if estimate_chromatic_affines is None
            else bool(estimate_chromatic_affines)
        )
        if run_chromatic_estimation:
            self._save_identity_chromatic_affines()
        self._load_global_normalization_vectors(
            gpu_id=0,
            recalculate=True,
            tile_indices=tile_indices,
            lowpass_sigma=lowpass_sigma,
        )
        if self._decode_run_key is None:
            temp_dir = Path(tempfile.mkdtemp())
        else:
            temp_dir = self._datastore.decoded_temporary_dir(self._decode_run_key)
            temp_dir.mkdir(parents=True, exist_ok=True)
        self._temp_dir = temp_dir

        # split the same set of tiles each iteration
        if tile_indices is not None:
            random_tiles = list(tile_indices)
        elif len(all_tiles) > n_random_tiles:
            random_tiles = sample(all_tiles, n_random_tiles)
        else:
            random_tiles = all_tiles
        chunk_size = (len(random_tiles) + self._num_gpus - 1) // self._num_gpus

        if self._verbose >= 1:
            iterator = trange(n_iterations, desc="Iterative normalization")
        else:
            iterator = range(n_iterations)

        for iteration in iterator:
            iteration_temp_dir = (
                self._datastore.decoded_temporary_dir(
                    self._decode_run_key,
                    iteration=iteration,
                )
                if self._decode_run_key is not None
                else temp_dir
            )
            iteration_temp_dir.mkdir(parents=True, exist_ok=True)
            self._temp_dir = iteration_temp_dir
            # launch one process per GPU
            processes = []
            for gpu in range(self._num_gpus):
                start = gpu * chunk_size
                end = min(start + chunk_size, len(random_tiles))
                subset = random_tiles[start:end]
                if not subset:
                    continue
                p = _start_gpu_worker_process(
                    target=_optimize_norm_worker,
                    args=(
                        self._datastore_path,
                        subset,
                        0,
                        self._n_merfish_bits,
                        self._zstride_level,
                        self._decode_mode,
                        iteration_temp_dir,
                        iteration,
                        lowpass_sigma,
                        magnitude_threshold,
                        minimum_pixels,
                        feature_predictor_threshold,
                        run_chromatic_estimation,
                    ),
                    physical_gpu_id=gpu,
                )
                processes.append(p)

            _join_gpu_workers(processes, "Iterative normalization")

            with cp.cuda.Device(0):
                # gather results and update
                self._load_all_barcodes()
                if not (self._is_3D):
                    radius_xy = self._datastore.voxel_size_zyx_um[-1]
                    radius_z = self._datastore.voxel_size_zyx_um[0]
                    self._remove_duplicates_within_tile(
                        radius_xy=radius_xy, radius_z=radius_z
                    )
                if run_chromatic_estimation:
                    self._estimate_chromatic_affines_from_barcodes()
                self._load_global_normalization_vectors(
                    gpu_id=0,
                    lowpass_sigma=lowpass_sigma,
                )
                self._iterative_normalization_vectors(gpu_id=0)
                del self._global_background_vector, self._global_normalization_vector
                gc.collect()
                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

        # cleanup temp files, etc.
        self._cleanup()
        self._optimize_normalization_weights = False
        if self._decode_run_key is None:
            shutil.rmtree(temp_dir)

    def decode_all_tiles(
        self,
        assign_to_cells: bool = True,
        lowpass_sigma: Sequence[float] | None = DEFAULT_DECODE_LOWPASS_SIGMA,
        magnitude_threshold: Sequence[float] | None = None,
        minimum_pixels: float | None = None,
        feature_predictor_threshold: float | None = 0.1,
        normalization_method: Literal["iterative", "global", "none"] = "iterative",
        duplicate_radius_xy: float | None = None,
        duplicate_radius_z: float | None = None,
        filter_method: Literal["blank_fraction", "lr"] = "blank_fraction",
        target_gross_misid_rate: float = 0.05,
        lr_fdr_target: float = 0.05,
    ) -> None:
        """Decode all tiles and apply the selected downstream transcript filter.

        Parameters
        ----------
        assign_to_cells: bool, default = True
            Assign codewords to cells
        lowpass_sigma : Sequence[float], default = (3, 1, 1)
            Lowpass sigma.
        magnitude_threshold: Sequence[float], optional
            Accept pixels with magnitudes between low and high values
        minimum_pixels : float, optional
            Minimum number of pixels for a barcode.
        feature_predictor_threshold : float, default 0.1
            feature_predictor threshold.
        normalization_method : {"iterative", "global", "none"}, default "iterative"
            Normalization source for pixel traces.
        duplicate_radius_xy : float, optional
            Override XY radius, in microns, for within-tile duplicate collapse.
        duplicate_radius_z : float, optional
            Override Z radius, in microns, for within-tile duplicate collapse.
        filter_method : {"blank_fraction", "lr"}, default "blank_fraction"
            Downstream filter to apply after exact transcript calling.
        target_gross_misid_rate : float, default 0.05
            Gross misidentification-rate target for blank-fraction filtering.
        lr_fdr_target: float, default = 0.05
            False discovery rate target for LR filtering.
        """

        if self._num_gpus < 1:
            raise RuntimeError("No GPUs allocated.")
        if magnitude_threshold is None:
            magnitude_threshold = DEFAULT_DECODE_MAGNITUDE_THRESHOLD
        if minimum_pixels is None:
            minimum_pixels = self._default_minimum_pixels()
        self._validate_filter_configuration(
            filter_method=filter_method,
            target_gross_misid_rate=float(target_gross_misid_rate),
            lr_fdr_target=float(lr_fdr_target),
        )
        all_tiles = list(range(len(self._datastore.tile_ids)))
        chunk_size = (len(all_tiles) + self._num_gpus - 1) // self._num_gpus

        processes = []
        for gpu in range(self._num_gpus):
            start = gpu * chunk_size
            end = min(start + chunk_size, len(all_tiles))
            subset = all_tiles[start:end]
            if not subset:
                continue
            p = _start_gpu_worker_process(
                target=decode_tiles_worker,
                args=(
                    self._datastore_path,
                    subset,
                    0,
                    self._n_merfish_bits,
                    self._verbose,
                    self._zstride_level,
                    self._decode_mode,
                    lowpass_sigma,
                    magnitude_threshold,
                    minimum_pixels,
                    feature_predictor_threshold,
                    normalization_method,
                ),
                physical_gpu_id=gpu,
            )
            processes.append(p)

        _join_gpu_workers(processes, "Tile decoding")

        # load all barcodes and filter
        self._load_tile_decoding = True
        self._load_all_barcodes()
        if self._verbose >= 1:
            print(f"Number of loaded barcodes: {len(self._df_barcodes_loaded)}")
            print(f"Verbosity:  {self._verbose}")
        self._apply_filter_method(
            filter_method=filter_method,
            target_gross_misid_rate=float(target_gross_misid_rate),
            lr_fdr_target=float(lr_fdr_target),
        )
        if not (self._is_3D):
            radius_xy = (
                self._datastore.voxel_size_zyx_um[-1]
                if duplicate_radius_xy is None
                else float(duplicate_radius_xy)
            )
            radius_z = (
                self._datastore.voxel_size_zyx_um[0]
                if duplicate_radius_z is None
                else float(duplicate_radius_z)
            )
            self._remove_duplicates_within_tile(radius_xy=radius_xy, radius_z=radius_z)

        if len(all_tiles) > 1:
            self._remove_duplicates_in_tile_overlap()
        if assign_to_cells:
            self._assign_cells()
        self._save_barcodes()

    @staticmethod
    def _validate_filter_configuration(
        filter_method: Literal["blank_fraction", "lr"],
        target_gross_misid_rate: float,
        lr_fdr_target: float,
    ) -> None:
        """
        Validate that the selected downstream filter uses the matching target.

        Parameters
        ----------
        filter_method : Literal['blank_fraction', 'lr']
            Function argument.
        target_gross_misid_rate : float
            Function argument.
        lr_fdr_target : float
            Function argument.

        Returns
        -------
        None
            Function result.
        """

        if filter_method == "blank_fraction":
            if lr_fdr_target != 0.05:
                raise ValueError(
                    "lr_fdr_target only applies when filter_method='lr'. "
                    "Use target_gross_misid_rate with filter_method='blank_fraction'."
                )
            return

        if filter_method == "lr":
            if target_gross_misid_rate != 0.05:
                raise ValueError(
                    "target_gross_misid_rate only applies when "
                    "filter_method='blank_fraction'. Use lr_fdr_target with "
                    "filter_method='lr'."
                )
            return

        raise ValueError("filter_method must be one of 'blank_fraction' or 'lr'.")

    def _apply_filter_method(
        self,
        filter_method: Literal["blank_fraction", "lr"],
        target_gross_misid_rate: float,
        lr_fdr_target: float,
    ) -> None:
        """
        Apply one supported downstream transcript filter to loaded barcodes.

        Parameters
        ----------
        filter_method : Literal['blank_fraction', 'lr']
            Function argument.
        target_gross_misid_rate : float
            Function argument.
        lr_fdr_target : float
            Function argument.

        Returns
        -------
        None
            Function result.
        """

        if self._verbose > 1:
            print(f"apply filter_method={filter_method}")

        if filter_method == "blank_fraction":
            self._filter_all_barcodes_blank_fraction(
                target_gross_misid_rate=float(target_gross_misid_rate)
            )
            return

        if filter_method == "lr":
            self._filter_all_barcodes_LR(lr_fdr_target=float(lr_fdr_target))
            return

        raise ValueError("filter_method must be one of 'blank_fraction' or 'lr'.")

    def optimize_filtering(
        self,
        assign_to_cells: bool = True,
        duplicate_radius_xy: float | None = None,
        duplicate_radius_z: float | None = None,
        filter_method: Literal["blank_fraction", "lr"] = "blank_fraction",
        target_gross_misid_rate: float = 0.05,
        lr_fdr_target: float = 0.05,
    ) -> None:
        """Re-apply downstream filtering to previously decoded exact-called transcripts.

        Parameters
        ----------
        assign_to_cells : bool, default False
            Assign barcodes to cells.
        duplicate_radius_xy : float, optional
            Override XY radius, in microns, for within-tile duplicate collapse.
        duplicate_radius_z : float, optional
            Override Z radius, in microns, for within-tile duplicate collapse.
        filter_method : {"blank_fraction", "lr"}, default "blank_fraction"
            Downstream filter to apply to decoded transcripts.
        target_gross_misid_rate : float, default 0.05
            Gross misidentification-rate target for blank-fraction filtering.
        lr_fdr_target : float, default 0.05
            False discovery rate target for LR filtering.

        Notes
        -----
        This method requires local decoded transcript parquet files produced by
        the exact two-threshold caller. Legacy decoded outputs without
        ``distance_min`` are rejected and must be regenerated.
        """

        if self._verbose >= 1:
            print("reprocess existing: load decoded transcripts")
        self._load_tile_decoding = True
        self._load_all_barcodes()
        if self._verbose >= 1:
            print(f"Number of loaded barcodes: {len(self._df_barcodes_loaded)}")
        self._load_tile_decoding = False
        self._validate_filter_configuration(
            filter_method=filter_method,
            target_gross_misid_rate=float(target_gross_misid_rate),
            lr_fdr_target=float(lr_fdr_target),
        )
        all_tiles = list(range(len(self._datastore.tile_ids)))
        if not (self._verbose == 0):
            self._verbose = 2
        self._apply_filter_method(
            filter_method=filter_method,
            target_gross_misid_rate=float(target_gross_misid_rate),
            lr_fdr_target=float(lr_fdr_target),
        )
        if len(all_tiles) or not (self._is_3D):
            if not (self._is_3D):
                radius_xy = (
                    self._datastore.voxel_size_zyx_um[-1]
                    if duplicate_radius_xy is None
                    else float(duplicate_radius_xy)
                )
                radius_z = (
                    self._datastore.voxel_size_zyx_um[0]
                    if duplicate_radius_z is None
                    else float(duplicate_radius_z)
                )
                self._remove_duplicates_within_tile(
                    radius_xy=radius_xy, radius_z=radius_z
                )
            else:
                self._remove_duplicates_in_tile_overlap()

        if not (self._verbose == 0):
            self._verbose = 1

        if assign_to_cells:
            self._assign_cells()
        self._save_barcodes()
        if self._verbose >= 1:
            print(f"Number of retained barcodes: {len(self._df_filtered_barcodes)}")


def time_stamp() -> str:
    """
    Time stamp.

    Returns
    -------
    str
        Function result.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
