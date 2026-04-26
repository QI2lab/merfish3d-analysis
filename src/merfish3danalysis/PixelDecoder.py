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

import gc
import shutil
import tempfile
import warnings
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from random import sample
from typing import Literal

import cupy as cp
import numpy as np
import pandas as pd
import rtree
from cucim.skimage.measure import label
from cucim.skimage.morphology import remove_small_objects
from cupyx.scipy.ndimage import gaussian_filter
from cuvs.distance import pairwise_distance
from roifile import roiread
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon
from skimage.measure import regionprops_table
from tqdm.auto import tqdm, trange

from merfish3danalysis.qi2labDataStore import qi2labDataStore

# filter warning from skimage
warnings.filterwarnings(
    "ignore",
    message="Only one label was provided to `remove_small_objects`. Did you mean to use a boolean array?",
)

# GPU helper functions


def decode_tiles_worker(
    datastore_path: Path,
    tile_indices: Sequence[int],
    gpu_id: int,
    merfish_bits: int,
    verbose: int,
    lowpass_sigma: Sequence[float],
    magnitude_threshold: Sequence[float],
    minimum_pixels: float,
    feature_predictor_threshold: float,
) -> None:
    """Worker that runs decode_one_tile on a subset of tiles under one GPU."""
    import cupy as cp
    import torch

    torch.cuda.set_device(gpu_id)
    cp.cuda.Device(gpu_id).use()
    cp.cuda.Stream.null.synchronize()

    local_datastore = qi2labDataStore(datastore_path)
    local_decoder = PixelDecoder(
        datastore=local_datastore,
        use_mask=False,
        merfish_bits=merfish_bits,
        num_gpus=1,
        verbose=0,
    )

    local_decoder._load_global_normalization_vectors(gpu_id=gpu_id)
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
            use_normalization=True,
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
    temp_dir: Path,
    iteration: int,
    lowpass_sigma: Sequence[float],
    magnitude_threshold: Sequence[float],
    minimum_pixels: float,
    feature_predictor_threshold: float,
) -> None:
    """Worker that runs one iteration of normalization-by-decoding on a GPU."""
    import cupy as cp
    import torch

    torch.cuda.set_device(gpu_id)
    cp.cuda.Device(gpu_id).use()
    cp.cuda.Stream.null.synchronize()

    local_datastore = qi2labDataStore(datastore_path)
    local_decoder = PixelDecoder(
        datastore=local_datastore,
        use_mask=False,
        merfish_bits=merfish_bits,
        num_gpus=1,
        verbose=0,
    )

    local_decoder._load_global_normalization_vectors(gpu_id=gpu_id)
    local_decoder._optimize_normalization_weights = True
    local_decoder._temp_dir = temp_dir

    # if iteration==0, skip use_normalization
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
            use_normalization=use_norm,
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
    """

    def __init__(
        self,
        datastore: qi2labDataStore,
        merfish_bits: int = 16,
        num_gpus: int = 1,
        verbose: int = 1,
        use_mask: bool | None = False,
        z_range: Sequence[int] | None = None,
    ) -> None:
        self._datastore_path = Path(datastore._datastore_path)
        self._datastore = datastore
        self._num_gpus = num_gpus
        self._verbose = verbose
        self._barcodes_filtered = False

        self._n_merfish_bits = merfish_bits

        if self._datastore.microscope_type == "2D":
            self._is_3D = False
        else:
            self._is_3D = True
        if z_range is None:
            self._z_crop = False
        else:
            self._z_crop = True
            self._z_range = [z_range[0], z_range[1]]

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
        self._global_normalization_loaded = False
        self._iterative_normalization_loaded = False
        self._blank_fraction_filter_results: dict[str, object] | None = None
        self._pixel_assignment_threshold = float(np.sqrt(2.0 - np.sqrt(2.0)))
        self._transcript_distance_threshold = float(np.sqrt(2.0 - 4.0 / np.sqrt(6.0)))

    @staticmethod
    def _compute_component_min_values(
        label_image: cp.ndarray, value_image: cp.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-component minima for a GPU label image."""

        labels_flat = cp.asarray(label_image).ravel()
        values_flat = cp.asarray(value_image, dtype=cp.float32).ravel()

        if labels_flat.shape != values_flat.shape:
            raise ValueError("Label and value images must have the same shape.")

        if labels_flat.size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        max_label = int(cp.max(labels_flat).get())
        min_values_by_label = cp.full(max_label + 1, cp.inf, dtype=cp.float32)
        cp.minimum.at(
            min_values_by_label,
            labels_flat.astype(cp.int32, copy=False),
            values_flat,
        )
        unique_labels = cp.unique(labels_flat)
        keep = unique_labels != 0

        return (
            cp.asnumpy(unique_labels[keep]).astype(np.int64, copy=False),
            cp.asnumpy(min_values_by_label[unique_labels[keep]]).astype(
                np.float32, copy=False
            ),
        )

    def _load_codebook(self) -> None:
        """Load the MERFISH codebook and remove one-bit rows from decoding."""

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
        self, gpu_id: int = 0, recalculate: bool = False
    ) -> None:
        """Load or calculate global normalization and background vectors.

        Parameters
        ----------
        gpu_id: int, default = 0
            GPU identifier
        recalculate : bool, default False
            Recompute global normalization/background vectors instead of
            reusing cached datastore values.
        """
        with cp.cuda.Device(gpu_id):
            normalization_vector = self._datastore.global_normalization_vector
            background_vector = self._datastore.global_background_vector
            if (
                not recalculate
                and normalization_vector is not None
                and background_vector is not None
            ):
                self._global_normalization_vector = cp.asarray(normalization_vector)
                self._global_background_vector = cp.asarray(background_vector)
                self._global_normalization_loaded = True
            else:
                self._global_normalization_vectors(gpu_id=gpu_id)

            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def _global_normalization_vectors(
        self,
        low_percentile_cut: float = 10.0,
        high_percentile_cut: float = 90.0,
        hot_pixel_threshold: int = 50000,
        gpu_id: int = 0,
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
        """

        with cp.cuda.Device(gpu_id):
            if len(self._datastore.tile_ids) > 5:
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

                    current_image = cp.where(
                        cp.asarray(feature_predictor_image, dtype=cp.float32) > 0.1,
                        cp.asarray(decon_image, dtype=cp.float32),
                        0.0,
                    )
                    current_image[current_image > hot_pixel_threshold] = cp.median(
                        current_image[current_image.shape[0] // 2, :, :]
                    ).astype(cp.float32)
                    if self._z_crop:
                        all_images.append(
                            cp.asnumpy(
                                current_image[self._z_range[0] : self._z_range[1], :]
                            ).astype(np.float32)
                        )
                    else:
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

            self._datastore.global_normalization_vector = (
                cp.asnumpy(normalization_vector).astype(np.float32).tolist()
            )
            self._datastore.global_background_vector = (
                cp.asnumpy(background_vector).astype(np.float32).tolist()
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
            normalization_vector = self._datastore.iterative_normalization_vector
            background_vector = self._datastore.iterative_background_vector

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

            bit_columns = [
                col
                for col in df_barcodes_loaded_no_blanks.columns
                if col.startswith("bit") and col.endswith("_mean_intensity")
            ]

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
            self._datastore.iterative_background_vector = (
                barcode_based_background_vector.astype(np.float32)
            )
            self._datastore.iterative_normalization_vector = (
                barcode_based_normalization_vector.astype(np.float32)
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
            self._datastore.iterative_normalization_vector = (
                barcode_based_normalization_vector
            )
            self._datastore.iterative_background_vector = (
                barcode_based_background_vector
            )

            self._iterative_normalization_loaded = True

            del df_barcodes_loaded_no_blanks
            gc.collect()
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def _load_bit_data(self, feature_predictor_threshold: float | None = 0.1) -> None:
        """Load raw data for all bits in the tile.

        Parameters
        ----------
        feature_predictor_threshold : float, default 0.1
            Threshold for feature_predictor image.
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

            if self._z_crop:
                current_mask = np.asarray(
                    feature_predictor_image[
                        self._z_range[0] : self._z_range[1], :
                    ].result(),
                    dtype=np.float32,
                )
                images.append(
                    np.where(
                        current_mask > feature_predictor_threshold,
                        np.asarray(
                            decon_image[
                                self._z_range[0] : self._z_range[1], :
                            ].result(),
                            dtype=np.float32,
                        ),
                        0,
                    )
                )
            else:
                current_mask = np.asarray(
                    feature_predictor_image.result(), dtype=np.float32
                )
                images.append(
                    np.where(
                        current_mask > feature_predictor_threshold,
                        np.asarray(decon_image.result(), dtype=np.float32),
                        0,
                    )
                )
            self._em_wvl.append(
                self._datastore.load_local_wavelengths_um(
                    tile=self._tile_idx,
                    bit=bit_id,
                )[1]
            )

        self._image_data = np.stack(images, axis=0)
        voxel_size_zyx_um = self._datastore.voxel_size_zyx_um
        self._pixel_size = voxel_size_zyx_um[1]
        self._axial_step = voxel_size_zyx_um[0]

        affine, origin, spacing = self._datastore.load_global_coord_xforms_um(
            tile=self._tile_idx
        )
        if affine is None or origin is None or spacing is None:
            if self._is_3D:
                affine = np.eye(4)
                origin = self._datastore.load_local_stage_position_zyx_um(
                    tile=self._tile_idx, round=0
                )
                spacing = self._datastore.voxel_size_zyx_um
            else:
                affine = np.eye(4)
                origin = self._datastore.load_local_stage_position_zyx_um(
                    tile=self._tile_idx, round=0
                )
                origin = [0, origin[0], origin[1]]
                spacing = self._datastore.voxel_size_zyx_um

        self._affine = affine
        self._origin = origin
        self._spacing = spacing

        del images
        gc.collect()

    def _lp_filter(self, gpu_id: int = 0, sigma: Sequence[int] = (3, 1, 1)) -> None:
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
                if self._is_3D:
                    image_data_cp = cp.asarray(self._image_data[i, :], dtype=cp.float32)
                    max_image_data = cp.asnumpy(
                        cp.max(image_data_cp, axis=(0, 1, 2))
                    ).astype(np.float32)
                    if max_image_data == 0:
                        self._image_data_lp[i, :, :, :] = 0
                    else:
                        self._image_data_lp[i, :, :, :] = cp.asnumpy(
                            gaussian_filter(image_data_cp, sigma=sigma)
                        ).astype(np.float32)
                        max_image_data_lp = np.max(
                            self._image_data_lp[i, :, :, :], axis=(0, 1, 2)
                        )
                        self._image_data_lp[i, :, :, :] = self._image_data_lp[
                            i, :, :, :
                        ] * (max_image_data / max_image_data_lp)
                else:
                    for z_idx in range(self._image_data.shape[1]):
                        image_data_cp = cp.asarray(
                            self._image_data[i, z_idx, :], dtype=cp.float32
                        )
                        max_image_data = cp.asnumpy(
                            cp.max(image_data_cp, axis=(0, 1))
                        ).astype(np.float32)
                        if max_image_data == 0:
                            self._image_data_lp[i, z_idx, :, :] = 0
                        else:
                            self._image_data_lp[i, z_idx, :, :] = cp.asnumpy(
                                gaussian_filter(
                                    image_data_cp, sigma=(sigma[1], sigma[2])
                                )
                            ).astype(np.float32)
                            max_image_data_lp = np.max(
                                self._image_data_lp[i, z_idx, :, :], axis=(0, 1)
                            )
                            self._image_data_lp[i, z_idx, :, :] = self._image_data_lp[
                                i, z_idx, :, :
                            ] * (max_image_data / max_image_data_lp)

            self._filter_type = "lp"

            del image_data_cp
            del self._image_data
            gc.collect()
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

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

        Returns
        -------
        registered_space_point : np.ndarray
            Registered space point.
        """

        physical_space_point = pixel_space_point * spacing + origin
        registered_space_point = (
            np.array(affine) @ np.array([*list(physical_space_point), 1])
        )[:-1]

        return registered_space_point

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
                    intensity_image = np.concatenate(
                        [
                            np.expand_dims(self._distance_image, axis=0),
                            self._image_data_lp,
                        ],
                        axis=0,
                    ).transpose(1, 2, 3, 0)
                else:
                    intensity_image = np.concatenate(
                        [
                            np.expand_dims(self._distance_image, axis=0),
                            self._image_data,
                        ],
                        axis=0,
                    ).transpose(1, 2, 3, 0)
            else:
                intensity_image = np.concatenate(
                    [
                        np.expand_dims(self._distance_image, axis=0),
                        self._scaled_pixel_images,
                    ],
                    axis=0,
                ).transpose(1, 2, 3, 0)

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
                codewords_label_image_cp, min_size=minimum_pixels
            )

            max_label = int(cp.max(codewords_label_image_cp).get())
            label_to_id = np.full(max_label + 1, -1, dtype=np.int16)
            label_to_distance_min = np.full(max_label + 1, np.nan, dtype=np.float32)

            if max_label > 0:
                labels_flat_cp = codewords_label_image_cp.ravel()
                decoded_flat_cp = decoded_image_cp.ravel()
                order_cp = cp.argsort(labels_flat_cp)
                labels_sorted_cp = labels_flat_cp[order_cp]
                decoded_sorted_cp = decoded_flat_cp[order_cp]
                uniq_labels_cp, first_idx_cp = cp.unique(
                    labels_sorted_cp, return_index=True
                )
                label_to_id_cp = cp.full(max_label + 1, -1, dtype=cp.int16)
                label_to_id_cp[uniq_labels_cp] = decoded_sorted_cp[first_idx_cp]
                label_to_id_cp[0] = -1
                label_to_id = cp.asnumpy(label_to_id_cp)

                distance_image_cp = cp.asarray(self._distance_image, dtype=cp.float32)
                component_labels, component_distance_min = (
                    self._compute_component_min_values(
                        codewords_label_image_cp, distance_image_cp
                    )
                )
                label_to_distance_min[component_labels] = component_distance_min

                del (
                    labels_flat_cp,
                    decoded_flat_cp,
                    order_cp,
                    labels_sorted_cp,
                    decoded_sorted_cp,
                    uniq_labels_cp,
                    first_idx_cp,
                    label_to_id_cp,
                    distance_image_cp,
                )

            # move arrays to CPU for the existing regionprops path
            codewords_label_image = cp.asnumpy(codewords_label_image_cp)

            del codewords_label_image_cp, decoded_image_cp
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

            # Map each region (component label) -> decoded_id
            region_labels = df_barcode["label"].to_numpy(dtype=np.int64)
            decoded_ids = label_to_id[region_labels].astype(np.int32, copy=False)
            df_barcode["decoded_id"] = decoded_ids
            df_barcode["distance_min"] = label_to_distance_min[region_labels]

            # Sanity: drop any region that somehow mapped to background
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

            if self._z_crop:
                df_barcode["z"] = df_barcode["z"] + self._z_range[0]

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
                )

            df_barcode["global_z"] = np.round(pts[:, 0], 2)
            df_barcode["global_y"] = np.round(pts[:, 1], 2)
            df_barcode["global_x"] = np.round(pts[:, 2], 2)

            if "intensity_mean-0" in df_barcode.columns:
                df_barcode = df_barcode.rename(
                    columns={"intensity_mean-0": "distance_mean"}
                )

            for i in range(1, self._n_merfish_bits + 1):
                src = f"intensity_mean-{i}"
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
                props,
                props_magnitude,
                df_magnitude,
                df_barcode,
            )
            gc.collect()
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def _save_barcodes(self) -> None:
        """Save barcodes to datastore."""

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
                    self._df_barcodes, tile=self._tile_idx
                )
            else:
                self._datastore.save_global_filtered_decoded_spots(
                    self._df_filtered_barcodes
                )

    def _reformat_barcodes_for_baysor(self) -> None:
        """Reformat barcodes for Baysor and save to datastore."""

        if self._barcodes_filtered:
            missing_columns = [
                col
                for col in [
                    "gene_id",
                    "global_z",
                    "global_y",
                    "global_x",
                    "cell_id",
                    "tile_idx",
                    "distance_mean",
                ]
                if col not in self._df_filtered_barcodes.columns
            ]
            if missing_columns:
                print(f"The following columns are missing: {missing_columns}")
            baysor_df = self._df_filtered_barcodes[
                [
                    "gene_id",
                    "global_z",
                    "global_y",
                    "global_x",
                    "cell_id",
                    "tile_idx",
                    "distance_mean",
                ]
            ].copy()
            baysor_df.rename(
                columns={
                    "gene_id": "feature_name",
                    "global_x": "x_location",
                    "global_y": "y_location",
                    "global_z": "z_location",
                    "barcode_id": "codeword_index",
                    "tile_idx": "fov_name",
                    "distance_mean": "qv",
                },
                inplace=True,
            )

            baysor_df["cell_id"] = baysor_df["cell_id"] + 1
            baysor_df["transcript_id"] = pd.util.hash_pandas_object(
                baysor_df, index=False
            )
            baysor_df["is_gene"] = ~baysor_df[
                "feature_name"
            ].str.lower().str.startswith("blank", na=False)
            self._datastore.save_spots_prepped_for_baysor(baysor_df)

    def _load_all_barcodes(self) -> None:
        """Load all barcodes from datastore."""

        if self._optimize_normalization_weights:
            decoded_dir_path = self._temp_dir

            tile_files = decoded_dir_path.glob("*.parquet")
            tile_files = sorted(tile_files, key=lambda x: x.name)

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
                tile_data.append(self._datastore.load_local_decoded_spots(tile_id))
            self._df_barcodes_loaded = pd.concat(tile_data)
        else:
            self._df_filtered_barcodes = (
                self._datastore.load_global_filtered_decoded_spots()
            )
            self._df_barcodes_loaded = self._df_filtered_barcodes.copy()
            self._barcodes_filtered = True

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
        """Filter transcripts using blank-fraction histograms in feature space."""

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
                "distance_mean",
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
                "distance_mean",
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
        return Polygon(roi.subpixel_coordinates[:, ::-1])

    def _assign_cells(self) -> None:
        """Assign cells to barcodes using Cellpose ROIs."""

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
        distance_mean = self._df_filtered_barcodes["distance_mean"].to_numpy(
            dtype=float, copy=False
        )

        tree = cKDTree(coords)
        pairs = tree.query_pairs(radius)

        rows_to_drop = set()
        distances = []
        for i, j in pairs:
            if tile_idxs[i] != tile_idxs[j]:
                if (distance_min[i], distance_mean[i], i) <= (
                    distance_min[j],
                    distance_mean[j],
                    j,
                ):
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
        Ties are broken deterministically by ``distance_mean`` and then the
        original row index (lower index wins).

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
        ``tile_idx``, ``gene_id``, ``distance_min``, ``distance_mean``.
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
        dmean = df["distance_mean"].to_numpy(dtype=float, copy=False)

        rows_to_drop: set[int] = set()

        # Union-Find (Disjoint Set)
        def uf_find(parent: np.ndarray, x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def uf_union(parent: np.ndarray, rank: np.ndarray, a: int, b: int) -> None:
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
                # Lexicographic: primary key is distance_min, then distance_mean, then row index.
                best_global = glob_members[
                    np.lexsort((glob_members, dmean[glob_members], dmin[glob_members]))
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
        """Display results using Napari."""

        import napari
        from qtpy.QtWidgets import QApplication

        def on_close_callback() -> None:
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
        """Cleanup memory."""
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
        lowpass_sigma: Sequence[float] | None = (3, 1, 1),
        magnitude_threshold: list[float, float] | None = (0.9, 10.0),
        minimum_pixels: float | None = 2.0,
        use_normalization: bool | None = True,
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
        magnitude_threshold: Sequence[float], default (0.9, 10.0)
            L2-norm threshold
        minimum_pixels : float, default 2.0
            Minimum number of pixels for a barcode.
        use_normalization : bool, default True
            Use normalization.
        feature_predictor_threshold : float, default 0.1
            feature_predictor threshold.

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
            if use_normalization:
                self._load_iterative_normalization_vectors(gpu_id=gpu_id)

            self._tile_idx = tile_idx
            self._load_bit_data(feature_predictor_threshold=feature_predictor_threshold)
            if not (np.any(lowpass_sigma == 0)):
                self._lp_filter(sigma=lowpass_sigma, gpu_id=gpu_id)
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
        minimum_pixels: float | None = 2.0,
        feature_predictor_threshold: float | None = 0.1,
        lowpass_sigma: Sequence[float] | None = (3, 1, 1),
        magnitude_threshold: Sequence[float] | None = (0.9, 10.0),
    ) -> None:
        """Iteratively refine normalization vectors using exact-called transcripts.

        Parameters
        ----------
        n_random_tiles : int, default 5
            Number of random tiles.
        n_iterations : int, default 10
            Number of iterations.
        minimum_pixels : float, default = 2.0
            Minimum number of pixels for a barcode.
        feature_predictor_threshold : float, default = 0.1
            feature_predictor threshold.
        lowpass_sigma : Sequence[float], default = (3, 1, 1)
            Lowpass sigma.
        magnitude_threshold: Sequence[float], default = (0.9,10.0)
            L2-norm threshold
        """
        if self._num_gpus < 1:
            raise RuntimeError("No GPUs allocated.")
        all_tiles = list(range(len(self._datastore.tile_ids)))

        # preload global normalization once
        self._iterative_background_vector = None
        self._iterative_normalization_vector = None
        self._global_background_vector = None
        self._optimize_normalization_weights = True
        self._load_global_normalization_vectors(gpu_id=0, recalculate=True)
        temp_dir = Path(tempfile.mkdtemp())
        self._temp_dir = temp_dir

        # split the same set of random tiles each iteration
        if len(all_tiles) > n_random_tiles:
            random_tiles = sample(all_tiles, n_random_tiles)
        else:
            random_tiles = all_tiles
        chunk_size = (len(random_tiles) + self._num_gpus - 1) // self._num_gpus

        if self._verbose >= 1:
            iterator = trange(n_iterations, desc="Iterative normalization")
        else:
            iterator = range(n_iterations)

        for iteration in iterator:
            # launch one process per GPU
            processes = []
            for gpu in range(self._num_gpus):
                start = gpu * chunk_size
                end = min(start + chunk_size, len(random_tiles))
                subset = random_tiles[start:end]
                if not subset:
                    continue
                p = mp.Process(
                    target=_optimize_norm_worker,
                    args=(
                        self._datastore_path,
                        subset,
                        gpu,
                        self._n_merfish_bits,
                        temp_dir,
                        iteration,
                        lowpass_sigma,
                        magnitude_threshold,
                        minimum_pixels,
                        feature_predictor_threshold,
                    ),
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            with cp.cuda.Device(0):
                # gather results and update
                self._load_all_barcodes()
                if not (self._is_3D):
                    radius_xy = self._datastore.voxel_size_zyx_um[-1]
                    radius_z = self._datastore.voxel_size_zyx_um[0]
                    self._remove_duplicates_within_tile(
                        radius_xy=radius_xy, radius_z=radius_z
                    )
                self._load_global_normalization_vectors(gpu_id=0)
                self._iterative_normalization_vectors(gpu_id=0)
                del self._global_background_vector, self._global_normalization_vector
                gc.collect()
                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

        # cleanup temp files, etc.
        self._cleanup()
        self._optimize_normalization_weights = False
        shutil.rmtree(self._temp_dir)

    def decode_all_tiles(
        self,
        assign_to_cells: bool = True,
        prep_for_baysor: bool = True,
        lowpass_sigma: Sequence[float] | None = (3, 1, 1),
        magnitude_threshold: Sequence[float] | None = (0.9, 10.0),
        minimum_pixels: float | None = 2.0,
        feature_predictor_threshold: float | None = 0.1,
        filter_method: Literal["blank_fraction", "lr"] = "blank_fraction",
        target_gross_misid_rate: float = 0.05,
        lr_fdr_target: float = 0.05,
    ) -> None:
        """Decode all tiles and apply the selected downstream transcript filter.

        Parameters
        ----------
        assign_to_cells: bool, default = True
            Assign codewords to cells
        prep_for_baysor: bool, default = True
            Create a baysor-compatible .parquet file
        lowpass_sigma : Sequence[float], default = (3, 1, 1)
            Lowpass sigma.
        magnitude_threshold: Sequence[float], default = (0.9, 10.0)
            Accept pixels with magnitudes between low and high values
        minimum_pixels : float, default 2.0
            Minimum number of pixels for a barcode.
        feature_predictor_threshold : float, default 0.1
            feature_predictor threshold.
        filter_method : {"blank_fraction", "lr"}, default "blank_fraction"
            Downstream filter to apply after exact transcript calling.
        target_gross_misid_rate : float, default 0.05
            Gross misidentification-rate target for blank-fraction filtering.
        lr_fdr_target: float, default = 0.05
            False discovery rate target for LR filtering.
        """

        if self._num_gpus < 1:
            raise RuntimeError("No GPUs allocated.")
        all_tiles = list(range(len(self._datastore.tile_ids)))
        chunk_size = (len(all_tiles) + self._num_gpus - 1) // self._num_gpus

        processes = []
        for gpu in range(self._num_gpus):
            start = gpu * chunk_size
            end = min(start + chunk_size, len(all_tiles))
            subset = all_tiles[start:end]
            if not subset:
                continue
            p = mp.Process(
                target=decode_tiles_worker,
                args=(
                    self._datastore_path,
                    subset,
                    gpu,
                    self._n_merfish_bits,
                    self._verbose,
                    lowpass_sigma,
                    magnitude_threshold,
                    minimum_pixels,
                    feature_predictor_threshold,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # load all barcodes and filter
        self._load_tile_decoding = True
        self._load_all_barcodes()
        if self._verbose >= 1:
            print(f"Number of loaded barcodes: {len(self._df_barcodes_loaded)}")
            print(f"Verbosity:  {self._verbose}")
        if filter_method == "blank_fraction":
            self._filter_all_barcodes_blank_fraction(
                target_gross_misid_rate=float(target_gross_misid_rate)
            )
        elif filter_method == "lr":
            self._filter_all_barcodes_LR(lr_fdr_target=float(lr_fdr_target))
        else:
            raise ValueError("filter_method must be either 'blank_fraction' or 'lr'.")
        if not (self._is_3D):
            radius_xy = self._datastore.voxel_size_zyx_um[-1]
            radius_z = self._datastore.voxel_size_zyx_um[0]
            self._remove_duplicates_within_tile(radius_xy=radius_xy, radius_z=radius_z)

        if len(all_tiles) > 1:
            self._remove_duplicates_in_tile_overlap()
        if assign_to_cells:
            self._assign_cells()
        self._save_barcodes()
        if self._verbose >= 1:
            print(f"Number of retained barcodes: {len(self._df_filtered_barcodes)}")
        if prep_for_baysor:
            self._reformat_barcodes_for_baysor()
        self._cleanup()

    def optimize_filtering(
        self,
        assign_to_cells: bool = False,
        prep_for_baysor: bool = True,
        filter_method: Literal["blank_fraction", "lr"] = "blank_fraction",
        target_gross_misid_rate: float = 0.05,
        lr_fdr_target: float = 0.05,
    ) -> None:
        """Re-apply downstream filtering to previously decoded exact-called transcripts.

        Parameters
        ----------
        assign_to_cells : bool, default False
            Assign barcodes to cells.
        prep_for_baysor : bool, default True
            Prepare barcodes for Baysor.
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

        self._load_tile_decoding = True
        self._load_all_barcodes()
        if self._verbose >= 1:
            print(f"Number of loaded barcodes: {len(self._df_barcodes_loaded)}")
        self._load_tile_decoding = False
        all_tiles = list(range(len(self._datastore.tile_ids)))
        if not (self._verbose == 0):
            self._verbose = 2
        if filter_method == "blank_fraction":
            self._filter_all_barcodes_blank_fraction(
                target_gross_misid_rate=float(target_gross_misid_rate)
            )
        elif filter_method == "lr":
            self._filter_all_barcodes_LR(lr_fdr_target=float(lr_fdr_target))
        else:
            raise ValueError("filter_method must be either 'blank_fraction' or 'lr'.")
        if len(all_tiles) or not (self._is_3D):
            if not (self._is_3D):
                radius_xy = self._datastore.voxel_size_zyx_um[-1]
                radius_z = self._datastore.voxel_size_zyx_um[0]
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
        if prep_for_baysor:
            self._reformat_barcodes_for_baysor()


def time_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
