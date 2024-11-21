"""
PixelDecoder: Perform pixel-based decoding for qi2lab widefield MERFISH data using GPU acceleration.

Shepherd 2024/03 - rework of GPU logic to reduce out-of-memory crashes
Shepherd 2024/01 - updates for qi2lab MERFISH file format v1.0
"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
import numpy as np
from pathlib import Path
import gc
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter
from cucim.skimage.measure import label
from cucim.skimage.morphology import remove_small_objects
from cupyx.scipy.spatial.distance import cdist
from skimage.measure import regionprops_table
from typing import Union, Optional, Sequence, Tuple
import pandas as pd
from random import sample
from tqdm import tqdm
from shapely.geometry import Point, Polygon
from roifile import roiread
import json
import rtree
from scipy.spatial import cKDTree
import warnings
import tempfile
import shutil

warnings.filterwarnings(
    "ignore",
    message="Only one label was provided to `remove_small_objects`. Did you mean to use a boolean array?",
)


class PixelDecoder:
    """
    Retrieve and process one tile from qi2lab 3D widefield zarr structure.
    Normalize codebook and data, perform plane-by-plane pixel decoding,
    extract barcode features, and save to disk.

    Parameters
    ----------
    dataset_path : Union[str, Path]
        Path to Zarr dataset
    use_mask: Optiona[bool] = False
        use mask stored in polyDT directory
    merfish_bits: int = 16
        number of merfish bits. Assumes that in codebook, MERFISH rounds are [0,merfish_bits].
    verbose: int = 1
        control verbosity. 0 - no output, 1 - tqdm bars, 2 - diagnostic outputs
    """

    def __init__(
        self,
        datastore: qi2labDataStore,
        use_mask: Optional[bool] = False,
        z_range: Optional[Sequence[int]] = None,
        include_blanks: Optional[bool] = True,
        merfish_bits: int = 16,
        verbose: int = 1,
    ):
        self._datastore = datastore
        self._verbose = verbose
        self._barcodes_filtered = False
        self._include_blanks = include_blanks

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
        self._distance_threshold = 0.5172  # default for HW4D4 code. TO DO: calculate based on self._num_on-bits
        self._magnitude_threshold = 0.9  # default for HW4D4 code

    def _load_codebook(self):
        """Load and parse codebook into gene_id and codeword matrix."""

        self._df_codebook = self._datastore.codebook.copy()
        self._df_codebook.fillna(0, inplace=True)

        self._blank_count = (
            self._df_codebook["gene_id"].str.lower().str.startswith("blank").sum()
        )

        if not (self._include_blanks):
            self._df_codebook.drop(
                self._df_codebook[self._df_codebook[0].str.startswith("Blank")].index,
                inplace=True,
            )

        self._codebook_matrix = self._df_codebook.iloc[:, 1:].to_numpy().astype(int)
        self._gene_ids = self._df_codebook.iloc[:, 0].tolist()

    def _normalize_codebook(self, include_errors: bool = False):
        """Normalize each codeword by L2 norm."""

        self._barcode_set = cp.asarray(
            self._codebook_matrix[:, 0 : self._n_merfish_bits]
        )
        magnitudes = cp.linalg.norm(self._barcode_set, axis=1, keepdims=True)
        magnitudes[magnitudes == 0] = 1  # ensure with smFISH rounds have magnitude 1

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
            return cp.asnumpy(all_barcodes)

    def _load_global_normalization_vectors(self):
        normalization_vector = self._datastore.global_normalization_vector
        background_vector = self._datastore.global_background_vector
        if normalization_vector is not None and background_vector is not None:
            self._global_normalization_vector = cp.asarray(normalization_vector)
            self._global_background_vector = cp.asarray(background_vector)
            self._global_normalization_loaded = True
        else:
            self._global_normalization_vectors()

    def _global_normalization_vectors(
        self,
        low_percentile_cut: float = 10.0,
        high_percentile_cut: float = 90.0,
        hot_pixel_threshold: int = 50000,
    ):
        if len(self._datastore.tile_ids) > 5:
            random_tiles = sample(self._datastore.tile_ids, 5)
        else:
            random_tiles = self._datastore.tile_ids

        normalization_vector = cp.ones(len(self._datastore.bit_ids), dtype=cp.float32)
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
                iterable_tiles = tqdm(random_tiles, desc="loading tiles", leave=False)
            else:
                iterable_tiles = random_tiles

            for tile_id in iterable_tiles:
                decon_image = self._datastore.load_local_registered_image(
                    tile=tile_id, bit=bit_id, return_future=False
                )
                ufish_image = self._datastore.load_local_ufish_image(
                    tile=tile_id, bit=bit_id, return_future=False
                )

                current_image = cp.where(
                    cp.asarray(ufish_image, dtype=cp.float32) > 0.1,
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
                iterable_tiles = random_tiles

            low_pixels = []
            for tile_idx, tile_id in iterable_tiles:
                current_image = cp.asarray(all_images[tile_idx, :], dtype=cp.float32)
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
                iterable_tiles = random_tiles

            high_pixels = []
            for tile_idx, tile_id in iterable_tiles:
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

    def _load_iterative_normalization_vectors(self):
        normalization_vector = self._datastore.iterative_normalization_vector
        background_vector = self._datastore.iterative_background_vector

        if normalization_vector is not None and background_vector is not None:
            background_vector = np.nan_to_num(background_vector, 0.0)
            normalization_vector = np.nan_to_num(normalization_vector, 1.0)
            self._iterative_normalization_vector = cp.asarray(normalization_vector)
            self._iterative_background_vector = cp.asarray(background_vector)
            self._iterative_normalization_loaded = True
        else:
            self._iterative_normalization_vectors()

    def _iterative_normalization_vectors(self):
        df_barcodes_loaded_no_blanks = self._df_barcodes_loaded[
            ~self._df_barcodes_loaded["gene_id"].str.startswith("Blank")
        ]

        bit_columns = [
            col
            for col in df_barcodes_loaded_no_blanks.columns
            if col.startswith("bit") and col.endswith("_mean_intensity")
        ]

        barcode_intensities = []
        barcode_background = []
        for index, row in df_barcodes_loaded_no_blanks.iterrows():
            selected_columns = [
                f'bit{row["on_bit_1"]:02d}_mean_intensity',
                f'bit{row["on_bit_2"]:02d}_mean_intensity',
                f'bit{row["on_bit_3"]:02d}_mean_intensity',
                f'bit{row["on_bit_4"]:02d}_mean_intensity',
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
                cp.asnumpy(self._global_background_vector[0 : self._n_merfish_bits]), 1
            )
            old_iterative_normalization_vector = np.round(
                cp.asnumpy(self._global_normalization_vector[0 : self._n_merfish_bits]),
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
            np.abs(barcode_based_background_vector - old_iterative_background_vector), 1
        )
        diff_iterative_normalization_vector = np.round(
            np.abs(
                barcode_based_normalization_vector - old_iterative_normalization_vector
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
            print("---")
            print("Background")
            print(diff_iterative_background_vector)
            print(barcode_based_background_vector)
            print("Foreground")
            print(diff_iterative_normalization_vector)
            print(barcode_based_normalization_vector)
            print("---")

        self._iterative_normalization_vector = barcode_based_normalization_vector
        self._iterative_background_vector = barcode_based_background_vector
        self._datastore.iterative_normalization_vector = (
            barcode_based_normalization_vector
        )
        self._datastore.iterative_background_vector = barcode_based_background_vector

        self._iterative_normalization_loaded = True

        del df_barcodes_loaded_no_blanks
        gc.collect()

    def _load_bit_data(self, ufish_threshold: Optional[float] = 0.1):
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
            ufish_image = self._datastore.load_local_ufish_image(
                tile=self._tile_idx,
                bit=bit_id,
            )

            if self._z_crop:
                current_mask = np.asarray(
                    ufish_image[self._z_range[0] : self._z_range[1], :].result(),
                    dtype=np.float32,
                )
                images.append(
                    np.where(
                        current_mask > ufish_threshold,
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
                current_mask = np.asarray(ufish_image.result(), dtype=np.float32)
                images.append(
                    np.where(
                        current_mask > ufish_threshold,
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

    def _lp_filter(self, sigma=(3, 1, 1)):
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
            iterable_lp = self._image_data_lp

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
                            gaussian_filter(image_data_cp, sigma=(sigma[1], sigma[2]))
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
        cp.get_default_memory_pool().free_all_blocks()

    @staticmethod
    def _scale_pixel_traces(
        pixel_traces: Union[np.ndarray, cp.ndarray],
        background_vector: Union[np.ndarray, cp.ndarray],
        normalization_vector: Union[np.ndarray, cp.ndarray],
        merfish_bits=16,
    ) -> cp.ndarray:
        if isinstance(pixel_traces, np.ndarray):
            pixel_traces = cp.asarray(pixel_traces, dtype=cp.float32)
        if isinstance(background_vector, np.ndarray):
            background_vector = cp.asarray(background_vector, dtype=cp.float32)
        if isinstance(normalization_vector, np.ndarray):
            normalization_vector = cp.asarray(normalization_vector, dtype=cp.float32)

        background_vector = background_vector[0:merfish_bits]
        normalization_vector = normalization_vector[0:merfish_bits]

        return (pixel_traces - background_vector[:, cp.newaxis]) / normalization_vector[
            :, cp.newaxis
        ]

    @staticmethod
    def _clip_pixel_traces(
        pixel_traces: Union[np.ndarray, cp.ndarray],
        clip_lower: float = 0.0,
        clip_upper: float = 1.0,
    ) -> cp.ndarray:
        return cp.clip(pixel_traces, clip_lower, clip_upper, pixel_traces)

    @staticmethod
    def _normalize_pixel_traces(
        pixel_traces: Union[np.ndarray, cp.ndarray],
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        if isinstance(pixel_traces, np.ndarray):
            pixel_traces = cp.asarray(pixel_traces, dtype=cp.float32)

        norms = cp.linalg.norm(pixel_traces, axis=0)
        norms = cp.where(norms == 0, np.inf, norms)
        normalized_traces = pixel_traces / norms
        norms = cp.where(norms == np.inf, -1, norms)

        # del pixel_traces
        # gc.collect()
        # cp.get_default_memory_pool().free_all_blocks()

        return normalized_traces, norms

    @staticmethod
    def _calculate_distances(
        pixel_traces: Union[np.ndarray, cp.ndarray],
        codebook_matrix: Union[np.ndarray, cp.ndarray],
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        if isinstance(pixel_traces, np.ndarray):
            pixel_traces = cp.asarray(pixel_traces, dtype=cp.float32)
        if isinstance(codebook_matrix, np.ndarray):
            codebook_matrix = cp.asarray(codebook_matrix, dtype=cp.float32)

        distances = cdist(
            cp.ascontiguousarray(pixel_traces.T),
            cp.ascontiguousarray(codebook_matrix),
            metric="euclidean",
        )

        min_indices = cp.argmin(distances, axis=1)
        min_distances = cp.min(distances, axis=1)

        del pixel_traces, codebook_matrix
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

        return min_distances, min_indices

    def _decode_pixels(
        self, distance_threshold: float = 0.5172, magnitude_threshold: float = 1.0
    ):
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
            iterable_z = tqdm(range(original_shape[1]), desc="decoding", leave=False)
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
                )
            elif self._global_normalization_loaded:
                scaled_pixel_traces = self._scale_pixel_traces(
                    scaled_pixel_traces,
                    self._global_background_vector,
                    self._global_normalization_vector,
                    self._n_merfish_bits,
                )

            scaled_pixel_traces = self._clip_pixel_traces(scaled_pixel_traces)
            normalized_pixel_traces, pixel_magnitude_trace = (
                self._normalize_pixel_traces(scaled_pixel_traces)
            )
            distance_trace, codebook_index_trace = self._calculate_distances(
                normalized_pixel_traces, self._decoding_matrix
            )

            del normalized_pixel_traces
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

            decoded_trace = cp.full(distance_trace.shape[0], -1, dtype=cp.int16)
            mask_trace = distance_trace < distance_threshold
            decoded_trace[mask_trace] = codebook_index_trace[mask_trace]
            decoded_trace[pixel_magnitude_trace <= magnitude_threshold] = -1

            self._decoded_image[z_idx, :] = cp.asnumpy(
                cp.reshape(cp.round(decoded_trace, 3), z_plane_shape[1:])
            )
            self._magnitude_image[z_idx, :] = cp.asnumpy(
                cp.reshape(cp.round(pixel_magnitude_trace, 3), z_plane_shape[1:])
            )
            self._scaled_pixel_images[:, z_idx, :] = cp.asnumpy(
                cp.reshape(cp.round(scaled_pixel_traces, 3), z_plane_shape)
            )
            self._distance_image[z_idx, :] = cp.asnumpy(
                cp.reshape(cp.round(distance_trace, 3), z_plane_shape[1:])
            )

            del (
                decoded_trace,
                pixel_magnitude_trace,
                scaled_pixel_traces,
                distance_trace,
            )
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

    @staticmethod
    def _warp_pixel(
        pixel_space_point: np.ndarray,
        spacing: np.ndarray,
        origin: np.ndarray,
        affine: np.ndarray,
    ) -> np.ndarray:
        physical_space_point = pixel_space_point * spacing + origin
        registered_space_point = (
            np.array(affine) @ np.array(list(physical_space_point) + [1])
        )[:-1]

        return registered_space_point

    def _extract_barcodes(self, minimum_pixels: int = 2, maximum_pixels: int = 200):
        if self._verbose > 1:
            print("extract barcodes")
        if self._verbose >= 1:
            iterable_barcode = tqdm(
                range(self._codebook_matrix.shape[0]), desc="barcode", leave=False
            )
        else:
            iterable_barcode = range(self._codebook_matrix.shape[0])
        decoded_image = cp.asarray(self._decoded_image, dtype=cp.int16)
        if self._optimize_normalization_weights:
            if self._filter_type == "lp":
                intensity_image = np.concatenate(
                    [np.expand_dims(self._distance_image, axis=0), self._image_data_lp],
                    axis=0,
                ).transpose(1, 2, 3, 0)
            else:
                intensity_image = np.concatenate(
                    [np.expand_dims(self._distance_image, axis=0), self._image_data],
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

        for barcode_index in iterable_barcode:
            on_bits_indices = np.where(self._codebook_matrix[barcode_index])[0]

            if len(on_bits_indices) == 1:
                break

            if self._is_3D:
                if self._verbose > 1:
                    print("")
                    print("label image")
                labeled_image = label(decoded_image == barcode_index, connectivity=3)

                if self._verbose > 1:
                    print("remove large")
                pixel_counts = cp.bincount(labeled_image.ravel())
                large_labels = cp.where(pixel_counts >= maximum_pixels)[0]
                large_label_mask = cp.zeros_like(labeled_image, dtype=bool)
                large_label_mask = cp.isin(labeled_image, large_labels)
                labeled_image[large_label_mask] = 0

                if self._verbose > 1:
                    print("remove small")
                labeled_image = remove_small_objects(
                    labeled_image, min_size=(minimum_pixels - 1)
                )
                if self._verbose > 1:
                    print("regionprops table")

                props = regionprops_table(
                    cp.asnumpy(labeled_image).astype(np.int32),
                    intensity_image=intensity_image,
                    properties=[
                        "area",
                        "centroid",
                        "intensity_mean",
                        "moments_normalized",
                        "inertia_tensor_eigvals",
                    ],
                )

                del labeled_image
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()

                df_barcode = pd.DataFrame(props)

                df_barcode["on_bit_1"] = on_bits_indices[0] + 1
                df_barcode["on_bit_2"] = on_bits_indices[1] + 1
                df_barcode["on_bit_3"] = on_bits_indices[2] + 1
                df_barcode["on_bit_4"] = on_bits_indices[3] + 1
                df_barcode["barcode_id"] = df_barcode.apply(
                    lambda x: (barcode_index + 1), axis=1
                )
                df_barcode["gene_id"] = df_barcode.apply(
                    lambda x: self._gene_ids[barcode_index], axis=1
                )
                df_barcode["tile_idx"] = self._tile_idx

                df_barcode.rename(columns={"centroid-0": "z"}, inplace=True)
                df_barcode.rename(columns={"centroid-1": "y"}, inplace=True)
                df_barcode.rename(columns={"centroid-2": "x"}, inplace=True)

                if self._z_crop:
                    df_barcode["z"] = df_barcode["z"] + self._z_range[0]

                df_barcode["tile_z"] = np.round(df_barcode["z"], 0).astype(int)
                df_barcode["tile_y"] = np.round(df_barcode["y"], 0).astype(int)
                df_barcode["tile_x"] = np.round(df_barcode["x"], 0).astype(int)
                pts = df_barcode[["z", "y", "x"]].to_numpy()
                for pt_idx, pt in enumerate(pts):
                    pts[pt_idx, :] = self._warp_pixel(
                        pts[pt_idx, :].copy(), self._spacing, self._origin, self._affine
                    )

                df_barcode["global_z"] = np.round(pts[:, 0], 2)
                df_barcode["global_y"] = np.round(pts[:, 1], 2)
                df_barcode["global_x"] = np.round(pts[:, 2], 2)

                df_barcode.rename(
                    columns={"intensity_mean-0": "distance_mean"}, inplace=True
                )
                for i in range(1, self._n_merfish_bits + 1):
                    df_barcode.rename(
                        columns={
                            "intensity_mean-" + str(i): "bit"
                            + str(i).zfill(2)
                            + "_mean_intensity"
                        },
                        inplace=True,
                    )

                on_bits = on_bits_indices + np.ones(4)

                signal_mean_columns = [
                    f"bit{int(bit):02d}_mean_intensity" for bit in on_bits
                ]
                bkd_mean_columns = [
                    f"bit{int(bit):02d}_mean_intensity"
                    for bit in range(1, self._n_merfish_bits + 1)
                    if bit not in on_bits
                ]

                df_barcode["signal_mean"] = df_barcode[signal_mean_columns].mean(axis=1)
                df_barcode["bkd_mean"] = df_barcode[bkd_mean_columns].mean(axis=1)
                df_barcode["s-b_mean"] = (
                    df_barcode["signal_mean"] - df_barcode["bkd_mean"]
                )

                del props
                gc.collect()

                if self._verbose > 1:
                    print("dataframe aggregation")
                if barcode_index == 0:
                    self._df_barcodes = df_barcode.copy()
                else:
                    self._df_barcodes = pd.concat([self._df_barcodes, df_barcode])
                    self._df_barcodes.reset_index(drop=True, inplace=True)

                del df_barcode
                gc.collect()
            else:
                for z_idx in range(decoded_image.shape[0]):
                    if self._verbose > 1:
                        print("")
                        print("label image")
                    labeled_image = label(
                        decoded_image[z_idx, :] == barcode_index, connectivity=2
                    )

                    if self._verbose > 1:
                        print("remove large")
                    pixel_counts = cp.bincount(labeled_image.ravel())
                    large_labels = cp.where(pixel_counts > maximum_pixels)[0]
                    large_label_mask = cp.zeros_like(labeled_image, dtype=bool)
                    large_label_mask = cp.isin(labeled_image, large_labels)
                    labeled_image[large_label_mask] = 0

                    if self._verbose > 1:
                        print("remove small")
                    labeled_image = remove_small_objects(
                        labeled_image, min_size=minimum_pixels
                    )
                    if self._verbose > 1:
                        print("regionprops table")
                    props = regionprops_table(
                        cp.asnumpy(labeled_image).astype(np.int32),
                        intensity_image=intensity_image[z_idx, :],
                        properties=[
                            "area",
                            "centroid",
                            "intensity_mean",
                            "moments_normalized",
                            "inertia_tensor_eigvals",
                        ],
                    )

                    del labeled_image
                    gc.collect()
                    cp.get_default_memory_pool().free_all_blocks()

                    df_barcode = pd.DataFrame(props)

                    df_barcode["on_bit_1"] = on_bits_indices[0] + 1
                    df_barcode["on_bit_2"] = on_bits_indices[1] + 1
                    df_barcode["on_bit_3"] = on_bits_indices[2] + 1
                    df_barcode["on_bit_4"] = on_bits_indices[3] + 1
                    df_barcode["barcode_id"] = df_barcode.apply(
                        lambda x: (barcode_index + 1), axis=1
                    )
                    df_barcode["gene_id"] = df_barcode.apply(
                        lambda x: self._gene_ids[barcode_index], axis=1
                    )
                    df_barcode["tile_idx"] = self._tile_idx

                    df_barcode["z"] = z_idx
                    df_barcode.rename(columns={"centroid-0": "y"}, inplace=True)
                    df_barcode.rename(columns={"centroid-1": "x"}, inplace=True)

                    if self._z_crop:
                        df_barcode["z"] = df_barcode["z"] + self._z_range[0]

                    df_barcode["tile_z"] = np.round(df_barcode["z"], 0).astype(int)
                    df_barcode["tile_y"] = np.round(df_barcode["y"], 0).astype(int)
                    df_barcode["tile_x"] = np.round(df_barcode["x"], 0).astype(int)

                    pts = df_barcode[["z", "y", "x"]].to_numpy()
                    for pt_idx, pt in enumerate(pts):
                        pts[pt_idx, :] = self._warp_pixel(
                            pts[pt_idx, :].copy(),
                            self._spacing,
                            self._origin,
                            self._affine,
                        )

                    df_barcode["global_z"] = np.round(pts[:, 0], 2)
                    df_barcode["global_y"] = np.round(pts[:, 1], 2)
                    df_barcode["global_x"] = np.round(pts[:, 2], 2)

                    df_barcode.rename(
                        columns={"intensity_mean-0": "distance_mean"}, inplace=True
                    )
                    for i in range(1, self._n_merfish_bits + 1):
                        df_barcode.rename(
                            columns={
                                "intensity_mean-" + str(i): "bit"
                                + str(i).zfill(2)
                                + "_mean_intensity"
                            },
                            inplace=True,
                        )

                    on_bits = on_bits_indices + np.ones(4)

                    signal_mean_columns = [
                        f"bit{int(bit):02d}_mean_intensity" for bit in on_bits
                    ]
                    bkd_mean_columns = [
                        f"bit{int(bit):02d}_mean_intensity"
                        for bit in range(1, self._n_merfish_bits + 1)
                        if bit not in on_bits
                    ]

                    df_barcode["signal_mean"] = df_barcode[signal_mean_columns].mean(
                        axis=1
                    )
                    df_barcode["bkd_mean"] = df_barcode[bkd_mean_columns].mean(axis=1)
                    df_barcode["s-b_mean"] = (
                        df_barcode["signal_mean"] - df_barcode["bkd_mean"]
                    )

                    del props
                    gc.collect()

                    if self._verbose > 1:
                        print("dataframe aggregation")
                    if barcode_index == 0:
                        self._df_barcodes = df_barcode.copy()
                    else:
                        self._df_barcodes = pd.concat([self._df_barcodes, df_barcode])
                        self._df_barcodes.reset_index(drop=True, inplace=True)

                    del df_barcode
                    gc.collect()

        del decoded_image, intensity_image
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    def _save_barcodes(self, format: str = "csv"):
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

    def _reformat_barcodes_for_baysor(self):
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
            baysor_df["is_gene"] = ~baysor_df["feature_name"].str.contains(
                "Blank", na=False
            )
            self._datastore.save_spots_prepped_for_baysor(baysor_df)

    def _load_all_barcodes(self):
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
            self._barcodes_filtered = True

    @staticmethod
    def calculate_fdr(df, threshold, blank_count, barcode_count, verbose):
        if threshold >= 0:
            df["prediction"] = df["predicted_probability"] > threshold

            coding = df[
                (~df["gene_id"].str.startswith("Blank"))
                & (df["predicted_probability"] > threshold)
            ].shape[0]
            noncoding = df[
                (df["gene_id"].str.startswith("Blank"))
                & (df["predicted_probability"] > threshold)
            ].shape[0]
        else:
            coding = df[(~df["gene_id"].str.startswith("Blank"))].shape[0]
            noncoding = df[(df["gene_id"].str.startswith("Blank"))].shape[0]

        if coding > 0:
            fdr = (noncoding / blank_count) / (coding / (barcode_count - blank_count))
        else:
            fdr = np.inf

        if verbose > 1:
            print(f"threshold: {threshold}")
            print(f"coding: {coding}")
            print(f"noncoding: {noncoding}")
            print(f"fdr: {fdr}")

        return fdr

    def _filter_all_barcodes(self, fdr_target: float = 0.05):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import classification_report
        from imblearn.over_sampling import SMOTE

        self._df_barcodes_loaded["X"] = ~self._df_barcodes_loaded[
            "gene_id"
        ].str.startswith("Blank")
        if self._is_3D:
            columns = [
                "X",
                "signal_mean",
                "s-b_mean",
                "distance_mean",
                "moments_normalized-0-0-2",
                "moments_normalized-0-0-3",
                "moments_normalized-0-1-1",
                "moments_normalized-0-1-2",
                "moments_normalized-0-1-3",
                "moments_normalized-0-2-0",
                "moments_normalized-0-2-1",
                "moments_normalized-0-2-3",
                "moments_normalized-0-3-0",
                "moments_normalized-0-3-1",
                "moments_normalized-0-3-2",
                "moments_normalized-0-3-3",
                "inertia_tensor_eigvals-0",
                "inertia_tensor_eigvals-1",
                "inertia_tensor_eigvals-2",
            ]
        else:
            columns = [
                "X",
                "signal_mean",
                "s-b_mean",
                "distance_mean",
                "moments_normalized-0-2",
                "moments_normalized-0-3",
                "moments_normalized-1-1",
                "moments_normalized-1-2",
                "moments_normalized-1-3",
                "moments_normalized-2-0",
                "moments_normalized-2-1",
                "moments_normalized-2-2",
                "moments_normalized-2-3",
                "moments_normalized-3-0",
                "moments_normalized-3-1",
                "moments_normalized-3-2",
                "moments_normalized-3-3",
                "inertia_tensor_eigvals-0",
                "inertia_tensor_eigvals-1",
            ]
        df_true = self._df_barcodes_loaded[self._df_barcodes_loaded["X"] == True][
            columns
        ]  # noqa
        df_false = self._df_barcodes_loaded[self._df_barcodes_loaded["X"] == False][
            columns
        ]  # noqa

        if len(df_false) > 0:
            df_true_sampled = df_true.sample(n=len(df_false), random_state=42)
            df_combined = pd.concat([df_true_sampled, df_false])
            x = df_combined.drop("X", axis=1)
            y = df_combined["X"]
            X_train, X_test, y_train, y_test = train_test_split(
                x, y, test_size=0.1, random_state=42
            )

            if self._verbose > 1:
                print("generating synthetic samples for class balance")
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            if self._verbose > 1:
                print("scaling features")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_resampled)
            X_test_scaled = scaler.transform(X_test)

            if self._verbose > 1:
                print("training classifier")
            # logistic = LogisticRegression(solver='liblinear', random_state=42)
            mlp = MLPClassifier(solver="adam", max_iter=10000, random_state=42)
            mlp.fit(X_train_scaled, y_train_resampled)
            predictions = mlp.predict(X_test_scaled)

            if self._verbose > 1:
                print(classification_report(y_test, predictions))

            if self._verbose > 1:
                print("predicting on full data")

            full_data_scaled = scaler.transform(self._df_barcodes_loaded[columns[1:]])
            self._df_barcodes_loaded["predicted_probability"] = mlp.predict_proba(
                full_data_scaled
            )[:, 1]

            if self._verbose > 1:
                print("filtering blanks")

            coarse_threshold = 0
            for threshold in np.arange(0, 1, 0.1):  # Coarse step: 0.1
                fdr = self.calculate_fdr(
                    self._df_barcodes_loaded,
                    threshold,
                    self._blank_count,
                    self._barcode_count,
                    self._verbose,
                )
                if fdr <= fdr_target:
                    coarse_threshold = threshold
                    break

            fine_threshold = coarse_threshold
            for threshold in np.arange(
                coarse_threshold - 0.1, coarse_threshold + 0.1, 0.01
            ):
                fdr = self.calculate_fdr(
                    self._df_barcodes_loaded,
                    threshold,
                    self._blank_count,
                    self._barcode_count,
                    self._verbose,
                )
                if fdr <= fdr_target:
                    fine_threshold = threshold
                    break

            df_above_threshold = self._df_barcodes_loaded[
                self._df_barcodes_loaded["predicted_probability"] > fine_threshold
            ]
            self._df_filtered_barcodes = df_above_threshold[
                [
                    "tile_idx",
                    "gene_id",
                    "global_z",
                    "global_y",
                    "global_x",
                    "distance_mean",
                ]
            ].copy()
            self._df_filtered_barcodes["cell_id"] = -1
            self._barcodes_filtered = True

            if self._verbose > 1:
                print(f"fdr : {fdr}")
                print(f"retained barcodes: {len(self._df_filtered_barcodes)}")

            del df_above_threshold, full_data_scaled
            del (
                mlp,
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
            self._df_filtered_barcodes.drop("X", axis=1, inplace=True)
            self._barcodes_filtered = True

    @staticmethod
    def _load_microjson(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        outlines = {}

        for feature in data["features"]:
            cell_id = feature["properties"]["cell_id"]
            coordinates = feature["geometry"]["coordinates"][0]
            outlines[cell_id] = np.array(coordinates)
        return outlines

    @staticmethod
    def _roi_to_shapely(roi):
        return Polygon(roi.subpixel_coordinates[:, ::-1])

    def _assign_cells(self):
        cellpose_roi_path = (
            self._datastore._datastore_path
            / Path("segmentation")
            / Path("cellpose")
            / Path("imagej_rois")
            / Path("global_coords_rois.zip")
        )

        try:
            rois = roiread(cellpose_roi_path)
        except Exception as e:
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
            except Exception as e:
                print(f"Failed to insert polygon into R-tree: {e}")

        def check_point(row):
            point = Point(row["global_y"], row["global_x"])

            candidate_ids = list(rtree_index.intersection(point.bounds))
            for candidate_id in candidate_ids:
                if shapely_polygons[candidate_id].contains(point):
                    return candidate_id + 1
            return 0

        self._df_filtered_barcodes["cell_id"] = self._df_filtered_barcodes.apply(
            check_point, axis=1
        )

    def _remove_duplicates_in_tile_overlap(self, radius: float = 0.75):
        self._df_filtered_barcodes.reset_index(drop=True, inplace=True)

        coords = self._df_filtered_barcodes[["global_z", "global_y", "global_x"]].values
        tile_idxs = self._df_filtered_barcodes["tile_idx"].values

        tree = cKDTree(coords)
        pairs = tree.query_pairs(radius)

        rows_to_drop = set()
        distances = []
        for i, j in pairs:
            if tile_idxs[i] != tile_idxs[j]:
                if (
                    self._df_filtered_barcodes.loc[i, "distance_mean"]
                    <= self._df_filtered_barcodes.loc[j, "distance_mean"]
                ):
                    rows_to_drop.add(j)
                    distances.append(self._df_filtered_barcodes.loc[j, "distance_mean"])
                else:
                    rows_to_drop.add(i)
                    distances.append(self._df_filtered_barcodes.loc[i, "distance_mean"])

        self._df_filtered_barcodes.drop(rows_to_drop, inplace=True)
        self._df_filtered_barcodes.reset_index(drop=True, inplace=True)

        avg_distance = np.mean(distances) if distances else 0
        dropped_count = len(rows_to_drop)

        if self._verbose > 1:
            print(
                "Average distance metric of dropped points (overlap): "
                + str(avg_distance)
            )
            print("Dropped points: " + str(dropped_count))

    def _display_results(self):
        import napari
        from qtpy.QtWidgets import QApplication

        def on_close_callback():
            viewer.layers.clear()
            gc.collect()

        viewer = napari.Viewer()
        app = QApplication.instance()

        app.lastWindowClosed.connect(on_close_callback)

        viewer.add_image(
            self._scaled_pixel_images,
            scale=[self._axial_step, self._pixel_size, self._pixel_size],
            name="pixels",
        )

        viewer.add_image(
            self._decoded_image,
            scale=[self._axial_step, self._pixel_size, self._pixel_size],
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

    def _cleanup(self):
        try:
            if self._filter_type == "lp":
                del self._image_data_lp
            else:
                del self._image_data
        except Exception:
            pass

        try:
            del (
                self._scaled_pixel_images,
                self._decoded_image,
                self._distance_image,
                self._magnitude_image,
            )
        except Exception:
            pass

        try:
            del self._df_barcodes
        except Exception:
            pass
        if self._barcodes_filtered:
            del self._df_filtered_barcodes

        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    def decode_one_tile(
        self,
        tile_idx: int = 0,
        display_results: bool = True,
        lowpass_sigma: Optional[Sequence[float]] = (3, 1, 1),
        minimum_pixels: Optional[float] = 9.0,
        use_normalization: Optional[bool] = True,
        ufish_threshold: Optional[float] = 0.1,
    ):
        if use_normalization:
            self._load_iterative_normalization_vectors()

        self._tile_idx = tile_idx
        self._load_bit_data(ufish_threshold=ufish_threshold)
        if not (np.any(lowpass_sigma == 0)):
            self._lp_filter(sigma=lowpass_sigma)
        self._decode_pixels(
            distance_threshold=self._distance_threshold,
            magnitude_threshold=self._magnitude_threshold,
        )
        if display_results:
            self._display_results()
        if not (self._optimize_normalization_weights):
            self._cleanup()
        else:
            self._extract_barcodes(minimum_pixels=minimum_pixels)

    def optimize_normalization_by_decoding(
        self,
        n_random_tiles: int = 10,
        n_iterations: int = 10,
        minimum_pixels: float = 9.0,
        ufish_threshold: float = 0.6,
    ):
        self._optimize_normalization_weights = True
        self._temp_dir = Path(tempfile.mkdtemp())

        if len(self._datastore.tile_ids) > n_random_tiles:
            random_tiles = sample(range(len(self._datastore.tile_ids)), n_random_tiles)
        else:
            random_tiles = range(len(self._datastore.tile_ids))

        if self._verbose >= 1:
            iterable_iteration = tqdm(range(n_iterations), desc="iteration", leave=True)
        else:
            iterable_iteration = range(n_iterations)

        self._load_global_normalization_vectors()
        self._iterative_background_vector = None
        self._iterative_normalization_vector = None
        for iteration in iterable_iteration:
            if self._verbose >= 1:
                iterable_tiles = tqdm(random_tiles, desc="tile", leave=True)
            else:
                iterable_tiles = random_tiles
            if iteration > 0:
                self._load_iterative_normalization_vectors()
            for tile_idx in iterable_tiles:
                if iteration == 0:
                    use_normalization = False
                else:
                    use_normalization = True
                self.decode_one_tile(
                    tile_idx=tile_idx,
                    display_results=False,
                    minimum_pixels=minimum_pixels,
                    ufish_threshold=ufish_threshold,
                    use_normalization=use_normalization,
                )
                self._save_barcodes(format="parquet")
            self._load_all_barcodes()
            if self._verbose >= 1:
                print("---")
                print("Total # of barcodes: " + str(len(self._df_barcodes_loaded)))
                print("---")
            self._iterative_normalization_vectors()
        self._cleanup()
        self._optimize_normalization_weights = False
        shutil.rmtree(self._temp_dir)

    def decode_all_tiles(
        self,
        assign_to_cells: bool = True,
        prep_for_baysor: bool = True,
        lowpass_sigma: Optional[Sequence[float]] = (3, 1, 1),
        minimum_pixels: Optional[float] = 2.0,
        ufish_threshold: Optional[float] = 0.6,
        fdr_target: Optional[float] = 0.05,
    ):
        if self._verbose >= 1:
            iterable_tile_id = enumerate(
                tqdm(self._datastore.tile_ids, desc="tile", leave=False)
            )
        else:
            iterable_tile_id = enumerate(self._datastore.tile_ids)

        self._optimize_normalization_weights = False
        self._load_iterative_normalization_vectors()

        if not (self._iterative_normalization_loaded):
            raise ValueError("Perform iterative normalization before decoding.")

        for tile_idx, _ in iterable_tile_id:
            self._tile_idx = tile_idx
            self._load_bit_data(ufish_threshold=ufish_threshold)
            if not (np.any(lowpass_sigma == 0)):
                self._lp_filter(sigma=lowpass_sigma)
            self._decode_pixels(
                distance_threshold=self._distance_threshold,
                magnitude_threshold=self._magnitude_threshold,
            )
            self._extract_barcodes(minimum_pixels=minimum_pixels)
            self._save_barcodes(format="parquet")
            self._cleanup()

        self._load_tile_decoding = True
        self._load_all_barcodes()
        self._load_tile_decoding = False
        self._verbose = 2
        self._filter_all_barcodes(fdr_target=fdr_target)
        self._verbose = 1
        self._remove_duplicates_in_tile_overlap()
        if assign_to_cells:
            self._assign_cells()
        self._save_barcodes(format="parquet")
        if prep_for_baysor:
            self._reformat_barcodes_for_baysor()
        self._cleanup()

    def optimize_filtering(
        self,
        assign_to_cells: bool = False,
        prep_for_baysor: bool = True,
        fdr_target: Optional[float] = 0.05,
    ):
        self._load_tile_decoding = True
        self._load_all_barcodes()
        self._load_tile_decoding = False
        self._verbose = 2
        self._filter_all_barcodes(fdr_target=fdr_target)
        self._verbose = 1
        self._remove_duplicates_in_tile_overlap()
        if assign_to_cells:
            self._assign_cells()
        self._save_barcodes(format="parquet")
        if prep_for_baysor:
            self._reformat_barcodes_for_baysor()
