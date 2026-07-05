from dataclasses import dataclass

import numpy as np
import pandas as pd

from merfish3danalysis.PixelDecoder import ChromaticAffineEstimationConfig, PixelDecoder


@dataclass
class _FakeDataStore:
    voxel_size_zyx_um: np.ndarray
    bit_ids: list[str]
    tile_ids: list[str]
    calibration: dict | None = None

    def load_local_wavelengths_um(self, tile, bit):
        del tile
        bit_index = self.bit_ids.index(bit) + 1
        if bit_index in (1, 2):
            return 0.49, 0.58
        return 0.63, 0.67

    def load_chromatic_affine_transforms_zyx_um(self):
        return self.calibration or {}

    def save_chromatic_affine_transforms_zyx_um(self, calibration):
        self.calibration = calibration


def _chromatic_affine_zyx_um() -> np.ndarray:
    affine = np.eye(4, dtype=np.float64)
    affine[0, 3] = 0.18
    affine[1, 1] = 0.9982
    affine[1, 3] = 0.42
    affine[2, 2] = 0.9982
    affine[2, 3] = -0.31
    return affine


def _transform_points(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    return (homogeneous @ affine.T)[:, :3]


def _empty_barcode_row() -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "on_bit_1": 1,
        "on_bit_2": 2,
        "on_bit_3": 3,
        "on_bit_4": 4,
        "gene_id": "GeneA",
    }
    for bit_index in range(1, 5):
        for suffix in ("center_z", "center_y", "center_x", "intensity_sum"):
            row[f"bit{bit_index:02d}_{suffix}"] = np.nan
    return row


def _add_bit_center(
    row: dict[str, float | int | str],
    bit_index: int,
    center_px: np.ndarray,
    weight: float,
) -> None:
    row[f"bit{bit_index:02d}_center_z"] = float(center_px[0])
    row[f"bit{bit_index:02d}_center_y"] = float(center_px[1])
    row[f"bit{bit_index:02d}_center_x"] = float(center_px[2])
    row[f"bit{bit_index:02d}_intensity_sum"] = float(weight)


def _make_chromatic_barcode_table(
    *,
    spacing_zyx_um: np.ndarray,
    true_affine_zyx_um: np.ndarray,
    n_good: int = 500,
    n_outliers: int = 300,
    n_same_wavelength_distractors: int = 500,
    n_blank_distractors: int = 900,
) -> pd.DataFrame:
    rng = np.random.default_rng(12345)
    inverse_affine = np.linalg.inv(true_affine_zyx_um)
    blank_affine = np.eye(4, dtype=np.float64)
    blank_affine[0, 3] = -0.8
    blank_affine[1, 3] = -5.0
    blank_affine[2, 3] = 4.0
    inverse_blank_affine = np.linalg.inv(blank_affine)
    rows = []

    reference_points_um = np.column_stack(
        [
            rng.uniform(2, 18, n_good),
            rng.uniform(0, 200, n_good),
            rng.uniform(0, 200, n_good),
        ]
    )
    red_points_um = _transform_points(inverse_affine, reference_points_um)
    for reference_point_um, red_point_um in zip(
        reference_points_um,
        red_points_um,
        strict=True,
    ):
        row = _empty_barcode_row()
        for bit_index in (1, 2):
            center_um = reference_point_um + rng.normal(0, 0.035, 3)
            _add_bit_center(
                row,
                bit_index,
                center_um / spacing_zyx_um,
                rng.uniform(80, 140),
            )
        for bit_index in (3, 4):
            center_um = red_point_um + rng.normal(0, 0.035, 3)
            _add_bit_center(
                row,
                bit_index,
                center_um / spacing_zyx_um,
                rng.uniform(80, 140),
            )
        rows.append(row)

    for _ in range(n_outliers):
        row = _empty_barcode_row()
        reference_point_um = np.array(
            [rng.uniform(2, 18), rng.uniform(0, 200), rng.uniform(0, 200)]
        )
        red_point_um = np.array(
            [rng.uniform(2, 18), rng.uniform(0, 200), rng.uniform(0, 200)]
        )
        for bit_index in (1, 2):
            _add_bit_center(
                row,
                bit_index,
                (reference_point_um + rng.normal(0, 0.035, 3)) / spacing_zyx_um,
                rng.uniform(80, 140),
            )
        for bit_index in (3, 4):
            _add_bit_center(
                row,
                bit_index,
                (red_point_um + rng.normal(0, 0.035, 3)) / spacing_zyx_um,
                rng.uniform(80, 140),
            )
        rows.append(row)

    for _ in range(n_same_wavelength_distractors):
        row = _empty_barcode_row()
        point_um = np.array(
            [rng.uniform(2, 18), rng.uniform(0, 200), rng.uniform(0, 200)]
        )
        if rng.random() < 0.5:
            row.update({"on_bit_1": 1, "on_bit_2": 2, "on_bit_3": 1, "on_bit_4": 2})
            bit_indices = (1, 2)
        else:
            row.update({"on_bit_1": 3, "on_bit_2": 4, "on_bit_3": 3, "on_bit_4": 4})
            bit_indices = (3, 4)
        for bit_index in bit_indices:
            _add_bit_center(
                row,
                bit_index,
                (point_um + rng.normal(0, 0.035, 3)) / spacing_zyx_um,
                rng.uniform(80, 140),
            )
        rows.append(row)

    blank_reference_points_um = np.column_stack(
        [
            rng.uniform(2, 18, n_blank_distractors),
            rng.uniform(0, 200, n_blank_distractors),
            rng.uniform(0, 200, n_blank_distractors),
        ]
    )
    blank_red_points_um = _transform_points(
        inverse_blank_affine, blank_reference_points_um
    )
    for reference_point_um, red_point_um in zip(
        blank_reference_points_um,
        blank_red_points_um,
        strict=True,
    ):
        row = _empty_barcode_row()
        row["gene_id"] = "Blank-1"
        for bit_index in (1, 2):
            _add_bit_center(
                row,
                bit_index,
                (reference_point_um + rng.normal(0, 0.015, 3)) / spacing_zyx_um,
                rng.uniform(200, 260),
            )
        for bit_index in (3, 4):
            _add_bit_center(
                row,
                bit_index,
                (red_point_um + rng.normal(0, 0.015, 3)) / spacing_zyx_um,
                rng.uniform(200, 260),
            )
        rows.append(row)

    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)


def test_chromatic_estimator_recovers_affine_with_distractors() -> None:
    spacing_zyx_um = np.array([0.32, 0.098, 0.098], dtype=np.float32)
    true_affine = _chromatic_affine_zyx_um()
    datastore = _FakeDataStore(
        voxel_size_zyx_um=spacing_zyx_um,
        bit_ids=["bit001", "bit002", "bit003", "bit004"],
        tile_ids=["tile0000"],
    )
    decoder = PixelDecoder.__new__(PixelDecoder)
    decoder._datastore = datastore
    decoder._n_merfish_bits = 4
    decoder._chromatic_affine_config = ChromaticAffineEstimationConfig(min_pairs=20)
    decoder._df_barcodes_loaded = _make_chromatic_barcode_table(
        spacing_zyx_um=spacing_zyx_um,
        true_affine_zyx_um=true_affine,
    )

    decoder._estimate_chromatic_affines_from_barcodes()

    channel = datastore.calibration["channels"]["wavelength_0.670000"]
    fit = channel["diagnostics"]["path_fits"][0]["fit"]
    estimated_affine = np.asarray(channel["affine_zyx_um"], dtype=np.float64)

    assert channel["status"] == "affine_estimated"
    assert channel["diagnostics"]["paired_transcripts"] == 800
    assert fit["candidate_pairs"] >= 500
    assert fit["used_pairs"] >= 300
    assert fit["median_residual_um"] < 0.12
    np.testing.assert_allclose(estimated_affine, true_affine, atol=0.006)


def test_chromatic_estimator_counts_only_valid_nonblank_cross_wavelength_rows() -> None:
    spacing_zyx_um = np.array([0.32, 0.098, 0.098], dtype=np.float32)
    true_affine = _chromatic_affine_zyx_um()
    datastore = _FakeDataStore(
        voxel_size_zyx_um=spacing_zyx_um,
        bit_ids=["bit001", "bit002", "bit003", "bit004"],
        tile_ids=["tile0000"],
    )
    barcode_table = _make_chromatic_barcode_table(
        spacing_zyx_um=spacing_zyx_um,
        true_affine_zyx_um=true_affine,
        n_good=9,
        n_outliers=0,
        n_same_wavelength_distractors=13,
        n_blank_distractors=21,
    )

    invalid_rows = []
    empty_gene_row = _empty_barcode_row()
    empty_gene_row["gene_id"] = ""
    invalid_rows.append(empty_gene_row)

    missing_gene_row = _empty_barcode_row()
    missing_gene_row["gene_id"] = None
    invalid_rows.append(missing_gene_row)

    invalid_on_bit_row = _empty_barcode_row()
    invalid_on_bit_row.update(
        {"on_bit_1": 99, "on_bit_2": 98, "on_bit_3": 97, "on_bit_4": 96}
    )
    for bit_index in range(1, 5):
        _add_bit_center(
            invalid_on_bit_row,
            bit_index,
            np.array([10.0, 20.0, 30.0]) / spacing_zyx_um,
            1000.0,
        )
    invalid_rows.append(invalid_on_bit_row)

    nan_center_row = _empty_barcode_row()
    for bit_index in range(1, 5):
        _add_bit_center(
            nan_center_row,
            bit_index,
            np.array([10.0, 20.0, 30.0]) / spacing_zyx_um,
            1000.0,
        )
    nan_center_row["bit03_center_y"] = np.nan
    nan_center_row["bit04_center_y"] = np.nan
    invalid_rows.append(nan_center_row)

    decoder = PixelDecoder.__new__(PixelDecoder)
    decoder._datastore = datastore
    decoder._n_merfish_bits = 4
    decoder._chromatic_affine_config = ChromaticAffineEstimationConfig(min_pairs=5)
    decoder._df_barcodes_loaded = pd.concat(
        [barcode_table, pd.DataFrame(invalid_rows)],
        ignore_index=True,
    )

    decoder._estimate_chromatic_affines_from_barcodes()

    channel = datastore.calibration["channels"]["wavelength_0.670000"]
    fit = channel["diagnostics"]["path_fits"][0]["fit"]
    estimated_affine = np.asarray(channel["affine_zyx_um"], dtype=np.float64)

    assert channel["status"] == "affine_estimated"
    assert channel["diagnostics"]["paired_transcripts"] == 9
    assert fit["candidate_pairs"] == 9
    assert fit["used_pairs"] >= 5
    np.testing.assert_allclose(estimated_affine, true_affine, atol=0.02)
