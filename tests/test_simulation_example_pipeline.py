from __future__ import annotations

import json
import os
import shutil
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pytest

SIMULATION_DATA_ROOT_ENV = "MERFISH3D_SIMULATION_ROOT"
SIMULATION_CACHE_ENV = "MERFISH3D_SIMULATION_CACHE"
SIMULATION_DATA_ZENODO_URL = (
    "https://zenodo.org/records/17274305/files/merfish3d_analysis-simulation.zip?download=1"
)
SIMULATION_DATA_ARCHIVE = "merfish3d_analysis-simulation.zip"
SIMULATION_DATA_EXTRACTED_DIR = "merfish3d_analysis-simulation"
DEFAULT_TESTS_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_SIMULATION_CACHE_DIR = DEFAULT_TESTS_DATA_DIR / "simulation_dataset"
PERFORMANCE_REPORT_ENV = "MERFISH3D_PERFORMANCE_REPORT"
DEFAULT_PERFORMANCE_REPORT = str(DEFAULT_TESTS_DATA_DIR / "simulation_performance.json")
AXIAL_SPACING_UM = ("0.315", "1.0", "1.5")
F1_RADIUS_BY_AXIAL_SPACING_UM = {
    "0.315": 0.5,
    "1.0": 1.0,
    "1.5": 1.5,
}
REQUIRED_SIMULATION_FILES = (
    "aligned_1.tiff",
    "bit_order.csv",
    "codebook.csv",
    "GT_spots.csv",
    "scan_metadata.csv",
)
RESULT_KEYS = {
    "F1 Score",
    "Precision",
    "Recall",
    "True Positives",
    "False Positives",
    "False Negatives",
}
BASELINE_PERFORMANCE_BY_AXIAL_SPACING_UM = {
    "0.315": {
        "f1_score": 0.946884148891677,
        "precision": 0.9496644295302014,
        "recall": 0.9441201000834029,
        "true_positives": 1132,
        "false_negatives": 67,
        "true_positives_per_second": 7.417473520406533,
    },
    "1.0": {
        "f1_score": 0.9577935645633097,
        "precision": 0.9597989949748744,
        "recall": 0.9557964970809008,
        "true_positives": 1146,
        "false_negatives": 53,
        "true_positives_per_second": 6.942682006642391,
    },
    "1.5": {
        "f1_score": 0.9656616415410386,
        "precision": 0.9697224558452481,
        "recall": 0.9616346955796498,
        "true_positives": 1153,
        "false_negatives": 46,
        "true_positives_per_second": 8.39388283894179,
    },
}
FLOAT_COMPARISON_EPSILON = 1e-9


@pytest.fixture(scope="session")
def simulation_dataset_root() -> Path:
    """Resolve simulation root from env var or auto-download from Zenodo."""

    configured_root = os.getenv(SIMULATION_DATA_ROOT_ENV)
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
        if not root.exists():
            pytest.skip(f"{SIMULATION_DATA_ROOT_ENV} points to a missing path: {root}")
        return _normalize_dataset_root(root)

    cache_root = Path(
        os.getenv(SIMULATION_CACHE_ENV, str(DEFAULT_SIMULATION_CACHE_DIR))
    ).expanduser()
    try:
        root = _download_simulation_dataset(cache_root)
    except Exception as exc:
        pytest.skip(
            "Auto-download of simulation data failed. "
            f"Set {SIMULATION_DATA_ROOT_ENV} to an extracted dataset. Error: {exc!r}"
        )
    return _normalize_dataset_root(root)


def _download_simulation_dataset(cache_root: Path) -> Path:
    """Download and extract simulation dataset into a local cache directory."""

    cache_root.mkdir(parents=True, exist_ok=True)
    extracted_root = cache_root / SIMULATION_DATA_EXTRACTED_DIR
    if (extracted_root / "example_16bit_flat").exists():
        return extracted_root

    archive_path = cache_root / SIMULATION_DATA_ARCHIVE
    partial_path = cache_root / f"{SIMULATION_DATA_ARCHIVE}.part"
    if not archive_path.exists():
        with urlopen(SIMULATION_DATA_ZENODO_URL, timeout=300) as response:
            with partial_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        partial_path.replace(archive_path)

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(cache_root)

    return extracted_root


def _normalize_dataset_root(root: Path) -> Path:
    """Normalize paths to the extracted simulation dataset root."""

    if (root / "merfish3d_analysis-simulation").exists():
        root = root / "merfish3d_analysis-simulation"

    example_dir = root / "example_16bit_flat"
    if not example_dir.exists():
        pytest.skip(
            f"Could not find 'example_16bit_flat' under dataset root: {root}. "
            "Point the env var to the extracted Zenodo dataset."
        )

    return root


@pytest.fixture(scope="session")
def simulation_api() -> dict[str, Any]:
    """Import callable APIs used by the former notebook workflow."""

    try:
        from merfish3danalysis.cli.statphysbio_simulation.calculate_F1 import (
            calculate_F1_with_radius,
        )
        from merfish3danalysis.cli.statphysbio_simulation.convert_simulation_to_experiment import (
            convert_simulation,
        )
        from merfish3danalysis.cli.statphysbio_simulation.convert_to_datastore import (
            convert_data,
        )
        from merfish3danalysis.cli.statphysbio_simulation.pixeldecode import (
            decode_pixels,
        )
        from merfish3danalysis.cli.statphysbio_simulation.register_and_deconvolve import (
            manage_data_registration_states,
        )
        from merfish3danalysis.qi2labDataStore import qi2labDataStore
    except Exception as exc:
        pytest.skip(f"Simulation API imports unavailable in this environment: {exc!r}")

    return {
        "calculate_F1_with_radius": calculate_F1_with_radius,
        "convert_simulation": convert_simulation,
        "convert_data": convert_data,
        "decode_pixels": decode_pixels,
        "manage_data_registration_states": manage_data_registration_states,
        "qi2labDataStore": qi2labDataStore,
    }


@pytest.fixture(scope="session")
def performance_records() -> list[dict[str, Any]]:
    """Collect and persist runtime/performance records across all approaches."""

    records: list[dict[str, Any]] = []
    yield records

    if not records:
        return

    report_path = Path(
        os.getenv(PERFORMANCE_REPORT_ENV, DEFAULT_PERFORMANCE_REPORT)
    ).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    ordered_records = sorted(records, key=lambda record: float(record["total_seconds"]))
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "records": ordered_records,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Simulation Performance Summary ===", flush=True)
    for record in ordered_records:
        print(
            (
                f"{record['axial_spacing_um']} um | "
                f"total={record['total_seconds']:.2f}s | "
                f"decode={record['timings_seconds']['decode_pixels']:.2f}s | "
                f"F1={record['f1_score']:.4f} | "
                f"TP/s={record['true_positives_per_second']:.2f}"
            ),
            flush=True,
        )
    print(f"Performance report written to: {report_path}", flush=True)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Use symlinks for large files when possible, with copy fallback."""

    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _prepare_case_workspace(source_case_dir: Path, work_root: Path) -> Path:
    """Create an isolated workspace with only files needed for this pipeline."""

    missing_files = [
        filename
        for filename in REQUIRED_SIMULATION_FILES
        if not (source_case_dir / filename).exists()
    ]
    if missing_files:
        pytest.skip(
            f"Missing required simulation files in {source_case_dir}: {missing_files}"
        )

    case_root = work_root / source_case_dir.name
    case_root.mkdir(parents=True, exist_ok=False)

    for filename in REQUIRED_SIMULATION_FILES:
        _link_or_copy(source_case_dir / filename, case_root / filename)

    return case_root


def _calculate_f1_from_datastore(
    case_root: Path, simulation_api: dict[str, Any], search_radius: float = 1.0
) -> dict[str, float]:
    import numpy as np
    import pandas as pd

    datastore = simulation_api["qi2labDataStore"](
        case_root / "sim_acquisition" / "qi2labdatastore"
    )
    gene_ids, _ = datastore.load_codebook_parsed()
    decoded_spots = datastore.load_global_filtered_decoded_spots()
    gt_spots = pd.read_csv(case_root / "GT_spots.csv")
    corrected_image = datastore.load_local_corrected_image(
        tile=0, round=0, return_future=False
    )

    if decoded_spots is None:
        raise RuntimeError("Decoded spots table is missing from datastore.")
    if corrected_image is None:
        raise RuntimeError("Could not load local corrected image for F1 calculation.")

    corrected_image = np.asarray(corrected_image)
    y_size = corrected_image.shape[-2]
    x_size = corrected_image.shape[-1]

    gt_coords_offset = np.asarray(
        [
            0.0,
            (y_size / 2) * datastore.voxel_size_zyx_um[1]
            - datastore.voxel_size_zyx_um[1] / 2,
            (x_size / 2) * datastore.voxel_size_zyx_um[2]
            - datastore.voxel_size_zyx_um[2] / 2,
        ],
        dtype=np.float64,
    )

    qi2lab_coords = decoded_spots[["global_z", "global_y", "global_x"]].to_numpy(
        dtype=np.float64
    )
    qi2lab_gene_ids = decoded_spots["gene_id"].to_numpy()

    gt_coords = gt_spots[["Z", "X", "Y"]].to_numpy(dtype=np.float64) + gt_coords_offset
    gene_ids = np.asarray(gene_ids)
    gt_gene_ids = gene_ids[(gt_spots["Gene_label"].to_numpy(dtype=int) - 1)]

    f1_results, _, _, _ = simulation_api["calculate_F1_with_radius"](
        qi2lab_coords=qi2lab_coords,
        qi2lab_gene_ids=qi2lab_gene_ids,
        gt_coords=gt_coords,
        gt_gene_ids=gt_gene_ids,
        radius=search_radius,
    )
    return f1_results


def _run_simulation_pipeline(
    case_root: Path, simulation_api: dict[str, Any], search_radius: float
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    acquisition_root = case_root / "sim_acquisition"
    case_label = case_root.name
    timings_seconds: dict[str, float] = {}

    start = time.perf_counter()
    print(f"[{case_label}] convert_simulation: start", flush=True)
    simulation_api["convert_simulation"](case_root)
    timings_seconds["convert_simulation"] = time.perf_counter() - start
    print(
        f"[{case_label}] convert_simulation: done ({timings_seconds['convert_simulation']:.2f}s)",
        flush=True,
    )

    start = time.perf_counter()
    print(f"[{case_label}] convert_data: start", flush=True)
    simulation_api["convert_data"](acquisition_root)
    timings_seconds["convert_data"] = time.perf_counter() - start
    print(
        f"[{case_label}] convert_data: done ({timings_seconds['convert_data']:.2f}s)",
        flush=True,
    )

    start = time.perf_counter()
    print(f"[{case_label}] preprocess (registration): start", flush=True)
    simulation_api["manage_data_registration_states"](acquisition_root)
    timings_seconds["preprocess_registration"] = time.perf_counter() - start
    print(
        f"[{case_label}] preprocess (registration): done ({timings_seconds['preprocess_registration']:.2f}s)",
        flush=True,
    )

    start = time.perf_counter()
    print(f"[{case_label}] decode_pixels: start", flush=True)
    simulation_api["decode_pixels"](acquisition_root)
    timings_seconds["decode_pixels"] = time.perf_counter() - start
    print(
        f"[{case_label}] decode_pixels: done ({timings_seconds['decode_pixels']:.2f}s)",
        flush=True,
    )

    start = time.perf_counter()
    print(f"[{case_label}] calculate_f1: start", flush=True)
    f1_results = _calculate_f1_from_datastore(case_root, simulation_api, search_radius)
    timings_seconds["calculate_f1"] = time.perf_counter() - start
    print(
        f"[{case_label}] calculate_f1: done ({timings_seconds['calculate_f1']:.2f}s) -> {f1_results}",
        flush=True,
    )

    total_seconds = sum(timings_seconds.values())
    timings_seconds["total"] = total_seconds
    performance_metrics = {
        "f1_score": float(f1_results["F1 Score"]),
        "precision": float(f1_results["Precision"]),
        "recall": float(f1_results["Recall"]),
        "true_positives": int(f1_results["True Positives"]),
        "false_positives": int(f1_results["False Positives"]),
        "false_negatives": int(f1_results["False Negatives"]),
        "true_positives_per_second": float(f1_results["True Positives"])
        / max(total_seconds, 1e-9),
        "decoded_spots_per_decode_second": float(
            f1_results["True Positives"] + f1_results["False Positives"]
        )
        / max(timings_seconds["decode_pixels"], 1e-9),
    }

    return f1_results, timings_seconds, performance_metrics


@pytest.mark.parametrize("axial_spacing_um", AXIAL_SPACING_UM)
def test_simulation_example_pipeline(
    simulation_dataset_root: Path,
    simulation_api: dict[str, Any],
    performance_records: list[dict[str, Any]],
    tmp_path: Path,
    axial_spacing_um: str,
) -> None:
    """Notebook pipeline rewritten as an API-driven pytest integration test."""

    source_case_dir = simulation_dataset_root / "example_16bit_flat" / axial_spacing_um
    if not source_case_dir.exists():
        pytest.skip(f"Dataset case missing for axial spacing {axial_spacing_um} um.")

    search_radius = F1_RADIUS_BY_AXIAL_SPACING_UM.get(axial_spacing_um)
    if search_radius is None:
        pytest.skip(f"No F1 search radius configured for axial spacing {axial_spacing_um}.")

    case_root = _prepare_case_workspace(source_case_dir, tmp_path)
    f1_results, timings_seconds, performance_metrics = _run_simulation_pipeline(
        case_root, simulation_api, search_radius
    )

    datastore_path = case_root / "sim_acquisition" / "qi2labdatastore"
    assert datastore_path.exists()
    assert isinstance(f1_results, dict)
    assert RESULT_KEYS.issubset(f1_results)

    baseline = BASELINE_PERFORMANCE_BY_AXIAL_SPACING_UM.get(axial_spacing_um)
    if baseline is None:
        pytest.skip(f"No baseline performance configured for {axial_spacing_um}.")

    assert (
        performance_metrics["f1_score"] + FLOAT_COMPARISON_EPSILON >= baseline["f1_score"]
    ), (
        f"{axial_spacing_um} F1 regressed: "
        f"{performance_metrics['f1_score']:.6f} < {baseline['f1_score']:.6f}"
    )
    assert (
        performance_metrics["precision"] + FLOAT_COMPARISON_EPSILON
        >= baseline["precision"]
    ), (
        f"{axial_spacing_um} precision regressed: "
        f"{performance_metrics['precision']:.6f} < {baseline['precision']:.6f}"
    )
    assert (
        performance_metrics["recall"] + FLOAT_COMPARISON_EPSILON >= baseline["recall"]
    ), (
        f"{axial_spacing_um} recall regressed: "
        f"{performance_metrics['recall']:.6f} < {baseline['recall']:.6f}"
    )
    assert performance_metrics["true_positives"] >= baseline["true_positives"], (
        f"{axial_spacing_um} true positives regressed: "
        f"{performance_metrics['true_positives']} < {baseline['true_positives']}"
    )
    assert performance_metrics["false_negatives"] <= baseline["false_negatives"], (
        f"{axial_spacing_um} false negatives regressed: "
        f"{performance_metrics['false_negatives']} > {baseline['false_negatives']}"
    )
    assert (
        performance_metrics["true_positives_per_second"] + FLOAT_COMPARISON_EPSILON
        >= baseline["true_positives_per_second"]
    ), (
        f"{axial_spacing_um} TP/s regressed: "
        f"{performance_metrics['true_positives_per_second']:.6f} "
        f"< {baseline['true_positives_per_second']:.6f}"
    )

    record = {
        "axial_spacing_um": axial_spacing_um,
        "f1_search_radius_um": search_radius,
        "timings_seconds": timings_seconds,
        "total_seconds": timings_seconds["total"],
        **performance_metrics,
    }
    performance_records.append(record)
    print(
        (
            f"[{axial_spacing_um}] performance summary: "
            f"total={timings_seconds['total']:.2f}s, "
            f"decode={timings_seconds['decode_pixels']:.2f}s, "
            f"F1={performance_metrics['f1_score']:.4f}, "
            f"TP/s={performance_metrics['true_positives_per_second']:.2f}"
        ),
        flush=True,
    )

    # TODO: Add explicit metric threshold assertions once target values are finalized.
