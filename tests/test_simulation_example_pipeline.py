import json
import os
import shutil
import time
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import pytest

LOCAL_SIMULATION_DATA_ROOT = Path("/media/dps/data/merfish3d_analysis-simulation")
DEFAULT_TESTS_DATA_DIR = Path(__file__).resolve().parent / "data"
PERFORMANCE_REPORT_ENV = "MERFISH3D_PERFORMANCE_REPORT"
DEFAULT_PERFORMANCE_REPORT = str(DEFAULT_TESTS_DATA_DIR / "simulation_performance.json")
AXIAL_SPACING_UM = ("0.315", "1.0", "1.5")
SIMULATION_DATASET_DIRS = {
    "cells": "example_16bit_cells",
    "uniform": "example_16bit_flat",
}
PREPROCESS_MODES = {
    "no-decon": False,
    "decon": True,
}
FEATURE_PREDICTOR_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)
DEFAULT_PREPROCESS_MODE = "decon"
DEFAULT_FEATURE_PREDICTOR_THRESHOLD = 0.5
DEFAULT_LOWPASS_SIGMA = (3.0, 1.0, 1.0)
DEFAULT_MAGNITUDE_THRESHOLD = (0.9, 10.0)
DEFAULT_MINIMUM_PIXELS_2D = 7
DEFAULT_MINIMUM_PIXELS_3D = 28
SIMULATION_AXIAL_NYQUIST_STEP_UM = 0.315
SIMULATION_2D_MAGNITUDE_THRESHOLD_BY_NYQUIST = {
    3.0: 0.7,
    5.0: 0.2,
}
F1_ABS_TOLERANCE = 0.02
COLLECT_FULL_BASELINES = False
F1_RADIUS_BY_AXIAL_SPACING_UM = {
    "0.315": 1.0,
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
STANDARD_SIMULATION_MATRIX = tuple(
    {
        "dataset_variant": dataset_variant,
        "dataset_dir": dataset_dir,
        "axial_spacing_um": axial_spacing_um,
        "preprocess_mode": DEFAULT_PREPROCESS_MODE,
        "decon_readout": PREPROCESS_MODES[DEFAULT_PREPROCESS_MODE],
        "feature_predictor_threshold": DEFAULT_FEATURE_PREDICTOR_THRESHOLD,
    }
    for (dataset_variant, dataset_dir), axial_spacing_um in product(
        SIMULATION_DATASET_DIRS.items(),
        AXIAL_SPACING_UM,
    )
)
FULL_SIMULATION_MATRIX = tuple(
    {
        "dataset_variant": dataset_variant,
        "dataset_dir": dataset_dir,
        "axial_spacing_um": axial_spacing_um,
        "preprocess_mode": preprocess_mode,
        "decon_readout": decon_readout,
        "feature_predictor_threshold": feature_predictor_threshold,
    }
    for (dataset_variant, dataset_dir), axial_spacing_um, (
        preprocess_mode,
        decon_readout,
    ), feature_predictor_threshold in product(
        SIMULATION_DATASET_DIRS.items(),
        AXIAL_SPACING_UM,
        PREPROCESS_MODES.items(),
        FEATURE_PREDICTOR_THRESHOLDS,
    )
)
FULL_SWEEP_BASE_SIMULATION_MATRIX = tuple(
    {
        "dataset_variant": dataset_variant,
        "dataset_dir": dataset_dir,
        "axial_spacing_um": axial_spacing_um,
        "preprocess_mode": preprocess_mode,
        "decon_readout": decon_readout,
    }
    for (dataset_variant, dataset_dir), axial_spacing_um, (
        preprocess_mode,
        decon_readout,
    ) in product(
        SIMULATION_DATASET_DIRS.items(),
        AXIAL_SPACING_UM,
        PREPROCESS_MODES.items(),
    )
)
STANDARD_EXPECTED_F1_SCORES = {
    ("cells", "0.315"): 0.9950,
    ("cells", "1.0"): 0.9641,
    ("cells", "1.5"): 0.9551,
    ("uniform", "0.315"): 0.9958,
    ("uniform", "1.0"): 0.9833,
    ("uniform", "1.5"): 0.9734,
}
FULL_EXPECTED_F1_SCORES = {
    ("cells", "0.315", "decon", 0.1): 0.9774058577405859,
    ("cells", "0.315", "decon", 0.2): 0.989139515455305,
    ("cells", "0.315", "decon", 0.3): 0.9899665551839465,
    ("cells", "0.315", "decon", 0.4): 0.9949832775919734,
    ("cells", "0.315", "decon", 0.5): 0.9949748743718593,
    ("cells", "0.315", "no-decon", 0.1): 0.9941225860621327,
    ("cells", "0.315", "no-decon", 0.2): 0.9941324392288349,
    ("cells", "0.315", "no-decon", 0.3): 0.9941324392288349,
    ("cells", "0.315", "no-decon", 0.4): 0.9949748743718593,
    ("cells", "0.315", "no-decon", 0.5): 0.9949748743718593,
    ("cells", "1.0", "decon", 0.1): 0.9093904448105437,
    ("cells", "1.0", "decon", 0.2): 0.9377593360995851,
    ("cells", "1.0", "decon", 0.3): 0.9443059019118869,
    ("cells", "1.0", "decon", 0.4): 0.9650582362728786,
    ("cells", "1.0", "decon", 0.5): 0.964076858813701,
    ("cells", "1.0", "no-decon", 0.1): 0.7495908346972175,
    ("cells", "1.0", "no-decon", 0.2): 0.798360655737705,
    ("cells", "1.0", "no-decon", 0.3): 0.8273026315789475,
    ("cells", "1.0", "no-decon", 0.4): 0.8535980148883375,
    ("cells", "1.0", "no-decon", 0.5): 0.8715824357912179,
    ("cells", "1.5", "decon", 0.1): 0.8973322554567502,
    ("cells", "1.5", "decon", 0.2): 0.9225753871230644,
    ("cells", "1.5", "decon", 0.3): 0.9262295081967213,
    ("cells", "1.5", "decon", 0.4): 0.9489291598023065,
    ("cells", "1.5", "decon", 0.5): 0.9551495016611296,
    ("cells", "1.5", "no-decon", 0.1): 0.9197080291970804,
    ("cells", "1.5", "no-decon", 0.2): 0.939344262295082,
    ("cells", "1.5", "no-decon", 0.3): 0.9420529801324503,
    ("cells", "1.5", "no-decon", 0.4): 0.9467554076539102,
    ("cells", "1.5", "no-decon", 0.5): 0.9524603836530442,
    ("uniform", "0.315", "decon", 0.1): 0.989517819706499,
    ("uniform", "0.315", "decon", 0.2): 0.9908103592314119,
    ("uniform", "0.315", "decon", 0.3): 0.9937317175094025,
    ("uniform", "0.315", "decon", 0.4): 0.9949832775919734,
    ("uniform", "0.315", "decon", 0.5): 0.9958228905597325,
    ("uniform", "0.315", "no-decon", 0.1): 0.9953994144709327,
    ("uniform", "0.315", "no-decon", 0.2): 0.9962390305056414,
    ("uniform", "0.315", "no-decon", 0.3): 0.9966555183946487,
    ("uniform", "0.315", "no-decon", 0.4): 0.995819397993311,
    ("uniform", "0.315", "no-decon", 0.5): 0.995819397993311,
    ("uniform", "1.0", "decon", 0.1): 0.9443983402489626,
    ("uniform", "1.0", "decon", 0.2): 0.9671790610718737,
    ("uniform", "1.0", "decon", 0.3): 0.9725685785536159,
    ("uniform", "1.0", "decon", 0.4): 0.9804248229904206,
    ("uniform", "1.0", "decon", 0.5): 0.9833055091819699,
    ("uniform", "1.0", "no-decon", 0.1): 0.8153023447141095,
    ("uniform", "1.0", "no-decon", 0.2): 0.8559636813867108,
    ("uniform", "1.0", "no-decon", 0.3): 0.884742951907131,
    ("uniform", "1.0", "no-decon", 0.4): 0.9077306733167082,
    ("uniform", "1.0", "no-decon", 0.5): 0.938877338877339,
    ("uniform", "1.5", "decon", 0.1): 0.9347471451876019,
    ("uniform", "1.5", "decon", 0.2): 0.9540229885057472,
    ("uniform", "1.5", "decon", 0.3): 0.9697972693421597,
    ("uniform", "1.5", "decon", 0.4): 0.9725913621262458,
    ("uniform", "1.5", "decon", 0.5): 0.973399833748961,
    ("uniform", "1.5", "no-decon", 0.1): 0.8751013787510138,
    ("uniform", "1.5", "no-decon", 0.2): 0.9020893076607948,
    ("uniform", "1.5", "no-decon", 0.3): 0.922824302134647,
    ("uniform", "1.5", "no-decon", 0.4): 0.9443298969072166,
    ("uniform", "1.5", "no-decon", 0.5): 0.9611570247933885,
}


@pytest.fixture(scope="session")
def simulation_dataset_root() -> Path:
    """Resolve the local simulation dataset root configured in this test file."""

    root = LOCAL_SIMULATION_DATA_ROOT.expanduser().resolve()
    if not root.exists():
        pytest.skip(f"Local simulation dataset root does not exist: {root}")
    return _normalize_dataset_root(root)


@pytest.fixture(scope="session")
def simulation_dataset_dirs(simulation_dataset_root: Path) -> dict[str, Path]:
    """Resolve local dataset directories for the simulation matrix."""

    resolved_dirs: dict[str, Path] = {}
    missing_variants: list[str] = []
    for variant, dataset_dir in SIMULATION_DATASET_DIRS.items():
        candidate = simulation_dataset_root / dataset_dir
        if candidate.exists():
            resolved_dirs[variant] = candidate
        else:
            missing_variants.append(f"{variant}:{dataset_dir}")

    if missing_variants:
        pytest.skip(
            "Missing required simulation dataset directories under "
            f"{simulation_dataset_root}: {missing_variants}"
        )

    return resolved_dirs


def _normalize_dataset_root(root: Path) -> Path:
    """Normalize the configured local simulation dataset root."""

    example_dir = root / "example_16bit_flat"
    if not example_dir.exists():
        pytest.skip(
            f"Could not find 'example_16bit_flat' under dataset root: {root}. "
            "Update LOCAL_SIMULATION_DATA_ROOT in this test file."
        )

    return root


def _default_minimum_pixels_per_rna(axial_spacing_um: str) -> int:
    """Return the default minimum-pixel threshold for a simulation spacing."""

    if axial_spacing_um == "0.315":
        return DEFAULT_MINIMUM_PIXELS_3D
    return DEFAULT_MINIMUM_PIXELS_2D


def _default_standard_magnitude_threshold(
    axial_spacing_um: str,
) -> tuple[float, float]:
    """Return the sampling-aware default magnitude threshold for simulations."""

    if axial_spacing_um == "0.315":
        return DEFAULT_MAGNITUDE_THRESHOLD

    nyquist_multiple = float(axial_spacing_um) / SIMULATION_AXIAL_NYQUIST_STEP_UM
    nearest_multiple = min(
        SIMULATION_2D_MAGNITUDE_THRESHOLD_BY_NYQUIST,
        key=lambda value: abs(value - nyquist_multiple),
    )
    lower_threshold = SIMULATION_2D_MAGNITUDE_THRESHOLD_BY_NYQUIST[nearest_multiple]
    return (lower_threshold, DEFAULT_MAGNITUDE_THRESHOLD[1])


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
            global_register_data,
        )
        from merfish3danalysis.DataRegistration import DataRegistration
        from merfish3danalysis.qi2labDataStore import qi2labDataStore
    except Exception as exc:
        pytest.skip(f"Simulation API imports unavailable in this environment: {exc!r}")

    return {
        "DataRegistration": DataRegistration,
        "calculate_F1_with_radius": calculate_F1_with_radius,
        "convert_simulation": convert_simulation,
        "convert_data": convert_data,
        "decode_pixels": decode_pixels,
        "global_register_data": global_register_data,
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
        dataset_variant = record.get("dataset_variant", "flat")
        preprocess_mode = record.get("preprocess_mode", "no-decon")
        feature_predictor_threshold = record.get(
            "feature_predictor_threshold", "unknown"
        )
        magnitude_threshold_lower = record.get("magnitude_threshold_lower")
        magnitude_suffix = (
            f":mag={magnitude_threshold_lower}"
            if magnitude_threshold_lower is not None
            else ""
        )
        print(
            (
                f"{dataset_variant}:{record['axial_spacing_um']} um:{preprocess_mode}:fp={feature_predictor_threshold}{magnitude_suffix} | "
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


def _run_simulation_preprocess(
    acquisition_root: Path,
    simulation_api: dict[str, Any],
    decon_readout: bool,
) -> None:
    datastore = simulation_api["qi2labDataStore"](acquisition_root / "qi2labdatastore")
    registration_factory = simulation_api["DataRegistration"](
        datastore=datastore,
        perform_optical_flow=False,
        bkd_subtract_fiducial=False,
        overwrite_registered=True,
        save_all_fiducial_registered=False,
        decon_readout=decon_readout,
        num_gpus=1,
        verbose=1,
    )
    registration_factory.register_all_tiles()

    simulation_api["global_register_data"](acquisition_root, create_max_proj_tiff=False)

    # Mark local registration complete only after global registration succeeds so
    # follow-on datastore opens do not validate a half-updated local state.
    datastore = simulation_api["qi2labDataStore"](
        acquisition_root / "qi2labdatastore", validate=False
    )
    datastore.datastore_state = {"LocalRegistered": True}


def _run_simulation_case_setup(
    case_root: Path,
    simulation_api: dict[str, Any],
    decon_readout: bool,
    case_label: str,
) -> dict[str, float]:
    acquisition_root = case_root / "sim_acquisition"
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
    print(
        f"[{case_label}] preprocess (registration, decon_readout={decon_readout}): start",
        flush=True,
    )
    _run_simulation_preprocess(
        acquisition_root,
        simulation_api,
        decon_readout=decon_readout,
    )
    timings_seconds["preprocess_registration"] = time.perf_counter() - start
    print(
        (
            f"[{case_label}] preprocess (registration, decon_readout={decon_readout}): "
            f"done ({timings_seconds['preprocess_registration']:.2f}s)"
        ),
        flush=True,
    )

    return timings_seconds


def _prepare_preprocessed_magnitude_case(
    simulation_dataset_dirs: dict[str, Path],
    simulation_api: dict[str, Any],
    case_spec: dict[str, Any],
    tmp_path: Path,
    *,
    run_calibration_decode: bool = True,
) -> dict[str, Any]:
    """Prepare one simulation case once before decode-only magnitude sweeps."""

    dataset_variant = case_spec["dataset_variant"]
    axial_spacing_um = case_spec["axial_spacing_um"]
    decon_readout = case_spec["decon_readout"]
    preprocess_mode = case_spec.get(
        "preprocess_mode",
        "decon" if decon_readout else "no-decon",
    )

    source_case_dir = simulation_dataset_dirs[dataset_variant] / axial_spacing_um
    if not source_case_dir.exists():
        pytest.skip(
            f"Dataset case missing for {dataset_variant} at {axial_spacing_um} um."
        )

    search_radius = F1_RADIUS_BY_AXIAL_SPACING_UM.get(axial_spacing_um)
    if search_radius is None:
        pytest.skip(
            f"No F1 search radius configured for axial spacing {axial_spacing_um}."
        )
    default_minimum_pixels = _default_minimum_pixels_per_rna(axial_spacing_um)
    default_magnitude_threshold = _default_standard_magnitude_threshold(axial_spacing_um)

    case_root = _prepare_case_workspace(source_case_dir, tmp_path)
    case_label = (
        f"{dataset_variant}-{axial_spacing_um}-{preprocess_mode}-"
        f"fp{DEFAULT_FEATURE_PREDICTOR_THRESHOLD}"
    )
    setup_timings = _run_simulation_case_setup(
        case_root,
        simulation_api,
        decon_readout=decon_readout,
        case_label=case_label,
    )
    if run_calibration_decode:
        default_minimum_pixels = _default_minimum_pixels_per_rna(axial_spacing_um)
        calibration_label = f"{case_label}-calibration"
        calibration_start = time.perf_counter()
        print(f"[{calibration_label}] decode_pixels: start", flush=True)
        simulation_api["decode_pixels"](
            case_root / "sim_acquisition",
            minimum_pixels_per_RNA=default_minimum_pixels,
            feature_predictor_threshold=DEFAULT_FEATURE_PREDICTOR_THRESHOLD,
            magnitude_threshold=default_magnitude_threshold,
            skip_optimization=False,
        )
        setup_timings["decode_calibration"] = time.perf_counter() - calibration_start
        print(
            (
                f"[{calibration_label}] decode_pixels: done "
                f"({setup_timings['decode_calibration']:.2f}s)"
            ),
            flush=True,
        )

    return {
        "case_root": case_root,
        "case_label": case_label,
        "case_spec": case_spec,
        "search_radius": search_radius,
        "setup_timings": setup_timings,
    }


def _run_simulation_decode_and_f1(
    case_root: Path,
    simulation_api: dict[str, Any],
    search_radius: float,
    feature_predictor_threshold: float,
    lowpass_sigma: tuple[float, float, float],
    magnitude_threshold: tuple[float, float],
    case_label: str,
    minimum_pixels_per_rna: int | None = None,
    skip_optimization: bool = False,
    duplicate_radius_xy: float | None = None,
    duplicate_radius_z: float | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    acquisition_root = case_root / "sim_acquisition"
    timings_seconds: dict[str, float] = {}

    start = time.perf_counter()
    print(f"[{case_label}] decode_pixels: start", flush=True)
    simulation_api["decode_pixels"](
        acquisition_root,
        minimum_pixels_per_RNA=minimum_pixels_per_rna,
        feature_predictor_threshold=feature_predictor_threshold,
        lowpass_sigma=lowpass_sigma,
        magnitude_threshold=magnitude_threshold,
        skip_optimization=skip_optimization,
        duplicate_radius_xy=duplicate_radius_xy,
        duplicate_radius_z=duplicate_radius_z,
    )
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


def _run_simulation_pipeline(
    case_root: Path,
    simulation_api: dict[str, Any],
    search_radius: float,
    decon_readout: bool,
    feature_predictor_threshold: float,
    lowpass_sigma: tuple[float, float, float] = DEFAULT_LOWPASS_SIGMA,
    magnitude_threshold: tuple[float, float] = (0.9, 10.0),
    minimum_pixels_per_rna: int | None = None,
    case_label: str | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    if case_label is None:
        case_label = case_root.name
    timings_seconds = _run_simulation_case_setup(
        case_root,
        simulation_api,
        decon_readout=decon_readout,
        case_label=case_label,
    )
    decode_results, decode_timings, performance_metrics = _run_simulation_decode_and_f1(
        case_root,
        simulation_api,
        search_radius,
        feature_predictor_threshold=feature_predictor_threshold,
        lowpass_sigma=lowpass_sigma,
        magnitude_threshold=magnitude_threshold,
        minimum_pixels_per_rna=minimum_pixels_per_rna,
        case_label=case_label,
    )
    timings_seconds.update(
        {key: value for key, value in decode_timings.items() if key != "total"}
    )
    timings_seconds["total"] = sum(timings_seconds.values())

    return decode_results, timings_seconds, performance_metrics


@pytest.fixture(
    params=STANDARD_SIMULATION_MATRIX,
    ids=[
        (
            f"{case['dataset_variant']}-"
            f"{case['axial_spacing_um']}-"
            f"{case['preprocess_mode']}-"
            f"fp{case['feature_predictor_threshold']}"
        )
        for case in STANDARD_SIMULATION_MATRIX
    ],
)
def simulation_standard_case_spec(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Parametrize the standard local simulation test matrix."""

    return request.param


@pytest.fixture
def simulation_standard_case_result(
    simulation_dataset_dirs: dict[str, Path],
    simulation_api: dict[str, Any],
    performance_records: list[dict[str, Any]],
    simulation_standard_case_spec: dict[str, Any],
    tmp_path: Path,
) -> dict[str, Any]:
    """Run one standard simulation pipeline case and return recorded metrics."""

    dataset_variant = simulation_standard_case_spec["dataset_variant"]
    axial_spacing_um = simulation_standard_case_spec["axial_spacing_um"]
    preprocess_mode = simulation_standard_case_spec["preprocess_mode"]
    decon_readout = simulation_standard_case_spec["decon_readout"]
    feature_predictor_threshold = simulation_standard_case_spec[
        "feature_predictor_threshold"
    ]

    source_case_dir = simulation_dataset_dirs[dataset_variant] / axial_spacing_um
    if not source_case_dir.exists():
        pytest.skip(
            f"Dataset case missing for {dataset_variant} at {axial_spacing_um} um."
        )

    search_radius = F1_RADIUS_BY_AXIAL_SPACING_UM.get(axial_spacing_um)
    if search_radius is None:
        pytest.skip(
            f"No F1 search radius configured for axial spacing {axial_spacing_um}."
        )
    default_minimum_pixels = _default_minimum_pixels_per_rna(axial_spacing_um)
    default_magnitude_threshold = _default_standard_magnitude_threshold(axial_spacing_um)

    case_root = _prepare_case_workspace(source_case_dir, tmp_path)
    case_label = (
        f"{dataset_variant}-{axial_spacing_um}-{preprocess_mode}-"
        f"fp{feature_predictor_threshold}"
    )
    f1_results, timings_seconds, performance_metrics = _run_simulation_pipeline(
        case_root,
        simulation_api,
        search_radius,
        decon_readout=decon_readout,
        feature_predictor_threshold=feature_predictor_threshold,
        magnitude_threshold=default_magnitude_threshold,
        minimum_pixels_per_rna=default_minimum_pixels,
        case_label=case_label,
    )

    record = {
        "dataset_variant": dataset_variant,
        "axial_spacing_um": axial_spacing_um,
        "preprocess_mode": preprocess_mode,
        "decon_readout": decon_readout,
        "feature_predictor_threshold": feature_predictor_threshold,
        "magnitude_threshold_lower": default_magnitude_threshold[0],
        "minimum_pixels_per_rna": default_minimum_pixels,
        "f1_search_radius_um": search_radius,
        "timings_seconds": timings_seconds,
        "total_seconds": timings_seconds["total"],
        **performance_metrics,
    }
    performance_records.append(record)
    print(
        (
            f"[{case_label}] performance summary: "
            f"total={timings_seconds['total']:.2f}s, "
            f"decode={timings_seconds['decode_pixels']:.2f}s, "
            f"F1={performance_metrics['f1_score']:.4f}, "
            f"TP/s={performance_metrics['true_positives_per_second']:.2f}"
        ),
        flush=True,
    )

    return {
        "case_root": case_root,
        "case_label": case_label,
        "case_spec": simulation_standard_case_spec,
        "f1_results": f1_results,
        "timings_seconds": timings_seconds,
        "performance_metrics": performance_metrics,
        "record": record,
    }


@pytest.fixture(
    params=FULL_SWEEP_BASE_SIMULATION_MATRIX,
    ids=[
        (
            f"{case['dataset_variant']}-"
            f"{case['axial_spacing_um']}-"
            f"{case['preprocess_mode']}"
        )
        for case in FULL_SWEEP_BASE_SIMULATION_MATRIX
    ],
)
def simulation_full_sweep_case_spec(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Parametrize the exhaustive local simulation base cases."""

    return request.param


@pytest.fixture
def simulation_full_preprocessed_case(
    simulation_dataset_dirs: dict[str, Path],
    simulation_api: dict[str, Any],
    simulation_full_sweep_case_spec: dict[str, Any],
    tmp_path: Path,
) -> dict[str, Any]:
    """Prepare one exhaustive simulation base case before FP sweeps."""

    return _prepare_preprocessed_magnitude_case(
        simulation_dataset_dirs,
        simulation_api,
        simulation_full_sweep_case_spec,
        tmp_path,
        run_calibration_decode=True,
    )


@pytest.fixture
def simulation_full_case_result(
    simulation_api: dict[str, Any],
    performance_records: list[dict[str, Any]],
    simulation_full_preprocessed_case: dict[str, Any],
) -> dict[str, Any]:
    """Run the exhaustive FP-threshold sweep on one calibrated base case."""

    case_root = simulation_full_preprocessed_case["case_root"]
    case_spec = simulation_full_preprocessed_case["case_spec"]
    case_label = simulation_full_preprocessed_case["case_label"]
    search_radius = simulation_full_preprocessed_case["search_radius"]
    setup_timings = simulation_full_preprocessed_case["setup_timings"]
    dataset_variant = case_spec["dataset_variant"]
    axial_spacing_um = case_spec["axial_spacing_um"]
    preprocess_mode = case_spec["preprocess_mode"]
    decon_readout = case_spec["decon_readout"]
    default_minimum_pixels = _default_minimum_pixels_per_rna(axial_spacing_um)
    default_magnitude_threshold = _default_standard_magnitude_threshold(axial_spacing_um)

    sweep_results: list[dict[str, Any]] = []
    for feature_predictor_threshold in FEATURE_PREDICTOR_THRESHOLDS:
        threshold_case_label = f"{case_label[:-5]}fp{feature_predictor_threshold}"
        f1_results, decode_timings, performance_metrics = _run_simulation_decode_and_f1(
            case_root,
            simulation_api,
            search_radius,
            feature_predictor_threshold=feature_predictor_threshold,
            lowpass_sigma=DEFAULT_LOWPASS_SIGMA,
            magnitude_threshold=default_magnitude_threshold,
            minimum_pixels_per_rna=default_minimum_pixels,
            case_label=threshold_case_label,
            skip_optimization=True,
        )
        timings_seconds = {
            **setup_timings,
            **{key: value for key, value in decode_timings.items() if key != "total"},
        }
        timings_seconds["total"] = sum(timings_seconds.values())
        record = {
            "dataset_variant": dataset_variant,
            "axial_spacing_um": axial_spacing_um,
            "preprocess_mode": preprocess_mode,
            "decon_readout": decon_readout,
            "feature_predictor_threshold": feature_predictor_threshold,
            "magnitude_threshold_lower": default_magnitude_threshold[0],
            "lowpass_sigma": list(DEFAULT_LOWPASS_SIGMA),
            "minimum_pixels_per_rna": default_minimum_pixels,
            "f1_search_radius_um": search_radius,
            "timings_seconds": timings_seconds,
            "total_seconds": timings_seconds["total"],
            **performance_metrics,
        }
        performance_records.append(record)
        print(
            (
                f"[{threshold_case_label}] performance summary: "
                f"total={timings_seconds['total']:.2f}s, "
                f"decode={timings_seconds['decode_pixels']:.2f}s, "
                f"F1={performance_metrics['f1_score']:.4f}, "
                f"TP/s={performance_metrics['true_positives_per_second']:.2f}"
            ),
            flush=True,
        )
        sweep_results.append(
            {
                "feature_predictor_threshold": feature_predictor_threshold,
                "f1_results": f1_results,
                "timings_seconds": timings_seconds,
                "performance_metrics": performance_metrics,
            }
        )

    return {
        "case_root": case_root,
        "case_label": case_label,
        "case_spec": case_spec,
        "results": sweep_results,
    }


def test_simulation_standard_matrix(
    simulation_standard_case_result: dict[str, Any],
) -> None:
    """Run the standard local simulation matrix with asserted baseline F1 values."""

    datastore_path = (
        simulation_standard_case_result["case_root"]
        / "sim_acquisition"
        / "qi2labdatastore"
    )
    case_spec = simulation_standard_case_result["case_spec"]
    expected_f1 = STANDARD_EXPECTED_F1_SCORES[
        (case_spec["dataset_variant"], case_spec["axial_spacing_um"])
    ]

    assert datastore_path.exists()
    assert isinstance(simulation_standard_case_result["f1_results"], dict)
    assert RESULT_KEYS.issubset(simulation_standard_case_result["f1_results"])
    assert (
        simulation_standard_case_result["performance_metrics"]["f1_score"]
        == pytest.approx(expected_f1, abs=F1_ABS_TOLERANCE)
    )
    assert simulation_standard_case_result["performance_metrics"]["true_positives"] >= 0
    assert simulation_standard_case_result["performance_metrics"]["false_positives"] >= 0
    assert simulation_standard_case_result["performance_metrics"]["false_negatives"] >= 0


@pytest.mark.simulation_exhaustive
def test_simulation_exhaustive_matrix(
    simulation_full_case_result: dict[str, Any],
) -> None:
    """Run the exhaustive local simulation matrix with asserted baseline F1 values."""

    datastore_path = (
        simulation_full_case_result["case_root"] / "sim_acquisition" / "qi2labdatastore"
    )
    case_spec = simulation_full_case_result["case_spec"]

    assert datastore_path.exists()
    assert len(simulation_full_case_result["results"]) == len(
        FEATURE_PREDICTOR_THRESHOLDS
    )
    collect_only = COLLECT_FULL_BASELINES
    for result in simulation_full_case_result["results"]:
        assert isinstance(result["f1_results"], dict)
        assert RESULT_KEYS.issubset(result["f1_results"])
        if not collect_only:
            expected_f1 = FULL_EXPECTED_F1_SCORES[
                (
                    case_spec["dataset_variant"],
                    case_spec["axial_spacing_um"],
                    case_spec["preprocess_mode"],
                    result["feature_predictor_threshold"],
                )
            ]
            assert result["performance_metrics"]["f1_score"] == pytest.approx(
                expected_f1, abs=F1_ABS_TOLERANCE
            )
        assert result["performance_metrics"]["true_positives"] >= 0
        assert result["performance_metrics"]["false_positives"] >= 0
        assert result["performance_metrics"]["false_negatives"] >= 0
