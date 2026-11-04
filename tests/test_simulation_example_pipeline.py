"""Simulation pipeline integration tests against the published Zenodo data.

The matrix converts StatPhysBio simulation outputs into qi2lab datastores,
runs preprocessing, decodes transcripts, and scores decoded features against
simulation ground truth. The optional exhaustive mode adds deconvolution-off
coverage to the standard Zenodo F1 integration matrix.
"""

import hashlib
import json
import os
import runpy
import shutil
import time
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from merfish3danalysis.utils.ufish import DEFAULT_UFISH_MODEL

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
DEFAULT_PREPROCESS_MODE = "decon"
DEFAULT_FEATURE_PREDICTOR_THRESHOLD = 0.5
F1_ACCURACY_ALLOWANCE = 0.02
DEFAULT_LOWPASS_SIGMA = (3.0, 1.0, 1.0)
SIMULATION_MAGNITUDE_THRESHOLDS = {
    "0.315": (0.9, 10.0),
    "1.0": (0.7, 10.0),
    "1.5": (0.2, 10.0),
}
SIMULATION_MINIMUM_PIXELS = {"0.315": 28, "1.0": 7, "1.5": 7}
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

# SHA-256 of each aligned_1.tiff in Zenodo record 17274305. Hashing the image
# payloads prevents an older server-side simulation set from silently being
# used for the published F1 accuracy measurements.
ZENODO_ALIGNED_IMAGE_SHA256 = {
    (
        "cells",
        "0.315",
    ): "2a60a8a2442c91c1b2e41bb8de5e21daba097c278be442c9056eb0481a41dfc5",
    (
        "cells",
        "1.0",
    ): "591f1a0231637dd68a3331b8ba3521f690ec89db4cf36f4e54529572687f9eec",
    (
        "cells",
        "1.5",
    ): "9f1a0b86fafa287105cac80038330246f127bbf46b6ead9470a7b258cf8cd02e",
    (
        "uniform",
        "0.315",
    ): "068e8d6974cf67c54ed45b96e3a1eb66364dbd8677be6588e0ad70f017b923f1",
    (
        "uniform",
        "1.0",
    ): "830fea90b1e0da349e3b8e91558ca470361ab536f0d1d86e08ca32b095a89c81",
    (
        "uniform",
        "1.5",
    ): "9e21b88fc5cb0cb39f4a9c9f90540145e903e6b6fcc6c1e94b012d6042a83b16",
}

# Reference accuracy measured from Zenodo record 17274305 on 2026-09-17.
# Tests enforce these scores minus the allowance as floors; improvements pass.
STANDARD_EXPECTED_F1_SCORES = {
    ("cells", "0.315", "no-chromatic"): 0.9848,
    ("cells", "0.315", "chromatic"): 0.9823,
    ("cells", "1.0", "no-chromatic"): 0.9623,
    ("cells", "1.0", "chromatic"): 0.9483,
    ("cells", "1.5", "no-chromatic"): 0.3867,
    ("cells", "1.5", "chromatic"): 0.3352,
    ("uniform", "0.315", "no-chromatic"): 0.9899,
    ("uniform", "0.315", "chromatic"): 0.9891,
    ("uniform", "1.0", "no-chromatic"): 0.9669,
    ("uniform", "1.0", "chromatic"): 0.9641,
    ("uniform", "1.5", "no-chromatic"): 0.6090,
    ("uniform", "1.5", "chromatic"): 0.5370,
}

EXHAUSTIVE_EXPECTED_F1_SCORES = {
    ("cells", "0.315", "decon"): 0.9823,
    ("cells", "0.315", "no-decon"): 0.9899,
    ("cells", "1.0", "decon"): 0.9483,
    ("cells", "1.0", "no-decon"): 0.9379,
    ("cells", "1.5", "decon"): 0.3352,
    ("cells", "1.5", "no-decon"): 0.8308,
    ("uniform", "0.315", "decon"): 0.9891,
    ("uniform", "0.315", "no-decon"): 0.9865,
    ("uniform", "1.0", "decon"): 0.9641,
    ("uniform", "1.0", "no-decon"): 0.9500,
    ("uniform", "1.5", "decon"): 0.5370,
    ("uniform", "1.5", "no-decon"): 0.7914,
}


def _total_seconds(record: dict[str, Any]) -> float:
    """Return elapsed seconds from a performance record."""
    return float(record["total_seconds"])


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    with path.open("rb") as file_handle:
        return hashlib.file_digest(file_handle, "sha256").hexdigest()


def _simulation_dataset_case_id(item: tuple[str, str]) -> str:
    """
    Return the pytest id for a simulation dataset fixture value.

    Parameters
    ----------
    item : tuple[str, str]
        Dataset variant and cached dataset directory name.

    Returns
    -------
    str
        Dataset variant name.
    """
    return item[0]


def _ufish_model_id(model: tuple[str, str | None]) -> str:
    """
    Return the pytest id for a U-FISH model fixture value.

    Parameters
    ----------
    model : tuple[str, str | None]
        Model display name and optional local weights path.

    Returns
    -------
    str
        Model display name.
    """
    return model[0]


DEFAULT_UFISH_MODELS = ((DEFAULT_UFISH_MODEL, DEFAULT_UFISH_MODEL),)
EXHAUSTIVE_SIMULATION_MATRIX = tuple(
    {
        "dataset_variant": dataset_variant,
        "dataset_dir": dataset_dir,
        "axial_spacing_um": axial_spacing_um,
        "ufish_model_name": ufish_model_name,
        "ufish_model": ufish_model,
        "preprocess_mode": preprocess_mode,
        "decon_readout": decon_readout,
        "registration_mode": registration_mode,
        "perform_deformable_registration": perform_deformable_registration,
        "chromatic_mode": chromatic_mode,
        "synthetic_chromatic_aberration": synthetic_chromatic_aberration,
        "estimate_chromatic_affines": estimate_chromatic_affines,
    }
    for (dataset_variant, dataset_dir), axial_spacing_um, (
        ufish_model_name,
        ufish_model,
    ), (
        preprocess_mode,
        decon_readout,
    ), (
        registration_mode,
        perform_deformable_registration,
    ), (
        chromatic_mode,
        synthetic_chromatic_aberration,
        estimate_chromatic_affines,
    ) in product(
        SIMULATION_DATASET_DIRS.items(),
        AXIAL_SPACING_UM,
        DEFAULT_UFISH_MODELS,
        PREPROCESS_MODES.items(),
        (("sofima", True),),
        (("chromatic", True, True),),
    )
)


@pytest.fixture(scope="session")
def simulation_dataset_root(pytestconfig: pytest.Config) -> Path:
    """Resolve the simulation dataset root passed to pytest."""
    configured_root = pytestconfig.getoption("--simulation-data-root")
    if not configured_root:
        pytest.skip("Pass --simulation-data-root with the extracted Zenodo dataset.")
    root = Path(configured_root).expanduser().resolve()
    if not root.is_dir():
        pytest.fail(f"Simulation dataset root is not a directory: {root}")
    return root


@pytest.fixture(scope="session")
def simulation_dataset_dirs(simulation_dataset_root: Path) -> dict[str, Path]:
    """Resolve variant paths; each selected case validates only its inputs."""
    return {
        variant: simulation_dataset_root / directory
        for variant, directory in SIMULATION_DATASET_DIRS.items()
    }


@pytest.fixture(scope="session")
def simulation_api() -> dict[str, Any]:
    """Import callable APIs used by the former notebook workflow."""
    from merfish3danalysis.cli.statphysbio_simulation.calculate_F1 import (
        calculate_F1,
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
    from merfish3danalysis.DataRegistration import DataRegistration
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    return {
        "DataRegistration": DataRegistration,
        "calculate_F1": calculate_F1,
        "convert_simulation": convert_simulation,
        "convert_data": convert_data,
        "decode_pixels": decode_pixels,
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

    ordered_records = sorted(records, key=_total_seconds)
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "records": ordered_records,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Simulation Performance Summary ===", flush=True)
    for record in ordered_records:
        dataset_variant = record.get("dataset_variant", "flat")
        preprocess_mode = record.get("preprocess_mode", "no-decon")
        ufish_model = record.get("ufish_model", "unknown")
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
                f"{dataset_variant}:{record['axial_spacing_um']} um:{ufish_model}:{preprocess_mode}:fp={feature_predictor_threshold}{magnitude_suffix} | "
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
        if not (source_case_dir / filename).is_file()
    ]
    if missing_files:
        pytest.fail(
            f"Missing required simulation files in {source_case_dir}: {missing_files}"
        )

    variant = next(
        name
        for name, directory in SIMULATION_DATASET_DIRS.items()
        if directory == source_case_dir.parent.name
    )
    image_path = source_case_dir / "aligned_1.tiff"
    expected_hash = ZENODO_ALIGNED_IMAGE_SHA256[(variant, source_case_dir.name)]
    actual_hash = _sha256_file(image_path)
    if actual_hash != expected_hash:
        pytest.fail(
            "Simulation data do not match Zenodo record 17274305: "
            f"{image_path}: expected {expected_hash}, got {actual_hash}"
        )

    case_root = work_root / source_case_dir.name
    case_root.mkdir(parents=True, exist_ok=False)

    for filename in REQUIRED_SIMULATION_FILES:
        _link_or_copy(source_case_dir / filename, case_root / filename)

    return case_root


def _run_simulation_preprocess(
    acquisition_root: Path,
    simulation_api: dict[str, Any],
    decon_readout: bool,
    ufish_model: str | None,
    perform_deformable_registration: bool,
) -> None:
    """Run simulation preprocess."""
    datastore = simulation_api["qi2labDataStore"](acquisition_root / "qi2labdatastore")
    registration_factory = simulation_api["DataRegistration"](
        datastore=datastore,
        perform_deformable_registration=perform_deformable_registration,
        overwrite_outputs=True,
        decon_readout=decon_readout,
        ufish_model=ufish_model,
        num_gpus=1,
        verbose=1,
    )
    registration_factory.register_all_tiles()
    registration_factory.global_register(create_max_proj_tiff=False)

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
    ufish_model: str | None,
    perform_deformable_registration: bool,
    case_label: str,
    synthetic_chromatic_aberration: bool = False,
) -> dict[str, float]:
    """Run simulation case setup."""
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
    simulation_api["convert_data"](
        acquisition_root,
        synthetic_chromatic_aberration=synthetic_chromatic_aberration,
    )
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
        ufish_model=ufish_model,
        perform_deformable_registration=perform_deformable_registration,
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
    ufish_model_name = case_spec["ufish_model_name"]
    ufish_model = case_spec["ufish_model"]
    decon_readout = case_spec["decon_readout"]
    perform_deformable_registration = case_spec.get(
        "perform_deformable_registration", False
    )
    synthetic_chromatic_aberration = case_spec.get(
        "synthetic_chromatic_aberration", False
    )
    estimate_chromatic_affines = case_spec.get("estimate_chromatic_affines", False)
    preprocess_mode = case_spec.get(
        "preprocess_mode",
        "decon" if decon_readout else "no-decon",
    )

    source_case_dir = simulation_dataset_dirs[dataset_variant] / axial_spacing_um
    search_radius = F1_RADIUS_BY_AXIAL_SPACING_UM.get(axial_spacing_um)
    if search_radius is None:
        pytest.skip(
            f"No F1 search radius configured for axial spacing {axial_spacing_um}."
        )
    default_minimum_pixels = SIMULATION_MINIMUM_PIXELS[axial_spacing_um]
    default_magnitude_threshold = SIMULATION_MAGNITUDE_THRESHOLDS[axial_spacing_um]

    case_root = _prepare_case_workspace(source_case_dir, tmp_path)
    case_label = (
        f"{dataset_variant}-{axial_spacing_um}-{ufish_model_name}-{preprocess_mode}-"
        f"{case_spec.get('registration_mode', 'affine')}-"
        f"fp{DEFAULT_FEATURE_PREDICTOR_THRESHOLD}"
    )
    setup_timings = _run_simulation_case_setup(
        case_root,
        simulation_api,
        decon_readout=decon_readout,
        ufish_model=ufish_model,
        perform_deformable_registration=perform_deformable_registration,
        case_label=case_label,
        synthetic_chromatic_aberration=synthetic_chromatic_aberration,
    )
    if run_calibration_decode:
        default_minimum_pixels = SIMULATION_MINIMUM_PIXELS[axial_spacing_um]
        calibration_label = f"{case_label}-calibration"
        calibration_start = time.perf_counter()
        print(f"[{calibration_label}] decode_pixels: start", flush=True)
        simulation_api["decode_pixels"](
            case_root / "sim_acquisition",
            minimum_pixels_per_RNA=default_minimum_pixels,
            feature_predictor_threshold=DEFAULT_FEATURE_PREDICTOR_THRESHOLD,
            magnitude_threshold=default_magnitude_threshold,
            skip_optimization=False,
            estimate_chromatic_affines=estimate_chromatic_affines,
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
    normalization_iterations: int = 3,
    estimate_chromatic_affines: bool = False,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Run simulation decode and f1."""
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
        normalization_iterations=normalization_iterations,
        estimate_chromatic_affines=estimate_chromatic_affines,
    )
    timings_seconds["decode_pixels"] = time.perf_counter() - start
    print(
        f"[{case_label}] decode_pixels: done ({timings_seconds['decode_pixels']:.2f}s)",
        flush=True,
    )

    start = time.perf_counter()
    print(f"[{case_label}] calculate_f1: start", flush=True)
    f1_results = simulation_api["calculate_F1"](case_root, search_radius=search_radius)
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
    ufish_model: str | None,
    feature_predictor_threshold: float,
    perform_deformable_registration: bool = False,
    synthetic_chromatic_aberration: bool = False,
    estimate_chromatic_affines: bool = False,
    lowpass_sigma: tuple[float, float, float] = DEFAULT_LOWPASS_SIGMA,
    magnitude_threshold: tuple[float, float] = (0.9, 10.0),
    minimum_pixels_per_rna: int | None = None,
    case_label: str | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Run simulation pipeline."""
    if case_label is None:
        case_label = case_root.name
    timings_seconds = _run_simulation_case_setup(
        case_root,
        simulation_api,
        decon_readout=decon_readout,
        ufish_model=ufish_model,
        perform_deformable_registration=perform_deformable_registration,
        case_label=case_label,
        synthetic_chromatic_aberration=synthetic_chromatic_aberration,
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
        estimate_chromatic_affines=estimate_chromatic_affines,
    )
    timings_seconds.update(
        {key: value for key, value in decode_timings.items() if key != "total"}
    )
    timings_seconds["total"] = sum(timings_seconds.values())

    return decode_results, timings_seconds, performance_metrics


@pytest.fixture(params=SIMULATION_DATASET_DIRS.items(), ids=_simulation_dataset_case_id)
def simulation_standard_dataset_case(
    request: pytest.FixtureRequest,
) -> tuple[str, str]:
    """
    Return one standard simulation dataset variant.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Active pytest fixture request.

    Returns
    -------
    tuple[str, str]
        Dataset variant name and cached dataset directory name.
    """
    return request.param


@pytest.fixture(params=AXIAL_SPACING_UM)
def simulation_standard_axial_spacing_um(
    request: pytest.FixtureRequest,
) -> str:
    """
    Return one standard axial spacing.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Active pytest fixture request.

    Returns
    -------
    str
        Axial spacing in microns.
    """
    return request.param


@pytest.fixture(params=DEFAULT_UFISH_MODELS, ids=_ufish_model_id)
def simulation_standard_ufish_model(
    request: pytest.FixtureRequest,
) -> tuple[str, str | None]:
    """
    Return one standard U-FISH model.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Active pytest fixture request.

    Returns
    -------
    tuple[str, str | None]
        Model display name and optional local weights path.
    """
    return request.param


@pytest.fixture(
    params=(
        pytest.param(("no-chromatic", False, False), id="no-chromatic"),
        pytest.param(("chromatic", True, True), id="chromatic"),
    )
)
def simulation_standard_chromatic_case(
    request: pytest.FixtureRequest,
) -> tuple[str, bool, bool]:
    """
    Return one standard chromatic-aberration condition.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Active pytest fixture request.

    Returns
    -------
    tuple[str, bool, bool]
        Case label, synthetic aberration flag, and estimator flag.
    """
    return request.param


def _assert_standard_result_valid(result: dict[str, Any]) -> None:
    """Validate stored outputs and independently calculate metrics from counts."""
    datastore_path = result["case_root"] / "sim_acquisition" / "qi2labdatastore"
    assert datastore_path.exists()
    assert isinstance(result["f1_results"], dict)
    assert RESULT_KEYS.issubset(result["f1_results"])
    metrics = result["performance_metrics"]
    tp, fp, fn = (
        metrics[key] for key in ("true_positives", "false_positives", "false_negatives")
    )
    assert all(isinstance(count, int) and count >= 0 for count in (tp, fp, fn))
    # Published acquisitions contain transcripts and must produce detections.
    assert tp + fn > 0
    assert tp + fp > 0
    assert metrics["precision"] == pytest.approx(tp / (tp + fp))
    assert metrics["recall"] == pytest.approx(tp / (tp + fn))
    assert metrics["f1_score"] == pytest.approx(2 * tp / (2 * tp + fp + fn))


def _run_standard_simulation_case(
    simulation_dataset_dirs: dict[str, Path],
    simulation_api: dict[str, Any],
    performance_records: list[dict[str, Any]],
    dataset_case: tuple[str, str],
    axial_spacing_um: str,
    ufish_model_case: tuple[str, str | None],
    chromatic_case: tuple[str, bool, bool],
    registration_case: tuple[str, bool],
    work_root: Path,
) -> dict[str, Any]:
    """Run one standard simulation pipeline case."""
    dataset_variant, _ = dataset_case
    ufish_model_name, ufish_model = ufish_model_case
    chromatic_mode, synthetic_chromatic_aberration, estimate_chromatic_affines = (
        chromatic_case
    )
    registration_mode, perform_deformable_registration = registration_case
    preprocess_mode = DEFAULT_PREPROCESS_MODE
    decon_readout = PREPROCESS_MODES[preprocess_mode]
    feature_predictor_threshold = DEFAULT_FEATURE_PREDICTOR_THRESHOLD

    source_case_dir = simulation_dataset_dirs[dataset_variant] / axial_spacing_um
    search_radius = F1_RADIUS_BY_AXIAL_SPACING_UM.get(axial_spacing_um)
    if search_radius is None:
        pytest.skip(
            f"No F1 search radius configured for axial spacing {axial_spacing_um}."
        )
    default_minimum_pixels = SIMULATION_MINIMUM_PIXELS[axial_spacing_um]
    default_magnitude_threshold = SIMULATION_MAGNITUDE_THRESHOLDS[axial_spacing_um]

    case_root = _prepare_case_workspace(source_case_dir, work_root)
    case_label = (
        f"{dataset_variant}-{axial_spacing_um}-{ufish_model_name}-{preprocess_mode}-"
        f"{registration_mode}-"
        f"{chromatic_mode}-"
        f"fp{feature_predictor_threshold}"
    )
    f1_results, timings_seconds, performance_metrics = _run_simulation_pipeline(
        case_root,
        simulation_api,
        search_radius,
        decon_readout=decon_readout,
        ufish_model=ufish_model,
        feature_predictor_threshold=feature_predictor_threshold,
        perform_deformable_registration=perform_deformable_registration,
        synthetic_chromatic_aberration=synthetic_chromatic_aberration,
        estimate_chromatic_affines=estimate_chromatic_affines,
        magnitude_threshold=default_magnitude_threshold,
        minimum_pixels_per_rna=default_minimum_pixels,
        case_label=case_label,
    )

    record = {
        "dataset_variant": dataset_variant,
        "axial_spacing_um": axial_spacing_um,
        "ufish_model": ufish_model_name,
        "preprocess_mode": preprocess_mode,
        "decon_readout": decon_readout,
        "registration_mode": registration_mode,
        "perform_deformable_registration": perform_deformable_registration,
        "chromatic_mode": chromatic_mode,
        "synthetic_chromatic_aberration": synthetic_chromatic_aberration,
        "estimate_chromatic_affines": estimate_chromatic_affines,
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
        "f1_results": f1_results,
        "timings_seconds": timings_seconds,
        "performance_metrics": performance_metrics,
        "record": record,
    }


@pytest.mark.integration
@pytest.mark.gpu
def test_zhuang_local_registration_on_published_simulation(
    simulation_dataset_dirs, simulation_api, tmp_path
):
    """Run the Zhuang example's local registration on a real acquisition."""
    case_root = _prepare_case_workspace(
        simulation_dataset_dirs["uniform"] / "0.315", tmp_path
    )
    simulation_api["convert_simulation"](case_root)
    acquisition_root = case_root / "sim_acquisition"
    simulation_api["convert_data"](acquisition_root)
    example = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "examples/zhuang_lab/02_register_and_deconvolve.py"
        )
    )
    result = CliRunner().invoke(example["app"], [str(acquisition_root), "--local-only"])
    assert result.exit_code == 0, (result.output, result.exception)
    datastore = simulation_api["qi2labDataStore"](acquisition_root / "qi2labdatastore")
    assert datastore.datastore_state["LocalRegistered"]
    for round_id in datastore.round_ids:
        transform = datastore.load_local_round_transform_zyx_um(0, round_id)
        assert transform is not None and np.isfinite(transform).all()
        assert (
            datastore.load_local_deconvolved_fiducial_image(0, round_id, None) is None
        )
    for bit in datastore.bit_ids:
        corrected = datastore.load_local_corrected_image(0, bit=bit, return_future=None)
        prediction = datastore.load_local_feature_predictor_image(0, bit, None)
        assert prediction is not None and prediction.shape == corrected.shape
        assert datastore.load_local_feature_predictor_spots(0, bit) is not None
        assert datastore.load_local_deconvolved_readout_image(0, bit, None) is None
    assert datastore.load_global_fiducial_projection(False) is None


@pytest.mark.integration
@pytest.mark.gpu
def test_qi2lab_fiducial_only_then_no_decon_preprocessing(
    simulation_dataset_dirs, simulation_api, tmp_path
):
    """Exercise both preprocessing modes on the published acquisition."""
    from merfish3danalysis.cli.qi2lab_microscopes import preprocess

    case_root = _prepare_case_workspace(
        simulation_dataset_dirs["uniform"] / "0.315", tmp_path
    )
    simulation_api["convert_simulation"](case_root)
    acquisition_root = case_root / "sim_acquisition"
    simulation_api["convert_data"](acquisition_root)
    runner = CliRunner()

    result = runner.invoke(
        preprocess.app, [str(acquisition_root), "--fiducial-registration-only"]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    datastore = simulation_api["qi2labDataStore"](acquisition_root / "qi2labdatastore")
    assert not datastore.datastore_state["LocalRegistered"]
    for round_id in datastore.round_ids:
        assert datastore.load_local_round_transform_zyx_um(0, round_id) is not None
        assert (
            datastore.load_local_deconvolved_fiducial_image(0, round_id, None)
            is not None
        )
    assert datastore.load_global_fiducial_projection(False) is not None
    for bit in datastore.bit_ids:
        assert datastore.load_local_deconvolved_readout_image(0, bit, None) is None
        assert datastore.load_local_feature_predictor_image(0, bit, None) is None
        assert datastore.load_local_feature_predictor_spots(0, bit) is None

    result = runner.invoke(preprocess.app, [str(acquisition_root), "--no-decon"])
    assert result.exit_code == 0, (result.output, result.exception)
    datastore = simulation_api["qi2labDataStore"](acquisition_root / "qi2labdatastore")
    assert datastore.datastore_state["LocalRegistered"]
    for bit in datastore.bit_ids:
        assert datastore.load_local_deconvolved_readout_image(0, bit, None) is None
        corrected = datastore.load_local_corrected_image(0, bit=bit, return_future=None)
        prediction = datastore.load_local_feature_predictor_image(0, bit, None)
        assert prediction is not None and prediction.shape == corrected.shape
        assert datastore.load_local_feature_predictor_spots(0, bit) is not None

    first_bit = datastore.bit_ids[0]
    original_prediction = datastore.load_local_feature_predictor_image(
        0, first_bit, False
    )
    original_spots = datastore.load_local_feature_predictor_spots(0, first_bit)
    result = runner.invoke(
        preprocess.app, [str(acquisition_root), "--fiducial-registration-only"]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    datastore = simulation_api["qi2labDataStore"](acquisition_root / "qi2labdatastore")
    assert datastore.datastore_state["LocalRegistered"]
    for bit in datastore.bit_ids:
        assert datastore.load_local_feature_predictor_image(0, bit, None) is not None
        assert datastore.load_local_feature_predictor_spots(0, bit) is not None
    np.testing.assert_array_equal(
        datastore.load_local_feature_predictor_image(0, first_bit, False),
        original_prediction,
    )
    import pandas as pd

    pd.testing.assert_frame_equal(
        datastore.load_local_feature_predictor_spots(0, first_bit), original_spots
    )


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize(
    "decode_run_key,normalization_features,check_exclusions",
    [
        pytest.param(None, "all", False, id="default-run-all-features"),
        pytest.param("integration_run", None, False, id="named-run-default-features"),
        pytest.param("exclusion_baseline", None, True, id="excluded-cells"),
    ],
)
def test_qi2lab_commands_decode_simulation_ground_truth(
    simulation_dataset_dirs,
    simulation_api,
    tmp_path,
    decode_run_key,
    normalization_features,
    check_exclusions,
):
    """Run the standard qi2lab commands on the published Nyquist acquisition."""
    from merfish3danalysis.cli.qi2lab_microscopes import pixeldecode, preprocess

    case_root = _prepare_case_workspace(
        simulation_dataset_dirs["uniform"] / "0.315", tmp_path
    )
    simulation_api["convert_simulation"](case_root)
    acquisition_root = case_root / "sim_acquisition"
    simulation_api["convert_data"](acquisition_root)

    runner = CliRunner()
    result = runner.invoke(preprocess.app, [str(acquisition_root)])
    assert result.exit_code == 0, (result.output, result.exception)

    # The 256x256 simulation has a centered stage position: XY is
    # -128 * 0.081 um rounded to 0.01 um, and recorded stage Z is zero.
    # Reopen the datastore; command completion alone is insufficient.
    datastore = simulation_api["qi2labDataStore"](
        acquisition_root / "qi2labdatastore", validate=False
    )
    affine, origin, spacing = datastore.load_global_coord_xforms_um(tile=0)
    np.testing.assert_array_equal(affine, np.eye(4))
    np.testing.assert_allclose(origin, [0, -10.37, -10.37])
    np.testing.assert_allclose(spacing, [0.315, 0.081, 0.081])
    assert datastore.datastore_state["LocalRegistered"]
    for bit in datastore.bit_ids:
        corrected = datastore.load_local_corrected_image(0, bit=bit, return_future=None)
        deconvolved = datastore.load_local_deconvolved_readout_image(0, bit, None)
        prediction = datastore.load_local_feature_predictor_image(0, bit, None)
        assert deconvolved is not None and deconvolved.shape == corrected.shape
        assert prediction is not None and prediction.shape == corrected.shape
        assert datastore.load_local_feature_predictor_spots(0, bit) is not None

    arguments = [
        str(acquisition_root),
        "--minimum-pixels-per-RNA",
        "28",
        "--magnitude-threshold",
        "0.9",
        "10",
    ]
    if normalization_features is not None:
        arguments.extend(["--normalization-features", normalization_features])
    if decode_run_key is not None:
        arguments.extend(["--decode-run-key", decode_run_key])
    result = runner.invoke(pixeldecode.app, arguments)
    assert result.exit_code == 0, (result.output, result.exception)
    datastore = simulation_api["qi2labDataStore"](acquisition_root / "qi2labdatastore")
    assert (
        datastore.load_local_decoded_spots(tile=0, decode_run_key=decode_run_key)
        is not None
    )
    assert (
        datastore.load_global_filtered_decoded_spots(decode_run_key=decode_run_key)
        is not None
    )
    other_run = "integration_run" if decode_run_key is None else None
    assert datastore.load_local_decoded_spots(tile=0, decode_run_key=other_run) is None
    assert (
        datastore.load_global_filtered_decoded_spots(decode_run_key=other_run) is None
    )
    for kind in ("global", "iterative"):
        normalization, background = datastore.load_decode_normalization_vectors(
            decode_run_key, kind
        )
        assert normalization is not None
        assert background is not None
        metadata = datastore.load_decode_normalization_metadata(decode_run_key, kind)
        assert metadata["normalization_features"] == (normalization_features or "cells")
    scores = simulation_api["calculate_F1"](
        case_root, search_radius=1.0, decode_run_key=decode_run_key
    )
    tp, fp, fn = (
        scores[key] for key in ("True Positives", "False Positives", "False Negatives")
    )
    assert tp > 0 and fp >= 0 and fn >= 0
    assert scores["Precision"] == pytest.approx(tp / (tp + fp))
    assert scores["Recall"] == pytest.approx(tp / (tp + fn))
    assert scores["F1 Score"] == pytest.approx(2 * tp / (2 * tp + fp + fn))
    assert scores["F1 Score"] >= (
        STANDARD_EXPECTED_F1_SCORES[("uniform", "0.315", "no-chromatic")]
        - F1_ACCURACY_ALLOWANCE
    )

    if decode_run_key is None:
        first_bit = datastore.bit_ids[0]
        deconvolved = datastore.load_local_deconvolved_readout_image(
            0, first_bit, False
        )
        result = runner.invoke(
            preprocess.app, [str(acquisition_root), "--no-decon", "--no-overwrite"]
        )
        assert result.exit_code == 0, (result.output, result.exception)
        datastore = simulation_api["qi2labDataStore"](
            acquisition_root / "qi2labdatastore"
        )
        assert not pixeldecode._readouts_are_deconvolved(datastore)
        for bit in datastore.bit_ids:
            corrected = datastore.load_local_corrected_image(
                0, bit=bit, return_future=False
            )
            np.testing.assert_array_equal(
                datastore.load_local_readout_image(0, bit, False), corrected
            )
            assert (
                datastore.load_local_image_metadata(
                    0, bit=bit, image_names=("feature_predictor_data",)
                )["source_image"]
                == "corrected_data"
            )
            assert datastore.load_local_feature_predictor_spots(0, bit) is not None
        np.testing.assert_array_equal(
            datastore.load_local_fusion_image(0, bit=first_bit)[:],
            datastore.load_local_corrected_image(0, bit=first_bit, return_future=False),
        )
        np.testing.assert_array_equal(
            datastore.load_local_deconvolved_readout_image(0, first_bit, False),
            deconvolved,
        )

        no_decon_run = "no_decon_run"
        result = runner.invoke(
            pixeldecode.app, [*arguments, "--decode-run-key", no_decon_run]
        )
        assert result.exit_code == 0, (result.output, result.exception)
        scores = simulation_api["calculate_F1"](
            case_root, search_radius=1.0, decode_run_key=no_decon_run
        )
        assert scores["F1 Score"] >= (
            EXHAUSTIVE_EXPECTED_F1_SCORES[("uniform", "0.315", "no-decon")]
            - F1_ACCURACY_ALLOWANCE
        )

    if check_exclusions:
        import pandas as pd
        from roifile import ImagejRoi, roiwrite
        from shapely.geometry import Point, Polygon

        from merfish3danalysis.PixelDecoder import PixelDecoder

        # Reuse the real first optimization iteration to select an abundant,
        # nonblank gene and a cell ROI containing only part of the field.
        baseline = pd.read_parquet(
            datastore.decoded_temporary_dir(decode_run_key, iteration=0)
            / "tile000_temp_decoded.parquet"
        )
        genes = baseline["gene_id"].astype("string")
        excluded_gene = (
            genes[~genes.str.lower().str.startswith("blank")].value_counts().index[0]
        )
        x_min, x_max = (
            baseline["global_x"].min() - 1,
            baseline["global_x"].quantile(0.75),
        )
        y_min, y_max = baseline["global_y"].min() - 1, baseline["global_y"].max() + 1
        roi = ImagejRoi.frompoints(
            np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
        )
        roiwrite(datastore.cellpose_global_rois_path(), [roi], mode="w")
        polygon = Polygon(next(iter(datastore.load_global_cellpose_roi_zip().values())))

        run = "excluded_cells"
        decoder = PixelDecoder(
            datastore,
            merfish_bits=datastore.num_bits,
            num_gpus=1,
            verbose=0,
            decode_run_key=run,
            normalization_features="cells",
        )
        decoder.optimize_normalization_by_decoding(
            n_random_tiles=1,
            n_iterations=1,
            minimum_pixels=28,
            magnitude_threshold=(0.9, 10),
            excluded_gene_ids=[excluded_gene, excluded_gene],
        )

        optimization = pd.read_parquet(
            datastore.decoded_temporary_dir(run, iteration=0)
            / "tile000_temp_decoded.parquet"
        )
        assert (genes == excluded_gene).any()
        assert not (optimization["gene_id"] == excluded_gene).any()
        inside = np.array(
            [
                polygon.covers(Point(x, y))
                for x, y in zip(
                    optimization["global_x"], optimization["global_y"], strict=True
                )
            ]
        )
        assert inside.any() and (~inside).any()
        pd.testing.assert_frame_equal(
            decoder._df_barcodes_loaded.reset_index(drop=True),
            optimization.loc[inside].reset_index(drop=True),
        )
        metadata = datastore.load_decode_normalization_metadata(run, "iterative")
        assert metadata["excluded_gene_ids"] == [excluded_gene]
        assert metadata["normalization_features"] == "cells"
        global_cells, _ = datastore.load_decode_normalization_vectors(run, "global")
        global_all, _ = datastore.load_decode_normalization_vectors(
            decode_run_key, "global"
        )
        assert not np.allclose(global_cells, global_all)

        # Exclusions apply to fitting only; the final decoder still calls that gene.
        decoder.decode_one_tile(
            tile_idx=0,
            minimum_pixels=28,
            magnitude_threshold=(0.9, 10),
            normalization_method="iterative",
        )
        assert (decoder.decoded_barcodes["gene_id"] == excluded_gene).any()


@pytest.mark.integration
@pytest.mark.gpu
def test_simulation_standard_matrix(
    simulation_dataset_dirs: dict[str, Path],
    simulation_api: dict[str, Any],
    performance_records: list[dict[str, Any]],
    simulation_standard_dataset_case: tuple[str, str],
    simulation_standard_axial_spacing_um: str,
    simulation_standard_ufish_model: tuple[str, str | None],
    simulation_standard_chromatic_case: tuple[str, bool, bool],
    tmp_path: Path,
) -> None:
    """Require ground-truth accuracy for both affine and SOFIMA processing."""
    affine_result = _run_standard_simulation_case(
        simulation_dataset_dirs,
        simulation_api,
        performance_records,
        simulation_standard_dataset_case,
        simulation_standard_axial_spacing_um,
        simulation_standard_ufish_model,
        simulation_standard_chromatic_case,
        ("affine", False),
        tmp_path / "affine",
    )
    _assert_standard_result_valid(affine_result)

    sofima_result = _run_standard_simulation_case(
        simulation_dataset_dirs,
        simulation_api,
        performance_records,
        simulation_standard_dataset_case,
        simulation_standard_axial_spacing_um,
        simulation_standard_ufish_model,
        simulation_standard_chromatic_case,
        ("sofima", True),
        tmp_path / "sofima",
    )
    _assert_standard_result_valid(sofima_result)

    dataset_variant, _ = simulation_standard_dataset_case
    chromatic_mode, _, _ = simulation_standard_chromatic_case
    expected_f1 = STANDARD_EXPECTED_F1_SCORES[
        (dataset_variant, simulation_standard_axial_spacing_um, chromatic_mode)
    ]
    minimum_f1 = expected_f1 - F1_ACCURACY_ALLOWANCE
    assert affine_result["performance_metrics"]["f1_score"] >= minimum_f1
    assert sofima_result["performance_metrics"]["f1_score"] >= minimum_f1

    affine_f1 = round(affine_result["performance_metrics"]["f1_score"], 3)
    sofima_f1 = round(sofima_result["performance_metrics"]["f1_score"], 3)
    assert sofima_f1 >= affine_f1, (
        "SOFIMA standard simulation integration failure: "
        f"{sofima_result['case_label']} F1={sofima_f1:.3f} "
        f"is lower than {affine_result['case_label']} F1={affine_f1:.3f}."
    )


@pytest.fixture(
    params=EXHAUSTIVE_SIMULATION_MATRIX,
    ids=[
        (
            f"{case['dataset_variant']}-"
            f"{case['axial_spacing_um']}-"
            f"{case['ufish_model_name']}-"
            f"{case['preprocess_mode']}-"
            f"{case['registration_mode']}"
        )
        for case in EXHAUSTIVE_SIMULATION_MATRIX
    ],
)
def simulation_exhaustive_case_spec(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Parametrize the exhaustive local simulation base cases."""
    return request.param


@pytest.fixture
def simulation_full_preprocessed_case(
    simulation_dataset_dirs: dict[str, Path],
    simulation_api: dict[str, Any],
    simulation_exhaustive_case_spec: dict[str, Any],
    tmp_path: Path,
) -> dict[str, Any]:
    """Prepare one exhaustive simulation case before decoding."""
    return _prepare_preprocessed_magnitude_case(
        simulation_dataset_dirs,
        simulation_api,
        simulation_exhaustive_case_spec,
        tmp_path,
        run_calibration_decode=True,
    )


@pytest.fixture
def simulation_full_case_result(
    simulation_api: dict[str, Any],
    performance_records: list[dict[str, Any]],
    simulation_full_preprocessed_case: dict[str, Any],
) -> dict[str, Any]:
    """Decode and score one exhaustive simulation case."""
    case_root = simulation_full_preprocessed_case["case_root"]
    case_spec = simulation_full_preprocessed_case["case_spec"]
    case_label = simulation_full_preprocessed_case["case_label"]
    search_radius = simulation_full_preprocessed_case["search_radius"]
    setup_timings = simulation_full_preprocessed_case["setup_timings"]
    dataset_variant = case_spec["dataset_variant"]
    axial_spacing_um = case_spec["axial_spacing_um"]
    ufish_model_name = case_spec["ufish_model_name"]
    preprocess_mode = case_spec["preprocess_mode"]
    decon_readout = case_spec["decon_readout"]
    registration_mode = case_spec["registration_mode"]
    perform_deformable_registration = case_spec["perform_deformable_registration"]
    chromatic_mode = case_spec["chromatic_mode"]
    synthetic_chromatic_aberration = case_spec["synthetic_chromatic_aberration"]
    estimate_chromatic_affines = case_spec["estimate_chromatic_affines"]
    default_minimum_pixels = SIMULATION_MINIMUM_PIXELS[axial_spacing_um]
    default_magnitude_threshold = SIMULATION_MAGNITUDE_THRESHOLDS[axial_spacing_um]

    f1_results, decode_timings, performance_metrics = _run_simulation_decode_and_f1(
        case_root,
        simulation_api,
        search_radius,
        feature_predictor_threshold=DEFAULT_FEATURE_PREDICTOR_THRESHOLD,
        lowpass_sigma=DEFAULT_LOWPASS_SIGMA,
        magnitude_threshold=default_magnitude_threshold,
        minimum_pixels_per_rna=default_minimum_pixels,
        case_label=case_label,
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
        "ufish_model": ufish_model_name,
        "preprocess_mode": preprocess_mode,
        "decon_readout": decon_readout,
        "registration_mode": registration_mode,
        "perform_deformable_registration": perform_deformable_registration,
        "chromatic_mode": chromatic_mode,
        "synthetic_chromatic_aberration": synthetic_chromatic_aberration,
        "estimate_chromatic_affines": estimate_chromatic_affines,
        "feature_predictor_threshold": DEFAULT_FEATURE_PREDICTOR_THRESHOLD,
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
        "case_spec": case_spec,
        "f1_results": f1_results,
        "timings_seconds": timings_seconds,
        "performance_metrics": performance_metrics,
        "record": record,
    }


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.simulation_exhaustive
def test_simulation_exhaustive_matrix(
    simulation_full_case_result: dict[str, Any],
) -> None:
    """Run the exhaustive simulation matrix with package-default settings."""
    _assert_standard_result_valid(simulation_full_case_result)
    performance_metrics = simulation_full_case_result["performance_metrics"]

    case_spec = simulation_full_case_result["case_spec"]
    expected_f1 = EXHAUSTIVE_EXPECTED_F1_SCORES[
        (
            case_spec["dataset_variant"],
            case_spec["axial_spacing_um"],
            case_spec["preprocess_mode"],
        )
    ]
    assert performance_metrics["f1_score"] >= expected_f1 - F1_ACCURACY_ALLOWANCE
