"""Simulation pipeline regression tests.

The matrix converts cached StatPhysBio simulation outputs into qi2lab
datastores, runs preprocessing, decodes transcripts, and scores decoded
features against simulation ground truth. The optional exhaustive mode expands
the matrix across feature-prediction probability thresholds and cached U-FISH
models so README result tables can be regenerated from the same report.
"""

import json
import os
import shutil
import time
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from merfish3danalysis.DataRegistration import DEFAULT_UFISH_MODEL, UFISH_MODEL_ALIASES

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


def _nearest_nyquist_multiple(
    thresholds_by_multiple: dict[float, float],
    nyquist_multiple: float,
) -> float:
    """Return the configured Nyquist multiple nearest to a measured multiple."""

    best_multiple = next(iter(thresholds_by_multiple))
    best_distance = abs(best_multiple - nyquist_multiple)
    for multiple in thresholds_by_multiple:
        distance = abs(multiple - nyquist_multiple)
        if distance < best_distance:
            best_multiple = multiple
            best_distance = distance
    return best_multiple


def _total_seconds(record: dict[str, Any]) -> float:
    """Return elapsed seconds from a performance record."""

    return float(record["total_seconds"])


def _ufish_label_from_weights_path(weights_path: Path) -> str:
    """
    Return a stable test id for a cached U-FISH ONNX weights file.

    Parameters
    ----------
    weights_path : Path
        Function argument.

    Returns
    -------
    str
        Function result.
    """

    label = weights_path.stem
    if label.startswith("v1.0.1-"):
        label = label[len("v1.0.1-") :]
    if label.endswith("_model"):
        label = label[: -len("_model")]
    return label.lower()


def _available_ufish_models() -> tuple[tuple[str, str | None], ...]:
    """
    Return every locally available distinct U-FISH model.

    Returns
    -------
    tuple[tuple[str, str | None], ...]
        Function result.
    """

    models: list[tuple[str, str | None]] = []
    seen_weights: set[str | None] = set()

    for alias, weights_file in UFISH_MODEL_ALIASES.items():
        if weights_file is None:
            continue
        weights_path = Path.home() / ".ufish" / weights_file
        if not weights_path.exists():
            continue
        resolved = str(weights_path)
        if resolved in seen_weights:
            continue
        models.append((alias, resolved))
        seen_weights.add(resolved)

    finetune_dir = Path.home() / ".ufish" / "finetune_models"
    if finetune_dir.exists():
        for weights_path in sorted(finetune_dir.glob("*.onnx")):
            resolved = str(weights_path)
            if resolved in seen_weights:
                continue
            models.append((_ufish_label_from_weights_path(weights_path), resolved))
            seen_weights.add(resolved)

    return tuple(models)


UFISH_MODELS = _available_ufish_models()
REGISTRATION_MODES = (
    (("affine", False), ("deformable", True))
    if os.environ.get("MERFISH3D_SIM_INCLUDE_DEFORMABLE", "").lower()
    in {"1", "true", "yes"}
    else (("affine", False),)
)
STANDARD_SIMULATION_MATRIX = tuple(
    {
        "dataset_variant": dataset_variant,
        "dataset_dir": dataset_dir,
        "axial_spacing_um": axial_spacing_um,
        "ufish_model_name": ufish_model_name,
        "ufish_model": ufish_model,
        "preprocess_mode": DEFAULT_PREPROCESS_MODE,
        "decon_readout": PREPROCESS_MODES[DEFAULT_PREPROCESS_MODE],
        "registration_mode": registration_mode,
        "perform_deformable_registration": perform_deformable_registration,
        "feature_predictor_threshold": DEFAULT_FEATURE_PREDICTOR_THRESHOLD,
    }
    for (dataset_variant, dataset_dir), axial_spacing_um, (
        ufish_model_name,
        ufish_model,
    ), (
        registration_mode,
        perform_deformable_registration,
    ) in product(
        SIMULATION_DATASET_DIRS.items(),
        AXIAL_SPACING_UM,
        UFISH_MODELS,
        REGISTRATION_MODES,
    )
)
FULL_SIMULATION_MATRIX = tuple(
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
        "feature_predictor_threshold": feature_predictor_threshold,
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
    ), feature_predictor_threshold in product(
        SIMULATION_DATASET_DIRS.items(),
        AXIAL_SPACING_UM,
        UFISH_MODELS,
        PREPROCESS_MODES.items(),
        REGISTRATION_MODES,
        FEATURE_PREDICTOR_THRESHOLDS,
    )
)
FULL_SWEEP_BASE_SIMULATION_MATRIX = tuple(
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
    ) in product(
        SIMULATION_DATASET_DIRS.items(),
        AXIAL_SPACING_UM,
        UFISH_MODELS,
        PREPROCESS_MODES.items(),
        REGISTRATION_MODES,
    )
)
STANDARD_EXPECTED_F1_SCORES = {
    ("cells", "0.315"): 0.984822934232715,
    ("cells", "1.0"): 0.9532710280373832,
    ("cells", "1.5"): 0.3768224299065421,
    ("uniform", "0.315"): 0.9899074852817493,
    ("uniform", "1.0"): 0.9672977624784854,
    ("uniform", "1.5"): 0.6160687815001483,
}
FULL_EXPECTED_F1_SCORES = {
    ("cells", "0.315", "decon", 0.1): 0.984822934232715,
    ("cells", "0.315", "decon", 0.2): 0.984822934232715,
    ("cells", "0.315", "decon", 0.3): 0.984822934232715,
    ("cells", "0.315", "decon", 0.4): 0.984822934232715,
    ("cells", "0.315", "decon", 0.5): 0.984822934232715,
    ("cells", "0.315", "no-decon", 0.1): 0.9881956155143339,
    ("cells", "0.315", "no-decon", 0.2): 0.9881956155143339,
    ("cells", "0.315", "no-decon", 0.3): 0.9881956155143339,
    ("cells", "0.315", "no-decon", 0.4): 0.9881956155143339,
    ("cells", "0.315", "no-decon", 0.5): 0.9881956155143339,
    ("cells", "1.0", "decon", 0.1): 0.9532710280373832,
    ("cells", "1.0", "decon", 0.2): 0.9532710280373832,
    ("cells", "1.0", "decon", 0.3): 0.9532710280373832,
    ("cells", "1.0", "decon", 0.4): 0.9532710280373832,
    ("cells", "1.0", "decon", 0.5): 0.9532710280373832,
    ("cells", "1.0", "no-decon", 0.1): 0.9540816326530613,
    ("cells", "1.0", "no-decon", 0.2): 0.9540816326530613,
    ("cells", "1.0", "no-decon", 0.3): 0.9540816326530613,
    ("cells", "1.0", "no-decon", 0.4): 0.9540816326530613,
    ("cells", "1.0", "no-decon", 0.5): 0.9540816326530613,
    ("cells", "1.5", "decon", 0.1): 0.3768224299065421,
    ("cells", "1.5", "decon", 0.2): 0.3768224299065421,
    ("cells", "1.5", "decon", 0.3): 0.3768224299065421,
    ("cells", "1.5", "decon", 0.4): 0.3768224299065421,
    ("cells", "1.5", "decon", 0.5): 0.3768224299065421,
    ("cells", "1.5", "no-decon", 0.1): 0.9051580698835274,
    ("cells", "1.5", "no-decon", 0.2): 0.9051580698835274,
    ("cells", "1.5", "no-decon", 0.3): 0.9051580698835274,
    ("cells", "1.5", "no-decon", 0.4): 0.9051580698835274,
    ("cells", "1.5", "no-decon", 0.5): 0.9051580698835274,
    ("uniform", "0.315", "decon", 0.1): 0.9899074852817493,
    ("uniform", "0.315", "decon", 0.2): 0.9899074852817493,
    ("uniform", "0.315", "decon", 0.3): 0.9899074852817493,
    ("uniform", "0.315", "decon", 0.4): 0.9899074852817493,
    ("uniform", "0.315", "decon", 0.5): 0.9899074852817493,
    ("uniform", "0.315", "no-decon", 0.1): 0.9882253994953742,
    ("uniform", "0.315", "no-decon", 0.2): 0.9882253994953742,
    ("uniform", "0.315", "no-decon", 0.3): 0.9882253994953742,
    ("uniform", "0.315", "no-decon", 0.4): 0.9882253994953742,
    ("uniform", "0.315", "no-decon", 0.5): 0.9882253994953742,
    ("uniform", "1.0", "decon", 0.1): 0.9672977624784854,
    ("uniform", "1.0", "decon", 0.2): 0.9672977624784854,
    ("uniform", "1.0", "decon", 0.3): 0.9672977624784854,
    ("uniform", "1.0", "decon", 0.4): 0.9672977624784854,
    ("uniform", "1.0", "decon", 0.5): 0.9672977624784854,
    ("uniform", "1.0", "no-decon", 0.1): 0.9598633646456021,
    ("uniform", "1.0", "no-decon", 0.2): 0.9598633646456021,
    ("uniform", "1.0", "no-decon", 0.3): 0.9598633646456021,
    ("uniform", "1.0", "no-decon", 0.4): 0.9598633646456021,
    ("uniform", "1.0", "no-decon", 0.5): 0.9598633646456021,
    ("uniform", "1.5", "decon", 0.1): 0.6160687815001483,
    ("uniform", "1.5", "decon", 0.2): 0.6160687815001483,
    ("uniform", "1.5", "decon", 0.3): 0.6160687815001483,
    ("uniform", "1.5", "decon", 0.4): 0.6160687815001483,
    ("uniform", "1.5", "decon", 0.5): 0.6160687815001483,
    ("uniform", "1.5", "no-decon", 0.1): 0.7897990726429677,
    ("uniform", "1.5", "no-decon", 0.2): 0.7897990726429677,
    ("uniform", "1.5", "no-decon", 0.3): 0.7897990726429677,
    ("uniform", "1.5", "no-decon", 0.4): 0.7897990726429677,
    ("uniform", "1.5", "no-decon", 0.5): 0.7897990726429677,
}


@pytest.fixture(scope="session")
def simulation_dataset_root() -> Path:
    """
    Resolve the local simulation dataset root configured in this test file.

    Returns
    -------
    Path
        Function result.
    """

    root = LOCAL_SIMULATION_DATA_ROOT.expanduser().resolve()
    if not root.exists():
        pytest.skip(f"Local simulation dataset root does not exist: {root}")
    return _normalize_dataset_root(root)


@pytest.fixture(scope="session")
def simulation_dataset_dirs(simulation_dataset_root: Path) -> dict[str, Path]:
    """
    Resolve local dataset directories for the simulation matrix.

    Parameters
    ----------
    simulation_dataset_root : Path
        Function argument.

    Returns
    -------
    dict[str, Path]
        Function result.
    """

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
    """
    Normalize the configured local simulation dataset root.

    Parameters
    ----------
    root : Path
        Function argument.

    Returns
    -------
    Path
        Function result.
    """

    example_dir = root / "example_16bit_flat"
    if not example_dir.exists():
        pytest.skip(
            f"Could not find 'example_16bit_flat' under dataset root: {root}. "
            "Update LOCAL_SIMULATION_DATA_ROOT in this test file."
        )

    return root


def _default_minimum_pixels_per_rna(axial_spacing_um: str) -> int:
    """
    Return the default minimum-pixel threshold for a simulation spacing.

    Parameters
    ----------
    axial_spacing_um : str
        Function argument.

    Returns
    -------
    int
        Function result.
    """

    if axial_spacing_um == "0.315":
        return DEFAULT_MINIMUM_PIXELS_3D
    return DEFAULT_MINIMUM_PIXELS_2D


def _default_standard_magnitude_threshold(
    axial_spacing_um: str,
) -> tuple[float, float]:
    """
    Return the sampling-aware default magnitude threshold for simulations.

    Parameters
    ----------
    axial_spacing_um : str
        Function argument.

    Returns
    -------
    tuple[float, float]
        Function result.
    """

    if axial_spacing_um == "0.315":
        return DEFAULT_MAGNITUDE_THRESHOLD

    nyquist_multiple = float(axial_spacing_um) / SIMULATION_AXIAL_NYQUIST_STEP_UM
    nearest_multiple = _nearest_nyquist_multiple(
        SIMULATION_2D_MAGNITUDE_THRESHOLD_BY_NYQUIST,
        nyquist_multiple,
    )
    lower_threshold = SIMULATION_2D_MAGNITUDE_THRESHOLD_BY_NYQUIST[nearest_multiple]
    return (lower_threshold, DEFAULT_MAGNITUDE_THRESHOLD[1])


@pytest.fixture(scope="session")
def simulation_api() -> dict[str, Any]:
    """
    Import callable APIs used by the former notebook workflow.

    Returns
    -------
    dict[str, Any]
        Function result.
    """

    try:
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
    except Exception as exc:
        pytest.skip(f"Simulation API imports unavailable in this environment: {exc!r}")

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
    """
    Collect and persist runtime/performance records across all approaches.

    Returns
    -------
    list[dict[str, Any]]
        Function result.
    """

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
    """
    Use symlinks for large files when possible, with copy fallback.

    Parameters
    ----------
    src : Path
        Function argument.
    dst : Path
        Function argument.

    Returns
    -------
    None
        Function result.
    """

    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _prepare_case_workspace(source_case_dir: Path, work_root: Path) -> Path:
    """
    Create an isolated workspace with only files needed for this pipeline.

    Parameters
    ----------
    source_case_dir : Path
        Function argument.
    work_root : Path
        Function argument.

    Returns
    -------
    Path
        Function result.
    """

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
    """
    Calculate f1 from datastore.

    Parameters
    ----------
    case_root : Path
        Function argument.
    simulation_api : dict[str, Any]
        Function argument.
    search_radius : float
        Function argument.

    Returns
    -------
    dict[str, float]
        Function result.
    """
    return simulation_api["calculate_F1"](
        case_root,
        search_radius=search_radius,
    )


def _run_simulation_preprocess(
    acquisition_root: Path,
    simulation_api: dict[str, Any],
    decon_readout: bool,
    ufish_model: str | None,
    perform_deformable_registration: bool,
) -> None:
    """
    Run simulation preprocess.

    Parameters
    ----------
    acquisition_root : Path
        Function argument.
    simulation_api : dict[str, Any]
        Function argument.
    decon_readout : bool
        Function argument.
    ufish_model : str | None
        Function argument.
    perform_deformable_registration : bool
        Function argument.

    Returns
    -------
    None
        Function result.
    """
    datastore = simulation_api["qi2labDataStore"](acquisition_root / "qi2labdatastore")
    save_all_fiducial_registered = os.environ.get(
        "MERFISH3D_SIM_SAVE_ALL_FIDUCIAL_REGISTERED", ""
    ).lower() in {"1", "true", "yes"}
    registration_factory = simulation_api["DataRegistration"](
        datastore=datastore,
        perform_deformable_registration=perform_deformable_registration,
        overwrite_registered=True,
        save_all_fiducial_registered=save_all_fiducial_registered,
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
    synthetic_chromatic_aberration_scale: float = 1.0,
) -> dict[str, float]:
    """
    Run simulation case setup.

    Parameters
    ----------
    case_root : Path
        Function argument.
    simulation_api : dict[str, Any]
        Function argument.
    decon_readout : bool
        Function argument.
    ufish_model : str | None
        Function argument.
    perform_deformable_registration : bool
        Function argument.
    case_label : str
        Function argument.
    synthetic_chromatic_aberration : bool
        Function argument.
    synthetic_chromatic_aberration_scale : float
        Function argument.

    Returns
    -------
    dict[str, float]
        Function result.
    """
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
        synthetic_chromatic_aberration_scale=synthetic_chromatic_aberration_scale,
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
    """
    Prepare one simulation case once before decode-only magnitude sweeps.

    Parameters
    ----------
    simulation_dataset_dirs : dict[str, Path]
        Function argument.
    simulation_api : dict[str, Any]
        Function argument.
    case_spec : dict[str, Any]
        Function argument.
    tmp_path : Path
        Function argument.
    run_calibration_decode : bool
        Function argument.

    Returns
    -------
    dict[str, Any]
        Function result.
    """

    dataset_variant = case_spec["dataset_variant"]
    axial_spacing_um = case_spec["axial_spacing_um"]
    ufish_model_name = case_spec["ufish_model_name"]
    ufish_model = case_spec["ufish_model"]
    decon_readout = case_spec["decon_readout"]
    perform_deformable_registration = case_spec.get(
        "perform_deformable_registration", False
    )
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
    default_magnitude_threshold = _default_standard_magnitude_threshold(
        axial_spacing_um
    )

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
    normalization_iterations: int = 3,
    estimate_chromatic_affines: bool = False,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """
    Run simulation decode and f1.

    Parameters
    ----------
    case_root : Path
        Function argument.
    simulation_api : dict[str, Any]
        Function argument.
    search_radius : float
        Function argument.
    feature_predictor_threshold : float
        Function argument.
    lowpass_sigma : tuple[float, float, float]
        Function argument.
    magnitude_threshold : tuple[float, float]
        Function argument.
    case_label : str
        Function argument.
    minimum_pixels_per_rna : int | None
        Function argument.
    skip_optimization : bool
        Function argument.
    duplicate_radius_xy : float | None
        Function argument.
    duplicate_radius_z : float | None
        Function argument.
    normalization_iterations : int
        Function argument.
    estimate_chromatic_affines : bool
        If True, enable chromatic affine estimation during iterative decoding.

    Returns
    -------
    tuple[dict[str, float], dict[str, float], dict[str, float]]
        Function result.
    """
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
    ufish_model: str | None,
    feature_predictor_threshold: float,
    perform_deformable_registration: bool = False,
    lowpass_sigma: tuple[float, float, float] = DEFAULT_LOWPASS_SIGMA,
    magnitude_threshold: tuple[float, float] = (0.9, 10.0),
    minimum_pixels_per_rna: int | None = None,
    case_label: str | None = None,
    synthetic_chromatic_aberration: bool = False,
    synthetic_chromatic_aberration_scale: float = 1.0,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """
    Run simulation pipeline.

    Parameters
    ----------
    case_root : Path
        Function argument.
    simulation_api : dict[str, Any]
        Function argument.
    search_radius : float
        Function argument.
    decon_readout : bool
        Function argument.
    ufish_model : str | None
        Function argument.
    feature_predictor_threshold : float
        Function argument.
    perform_deformable_registration : bool
        Function argument.
    lowpass_sigma : tuple[float, float, float]
        Function argument.
    magnitude_threshold : tuple[float, float]
        Function argument.
    minimum_pixels_per_rna : int | None
        Function argument.
    case_label : str | None
        Function argument.
    synthetic_chromatic_aberration : bool
        Function argument.
    synthetic_chromatic_aberration_scale : float
        Function argument.

    Returns
    -------
    tuple[dict[str, float], dict[str, float], dict[str, float]]
        Function result.
    """
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
        synthetic_chromatic_aberration_scale=synthetic_chromatic_aberration_scale,
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
            f"{case['ufish_model_name']}-"
            f"{case['preprocess_mode']}-"
            f"{case['registration_mode']}-"
            f"fp{case['feature_predictor_threshold']}"
        )
        for case in STANDARD_SIMULATION_MATRIX
    ],
)
def simulation_standard_case_spec(request: pytest.FixtureRequest) -> dict[str, Any]:
    """
    Parametrize the standard local simulation test matrix.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Function argument.

    Returns
    -------
    dict[str, Any]
        Function result.
    """

    return request.param


@pytest.fixture
def simulation_standard_case_result(
    simulation_dataset_dirs: dict[str, Path],
    simulation_api: dict[str, Any],
    performance_records: list[dict[str, Any]],
    simulation_standard_case_spec: dict[str, Any],
    tmp_path: Path,
) -> dict[str, Any]:
    """
    Run one standard simulation pipeline case and return recorded metrics.

    Parameters
    ----------
    simulation_dataset_dirs : dict[str, Path]
        Function argument.
    simulation_api : dict[str, Any]
        Function argument.
    performance_records : list[dict[str, Any]]
        Function argument.
    simulation_standard_case_spec : dict[str, Any]
        Function argument.
    tmp_path : Path
        Function argument.

    Returns
    -------
    dict[str, Any]
        Function result.
    """

    dataset_variant = simulation_standard_case_spec["dataset_variant"]
    axial_spacing_um = simulation_standard_case_spec["axial_spacing_um"]
    ufish_model_name = simulation_standard_case_spec["ufish_model_name"]
    ufish_model = simulation_standard_case_spec["ufish_model"]
    preprocess_mode = simulation_standard_case_spec["preprocess_mode"]
    decon_readout = simulation_standard_case_spec["decon_readout"]
    registration_mode = simulation_standard_case_spec["registration_mode"]
    perform_deformable_registration = simulation_standard_case_spec[
        "perform_deformable_registration"
    ]
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
    default_magnitude_threshold = _default_standard_magnitude_threshold(
        axial_spacing_um
    )

    case_root = _prepare_case_workspace(source_case_dir, tmp_path)
    case_label = (
        f"{dataset_variant}-{axial_spacing_um}-{ufish_model_name}-{preprocess_mode}-"
        f"{registration_mode}-"
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
            f"{case['ufish_model_name']}-"
            f"{case['preprocess_mode']}-"
            f"{case['registration_mode']}"
        )
        for case in FULL_SWEEP_BASE_SIMULATION_MATRIX
    ],
)
def simulation_full_sweep_case_spec(request: pytest.FixtureRequest) -> dict[str, Any]:
    """
    Parametrize the exhaustive local simulation base cases.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Function argument.

    Returns
    -------
    dict[str, Any]
        Function result.
    """

    return request.param


@pytest.fixture
def simulation_full_preprocessed_case(
    simulation_dataset_dirs: dict[str, Path],
    simulation_api: dict[str, Any],
    simulation_full_sweep_case_spec: dict[str, Any],
    tmp_path: Path,
) -> dict[str, Any]:
    """
    Prepare one exhaustive simulation base case before FP sweeps.

    Parameters
    ----------
    simulation_dataset_dirs : dict[str, Path]
        Function argument.
    simulation_api : dict[str, Any]
        Function argument.
    simulation_full_sweep_case_spec : dict[str, Any]
        Function argument.
    tmp_path : Path
        Function argument.

    Returns
    -------
    dict[str, Any]
        Function result.
    """

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
    """
    Run the exhaustive FP-threshold sweep on one calibrated base case.

    Parameters
    ----------
    simulation_api : dict[str, Any]
        Function argument.
    performance_records : list[dict[str, Any]]
        Function argument.
    simulation_full_preprocessed_case : dict[str, Any]
        Function argument.

    Returns
    -------
    dict[str, Any]
        Function result.
    """

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
    default_minimum_pixels = _default_minimum_pixels_per_rna(axial_spacing_um)
    default_magnitude_threshold = _default_standard_magnitude_threshold(
        axial_spacing_um
    )

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
            "ufish_model": ufish_model_name,
            "preprocess_mode": preprocess_mode,
            "decon_readout": decon_readout,
            "registration_mode": registration_mode,
            "perform_deformable_registration": perform_deformable_registration,
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
    """
    Run the standard simulation matrix with package-default baselines.

    Parameters
    ----------
    simulation_standard_case_result : dict[str, Any]
        Function argument.

    Returns
    -------
    None
        Function result.
    """

    datastore_path = (
        simulation_standard_case_result["case_root"]
        / "sim_acquisition"
        / "qi2labdatastore"
    )
    case_spec = simulation_standard_case_result["case_spec"]
    ufish_model_name = case_spec["ufish_model_name"]
    expected_f1 = (
        STANDARD_EXPECTED_F1_SCORES[
            (case_spec["dataset_variant"], case_spec["axial_spacing_um"])
        ]
        if ufish_model_name == DEFAULT_UFISH_MODEL
        else None
    )

    assert datastore_path.exists()
    assert isinstance(simulation_standard_case_result["f1_results"], dict)
    assert RESULT_KEYS.issubset(simulation_standard_case_result["f1_results"])
    if expected_f1 is not None:
        assert simulation_standard_case_result["performance_metrics"][
            "f1_score"
        ] == pytest.approx(expected_f1, abs=F1_ABS_TOLERANCE)
    assert simulation_standard_case_result["performance_metrics"]["true_positives"] >= 0
    assert (
        simulation_standard_case_result["performance_metrics"]["false_positives"] >= 0
    )
    assert (
        simulation_standard_case_result["performance_metrics"]["false_negatives"] >= 0
    )


def _apply_affine_to_points(affine_zyx_um: np.ndarray, points_zyx_um: np.ndarray):
    """
    Apply one homogeneous ZYX affine to physical points.

    Parameters
    ----------
    affine_zyx_um : numpy.ndarray
        Homogeneous 4x4 affine in Z, Y, X physical coordinates.
    points_zyx_um : numpy.ndarray
        Points in Z, Y, X physical coordinates.

    Returns
    -------
    numpy.ndarray
        Transformed points in Z, Y, X physical coordinates.
    """

    homogeneous = np.concatenate(
        [points_zyx_um, np.ones((points_zyx_um.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    return (homogeneous @ affine_zyx_um.T)[:, :3]


def _format_matrix_fixed(matrix: np.ndarray) -> str:
    """
    Format a matrix without scientific notation.

    Parameters
    ----------
    matrix : numpy.ndarray
        Matrix to format.

    Returns
    -------
    str
        Fixed-decimal matrix representation.
    """

    return np.array2string(
        np.asarray(matrix, dtype=np.float64),
        precision=8,
        suppress_small=True,
        floatmode="fixed",
    )


@pytest.mark.simulation_exhaustive
@pytest.mark.parametrize(
    ("chromatic_shift_label", "chromatic_shift_scale"),
    (
        ("smaller", 0.5),
        ("larger", 1.0),
        ("larger_x2", 2.0),
        ("larger_x4", 4.0),
        ("larger_x8", 8.0),
        ("larger_x16", 16.0),
    ),
)
@pytest.mark.parametrize("axial_spacing_um", AXIAL_SPACING_UM)
def test_simulation_chromatic_affine_recovery_cells_decon(
    simulation_dataset_dirs: dict[str, Path],
    simulation_api: dict[str, Any],
    tmp_path: Path,
    axial_spacing_um: str,
    chromatic_shift_label: str,
    chromatic_shift_scale: float,
) -> None:
    """
    Verify iterative decoding recovers known synthetic chromatic affines.

    Parameters
    ----------
    simulation_dataset_dirs : dict[str, Path]
        Function argument.
    simulation_api : dict[str, Any]
        Function argument.
    tmp_path : Path
        Function argument.
    axial_spacing_um : str
        Function argument.
    chromatic_shift_label : str
        Function argument.
    chromatic_shift_scale : float
        Function argument.

    Returns
    -------
    None
        Function result.
    """

    try:
        from merfish3danalysis.cli.statphysbio_simulation.convert_to_datastore import (
            synthetic_chromatic_affines_zyx_um,
        )
    except Exception as exc:
        pytest.skip(f"Synthetic chromatic helper unavailable: {exc!r}")

    default_model = next(
        (
            (ufish_model_name, ufish_model)
            for ufish_model_name, ufish_model in UFISH_MODELS
            if ufish_model_name == DEFAULT_UFISH_MODEL
        ),
        None,
    )
    if default_model is None:
        pytest.skip(f"Default U-FISH model is unavailable: {DEFAULT_UFISH_MODEL}")
    ufish_model_name, ufish_model = default_model

    source_case_dir = simulation_dataset_dirs["cells"] / axial_spacing_um
    if not source_case_dir.exists():
        pytest.skip(f"Dataset case missing: {source_case_dir}")

    search_radius = F1_RADIUS_BY_AXIAL_SPACING_UM[axial_spacing_um]
    default_minimum_pixels = _default_minimum_pixels_per_rna(axial_spacing_um)
    default_magnitude_threshold = _default_standard_magnitude_threshold(
        axial_spacing_um
    )
    case_root = _prepare_case_workspace(source_case_dir, tmp_path)
    case_label = (
        f"cells-{axial_spacing_um}-{ufish_model_name}-decon-affine-"
        f"synthetic-chromatic-{chromatic_shift_label}"
    )
    setup_timings = _run_simulation_case_setup(
        case_root,
        simulation_api,
        decon_readout=True,
        ufish_model=ufish_model,
        perform_deformable_registration=False,
        case_label=case_label,
        synthetic_chromatic_aberration=True,
        synthetic_chromatic_aberration_scale=chromatic_shift_scale,
    )
    f1_results, _decode_timings, performance_metrics = _run_simulation_decode_and_f1(
        case_root,
        simulation_api,
        search_radius,
        feature_predictor_threshold=DEFAULT_FEATURE_PREDICTOR_THRESHOLD,
        lowpass_sigma=DEFAULT_LOWPASS_SIGMA,
        magnitude_threshold=default_magnitude_threshold,
        minimum_pixels_per_rna=default_minimum_pixels,
        case_label=case_label,
        normalization_iterations=10,
        estimate_chromatic_affines=True,
    )
    print(
        (
            f"[{case_label}] setup_seconds={sum(setup_timings.values()):.2f}, "
            f"F1={performance_metrics['f1_score']:.4f}, results={f1_results}"
        ),
        flush=True,
    )

    datastore = simulation_api["qi2labDataStore"](
        case_root / "sim_acquisition" / "qi2labdatastore",
        validate=False,
    )
    calibration = datastore.load_chromatic_affine_transforms_zyx_um()
    corrected_image = np.asarray(
        datastore.load_local_corrected_image(tile=0, bit=0, return_future=False)
    )
    spacing = np.asarray(datastore.voxel_size_zyx_um, dtype=np.float32)
    readout_relative_affines = synthetic_chromatic_affines_zyx_um(
        tuple(int(v) for v in corrected_image.shape),
        datastore.voxel_size_zyx_um,
        (0.580, 0.670),
        shift_scale=chromatic_shift_scale,
    )
    assert np.allclose(readout_relative_affines[0.580], np.eye(4, dtype=np.float32))
    red_radial_scale = readout_relative_affines[0.670][1, 1]
    assert red_radial_scale < 1.0

    expected_affines = {
        0.580: np.eye(4, dtype=np.float32),
        0.670: readout_relative_affines[0.670],
    }
    recovered_red_to_green = datastore.load_chromatic_affine_transform_zyx_um(
        wavelength_um=0.670
    )
    print(
        (
            f"[{case_label}] actual_green_readout_reference_zyx_um=\n"
            f"{_format_matrix_fixed(readout_relative_affines[0.580])}\n"
            f"[{case_label}] actual_red_to_green_zyx_um=\n"
            f"{_format_matrix_fixed(readout_relative_affines[0.670])}\n"
            f"[{case_label}] expected_red_to_green_zyx_um=\n"
            f"{_format_matrix_fixed(expected_affines[0.670])}\n"
            f"[{case_label}] recovered_red_to_green_zyx_um=\n"
            f"{_format_matrix_fixed(recovered_red_to_green)}"
        ),
        flush=True,
    )

    shape = np.asarray(corrected_image.shape, dtype=np.float32)
    source_points_px = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, shape[2] - 1.0],
            [0.0, shape[1] - 1.0, 0.0],
            [0.0, shape[1] - 1.0, shape[2] - 1.0],
            [shape[0] - 1.0, 0.0, 0.0],
            [shape[0] - 1.0, 0.0, shape[2] - 1.0],
            [shape[0] - 1.0, shape[1] - 1.0, 0.0],
            [shape[0] - 1.0, shape[1] - 1.0, shape[2] - 1.0],
            (shape - 1.0) / 2.0,
        ],
        dtype=np.float32,
    )
    source_points_um = source_points_px * spacing

    expected_f1 = STANDARD_EXPECTED_F1_SCORES[("cells", axial_spacing_um)]
    assert performance_metrics["f1_score"] >= expected_f1 - 0.03
    reference_affine = datastore.load_chromatic_affine_transform_zyx_um(
        wavelength_um=0.580
    )
    np.testing.assert_allclose(reference_affine, np.eye(4), atol=1e-6)

    radial_shifts_px = {}
    for wavelength in (0.670,):
        recovered = recovered_red_to_green
        assert not np.allclose(recovered, np.eye(4, dtype=np.float32))
        assert recovered[1, 1] < 1.0
        assert recovered[2, 2] < 1.0
        assert recovered[0, 3] < 0.0

        expected_points_um = _apply_affine_to_points(
            expected_affines[wavelength],
            source_points_um,
        )
        recovered_points_um = _apply_affine_to_points(recovered, source_points_um)
        error_px = (recovered_points_um - expected_points_um) / spacing
        xy_error_px = np.linalg.norm(error_px[:, 1:3], axis=1)

        assert float(np.max(xy_error_px)) <= 0.75
        assert float(np.max(np.abs(error_px[:, 0]))) <= 0.5

        reference_points_um = source_points_um
        expected_shift_px = (expected_points_um - reference_points_um) / spacing
        recovered_shift_px = (recovered_points_um - reference_points_um) / spacing
        expected_radial_shift = np.linalg.norm(expected_shift_px[:, 1:3], axis=1)
        recovered_radial_shift = np.linalg.norm(recovered_shift_px[:, 1:3], axis=1)
        assert float(np.max(recovered_radial_shift)) > 0.0
        assert np.sign(float(np.mean(recovered[0, 3]))) == np.sign(
            float(expected_affines[wavelength][0, 3])
        )
        radial_shifts_px[wavelength] = float(np.max(recovered_radial_shift))
        assert radial_shifts_px[wavelength] == pytest.approx(
            float(np.max(expected_radial_shift)),
            abs=0.75,
        )

    assert radial_shifts_px[0.670] > 0.0
    assert calibration["channels"]["wavelength_0.580000"]["status"] == (
        "identity_reference"
    )
    assert calibration["channels"]["wavelength_0.670000"]["status"] == (
        "affine_estimated"
    )


@pytest.mark.simulation_exhaustive
def test_simulation_exhaustive_matrix(
    simulation_full_case_result: dict[str, Any],
) -> None:
    """
    Run the exhaustive simulation matrix with package-default baselines.

    Parameters
    ----------
    simulation_full_case_result : dict[str, Any]
        Function argument.

    Returns
    -------
    None
        Function result.
    """

    datastore_path = (
        simulation_full_case_result["case_root"] / "sim_acquisition" / "qi2labdatastore"
    )
    case_spec = simulation_full_case_result["case_spec"]
    ufish_model_name = case_spec["ufish_model_name"]

    assert datastore_path.exists()
    assert len(simulation_full_case_result["results"]) == len(
        FEATURE_PREDICTOR_THRESHOLDS
    )
    collect_only = COLLECT_FULL_BASELINES or ufish_model_name != DEFAULT_UFISH_MODEL
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
