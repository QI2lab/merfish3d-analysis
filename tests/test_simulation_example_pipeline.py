import json
import os
import shutil
import time
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

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
STANDARD_SIMULATION_MATRIX = tuple(
    {
        "dataset_variant": dataset_variant,
        "dataset_dir": dataset_dir,
        "axial_spacing_um": axial_spacing_um,
        "ufish_model_name": ufish_model_name,
        "ufish_model": ufish_model,
        "preprocess_mode": DEFAULT_PREPROCESS_MODE,
        "decon_readout": PREPROCESS_MODES[DEFAULT_PREPROCESS_MODE],
        "feature_predictor_threshold": DEFAULT_FEATURE_PREDICTOR_THRESHOLD,
    }
    for (dataset_variant, dataset_dir), axial_spacing_um, (
        ufish_model_name,
        ufish_model,
    ) in product(
        SIMULATION_DATASET_DIRS.items(),
        AXIAL_SPACING_UM,
        UFISH_MODELS,
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
        "feature_predictor_threshold": feature_predictor_threshold,
    }
    for (dataset_variant, dataset_dir), axial_spacing_um, (
        ufish_model_name,
        ufish_model,
    ), (
        preprocess_mode,
        decon_readout,
    ), feature_predictor_threshold in product(
        SIMULATION_DATASET_DIRS.items(),
        AXIAL_SPACING_UM,
        UFISH_MODELS,
        PREPROCESS_MODES.items(),
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
    }
    for (dataset_variant, dataset_dir), axial_spacing_um, (
        ufish_model_name,
        ufish_model,
    ), (
        preprocess_mode,
        decon_readout,
    ) in product(
        SIMULATION_DATASET_DIRS.items(),
        AXIAL_SPACING_UM,
        UFISH_MODELS,
        PREPROCESS_MODES.items(),
    )
)
STANDARD_EXPECTED_F1_SCORES = {
    ("cells", "0.315"): 0.9933222036727879,
    ("cells", "1.0"): 0.9710391822827938,
    ("cells", "1.5"): 0.42944785276073616,
    ("uniform", "0.315"): 0.9954070981210855,
    ("uniform", "1.0"): 0.9811951525282072,
    ("uniform", "1.5"): 0.6323404255319149,
}
FULL_EXPECTED_F1_SCORES = {
    ("cells", "0.315", "decon", 0.1): 0.9924686192468619,
    ("cells", "0.315", "decon", 0.2): 0.994142259414226,
    ("cells", "0.315", "decon", 0.3): 0.9933110367892977,
    ("cells", "0.315", "decon", 0.4): 0.9949832775919734,
    ("cells", "0.315", "decon", 0.5): 0.9933222036727879,
    ("cells", "0.315", "no-decon", 0.1): 0.9941225860621327,
    ("cells", "0.315", "no-decon", 0.2): 0.9916107382550335,
    ("cells", "0.315", "no-decon", 0.3): 0.9949748743718593,
    ("cells", "0.315", "no-decon", 0.4): 0.9949748743718593,
    ("cells", "0.315", "no-decon", 0.5): 0.9958158995815899,
    ("cells", "1.0", "decon", 0.1): 0.9635157545605307,
    ("cells", "1.0", "decon", 0.2): 0.9716666666666667,
    ("cells", "1.0", "decon", 0.3): 0.9791840133222314,
    ("cells", "1.0", "decon", 0.4): 0.9824561403508771,
    ("cells", "1.0", "decon", 0.5): 0.9710391822827938,
    ("cells", "1.0", "no-decon", 0.1): 0.7801302931596091,
    ("cells", "1.0", "no-decon", 0.2): 0.8346972176759412,
    ("cells", "1.0", "no-decon", 0.3): 0.8857615894039735,
    ("cells", "1.0", "no-decon", 0.4): 0.9202657807308969,
    ("cells", "1.0", "no-decon", 0.5): 0.9507923269391159,
    ("cells", "1.5", "decon", 0.1): 0.5936981757877281,
    ("cells", "1.5", "decon", 0.2): 0.5777777777777778,
    ("cells", "1.5", "decon", 0.3): 0.5376344086021505,
    ("cells", "1.5", "decon", 0.4): 0.49479768786127165,
    ("cells", "1.5", "decon", 0.5): 0.42944785276073616,
    ("cells", "1.5", "no-decon", 0.1): 0.9210740439381611,
    ("cells", "1.5", "no-decon", 0.2): 0.9377049180327869,
    ("cells", "1.5", "no-decon", 0.3): 0.9627174813587407,
    ("cells", "1.5", "no-decon", 0.4): 0.970906068162926,
    ("cells", "1.5", "no-decon", 0.5): 0.9741019214703426,
    ("uniform", "0.315", "decon", 0.1): 0.9937264742785445,
    ("uniform", "0.315", "decon", 0.2): 0.9941471571906355,
    ("uniform", "0.315", "decon", 0.3): 0.9941471571906355,
    ("uniform", "0.315", "decon", 0.4): 0.9954032595068951,
    ("uniform", "0.315", "decon", 0.5): 0.9954070981210855,
    ("uniform", "0.315", "no-decon", 0.1): 0.9953955629970699,
    ("uniform", "0.315", "no-decon", 0.2): 0.9966555183946487,
    ("uniform", "0.315", "no-decon", 0.3): 0.995819397993311,
    ("uniform", "0.315", "no-decon", 0.4): 0.9962390305056414,
    ("uniform", "0.315", "no-decon", 0.5): 0.9974937343358397,
    ("uniform", "1.0", "decon", 0.1): 0.9575353871773523,
    ("uniform", "1.0", "decon", 0.2): 0.9737390579408086,
    ("uniform", "1.0", "decon", 0.3): 0.9770546516478932,
    ("uniform", "1.0", "decon", 0.4): 0.9795577805590322,
    ("uniform", "1.0", "decon", 0.5): 0.9811951525282072,
    ("uniform", "1.0", "no-decon", 0.1): 0.8148148148148148,
    ("uniform", "1.0", "no-decon", 0.2): 0.8653128885205139,
    ("uniform", "1.0", "no-decon", 0.3): 0.9052544476623914,
    ("uniform", "1.0", "no-decon", 0.4): 0.9451371571072318,
    ("uniform", "1.0", "no-decon", 0.5): 0.9575707154742097,
    ("uniform", "1.5", "decon", 0.1): 0.6759455370650529,
    ("uniform", "1.5", "decon", 0.2): 0.6945954589062999,
    ("uniform", "1.5", "decon", 0.3): 0.7006993006993008,
    ("uniform", "1.5", "decon", 0.4): 0.6717674062739096,
    ("uniform", "1.5", "decon", 0.5): 0.6323404255319149,
    ("uniform", "1.5", "no-decon", 0.1): 0.8992688870836718,
    ("uniform", "1.5", "no-decon", 0.2): 0.9317995069843878,
    ("uniform", "1.5", "no-decon", 0.3): 0.9591752577319588,
    ("uniform", "1.5", "no-decon", 0.4): 0.9763583575279967,
    ("uniform", "1.5", "no-decon", 0.5): 0.9791492910758965,
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
    nearest_multiple = min(
        SIMULATION_2D_MAGNITUDE_THRESHOLD_BY_NYQUIST,
        key=lambda value: abs(value - nyquist_multiple),
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
    ufish_model: str | None,
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

    Returns
    -------
    None
        Function result.
    """
    datastore = simulation_api["qi2labDataStore"](acquisition_root / "qi2labdatastore")
    registration_factory = simulation_api["DataRegistration"](
        datastore=datastore,
        perform_optical_flow=False,
        overwrite_registered=True,
        save_all_fiducial_registered=False,
        decon_readout=decon_readout,
        ufish_model=ufish_model,
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
    ufish_model: str | None,
    case_label: str,
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
    case_label : str
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
        ufish_model=ufish_model,
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
        f"fp{DEFAULT_FEATURE_PREDICTOR_THRESHOLD}"
    )
    setup_timings = _run_simulation_case_setup(
        case_root,
        simulation_api,
        decon_readout=decon_readout,
        ufish_model=ufish_model,
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
    lowpass_sigma: tuple[float, float, float] = DEFAULT_LOWPASS_SIGMA,
    magnitude_threshold: tuple[float, float] = (0.9, 10.0),
    minimum_pixels_per_rna: int | None = None,
    case_label: str | None = None,
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
    lowpass_sigma : tuple[float, float, float]
        Function argument.
    magnitude_threshold : tuple[float, float]
        Function argument.
    minimum_pixels_per_rna : int | None
        Function argument.
    case_label : str | None
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
            f"{case['ufish_model_name']}-"
            f"{case['preprocess_mode']}-"
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
        f"fp{feature_predictor_threshold}"
    )
    f1_results, timings_seconds, performance_metrics = _run_simulation_pipeline(
        case_root,
        simulation_api,
        search_radius,
        decon_readout=decon_readout,
        ufish_model=ufish_model,
        feature_predictor_threshold=feature_predictor_threshold,
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
            f"{case['preprocess_mode']}"
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
