from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from pyinstrument import Profiler

SIMULATION_DATA_ROOT_ENV = "MERFISH3D_SIMULATION_ROOT"
SIMULATION_CACHE_ENV = "MERFISH3D_SIMULATION_CACHE"
SIMULATION_DATA_ZENODO_URL = "https://zenodo.org/records/17274305/files/merfish3d_analysis-simulation.zip?download=1"
SIMULATION_DATA_ARCHIVE = "merfish3d_analysis-simulation.zip"
SIMULATION_DATA_EXTRACTED_DIR = "merfish3d_analysis-simulation"
AXIAL_SPACING_UM = ("0.315", "1.0", "1.5")
BASELINE_SPACING_UM = "0.315"
REQUIRED_SIMULATION_FILES = (
    "aligned_1.tiff",
    "bit_order.csv",
    "codebook.csv",
    "GT_spots.csv",
    "scan_metadata.csv",
)


@dataclass
class FunctionTiming:
    file_path: str
    function: str
    line_no: int
    calls: int
    self_time: float
    inclusive_time: float


def _download_simulation_dataset(cache_root: Path) -> Path:
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
    if (root / "merfish3d_analysis-simulation").exists():
        root = root / "merfish3d_analysis-simulation"

    example_dir = root / "example_16bit_flat"
    if not example_dir.exists():
        raise FileNotFoundError(
            f"Could not find 'example_16bit_flat' under dataset root: {root}"
        )

    return root


def _resolve_dataset_root(default_cache_root: Path) -> Path:
    configured_root = os.getenv(SIMULATION_DATA_ROOT_ENV)
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(
                f"{SIMULATION_DATA_ROOT_ENV} points to missing path: {root}"
            )
        return _normalize_dataset_root(root)

    cache_root = Path(
        os.getenv(SIMULATION_CACHE_ENV, str(default_cache_root))
    ).expanduser()
    root = _download_simulation_dataset(cache_root)
    return _normalize_dataset_root(root)


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _prepare_case_workspace(source_case_dir: Path, work_root: Path) -> Path:
    missing_files = [
        filename
        for filename in REQUIRED_SIMULATION_FILES
        if not (source_case_dir / filename).exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required simulation files in {source_case_dir}: {missing_files}"
        )

    case_root = work_root / source_case_dir.name
    if case_root.exists():
        shutil.rmtree(case_root)
    case_root.mkdir(parents=True, exist_ok=False)

    for filename in REQUIRED_SIMULATION_FILES:
        _link_or_copy(source_case_dir / filename, case_root / filename)

    return case_root


def _load_simulation_api() -> dict[str, Any]:
    from merfish3danalysis.cli.statphysbio_simulation.convert_simulation_to_experiment import (
        convert_simulation,
    )
    from merfish3danalysis.cli.statphysbio_simulation.convert_to_datastore import (
        convert_data,
    )
    from merfish3danalysis.cli.statphysbio_simulation.register_and_deconvolve import (
        manage_data_registration_states,
    )

    return {
        "convert_simulation": convert_simulation,
        "convert_data": convert_data,
        "manage_data_registration_states": manage_data_registration_states,
    }


@contextlib.contextmanager
def _inline_dataregistration_processes(enabled: bool):
    """Optionally patch DataRegistration to run worker functions inline.

    This allows pyinstrument in the parent process to capture detailed timings
    that would otherwise be hidden inside multiprocessing workers.
    """

    if not enabled:
        yield
        return

    import merfish3danalysis.DataRegistration as dr_module

    cls = dr_module.DataRegistration
    original_generate = cls._generate_registrations
    original_apply_bits = cls._apply_registration_to_bits

    def _generate_registrations_inline(self):
        test = self._datastore.load_local_registered_image(
            tile=self._tile_id, round=self._round_ids[0]
        )
        has_reg_decon_data = test is not None

        if not has_reg_decon_data or self._overwrite_registered:
            dr_module._apply_first_polyDT_on_gpu(self, 0)

        if self._num_gpus == 0:
            raise RuntimeError(
                "No GPUs detected. Cannot run _generate_registrations()."
            )

        all_rounds = list(self._round_ids[1:])
        chunk_size = (len(all_rounds) + self._num_gpus - 1) // self._num_gpus
        for gpu_id in range(self._num_gpus):
            start = gpu_id * chunk_size
            end = min(start + chunk_size, len(all_rounds))
            if start >= end:
                break

            subset = all_rounds[start:end]
            old_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            try:
                dr_module._apply_polyDT_on_gpu(self, subset, 0)
            finally:
                if old_vis is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old_vis

    def _apply_registration_to_bits_inline(self):
        if self._num_gpus == 0:
            raise RuntimeError(
                "No GPUs detected. Cannot run _apply_registration_to_bits()."
            )

        all_bits = list(self._bit_ids)
        chunk_size = (len(all_bits) + self._num_gpus - 1) // self._num_gpus
        for gpu_id in range(self._num_gpus):
            start = gpu_id * chunk_size
            end = min(start + chunk_size, len(all_bits))
            if start >= end:
                break

            subset = all_bits[start:end]
            old_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            try:
                dr_module._apply_bits_on_gpu(self, subset, 0)
            finally:
                if old_vis is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old_vis

    cls._generate_registrations = _generate_registrations_inline
    cls._apply_registration_to_bits = _apply_registration_to_bits_inline
    try:
        yield
    finally:
        cls._generate_registrations = original_generate
        cls._apply_registration_to_bits = original_apply_bits


def _frame_key(file_path: str, function: str, line_no: int) -> str:
    return f"{file_path}:{line_no}:{function}"


def _collect_function_timings(
    frame: Any,
    *,
    repo_root: Path,
    stats_by_key: dict[str, FunctionTiming],
) -> None:
    file_path = frame.file_path
    if file_path:
        resolved_file = Path(file_path).resolve()
        if resolved_file.is_relative_to(repo_root):
            key = _frame_key(
                str(resolved_file),
                frame.function or "<unknown>",
                int(frame.line_no or 0),
            )
            if key not in stats_by_key:
                stats_by_key[key] = FunctionTiming(
                    file_path=str(resolved_file),
                    function=frame.function or "<unknown>",
                    line_no=int(frame.line_no or 0),
                    calls=0,
                    self_time=0.0,
                    inclusive_time=0.0,
                )
            stats = stats_by_key[key]
            stats.calls += 1
            stats.self_time += float(frame.absorbed_time)
            stats.inclusive_time += float(frame.time)

    for child in frame.children:
        _collect_function_timings(
            child,
            repo_root=repo_root,
            stats_by_key=stats_by_key,
        )


def _timings_to_serializable(
    stats_by_key: dict[str, FunctionTiming],
) -> dict[str, dict]:
    return {key: asdict(value) for key, value in stats_by_key.items()}


def _compare_to_baseline(
    baseline_stats: dict[str, FunctionTiming],
    target_stats: dict[str, FunctionTiming],
) -> list[dict[str, Any]]:
    all_keys = set(baseline_stats) | set(target_stats)
    regressions: list[dict[str, Any]] = []

    for key in all_keys:
        baseline = baseline_stats.get(key)
        target = target_stats.get(key)

        baseline_self = baseline.self_time if baseline else 0.0
        target_self = target.self_time if target else 0.0
        baseline_inclusive = baseline.inclusive_time if baseline else 0.0
        target_inclusive = target.inclusive_time if target else 0.0
        self_delta = target_self - baseline_self
        inclusive_delta = target_inclusive - baseline_inclusive

        if self_delta <= 0 and inclusive_delta <= 0:
            continue

        reference = target if target is not None else baseline
        assert reference is not None
        regressions.append(
            {
                "file_path": reference.file_path,
                "function": reference.function,
                "line_no": reference.line_no,
                "baseline_calls": baseline.calls if baseline else 0,
                "target_calls": target.calls if target else 0,
                "baseline_self_seconds": baseline_self,
                "target_self_seconds": target_self,
                "self_seconds_delta": self_delta,
                "baseline_inclusive_seconds": baseline_inclusive,
                "target_inclusive_seconds": target_inclusive,
                "inclusive_seconds_delta": inclusive_delta,
            }
        )

    regressions.sort(
        key=lambda row: (row["self_seconds_delta"], row["inclusive_seconds_delta"]),
        reverse=True,
    )
    return regressions


def _write_markdown_report(
    output_path: Path,
    *,
    summary: dict[str, Any],
    comparisons: dict[str, list[dict[str, Any]]],
    top_n: int,
) -> None:
    lines: list[str] = []
    lines.append("# Preprocess Profiling Comparison (pyinstrument)")
    lines.append("")
    lines.append(f"- Generated: {summary['generated_at_utc']}")
    lines.append(f"- Baseline spacing: `{BASELINE_SPACING_UM}`")
    lines.append("")
    lines.append("## Wall Time")
    lines.append("")
    lines.append("| Spacing (um) | Preprocess wall time (s) |")
    lines.append("|---|---:|")
    for spacing in AXIAL_SPACING_UM:
        wall_time = summary["profiles"][spacing]["preprocess_wall_seconds"]
        lines.append(f"| {spacing} | {wall_time:.2f} |")
    lines.append("")

    for spacing in AXIAL_SPACING_UM:
        if spacing == BASELINE_SPACING_UM:
            continue
        lines.append(
            f"## Functions Slower Than `{BASELINE_SPACING_UM}` for `{spacing}`"
        )
        lines.append("")
        rows = comparisons[spacing][:top_n]
        if not rows:
            lines.append("No slower functions detected.")
            lines.append("")
            continue

        lines.append(
            "| Function | File | Baseline self (s) | Target self (s) | Delta self (s) | "
            "Baseline incl (s) | Target incl (s) | Delta incl (s) |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            file_name = Path(row["file_path"]).name
            lines.append(
                f"| `{row['function']}` | `{file_name}:{row['line_no']}` | "
                f"{row['baseline_self_seconds']:.3f} | {row['target_self_seconds']:.3f} | "
                f"{row['self_seconds_delta']:.3f} | "
                f"{row['baseline_inclusive_seconds']:.3f} | {row['target_inclusive_seconds']:.3f} | "
                f"{row['inclusive_seconds_delta']:.3f} |"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(
    output_dir: Path,
    keep_workspaces: bool = False,
    inline_registration_processes: bool = True,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    tests_data_dir = repo_root / "tests" / "data"
    default_cache_root = tests_data_dir / "simulation_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    workspaces_dir = output_dir / "workspaces"
    workspaces_dir.mkdir(parents=True, exist_ok=True)

    print("Resolving dataset root...", flush=True)
    dataset_root = _resolve_dataset_root(default_cache_root)
    print(f"Dataset root: {dataset_root}", flush=True)

    simulation_api = _load_simulation_api()
    profile_summaries: dict[str, Any] = {}
    function_timings: dict[str, dict[str, FunctionTiming]] = {}

    for spacing in AXIAL_SPACING_UM:
        print(f"[{spacing}] preparing workspace", flush=True)
        source_case_dir = dataset_root / "example_16bit_flat" / spacing
        case_root = _prepare_case_workspace(source_case_dir, workspaces_dir)

        print(f"[{spacing}] convert_simulation", flush=True)
        simulation_api["convert_simulation"](case_root)
        acquisition_root = case_root / "sim_acquisition"

        print(f"[{spacing}] convert_data", flush=True)
        simulation_api["convert_data"](acquisition_root)

        print(f"[{spacing}] profile manage_data_registration_states", flush=True)
        profiler = Profiler(interval=0.001)
        start = time.perf_counter()
        profiler.start()
        with _inline_dataregistration_processes(inline_registration_processes):
            simulation_api["manage_data_registration_states"](acquisition_root)
        profiler.stop()
        preprocess_wall_seconds = time.perf_counter() - start

        session = profiler.last_session
        spacing_slug = spacing.replace(".", "p")
        text_output_path = output_dir / f"preprocess_profile_{spacing_slug}.txt"
        html_output_path = output_dir / f"preprocess_profile_{spacing_slug}.html"

        text_output_path.write_text(
            profiler.output_text(unicode=True, color=False, show_all=True),
            encoding="utf-8",
        )
        html_output_path.write_text(profiler.output_html(), encoding="utf-8")

        stats_by_key: dict[str, FunctionTiming] = {}
        _collect_function_timings(
            session.root_frame(),
            repo_root=repo_root,
            stats_by_key=stats_by_key,
        )
        function_timings[spacing] = stats_by_key

        top_self_functions = sorted(
            stats_by_key.values(), key=lambda item: item.self_time, reverse=True
        )[:25]
        profile_summaries[spacing] = {
            "preprocess_wall_seconds": preprocess_wall_seconds,
            "sample_count": int(session.sample_count),
            "duration": float(session.duration),
            "profile_text": str(text_output_path),
            "profile_html": str(html_output_path),
            "top_self_time_functions": [asdict(item) for item in top_self_functions],
            "all_function_timings_by_key": _timings_to_serializable(stats_by_key),
        }

        if not keep_workspaces:
            shutil.rmtree(case_root, ignore_errors=True)

    baseline_timings = function_timings[BASELINE_SPACING_UM]
    comparisons: dict[str, list[dict[str, Any]]] = {}
    for spacing in AXIAL_SPACING_UM:
        if spacing == BASELINE_SPACING_UM:
            continue
        comparisons[spacing] = _compare_to_baseline(
            baseline_timings,
            function_timings[spacing],
        )

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_root": str(dataset_root),
        "baseline_spacing_um": BASELINE_SPACING_UM,
        "profiles": profile_summaries,
        "slower_than_baseline_by_spacing": comparisons,
    }

    summary_path = output_dir / "preprocess_pyinstrument_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    markdown_path = output_dir / "preprocess_pyinstrument_comparison.md"
    _write_markdown_report(
        markdown_path,
        summary=summary,
        comparisons=comparisons,
        top_n=25,
    )

    print("\nPreprocess profiling complete.", flush=True)
    print(f"Summary JSON: {summary_path}", flush=True)
    print(f"Comparison report: {markdown_path}", flush=True)
    for spacing in AXIAL_SPACING_UM:
        wall_time = profile_summaries[spacing]["preprocess_wall_seconds"]
        print(f"{spacing} um preprocess: {wall_time:.2f}s", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile preprocessing by axial spacing using pyinstrument."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "pyinstrument_preprocess",
        help="Directory to store profiling artifacts and comparison outputs.",
    )
    parser.add_argument(
        "--keep-workspaces",
        action="store_true",
        help="Keep generated per-case workspaces under output-dir/workspaces.",
    )
    parser.add_argument(
        "--keep-multiprocessing",
        action="store_true",
        help="Do not patch DataRegistration multiprocessing during profiling.",
    )
    args = parser.parse_args()

    run_benchmark(
        args.output_dir,
        keep_workspaces=args.keep_workspaces,
        inline_registration_processes=not args.keep_multiprocessing,
    )


if __name__ == "__main__":
    main()
