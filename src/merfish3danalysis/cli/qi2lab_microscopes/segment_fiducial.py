"""Run Cellpose and save segmentation outputs in datastore coordinates.

Shepherd 2025/10 - change to CLI.
Shepherd 2025/07 - refactor for CellposeSAM
Shepherd 2024/12 - refactor
Shepherd 2024/11 - created script to run cellpose given determined parameters.
"""

import os
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Annotated

import numpy as np
import typer
from cellpose import io, models, transforms
from roifile import ImagejRoi, roiread, roiwrite

from merfish3danalysis.cli.qi2lab_microscopes._common import qi2lab_datastore_path
from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.cellpose_rois import extract_pixel_rois, global_rois
from merfish3danalysis.utils.spacing import round_spacing_um

app = typer.Typer()
app.pretty_exceptions_enable = False


@app.command()
def run_cellpose(
    root_path: Path,
    normalization: tuple[float, float] = (1.0, 99.0),
    diameter: float | None = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_size: int = 15,
    pretrained_model: str = "cpsam_v2",
    roi_multiprocessing: bool = True,
    save_outputs: bool = True,
    use_gpu: bool = True,
    min_cell_area_um2: Annotated[
        float,
        typer.Option(
            help="Minimum exported outline area in square microns; 0 disables filtering."
        ),
    ] = 0.0,
    outlines_only: Annotated[
        bool,
        typer.Option(
            help="Rebuild global outlines from saved pixel ROIs without running Cellpose."
        ),
    ] = False,
    roi_workers: Annotated[
        int,
        typer.Option(
            help="CPU workers for ROI extraction/export; 0 uses up to 8 available CPUs."
        ),
    ] = 0,
) -> None:
    """Run Cellpose and save masks plus ImageJ ROIs.

    Parameters
    ----------
    root_path : Path
        Experiment root directory.
    normalization : tuple[float, float], default=(1.0, 99.0)
        Percentile normalization range.
    diameter : float | None, default=None
        Cell size in pixels. The Cellpose GUI default is blank, which passes
        None and does not force a cell size.
    flow_threshold : float, default=0.4
        Flow threshold.
    cellprob_threshold : float, default=0.0
        Cell-probability threshold.
    min_size : int, default=15
        Minimum mask size. This matches the Cellpose GUI default.
    pretrained_model : str, default="cpsam_v2"
        Cellpose model name or path. Built-in GUI names include cpsam_v2,
        cpdino, cpdino-vitb, and cpsam.
    roi_multiprocessing : bool, default=True
        Enable parallel ROI extraction and export. Workers use threads to
        share the mask image without copying it between processes.
    save_outputs : bool, default=True
        Save mask image and ImageJ ROIs after Cellpose finishes. Disable this
        for fast parameter comparison against the Cellpose GUI.
    use_gpu : bool, default=True
        Run Cellpose on CUDA. If True and CUDA is unavailable to PyTorch, raise
        an error instead of silently falling back to slow CPU inference.
    min_cell_area_um2 : float, default=0.0
        Minimum global XY outline area in square microns. Outlines smaller
        than this are excluded from the global ROI ZIP used by the viewer and
        decoder. Zero disables filtering. Raw masks and pixel ROIs are retained.
    outlines_only : bool, default=False
        Regenerate global ROIs from saved pixel ROIs without running Cellpose.
        Requires save_outputs=True.
    roi_workers : int, default=0
        Number of CPU workers for ROI extraction and export. Zero selects up
        to eight available CPUs; roi_multiprocessing=False forces one worker.
    """
    if not np.isfinite(min_cell_area_um2) or min_cell_area_um2 < 0:
        raise typer.BadParameter(
            "must be finite and non-negative", param_hint="--min-cell-area-um2"
        )
    if outlines_only and not save_outputs:
        raise typer.BadParameter("--outlines-only requires --save-outputs")
    if roi_workers < 0:
        raise typer.BadParameter("must be non-negative", param_hint="--roi-workers")
    available_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    )
    workers = (roi_workers or min(8, available_cpus)) if roi_multiprocessing else 1

    # initialize datastore
    datastore_path = qi2lab_datastore_path(root_path)
    datastore = qi2labDataStore(datastore_path)
    print(f"Using datastore at {datastore_path}")

    fused_image_path = datastore._image_store_path(
        datastore._fused_root_path / f"fused_{datastore.fiducial_folder_name}_zyx"
    )
    if not fused_image_path.exists():
        raise FileNotFoundError(
            f"Globally registered fused image not found: {fused_image_path}"
        )

    attributes = datastore._read_extra_attributes(fused_image_path)
    affine_zyx_um = np.asarray(attributes["affine_zyx_um"], dtype=np.float32)
    origin_zyx_um = np.asarray(attributes["origin_zyx_um"], dtype=np.float32)
    spacing_zyx_um = round_spacing_um(attributes["spacing_zyx_um"]).astype(np.float32)

    imagej_roi_path_dir = datastore_path / "segmentation" / "cellpose" / "imagej_rois"
    cellpose_roi_path = imagej_roi_path_dir / "pixel_spacing_rois.zip"
    global_roi_path = imagej_roi_path_dir / "global_coords_rois.zip"
    if outlines_only:
        if not cellpose_roi_path.exists():
            raise FileNotFoundError(
                f"Saved pixel-space ROIs not found: {cellpose_roi_path}. "
                "Run qi2lab-segment without --outlines-only first."
            )
        _save_global_rois(
            roiread(cellpose_roi_path),
            global_roi_path,
            spacing_zyx_um,
            origin_zyx_um,
            affine_zyx_um,
            min_cell_area_um2,
            workers=workers,
        )
        datastore_state = datastore.datastore_state
        datastore_state.update({"SegmentedCells": True})
        datastore.datastore_state = datastore_state
        return

    max_projection_path = (
        datastore_path
        / "segmentation"
        / "cellpose"
        / "fiducial_max_projection.ome.tiff"
    )
    if max_projection_path.exists():
        print(
            f"Loading fused fiducial max projection from {max_projection_path}",
            flush=True,
        )
        fiducial_max_projection = io.imread_2D(str(max_projection_path))
    else:
        print(
            "Max projection TIFF not found; loading full fused Zarr to compute one. "
            "This can be slow for large datasets.",
            flush=True,
        )
        loaded = datastore.load_global_fiducial_image(return_future=False)
        if loaded is None:
            raise RuntimeError(
                "Could not load globally registered fused fiducial image."
            )
        fiducial_fused, affine_zyx_um, origin_zyx_um, spacing_zyx_um = loaded
        fiducial_max_projection = np.max(np.squeeze(fiducial_fused), axis=0)
        del fiducial_fused
    fiducial_max_projection = _prepare_cellpose_input_image(fiducial_max_projection)
    print(
        "Prepared Cellpose input image "
        f"shape={tuple(int(v) for v in fiducial_max_projection.shape)} "
        f"dtype={fiducial_max_projection.dtype} "
        f"min={float(np.min(fiducial_max_projection)):.3f} "
        f"max={float(np.max(fiducial_max_projection)):.3f}.",
        flush=True,
    )

    # initialize cellpose model and options
    import torch

    torch.sparse.check_sparse_tensor_invariants.disable()

    cuda_available = torch.cuda.is_available()
    if use_gpu and not cuda_available:
        raise RuntimeError(
            "Cellpose GPU mode was requested, but torch.cuda.is_available() is False. "
            'Run `uv run python -c "import torch; print(torch.cuda.is_available())"` '
            "to verify the environment, or rerun with `--no-use-gpu` for slow CPU mode."
        )
    use_bfloat16 = False
    if use_gpu and cuda_available:
        major, _minor = torch.cuda.get_device_capability(0)
        use_bfloat16 = major >= 8
        print(
            "Using Cellpose GPU "
            f"{torch.cuda.get_device_name(0)!r}; use_bfloat16={use_bfloat16}.",
            flush=True,
        )
    else:
        print("Using Cellpose CPU mode.", flush=True)

    normalize = {
        **models.normalize_default,
        "percentile": list(normalization),
        "norm3D": True,
        "sharpen_radius": 0.0,
        "smooth_radius": 0.0,
        "tile_norm_blocksize": 0.0,
        "tile_norm_smooth3D": 0.0,
        "invert": False,
    }
    print(f"Loading Cellpose model {pretrained_model!r}.", flush=True)
    model = models.CellposeModel(
        gpu=use_gpu,
        pretrained_model=pretrained_model,
        use_bfloat16=use_bfloat16,
    )
    print(f"Loaded Cellpose model from {model.pretrained_model!r}.", flush=True)

    # run cellpose on fiducial max projection
    print(
        "Running Cellpose "
        f"image_shape={tuple(int(v) for v in fiducial_max_projection.shape)} "
        f"diameter={diameter} flow_threshold={flow_threshold} "
        f"cellprob_threshold={cellprob_threshold} min_size={min_size} "
        f"normalize={normalize!r}.",
        flush=True,
    )
    masks, _, _ = model.eval(
        fiducial_max_projection,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize=normalize,
        do_3D=False,
        stitch_threshold=0.0,
        anisotropy=1.0,
        flow3D_smooth=0.0,
        min_size=min_size,
        z_axis=None,
    )
    mask_labels = np.unique(masks)
    mask_count = len(mask_labels) - int(mask_labels[0] == 0) if mask_labels.size else 0
    max_label_id = int(mask_labels[-1]) if mask_labels.size else 0
    print(
        f"Cellpose finished; masks={mask_count} max_label_id={max_label_id}.",
        flush=True,
    )
    if not save_outputs:
        print(
            "Skipping Cellpose mask/ROI outputs because save_outputs=False.", flush=True
        )
        return

    # save masks
    step_start = perf_counter()
    print(
        "Saving Cellpose mask image "
        f"shape={tuple(int(v) for v in masks.shape)} dtype={masks.dtype}.",
        flush=True,
    )
    datastore.save_global_cellpose_segmentation_image(masks, downsampling=[1, 3.5, 3.5])
    print(
        f"Saved Cellpose mask image in {perf_counter() - step_start:.1f} s.", flush=True
    )

    # save pixel spaced ROIs
    step_start = perf_counter()
    imagej_roi_path_dir.mkdir(exist_ok=True)
    print(
        f"Extracting pixel-space ImageJ ROIs with {workers} workers.",
        flush=True,
    )
    pixel_spacing_rois = extract_pixel_rois(
        masks, mask_labels[mask_labels != 0], workers=workers
    )
    print(
        f"Extracted {len(pixel_spacing_rois)} pixel-space ROIs in "
        f"{perf_counter() - step_start:.1f} s.",
        flush=True,
    )
    step_start = perf_counter()
    print(f"Saving pixel-space ImageJ ROIs to {cellpose_roi_path}.", flush=True)
    roiwrite(cellpose_roi_path, pixel_spacing_rois, mode="w")
    print(
        f"Saved pixel-space ImageJ ROIs in {perf_counter() - step_start:.1f} s.",
        flush=True,
    )

    _save_global_rois(
        pixel_spacing_rois,
        global_roi_path,
        spacing_zyx_um,
        origin_zyx_um,
        affine_zyx_um,
        min_cell_area_um2,
        workers=workers,
    )

    # update datastore state
    datastore_state = datastore.datastore_state
    datastore_state.update({"SegmentedCells": True})
    datastore.datastore_state = datastore_state


def _save_global_rois(
    pixel_spacing_rois: list[ImagejRoi],
    global_roi_path: Path,
    spacing_zyx_um: np.ndarray,
    origin_zyx_um: np.ndarray,
    affine_zyx_um: np.ndarray,
    min_cell_area_um2: float,
    *,
    workers: int = 1,
) -> None:
    """Stream parallel, vectorized outline filtering into a replacement ZIP."""
    step_start = perf_counter()
    print(
        f"Warping, filtering, and saving global ImageJ ROIs to {global_roi_path} "
        f"with {workers} workers.",
        flush=True,
    )
    kept_count = 0

    def retained_rois() -> Iterator[ImagejRoi]:
        nonlocal kept_count
        for roi in global_rois(
            pixel_spacing_rois,
            spacing_zyx_um,
            origin_zyx_um,
            affine_zyx_um,
            min_cell_area_um2,
            workers=workers,
        ):
            kept_count += 1
            yield roi
            if kept_count % 10000 == 0:
                print(f"Saved {kept_count} global ImageJ ROIs.", flush=True)

    with TemporaryDirectory(dir=global_roi_path.parent) as temporary_directory:
        temporary_path = Path(temporary_directory) / global_roi_path.name
        roiwrite(temporary_path, retained_rois(), mode="w")
        temporary_path.replace(global_roi_path)
    print(
        f"Saved {kept_count}/{len(pixel_spacing_rois)} ImageJ ROIs; "
        f"removed {len(pixel_spacing_rois) - kept_count} with "
        f"min_cell_area_um2={min_cell_area_um2} in "
        f"{perf_counter() - step_start:.1f} s.",
        flush=True,
    )


def warp_points(
    pixel_space_points: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
    affine: np.ndarray,
) -> np.ndarray:
    """
    Warp points from pixel space to global space using known transforms.

    Parameters
    ----------
    pixel_space_points : numpy.ndarray
        Points in pixel Z, Y, X coordinates.
    spacing : numpy.ndarray
        Pixel spacing in microns.
    origin : numpy.ndarray
        Physical origin in microns.
    affine : numpy.ndarray
        Homogeneous affine transform.

    Returns
    -------
    numpy.ndarray
        Warped physical Z, Y, X coordinates.
    """
    spacing = round_spacing_um(spacing).astype(np.asarray(spacing).dtype)
    physical_space_points = pixel_space_points * spacing + origin
    homogeneous_points = np.column_stack(
        (
            physical_space_points,
            np.ones(physical_space_points.shape[0], dtype=physical_space_points.dtype),
        )
    )
    return (np.asarray(affine) @ homogeneous_points.T).T[:, :3]


def _prepare_cellpose_input_image(image: np.ndarray) -> np.ndarray:
    """Prepare image axes for 2D Cellpose evaluation without intensity scaling."""
    return transforms.convert_image(np.asarray(image), do_3D=False)


def main() -> None:
    """Run the Typer app."""
    app()


if __name__ == "__main__":
    main()
