"""Run Cellpose and save segmentation outputs in datastore coordinates.

Shepherd 2025/10 - change to CLI.
Shepherd 2025/07 - refactor for CellposeSAM
Shepherd 2024/12 - refactor
Shepherd 2024/11 - created script to run cellpose given determined parameters.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import typer
from cellpose import io, models, transforms
from roifile import ImagejRoi, roiread, roiwrite

from merfish3danalysis.cli.qi2lab_microscopes._common import qi2lab_datastore_path
from merfish3danalysis.qi2labDataStore import qi2labDataStore

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
        Use Cellpose multiprocessing for ImageJ ROI outline extraction.
    save_outputs : bool, default=True
        Save mask image and ImageJ ROIs after Cellpose finishes. Disable this
        for fast parameter comparison against the Cellpose GUI.
    use_gpu : bool, default=True
        Run Cellpose on CUDA. If True and CUDA is unavailable to PyTorch, raise
        an error instead of silently falling back to slow CPU inference.
    """

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
    spacing_zyx_um = np.asarray(attributes["spacing_zyx_um"], dtype=np.float32)

    max_projection_path = (
        datastore_path / "segmentation" / "cellpose" / "fiducial_max_projection.ome.tiff"
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
    imagej_roi_path_dir = datastore_path / "segmentation" / "cellpose" / "imagej_rois"
    imagej_roi_path_dir.mkdir(exist_ok=True)
    imagej_roi_path = imagej_roi_path_dir / "pixel_spacing"
    print(
        f"Saving pixel-space ImageJ ROIs to {imagej_roi_path}_rois.zip "
        f"multiprocessing={roi_multiprocessing}.",
        flush=True,
    )
    io.save_rois(masks, str(imagej_roi_path), multiprocessing=roi_multiprocessing)
    print(
        f"Saved pixel-space ImageJ ROIs in {perf_counter() - step_start:.1f} s.",
        flush=True,
    )

    # load pixel spaced ROIs
    step_start = perf_counter()
    cellpose_roi_path = imagej_roi_path_dir / "pixel_spacing_rois.zip"
    print(f"Loading pixel-space ImageJ ROIs from {cellpose_roi_path}.", flush=True)
    pixel_spacing_rois = roiread(cellpose_roi_path)
    print(
        f"Loaded {len(pixel_spacing_rois)} pixel-space ROIs in "
        f"{perf_counter() - step_start:.1f} s.",
        flush=True,
    )

    step_start = perf_counter()
    print("Warping ImageJ ROIs into global coordinates.", flush=True)
    global_spacing_rois = []
    for cell_idx, pixel_spaced_roi in enumerate(pixel_spacing_rois):
        roi = _global_roi_from_pixel_roi(
            pixel_spaced_roi,
            cell_idx,
            spacing_zyx_um,
            origin_zyx_um,
            affine_zyx_um,
        )
        global_spacing_rois.append(roi)
        del roi
        if (cell_idx + 1) % 1000 == 0:
            print(
                f"Warped {cell_idx + 1}/{len(pixel_spacing_rois)} ROIs.",
                flush=True,
            )
    print(
        f"Warped {len(global_spacing_rois)} ImageJ ROIs in "
        f"{perf_counter() - step_start:.1f} s.",
        flush=True,
    )

    # write global coordinate ROIs
    step_start = perf_counter()
    global_roi_path = imagej_roi_path_dir / "global_coords_rois.zip"
    print(f"Saving global-coordinate ImageJ ROIs to {global_roi_path}.", flush=True)
    roiwrite(global_roi_path, global_spacing_rois)
    print(
        f"Saved global-coordinate ImageJ ROIs in {perf_counter() - step_start:.1f} s.",
        flush=True,
    )


def _global_roi_from_pixel_roi(
    pixel_spaced_roi: ImagejRoi,
    cell_idx: int,
    spacing_zyx_um: np.ndarray,
    origin_zyx_um: np.ndarray,
    affine_zyx_um: np.ndarray,
) -> ImagejRoi:
    """Warp one pixel-space ROI into global ImageJ xy coordinates.

    Pixel-space ROIs are stored as xy points. The global transform expects zyx
    points, so this pads a dummy z plane, flips xy to yx for the transform, then
    drops z and flips back to xy for ImageJ ROI storage.
    """
    pixel_coordinates = pixel_spaced_roi.coordinates().astype(np.float32)
    global_coordinates_padded = warp_points(
        np.column_stack(
            (
                np.full(pixel_coordinates.shape[0], 10, dtype=np.float32),
                pixel_coordinates[:, 1],
                pixel_coordinates[:, 0],
            )
        ),
        spacing_zyx_um,
        origin_zyx_um,
        affine_zyx_um,
    )
    roi = ImagejRoi.frompoints(
        np.round(global_coordinates_padded[:, 1:][:, ::-1], 2).astype(np.float32)
    )
    roi.name = "cell_" + str(cell_idx).zfill(7)
    return roi


def warp_points(
    pixel_space_points: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
    affine: np.ndarray,
) -> np.ndarray:
    """Warp points from pixel space to global space using known transforms."""
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
