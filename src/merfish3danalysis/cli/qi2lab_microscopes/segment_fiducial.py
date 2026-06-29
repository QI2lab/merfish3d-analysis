"""
Run cellpose, save ROIs, reload ROIs, warp ROIs to global system, save again.

Eventually, this will be re-integrated back into the library - but it is
helpful to see it split out for now.

IMPORTANT: You must optimize the cellpose parameters on your own using the GUI,
then fill in the dictionary at the bottom of the script.

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

from merfish3danalysis.qi2labDataStore import qi2labDataStore

app = typer.Typer()
app.pretty_exceptions_enable = False


@app.command()
def run_cellpose(
    root_path: Path,
    normalization: tuple[float, float] = [1.0, 99.0],
    diameter: int = 30,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    niter: int = 0,
    min_size: int = 15,
    pretrained_model: str = "cpsam_v2",
    roi_multiprocessing: bool = False,
    use_gpu: bool = True,
    zstride_level: int = 0,
) -> None:
    """Run cellpose and save ROIs

    Parameters
    ----------
    root_path: Path
        path to experiment.
    normalization: tuple[float,float], default = [1.0,99.0]
        normalization values [low,high].
    diameter: int, default = 30
        cell size in integers.
    flow_threshold: float, default = 0.4
        flow threshold.
    cellprob_threshold: float, default = 0.0
        cell probability threshold.
    niter : int, default=0
        Number of Cellpose dynamics iterations. The GUI default is 0, which
        lets Cellpose choose the iteration count from the diameter.
    min_size : int, default=15
        Minimum mask size. This matches the Cellpose GUI default.
    pretrained_model : str, default="cpsam_v2"
        Cellpose model name or path. Built-in GUI names include cpsam_v2,
        cpdino, cpdino-vitb, and cpsam.
    roi_multiprocessing : bool, default=False
        Use Cellpose multiprocessing for ImageJ ROI outline extraction.
    use_gpu : bool, default=True
        Run Cellpose on CUDA. If True and CUDA is unavailable to PyTorch, raise
        an error instead of silently falling back to slow CPU inference.
    zstride_level: int, default = 0
        look for a skip z dataset.
    """

    # initialize datastore
    if zstride_level == 0:
        datastore_path = root_path / Path(r"qi2labdatastore")
    else:
        datastore_path = root_path / Path(f"qi2labdatastore_zstride0{zstride_level}")
    datastore = qi2labDataStore(datastore_path)
    print(f"Using datastore at {datastore_path}")

    fused_image_path = datastore._image_store_path(
        datastore._fused_root_path / Path(f"fused_{datastore.fiducial_folder_name}_zyx")
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
        datastore_path
        / Path("segmentation")
        / Path("cellpose")
        / Path("fiducial_max_projection.ome.tiff")
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
        loaded = datastore.load_global_fidicual_image(return_future=False)
        if loaded is None:
            raise RuntimeError(
                "Could not load globally registered fused fiducial image."
            )
        fiducial_fused, affine_zyx_um, origin_zyx_um, spacing_zyx_um = loaded
        fiducial_max_projection = np.max(np.squeeze(fiducial_fused), axis=0)
        del fiducial_fused
    fiducial_max_projection = _prepare_cellpose_gui_image(fiducial_max_projection)
    print(
        "Prepared Cellpose GUI-style image "
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
        "percentile": normalization,
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
        f"cellprob_threshold={cellprob_threshold} niter={niter} min_size={min_size} "
        f"normalize={normalize!r}.",
        flush=True,
    )
    masks, _, _ = model.eval(
        fiducial_max_projection,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        niter=niter,
        normalize=normalize,
        min_size=min_size,
        channel_axis=-1,
        z_axis=None,
    )
    print(
        f"Cellpose finished; labels={int(np.max(masks)) if masks.size else 0}.",
        flush=True,
    )

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
    imagej_roi_path_dir = (
        datastore_path / Path("segmentation") / Path("cellpose") / Path("imagej_rois")
    )
    if not (imagej_roi_path_dir.exists()):
        imagej_roi_path_dir.mkdir()
    imagej_roi_path = imagej_roi_path_dir / Path("pixel_spacing")
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
    cellpose_roi_path = imagej_roi_path_dir / Path("pixel_spacing_rois.zip")
    print(f"Loading pixel-space ImageJ ROIs from {cellpose_roi_path}.", flush=True)
    pixel_spacing_rois = roiread(cellpose_roi_path)
    print(
        f"Loaded {len(pixel_spacing_rois)} pixel-space ROIs in "
        f"{perf_counter() - step_start:.1f} s.",
        flush=True,
    )

    # warp ROIs into global coordinates
    # the ROIs are in (x,y) format. So we have to (1) fake a z dimension,
    # (2) flip xy to yx, (3) warp, (4) remove the z, (5) flip back to (x,y)
    # When we load to check if RNA are in an ROI, need to remember to flip
    # back to (y,x).
    step_start = perf_counter()
    print("Warping ImageJ ROIs into global coordinates.", flush=True)
    global_spacing_rois = []
    for cell_idx, pixel_spaced_roi in enumerate(pixel_spacing_rois):
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
        global_coordinates = global_coordinates_padded[:, 1:]
        roi = ImagejRoi.frompoints(
            np.round(global_coordinates[:, ::-1], 2).astype(np.float32)
        )
        roi.name = "cell_" + str(cell_idx).zfill(7)
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
    global_roi_path = imagej_roi_path_dir / Path("global_coords_rois.zip")
    print(f"Saving global-coordinate ImageJ ROIs to {global_roi_path}.", flush=True)
    pixel_spacing_rois = roiwrite(global_roi_path, global_spacing_rois)
    print(
        f"Saved global-coordinate ImageJ ROIs in {perf_counter() - step_start:.1f} s.",
        flush=True,
    )


def warp_point(
    pixel_space_point: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
    affine: np.ndarray,
) -> np.ndarray:
    """Warp point from pixel space to global space using known transforms.

    Parameters
    ----------
    pixel_space_point : np.ndarray
        point in the image coordinate system, zyx order
    spacing: np.ndarray
        pixel size in microns, zyx order
    origin: np.ndarray
        world coordinate origin (um), zyx order
    affine: np.ndarray
        4x4 affine matrix (um), zyx order

    Returns
    -------
    registered_space_point: np.ndarray
        point in the world coordinate system (um), zyx order

    """

    physical_space_point = pixel_space_point * spacing + origin
    registered_space_point = (
        np.array(affine) @ np.array([*list(physical_space_point), 1])
    )[:-1]

    return registered_space_point


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


def _prepare_cellpose_gui_image(image: np.ndarray) -> np.ndarray:
    """Match Cellpose GUI image preparation before 2D model evaluation."""
    gui_image = transforms.convert_image(np.asarray(image), do_3D=False)
    image_min = float(np.min(gui_image))
    image_max = float(np.max(gui_image))
    gui_image = gui_image.astype(np.float32, copy=False)
    gui_image -= image_min
    if image_max > image_min + 1e-3:
        gui_image /= image_max - image_min
    gui_image *= 255.0
    return gui_image


def main() -> None:
    """
    Main.

    Returns
    -------
    None
        Function result.
    """
    app()


if __name__ == "__main__":
    main()
