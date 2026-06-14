"""Chromatic aberration calibration from multi-channel bead images."""

import json
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tifffile import TiffFile, imread

UFISH_MODEL_ALIASES = {
    "merfish": "finetune_models/v1.0.1-MERFISH_model.onnx",
    "seqfish": "finetune_models/v1.0.1-seqFISH_model.onnx",
    "simfish": "finetune_models/v1.0.1-simfish_model.onnx",
    "smfish": "finetune_models/v1.0.1-simfish_model.onnx",
    "deepspot": "finetune_models/v1.0.1-deepspot_model.onnx",
    "exseq": "finetune_models/v1.0.1-ExSeq_model.onnx",
}
DEFAULT_UFISH_MODEL = "simfish"


def parse_csv_floats(value: str | Sequence[float] | None) -> tuple[float, ...] | None:
    """
    Parse a comma-separated float string.

    Parameters
    ----------
    value : str or Sequence[float] or None
        Input value to parse.

    Returns
    -------
    tuple[float, ...] or None
        Parsed floats, or None when ``value`` is None.
    """

    if value is None:
        return None
    if isinstance(value, str):
        return tuple(float(v.strip()) for v in value.split(",") if v.strip())
    return tuple(float(v) for v in value)


def _xml_float(value: str | None) -> float | None:
    """
    Parse an XML float value.

    Parameters
    ----------
    value : str or None
        XML attribute value.

    Returns
    -------
    float or None
        Parsed float, or None if parsing fails.
    """

    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _wavelength_to_um(value: float | None) -> float | None:
    """
    Convert wavelength metadata to microns.

    Parameters
    ----------
    value : float or None
        Wavelength value. OME files often store wavelengths in nanometers.

    Returns
    -------
    float or None
        Wavelength in microns.
    """

    if value is None:
        return None
    if value > 10:
        return value / 1000.0
    return value


def _parse_ome_metadata(
    ome_metadata: str | None,
) -> tuple[tuple[float, float, float] | None, list[float] | None, list[str] | None]:
    """
    Parse spacing and channel metadata from OME XML.

    Parameters
    ----------
    ome_metadata : str or None
        OME XML metadata string.

    Returns
    -------
    tuple
        ``(spacing_zyx_um, wavelengths_um, channel_names)``. Missing values are
        returned as None.
    """

    if not ome_metadata:
        return None, None, None

    try:
        root = ET.fromstring(ome_metadata)
    except ET.ParseError:
        return None, None, None

    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag.split("}", 1)[0] + "}"
    pixels = root.find(f".//{namespace}Pixels")
    if pixels is None:
        return None, None, None

    size_z = _xml_float(pixels.attrib.get("PhysicalSizeZ"))
    size_y = _xml_float(pixels.attrib.get("PhysicalSizeY"))
    size_x = _xml_float(pixels.attrib.get("PhysicalSizeX"))
    spacing = None
    if size_z is not None and size_y is not None and size_x is not None:
        spacing = (float(size_z), float(size_y), float(size_x))

    wavelengths = []
    names = []
    for channel in pixels.findall(f"{namespace}Channel"):
        name = channel.attrib.get("Name") or channel.attrib.get("ID") or ""
        names.append(str(name))
        emission = _wavelength_to_um(_xml_float(channel.attrib.get("EmissionWavelength")))
        excitation = _wavelength_to_um(
            _xml_float(channel.attrib.get("ExcitationWavelength"))
        )
        wavelengths.append(emission if emission is not None else excitation)

    if not wavelengths or any(v is None for v in wavelengths):
        wavelengths = None
    else:
        wavelengths = [float(v) for v in wavelengths]
    if not names:
        names = None
    return spacing, wavelengths, names


def load_bead_channel_stack(
    image_path: Path | str,
    *,
    channel_axis: str | None = None,
    voxel_size_zyx_um: Sequence[float] | None = None,
    wavelengths_um: Sequence[float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load a bead image as a channel stack in C, Z, Y, X order.

    Parameters
    ----------
    image_path : pathlib.Path or str
        Input bead image file. OME-TIFF/TIFF files are supported.
    channel_axis : str or None, default=None
        Optional axis string for the input array. If omitted, tifffile series
        axis metadata are used.
    voxel_size_zyx_um : Sequence[float] or None, default=None
        Optional voxel spacing override in Z, Y, X microns.
    wavelengths_um : Sequence[float] or None, default=None
        Optional channel wavelengths in microns.

    Returns
    -------
    tuple[numpy.ndarray, dict[str, Any]]
        Channel stack in C, Z, Y, X order and metadata.
    """

    image_path = Path(image_path)
    with TiffFile(image_path) as tif:
        axes = channel_axis or tif.series[0].axes
        ome_spacing, ome_wavelengths, channel_names = _parse_ome_metadata(
            tif.ome_metadata
        )
    image = np.asarray(imread(image_path))
    axes = str(axes).upper()

    if len(axes) != image.ndim:
        raise ValueError(
            f"Axis string {axes!r} has length {len(axes)} but image has "
            f"{image.ndim} dimensions."
        )
    for axis_index in reversed(range(len(axes))):
        if axes[axis_index] in "CZYX":
            continue
        if image.shape[axis_index] != 1:
            raise ValueError(
                f"Unsupported non-singleton axis {axes[axis_index]!r} in bead image."
            )
        image = np.take(image, 0, axis=axis_index)
        axes = axes[:axis_index] + axes[axis_index + 1 :]

    if "C" not in axes:
        raise ValueError("Bead calibration image must include a channel axis.")
    if "Y" not in axes or "X" not in axes:
        raise ValueError("Bead calibration image must include Y and X axes.")

    if "Z" not in axes:
        image = np.expand_dims(image, axis=0)
        axes = "Z" + axes

    order = [axes.index(axis) for axis in "CZYX"]
    stack = np.transpose(image, order)
    if stack.ndim != 4:
        raise ValueError(f"Expected CZYX stack after axis normalization, got {stack.shape}.")

    spacing = tuple(float(v) for v in (voxel_size_zyx_um or ome_spacing or (1, 1, 1)))
    if len(spacing) != 3:
        raise ValueError("voxel_size_zyx_um must contain three values.")

    wavelengths = wavelengths_um or ome_wavelengths
    if wavelengths is None:
        raise ValueError(
            "Channel wavelengths are required to generate PSFs. Provide OME "
            "wavelength metadata or pass wavelengths_um."
        )
    wavelengths = [float(v) for v in wavelengths]
    if len(wavelengths) != stack.shape[0]:
        raise ValueError(
            f"Expected {stack.shape[0]} wavelengths, got {len(wavelengths)}."
        )

    if channel_names is None or len(channel_names) != stack.shape[0]:
        channel_names = [f"channel{i:03d}" for i in range(stack.shape[0])]

    metadata = {
        "axes": "CZYX",
        "voxel_size_zyx_um": spacing,
        "wavelengths_um": wavelengths,
        "channel_names": channel_names,
    }
    return stack.astype(np.float32, copy=False), metadata


def _resolve_ufish_weights_path(model: str | Path | None) -> Path | str:
    """
    Resolve a U-FISH model alias or local path.

    Parameters
    ----------
    model : str or pathlib.Path or None
        Model alias or path.

    Returns
    -------
    pathlib.Path or str
        Local path or U-FISH weights filename.
    """

    if model is None:
        model = DEFAULT_UFISH_MODEL
    model_str = str(model).strip() or DEFAULT_UFISH_MODEL
    model_path = Path(model_str).expanduser()
    if model_path.exists():
        return model_path
    weights_file = UFISH_MODEL_ALIASES.get(model_str.lower(), model_str)
    local_path = Path.home() / ".ufish" / weights_file
    if local_path.exists():
        return local_path
    return weights_file


def _load_ufish_model(ufish: Any, model: str | Path | None) -> None:
    """
    Load U-FISH weights.

    Parameters
    ----------
    ufish : Any
        U-FISH object.
    model : str or pathlib.Path or None
        Model alias or path.

    Returns
    -------
    None
        Model weights are loaded in place.
    """

    weights = _resolve_ufish_weights_path(model)
    if isinstance(weights, Path):
        ufish.load_weights_from_path(weights)
    else:
        ufish.load_weights(weights_file=weights)


def _roi_sum(image: np.ndarray, point_zyx: Sequence[float], radius_zyx: Sequence[int]) -> float:
    """
    Sum image pixels in a clipped ROI around one point.

    Parameters
    ----------
    image : numpy.ndarray
        Image in Z, Y, X order.
    point_zyx : Sequence[float]
        Center point in Z, Y, X pixels.
    radius_zyx : Sequence[int]
        ROI half-widths in Z, Y, X pixels.

    Returns
    -------
    float
        Sum of ROI pixel values.
    """

    center = np.round(np.asarray(point_zyx, dtype=float)).astype(int)
    radius = np.asarray(radius_zyx, dtype=int)
    start = np.maximum(center - radius, 0)
    stop = np.minimum(center + radius + 1, image.shape)
    roi = image[start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]]
    return float(np.sum(roi))


def detect_bead_centroids(
    image_zyx: np.ndarray,
    *,
    gpu_id: int = 0,
    ufish_model: str | Path | None = None,
    min_intensity_quantile: float = 0.5,
    max_beads: int | None = None,
) -> pd.DataFrame:
    """
    Detect bead centroids in one channel using U-FISH.

    Parameters
    ----------
    image_zyx : numpy.ndarray
        Deconvolved bead image in Z, Y, X order.
    gpu_id : int, default=0
        CUDA device ID.
    ufish_model : str or pathlib.Path or None, default=None
        U-FISH model alias or path.
    min_intensity_quantile : float, default=0.5
        Quantile threshold on local deconvolved intensity.
    max_beads : int or None, default=None
        Optional maximum number of beads to keep after sorting by brightness.

    Returns
    -------
    pandas.DataFrame
        Bead centroid table with columns ``z``, ``y``, ``x`` and
        ``sum_decon_pixels``.
    """

    from ufish.api import UFish

    ufish = UFish(device=f"cuda:{int(gpu_id)}")
    _load_ufish_model(ufish, ufish_model)
    loc, _prediction = ufish.predict(
        image_zyx.astype(np.float32, copy=False),
        axes="zyx",
        blend_3d=False,
        batch_size=1,
    )
    if loc.empty:
        return pd.DataFrame(columns=["z", "y", "x", "sum_decon_pixels"])
    loc = loc.rename(columns={"axis-0": "z", "axis-1": "y", "axis-2": "x"})
    loc = loc.dropna(subset=["z", "y", "x"]).copy()
    loc["sum_decon_pixels"] = [
        _roi_sum(image_zyx, row[["z", "y", "x"]].to_numpy(), (2, 2, 2))
        for _idx, row in loc.iterrows()
    ]
    if not loc.empty:
        threshold = loc["sum_decon_pixels"].quantile(float(min_intensity_quantile))
        loc = loc[loc["sum_decon_pixels"] >= threshold]
    loc = loc.sort_values("sum_decon_pixels", ascending=False)
    if max_beads is not None:
        loc = loc.head(int(max_beads))
    return loc.reset_index(drop=True)


def deconvolve_channels(
    stack_czyx: np.ndarray,
    psfs: Sequence[np.ndarray],
    *,
    gpu_id: int = 0,
    crop_yx: int = 2048,
) -> np.ndarray:
    """
    Deconvolve every channel using RLGC.

    Parameters
    ----------
    stack_czyx : numpy.ndarray
        Channel stack in C, Z, Y, X order.
    psfs : Sequence[numpy.ndarray]
        One PSF per channel or one shared PSF.
    gpu_id : int, default=0
        CUDA device ID.
    crop_yx : int, default=2048
        Initial RLGC lateral crop size.

    Returns
    -------
    numpy.ndarray
        Deconvolved channel stack in C, Z, Y, X order.
    """

    from merfish3danalysis.utils.rlgc import chunked_rlgc

    if len(psfs) not in (1, stack_czyx.shape[0]):
        raise ValueError("Provide either one shared PSF or one PSF per channel.")
    outputs = []
    for channel_index, image in enumerate(stack_czyx):
        psf = psfs[0] if len(psfs) == 1 else psfs[channel_index]
        outputs.append(
            chunked_rlgc(
                np.asarray(image, dtype=np.float32),
                np.asarray(psf, dtype=np.float32),
                gpu_id=gpu_id,
                crop_yx=crop_yx,
                release_memory=True,
            )
        )
    return np.stack(outputs, axis=0)


def generate_channel_psfs(
    *,
    num_z: int,
    voxel_size_zyx_um: Sequence[float],
    wavelengths_um: Sequence[float],
    na: float = 1.35,
    ri: float = 1.51,
    psf_nx: int = 51,
) -> list[np.ndarray]:
    """
    Generate one vectorial PSF per channel.

    Parameters
    ----------
    num_z : int
        Number of z planes in the bead image.
    voxel_size_zyx_um : Sequence[float]
        Voxel spacing in Z, Y, X microns.
    wavelengths_um : Sequence[float]
        Channel emission wavelengths in microns.
    na : float, default=1.35
        Objective numerical aperture.
    ri : float, default=1.51
        Immersion refractive index.
    psf_nx : int, default=51
        Lateral PSF support in pixels.

    Returns
    -------
    list[numpy.ndarray]
        Normalized PSFs in Z, Y, X order.
    """

    from psfmodels import make_psf

    spacing = tuple(float(v) for v in voxel_size_zyx_um)
    if len(spacing) != 3:
        raise ValueError("voxel_size_zyx_um must contain three values.")

    psfs = []
    for wavelength_um in wavelengths_um:
        psf = make_psf(
            z=int(num_z),
            nx=int(psf_nx),
            dxy=spacing[1],
            dz=spacing[0],
            NA=float(na),
            wvl=float(wavelength_um),
            ns=1.47,
            ni=float(ri),
            ni0=float(ri),
            model="vectorial",
        ).astype(np.float32)
        psf_sum = np.sum(psf, axis=(0, 1, 2))
        psfs.append(psf / (psf_sum if psf_sum != 0 else 1.0))
    return psfs


def _mutual_nearest_matches(
    reference_points_um: np.ndarray,
    moving_points_um: np.ndarray,
    radius_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match points by mutual nearest neighbors within a radius.

    Parameters
    ----------
    reference_points_um : numpy.ndarray
        Reference points in physical Z, Y, X microns.
    moving_points_um : numpy.ndarray
        Moving points in physical Z, Y, X microns.
    radius_um : float
        Maximum matching distance.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Matched reference and moving points.
    """

    if reference_points_um.size == 0 or moving_points_um.size == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty

    reference_tree = cKDTree(reference_points_um)
    moving_tree = cKDTree(moving_points_um)
    dist_to_moving, moving_indices = moving_tree.query(
        reference_points_um,
        distance_upper_bound=float(radius_um),
    )
    dist_to_ref, reference_indices = reference_tree.query(
        moving_points_um,
        distance_upper_bound=float(radius_um),
    )
    ref_matches = []
    mov_matches = []
    for reference_index, moving_index in enumerate(moving_indices):
        if moving_index >= moving_points_um.shape[0]:
            continue
        if not np.isfinite(dist_to_moving[reference_index]):
            continue
        if reference_indices[moving_index] != reference_index:
            continue
        if not np.isfinite(dist_to_ref[moving_index]):
            continue
        ref_matches.append(reference_index)
        mov_matches.append(int(moving_index))
    if not ref_matches:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty
    return (
        reference_points_um[np.asarray(ref_matches, dtype=np.int64)],
        moving_points_um[np.asarray(mov_matches, dtype=np.int64)],
    )


def fit_affine_source_to_reference(
    source_points_zyx_um: np.ndarray,
    reference_points_zyx_um: np.ndarray,
    *,
    max_iterations: int = 3,
    outlier_threshold_um: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Fit an affine transform mapping source points to reference points.

    Parameters
    ----------
    source_points_zyx_um : numpy.ndarray
        Moving/source points in physical Z, Y, X microns.
    reference_points_zyx_um : numpy.ndarray
        Fixed/reference points in physical Z, Y, X microns.
    max_iterations : int, default=3
        Number of residual rejection/refit iterations.
    outlier_threshold_um : float, default=1.0
        Residual threshold for keeping matches.

    Returns
    -------
    tuple[numpy.ndarray, dict[str, Any]]
        4x4 affine transform and fit diagnostics.
    """

    source = np.asarray(source_points_zyx_um, dtype=np.float64)
    reference = np.asarray(reference_points_zyx_um, dtype=np.float64)
    if source.shape[0] < 4:
        raise ValueError("At least four matched beads are required for 3D affine fitting.")

    keep = np.ones(source.shape[0], dtype=bool)
    affine = np.eye(4, dtype=np.float64)
    for _iteration in range(max(1, int(max_iterations))):
        design = np.concatenate(
            [source[keep], np.ones((int(np.sum(keep)), 1), dtype=np.float64)],
            axis=1,
        )
        solution, *_ = np.linalg.lstsq(design, reference[keep], rcond=None)
        affine = np.eye(4, dtype=np.float64)
        affine[:3, :3] = solution[:3, :].T
        affine[:3, 3] = solution[3, :]
        predicted = (np.concatenate([source, np.ones((source.shape[0], 1))], axis=1) @ affine.T)[
            :, :3
        ]
        residuals = np.linalg.norm(predicted - reference, axis=1)
        next_keep = residuals <= float(outlier_threshold_um)
        if np.sum(next_keep) < 4 or np.array_equal(next_keep, keep):
            break
        keep = next_keep

    predicted = (np.concatenate([source, np.ones((source.shape[0], 1))], axis=1) @ affine.T)[
        :, :3
    ]
    residuals = np.linalg.norm(predicted - reference, axis=1)
    kept_residuals = residuals[keep]
    diagnostics = {
        "matched_beads": int(source.shape[0]),
        "used_beads": int(np.sum(keep)),
        "median_residual_um": float(np.median(kept_residuals)),
        "p95_residual_um": float(np.percentile(kept_residuals, 95)),
        "max_residual_um": float(np.max(kept_residuals)),
    }
    return affine.astype(np.float32), diagnostics


def estimate_chromatic_affines(
    centroid_tables: Sequence[pd.DataFrame],
    *,
    voxel_size_zyx_um: Sequence[float],
    wavelengths_um: Sequence[float],
    channel_names: Sequence[str] | None = None,
    match_radius_um: float = 1.0,
    outlier_threshold_um: float = 1.0,
) -> dict[str, Any]:
    """
    Estimate channel chromatic affine transforms relative to the blue channel.

    Parameters
    ----------
    centroid_tables : Sequence[pandas.DataFrame]
        One centroid table per channel with ``z``, ``y`` and ``x`` columns.
    voxel_size_zyx_um : Sequence[float]
        Voxel spacing in Z, Y, X microns.
    wavelengths_um : Sequence[float]
        Channel wavelengths in microns.
    channel_names : Sequence[str] or None, default=None
        Optional channel names.
    match_radius_um : float, default=1.0
        Mutual nearest-neighbor match radius in microns.
    outlier_threshold_um : float, default=1.0
        Affine residual rejection threshold in microns.

    Returns
    -------
    dict[str, Any]
        Calibration metadata with one affine per channel.
    """

    spacing = np.asarray(voxel_size_zyx_um, dtype=np.float32)
    wavelengths = np.asarray(wavelengths_um, dtype=np.float32)
    reference_index = int(np.argmin(wavelengths))
    if channel_names is None:
        channel_names = [f"channel{i:03d}" for i in range(len(centroid_tables))]

    reference_points_px = centroid_tables[reference_index][["z", "y", "x"]].to_numpy(
        dtype=np.float32
    )
    reference_points_um = reference_points_px * spacing
    channels = {}
    for channel_index, table in enumerate(centroid_tables):
        channel_key = str(channel_names[channel_index])
        if channel_index == reference_index:
            channels[channel_key] = {
                "channel_index": channel_index,
                "channel_name": channel_key,
                "wavelength_um": float(wavelengths[channel_index]),
                "reference_channel": True,
                "affine_zyx_um": np.eye(4, dtype=np.float32).tolist(),
                "diagnostics": {
                    "matched_beads": int(reference_points_um.shape[0]),
                    "used_beads": int(reference_points_um.shape[0]),
                    "median_residual_um": 0.0,
                    "p95_residual_um": 0.0,
                    "max_residual_um": 0.0,
                },
            }
            continue

        moving_points_px = table[["z", "y", "x"]].to_numpy(dtype=np.float32)
        moving_points_um = moving_points_px * spacing
        reference_matches, moving_matches = _mutual_nearest_matches(
            reference_points_um,
            moving_points_um,
            radius_um=match_radius_um,
        )
        try:
            affine, diagnostics = fit_affine_source_to_reference(
                moving_matches,
                reference_matches,
                outlier_threshold_um=outlier_threshold_um,
            )
            status = "ok"
        except ValueError as exc:
            affine = np.eye(4, dtype=np.float32)
            diagnostics = {
                "matched_beads": int(reference_matches.shape[0]),
                "used_beads": 0,
                "median_residual_um": None,
                "p95_residual_um": None,
                "max_residual_um": None,
                "error": str(exc),
            }
            status = "identity_fallback"

        channels[channel_key] = {
            "channel_index": channel_index,
            "channel_name": channel_key,
            "wavelength_um": float(wavelengths[channel_index]),
            "reference_channel": False,
            "affine_zyx_um": np.asarray(affine, dtype=np.float32).tolist(),
            "diagnostics": diagnostics,
            "status": status,
        }

    return {
        "reference_channel_index": reference_index,
        "reference_channel_name": str(channel_names[reference_index]),
        "reference_wavelength_um": float(wavelengths[reference_index]),
        "voxel_size_zyx_um": [float(v) for v in spacing],
        "channels": channels,
    }


def save_calibration_json(calibration: dict[str, Any], output_path: Path | str) -> None:
    """
    Save chromatic calibration metadata as JSON.

    Parameters
    ----------
    calibration : dict[str, Any]
        Calibration metadata.
    output_path : pathlib.Path or str
        Output JSON path.

    Returns
    -------
    None
        JSON is written to disk.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(calibration, handle, indent=2)


def run_chromatic_calibration(
    image_path: Path | str,
    *,
    output_path: Path | str,
    datastore: Any | None = None,
    channel_axis: str | None = None,
    voxel_size_zyx_um: Sequence[float] | None = None,
    wavelengths_um: Sequence[float] | None = None,
    na: float = 1.35,
    ri: float = 1.51,
    psf_nx: int = 51,
    gpu_id: int = 0,
    crop_yx: int = 2048,
    ufish_model: str | Path | None = None,
    match_radius_um: float = 1.0,
    outlier_threshold_um: float = 1.0,
    min_intensity_quantile: float = 0.5,
    max_beads: int | None = None,
    save_intermediates: bool = False,
) -> dict[str, Any]:
    """
    Run full chromatic calibration from a multi-channel bead image.

    Parameters
    ----------
    image_path : pathlib.Path or str
        Multi-channel bead image.
    output_path : pathlib.Path or str
        Output JSON path.
    datastore : Any or None, default=None
        Optional qi2labDataStore. When provided, calibration metadata are saved
        into its calibration sidecar.
    channel_axis : str or None, default=None
        Optional input image axis string.
    voxel_size_zyx_um : Sequence[float] or None, default=None
        Optional voxel spacing override.
    wavelengths_um : Sequence[float] or None, default=None
        Optional channel emission wavelengths in microns.
    na : float, default=1.35
        Objective numerical aperture used to generate channel PSFs.
    ri : float, default=1.51
        Immersion refractive index used to generate channel PSFs.
    psf_nx : int, default=51
        Lateral PSF support in pixels.
    gpu_id : int, default=0
        CUDA device ID.
    crop_yx : int, default=2048
        Initial RLGC crop size.
    ufish_model : str or pathlib.Path or None, default=None
        U-FISH model alias or path.
    match_radius_um : float, default=1.0
        Bead matching radius in microns.
    outlier_threshold_um : float, default=1.0
        Affine residual rejection threshold in microns.
    min_intensity_quantile : float, default=0.5
        Quantile threshold for keeping bright U-FISH bead calls.
    max_beads : int or None, default=None
        Optional maximum number of beads per channel.
    save_intermediates : bool, default=False
        If True, save deconvolved stack and centroid CSV files next to JSON.

    Returns
    -------
    dict[str, Any]
        Calibration metadata.
    """

    stack, metadata = load_bead_channel_stack(
        image_path,
        channel_axis=channel_axis,
        voxel_size_zyx_um=voxel_size_zyx_um,
        wavelengths_um=wavelengths_um,
    )
    psfs = generate_channel_psfs(
        num_z=stack.shape[1],
        voxel_size_zyx_um=metadata["voxel_size_zyx_um"],
        wavelengths_um=metadata["wavelengths_um"],
        na=na,
        ri=ri,
        psf_nx=psf_nx,
    )
    deconvolved = deconvolve_channels(stack, psfs, gpu_id=gpu_id, crop_yx=crop_yx)
    centroid_tables = [
        detect_bead_centroids(
            deconvolved[channel_index],
            gpu_id=gpu_id,
            ufish_model=ufish_model,
            min_intensity_quantile=min_intensity_quantile,
            max_beads=max_beads,
        )
        for channel_index in range(deconvolved.shape[0])
    ]
    calibration = estimate_chromatic_affines(
        centroid_tables,
        voxel_size_zyx_um=metadata["voxel_size_zyx_um"],
        wavelengths_um=metadata["wavelengths_um"],
        channel_names=metadata["channel_names"],
        match_radius_um=match_radius_um,
        outlier_threshold_um=outlier_threshold_um,
    )
    calibration["source_image"] = str(Path(image_path))
    save_calibration_json(calibration, output_path)
    if datastore is not None:
        datastore.save_chromatic_affine_transforms_zyx_um(calibration)

    if save_intermediates:
        output_path = Path(output_path)
        np.save(output_path.with_suffix(".deconvolved.npy"), deconvolved)
        for channel_index, table in enumerate(centroid_tables):
            table.to_csv(
                output_path.with_name(
                    f"{output_path.stem}_channel{channel_index:03d}_beads.csv"
                ),
                index=False,
            )
    return calibration


__all__ = [
    "estimate_chromatic_affines",
    "fit_affine_source_to_reference",
    "generate_channel_psfs",
    "load_bead_channel_stack",
    "run_chromatic_calibration",
]
