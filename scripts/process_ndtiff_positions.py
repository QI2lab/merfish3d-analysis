"""Export camera-corrected NDTiff positions and selected U-FISH predictions."""

import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from tifffile import imwrite

from merfish3danalysis.utils.psf import (
    QI2LAB_EMISSION_WAVELENGTHS_UM,
    generate_qi2lab_psf,
)

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

QI2LAB_DECON_CROP_YX = 2048


@dataclass(frozen=True)
class DatasetLayout:
    """Axes needed to read each position, channel, and Z plane."""

    channels: tuple[str, ...]
    channel_coordinates: tuple[int | str, ...]
    positions: tuple[int | None, ...]
    z_values: tuple[int | None, ...]
    fixed_coordinates: dict[str, int | str]


def _load_dataset(dataset_path: Path) -> Any:
    """Open an NDTiff dataset."""
    from ndstorage import Dataset

    dataset = Dataset(str(dataset_path))
    _alias_zero_suffixed_stack_reader(dataset)
    return dataset


def _expand_ufish_channel_arguments(arguments: Sequence[str]) -> list[str]:
    """Expand consecutive U-FISH channel values into repeated CLI options."""
    expanded_arguments = []
    argument_index = 0
    while argument_index < len(arguments):
        argument = arguments[argument_index]
        if argument != "--ufish-channel":
            expanded_arguments.append(argument)
            argument_index += 1
            continue

        expanded_arguments.append(argument)
        argument_index += 1
        if argument_index >= len(arguments) or arguments[argument_index].startswith(
            "-"
        ):
            continue

        expanded_arguments.append(arguments[argument_index])
        argument_index += 1
        while argument_index < len(arguments) and not arguments[
            argument_index
        ].startswith("-"):
            expanded_arguments.extend(["--ufish-channel", arguments[argument_index]])
            argument_index += 1

    return expanded_arguments


def _alias_zero_suffixed_stack_reader(dataset: Any) -> None:
    """Alias an indexed first stack to its on-disk ``_0`` reader when needed."""
    readers = getattr(dataset, "_readers_by_filename", None)
    index = getattr(dataset, "index", None)
    if not isinstance(readers, dict) or not isinstance(index, dict):
        return

    indexed_filenames = {entry["filename"] for entry in index.values()}
    for filename in indexed_filenames - readers.keys():
        path = Path(filename)
        zero_suffixed_name = f"{path.stem}_0{path.suffix}"
        if zero_suffixed_name in readers:
            readers[filename] = readers[zero_suffixed_name]


def _axis_values(
    coordinates: Sequence[dict[str, int | str]], axis: str
) -> tuple[int | str, ...]:
    """Return sorted unique values for one NDTiff axis."""
    values = {coordinate[axis] for coordinate in coordinates if axis in coordinate}
    try:
        return tuple(sorted(values))
    except TypeError as error:
        raise ValueError(
            f"NDTiff axis {axis!r} mixes incompatible value types."
        ) from error


def inspect_dataset(dataset: Any) -> DatasetLayout:
    """Discover channels, positions, Z planes, and singleton coordinates."""
    coordinates = dataset.get_image_coordinates_list()
    if not coordinates:
        raise ValueError("The NDTiff dataset does not contain any images.")

    channel_coordinates = _axis_values(coordinates, "channel")
    if not channel_coordinates:
        raise ValueError("The NDTiff dataset does not contain a channel axis.")

    channels = tuple(str(channel) for channel in dataset.get_channel_names())
    if not channels:
        summary_metadata = getattr(dataset, "summary_metadata", None)
        summary_channels = (
            summary_metadata.get("ChNames")
            if isinstance(summary_metadata, dict)
            else None
        )
        if isinstance(summary_channels, (list, tuple)):
            channels = tuple(str(channel) for channel in summary_channels)
    if len(channels) != len(channel_coordinates):
        raise ValueError(
            "Could not map NDTiff channel names to channel coordinates: "
            f"names={channels!r}, coordinates={channel_coordinates!r}."
        )
    if len(set(channels)) != len(channels):
        raise ValueError(f"NDTiff channel names are not unique: {channels!r}.")

    if all(isinstance(value, str) for value in channel_coordinates):
        coordinate_by_name = {str(value): value for value in channel_coordinates}
        if set(channels) == set(coordinate_by_name):
            channel_coordinates = tuple(coordinate_by_name[name] for name in channels)

    position_values = _axis_values(coordinates, "position")
    if position_values:
        positions = []
        for position in position_values:
            if not isinstance(position, (int, np.integer)) or int(position) < 0:
                raise ValueError(
                    "Position coordinates must be non-negative integers; "
                    f"found {position!r}."
                )
            positions.append(int(position))
        normalized_positions: tuple[int | None, ...] = tuple(positions)
    else:
        normalized_positions = (None,)

    raw_z_values = _axis_values(coordinates, "z")
    if raw_z_values:
        z_values = []
        for z_value in raw_z_values:
            if not isinstance(z_value, (int, np.integer)):
                raise ValueError(f"Z coordinates must be integers; found {z_value!r}.")
            z_values.append(int(z_value))
        normalized_z_values: tuple[int | None, ...] = tuple(z_values)
    else:
        normalized_z_values = (None,)

    fixed_coordinates: dict[str, int | str] = {}
    axis_names = {axis for coordinate in coordinates for axis in coordinate}
    for axis in sorted(axis_names - {"channel", "position", "z"}):
        values = _axis_values(coordinates, axis)
        if len(values) != 1:
            raise ValueError(
                f"Axis {axis!r} has {len(values)} values. This script only supports "
                "singleton axes other than channel, position, and z."
            )
        fixed_coordinates[axis] = values[0]

    return DatasetLayout(
        channels=channels,
        channel_coordinates=channel_coordinates,
        positions=normalized_positions,
        z_values=normalized_z_values,
        fixed_coordinates=fixed_coordinates,
    )


def camera_parameters(metadata: dict[str, Any]) -> tuple[str, float, float]:
    """Return camera name, electrons per ADU, and offset from image metadata."""
    camera_id = metadata.get("Camera-CameraName")
    alternate_camera_id = metadata.get("Core-Camera")

    if camera_id == "C13440-20CU" or alternate_camera_id == "C13440-20CU":
        try:
            gain = float(metadata["Camera-CONVERSION FACTOR COEFF"])
            offset = float(metadata["Camera-CONVERSION FACTOR OFFSET"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "ORCA-Flash4.0 V3 metadata is missing a valid conversion factor "
                "coefficient or offset."
            ) from error
        return "orcav3", gain, offset

    if (
        camera_id == "Blackfly S BFS-U3-200S6M"
        or alternate_camera_id == "Blackfly S BFS-U3-200S6M"
    ):
        return "flir", 0.03, 0.0

    raise ValueError(
        "Unsupported camera metadata: "
        f"Camera-CameraName={camera_id!r}, Core-Camera={alternate_camera_id!r}."
    )


def voxel_size_zyx_um(
    dataset: Any,
    *,
    layout: DatasetLayout,
) -> tuple[float, float, float]:
    """Derive uniform ZYX voxel spacing in micrometers from NDTiff metadata."""
    if len(layout.z_values) < 2:
        raise ValueError(
            "At least two Z planes are required to infer PhysicalSizeZ from NDTiff "
            "metadata."
        )

    first_coordinates = _image_coordinates(
        channel_coordinate=layout.channel_coordinates[0],
        position=layout.positions[0],
        z_value=layout.z_values[0],
        fixed_coordinates=layout.fixed_coordinates,
    )
    first_metadata = dataset.read_metadata(**first_coordinates)
    try:
        pixel_size_yx = float(first_metadata["PixelSizeUm"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "NDTiff image metadata must contain a numeric PixelSizeUm value to "
            "write calibrated OME-TIFFs."
        ) from error
    if not np.isfinite(pixel_size_yx) or pixel_size_yx <= 0:
        raise ValueError(f"Invalid NDTiff PixelSizeUm value: {pixel_size_yx!r}.")

    summary_metadata = getattr(dataset, "summary_metadata", None)
    summary_z_step = (
        summary_metadata.get("z-step_um")
        if isinstance(summary_metadata, dict)
        else None
    )
    try:
        z_size_um = float(summary_z_step)
    except (TypeError, ValueError):
        z_size_um = _z_spacing_from_plane_metadata(dataset, layout)
    if not np.isfinite(z_size_um) or z_size_um <= 0:
        raise ValueError(f"Invalid NDTiff axial pixel size: {z_size_um!r}.")

    return z_size_um, pixel_size_yx, pixel_size_yx


def _z_spacing_from_plane_metadata(dataset: Any, layout: DatasetLayout) -> float:
    """Calculate uniform Z spacing from intended plane positions."""
    intended_z_positions = []
    for z_value in layout.z_values:
        coordinates = _image_coordinates(
            channel_coordinate=layout.channel_coordinates[0],
            position=layout.positions[0],
            z_value=z_value,
            fixed_coordinates=layout.fixed_coordinates,
        )
        metadata = dataset.read_metadata(**coordinates)
        try:
            intended_z = float(metadata["ZPosition_um_Intended"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "NDTiff metadata must contain either a numeric summary z-step_um "
                "or per-plane ZPosition_um_Intended values."
            ) from error
        intended_z_positions.append(intended_z)

    z_steps = np.abs(np.diff(np.asarray(intended_z_positions, dtype=np.float64)))
    if np.any(~np.isfinite(z_steps)) or np.any(z_steps <= 0):
        raise ValueError(
            "NDTiff intended Z positions do not define a positive axial pixel size: "
            f"{intended_z_positions!r}."
        )
    if not np.allclose(z_steps, z_steps[0], rtol=1e-4, atol=1e-6):
        raise ValueError(
            "NDTiff intended Z positions are not uniformly spaced: "
            f"{intended_z_positions!r}."
        )
    return float(z_steps[0])


def _optical_channel_index(channel_name: str, fallback_index: int) -> int:
    """Map a metadata channel name onto qi2lab blue, yellow, or red optics."""
    normalized_name = channel_name.casefold()
    channel_tokens = (
        (0, ("blue", "488")),
        (1, ("yellow", "561", "565")),
        (2, ("red", "635", "647")),
    )
    for optical_index, tokens in channel_tokens:
        if any(token in normalized_name for token in tokens):
            return optical_index
    if fallback_index < len(QI2LAB_EMISSION_WAVELENGTHS_UM):
        return fallback_index
    raise ValueError(
        f"Cannot assign qi2lab optical parameters to channel {channel_name!r}. "
        "Only blue, yellow, and red microscope channels are supported."
    )


def generate_channel_psfs(
    channels: Sequence[str],
    *,
    z_depth: int,
    voxel_size_zyx_um: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    """Generate normalized channel PSFs using qi2lab microscope parameters."""
    channel_psfs = {}
    for fallback_index, channel in enumerate(channels):
        optical_index = _optical_channel_index(channel, fallback_index)
        emission_wavelength_um = QI2LAB_EMISSION_WAVELENGTHS_UM[optical_index]
        channel_psfs[channel] = generate_qi2lab_psf(
            z_depth=z_depth,
            voxel_size_zyx_um=voxel_size_zyx_um,
            emission_wavelength_um=emission_wavelength_um,
        )
    return channel_psfs


def deconvolve_stack(
    image_zyx: np.ndarray,
    psf_zyx: np.ndarray,
    *,
    gpu_id: int,
) -> np.ndarray:
    """Run repository-standard chunked RLGC and return a uint16 stack."""
    from merfish3danalysis.utils.rlgc import chunked_rlgc

    deconvolved = chunked_rlgc(
        image=image_zyx,
        psf=psf_zyx,
        gpu_id=gpu_id,
        crop_yx=QI2LAB_DECON_CROP_YX,
        crop_z=None,
        release_memory=True,
    )
    return np.asarray(deconvolved).clip(0, np.iinfo(np.uint16).max).astype(np.uint16)


def correct_camera_image(
    raw_image: np.ndarray, *, gain: float, offset: float
) -> np.ndarray:
    """Apply camera offset and gain and return a clipped uint16 image."""
    corrected = (np.asarray(raw_image, dtype=np.float32) - offset) * gain
    return corrected.clip(0, np.iinfo(np.uint16).max).astype(np.uint16)


def _image_coordinates(
    *,
    channel_coordinate: int | str,
    position: int | None,
    z_value: int | None,
    fixed_coordinates: dict[str, int | str],
) -> dict[str, int | str]:
    """Build ndstorage keyword coordinates for one image plane."""
    coordinates = dict(fixed_coordinates)
    coordinates["channel"] = channel_coordinate
    if position is not None:
        coordinates["position"] = position
    if z_value is not None:
        coordinates["z"] = z_value
    return coordinates


def read_corrected_stack(
    dataset: Any,
    *,
    channel: str,
    channel_coordinate: int | str,
    position: int | None,
    z_values: Sequence[int | None],
    fixed_coordinates: dict[str, int | str],
    gain: float,
    offset: float,
) -> np.ndarray:
    """Read and camera-correct one channel as a ZYX volume."""
    planes = []
    expected_shape: tuple[int, int] | None = None
    for z_value in z_values:
        coordinates = _image_coordinates(
            channel_coordinate=channel_coordinate,
            position=position,
            z_value=z_value,
            fixed_coordinates=fixed_coordinates,
        )
        if not dataset.has_image(**coordinates):
            raise ValueError(
                f"Missing image for channel={channel!r}, position={position!r}, "
                f"z={z_value!r}."
            )
        plane = np.asarray(dataset.read_image(**coordinates))
        if plane.ndim != 2:
            raise ValueError(
                f"Expected a 2D image plane for {coordinates!r}; got shape "
                f"{plane.shape!r}."
            )
        if expected_shape is None:
            expected_shape = plane.shape
        elif plane.shape != expected_shape:
            raise ValueError(
                f"Image shape changed within channel {channel!r} at position "
                f"{position!r}: expected {expected_shape!r}, got {plane.shape!r}."
            )
        planes.append(correct_camera_image(plane, gain=gain, offset=offset))
    return np.stack(planes, axis=0)


def _validate_channel_filenames(channels: Sequence[str]) -> None:
    """Ensure metadata channel names can be used unchanged as TIFF stems."""
    invalid = [
        channel
        for channel in channels
        if not channel
        or channel in {".", ".."}
        or Path(channel).name != channel
        or "\x00" in channel
    ]
    if invalid:
        raise ValueError(
            "Channel names must be safe single filename components; invalid names: "
            f"{invalid!r}."
        )


def _planned_output_paths(
    output_dir: Path,
    layout: DatasetLayout,
    ufish_channels: set[str],
    *,
    deconvolve: bool,
) -> list[Path]:
    """Return every TIFF path that the processing run will write."""
    paths = []
    for position_index, position in enumerate(layout.positions):
        output_position = position_index if position is None else position
        position_dir = output_dir / f"pos{output_position:04d}"
        for channel in layout.channels:
            paths.append(position_dir / f"{channel}.ome.tif")
            if deconvolve:
                paths.append(position_dir / f"{channel}_decon.ome.tif")
            if channel in ufish_channels:
                paths.append(position_dir / f"{channel}_ufish.ome.tif")
    if len(paths) != len(set(paths)):
        duplicates = [str(path) for path, count in _counts(paths).items() if count > 1]
        raise ValueError(
            "Channel names produce colliding output filenames: " + ", ".join(duplicates)
        )
    return paths


def _counts(paths: Sequence[Path]) -> dict[Path, int]:
    """Count paths while retaining their first-seen order."""
    counts: dict[Path, int] = defaultdict(int)
    for path in paths:
        counts[path] += 1
    return dict(counts)


def _write_stack(
    path: Path,
    stack: np.ndarray,
    *,
    voxel_size_zyx_um: tuple[float, float, float],
    channel_name: str,
) -> None:
    """Write one uncompressed, calibrated ZYX OME-TIFF stack."""
    z_size_um, y_size_um, x_size_um = voxel_size_zyx_um
    stack_array = np.asarray(stack)
    imwrite(
        path,
        stack_array,
        ome=True,
        photometric="minisblack",
        resolution=(1e4 / x_size_um, 1e4 / y_size_um),
        resolutionunit="CENTIMETER",
        metadata={
            "axes": "ZYX",
            "SignificantBits": int(stack_array.dtype.itemsize * 8),
            "PhysicalSizeX": x_size_um,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": y_size_um,
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZ": z_size_um,
            "PhysicalSizeZUnit": "µm",
            "Channel": {"Name": [channel_name]},
        },
    )


def load_default_ufish(gpu_id: int) -> Any:
    """Load the repository's default simfish U-FISH model."""
    from ufish.api import UFish

    from merfish3danalysis.utils.chromatic import _load_ufish_model

    ufish = UFish(device=f"cuda:{gpu_id}")
    _load_ufish_model(ufish, None)
    return ufish


def process_dataset(
    dataset: Any,
    *,
    output_dir: Path,
    ufish_channels: Sequence[str],
    z_start: int,
    z_stop: int,
    overwrite: bool,
    ufish: Any | None,
    gpu_id: int = 0,
    deconvolve: bool = False,
) -> None:
    """Process all positions and channels from an open NDTiff dataset."""
    layout = inspect_dataset(dataset)
    _validate_channel_filenames(layout.channels)

    requested_ufish_channels = tuple(dict.fromkeys(ufish_channels))
    missing_channels = sorted(set(requested_ufish_channels) - set(layout.channels))
    if missing_channels:
        raise ValueError(
            "Requested U-FISH channel names are absent from NDTiff metadata: "
            f"{missing_channels!r}. Available channels: {list(layout.channels)!r}."
        )
    effective_z_stop = len(layout.z_values) if z_stop == -1 else z_stop
    if z_start < 0 or effective_z_stop <= z_start:
        raise ValueError(
            "The Z range must satisfy 0 <= z_start < z_stop, with -1 allowed "
            "as an end-of-stack z_stop; "
            f"received [{z_start}, {z_stop})."
        )
    if effective_z_stop > len(layout.z_values):
        raise ValueError(
            f"Z stop {effective_z_stop} exceeds the dataset depth "
            f"{len(layout.z_values)}."
        )
    selected_z_values = layout.z_values[z_start:effective_z_stop]

    ufish_channel_set = set(requested_ufish_channels)
    planned_paths = _planned_output_paths(
        output_dir,
        layout,
        ufish_channel_set,
        deconvolve=deconvolve,
    )
    existing_paths = [path for path in planned_paths if path.exists()]
    if existing_paths and not overwrite:
        preview = ", ".join(str(path) for path in existing_paths[:5])
        suffix = " ..." if len(existing_paths) > 5 else ""
        raise FileExistsError(
            f"Refusing to overwrite {len(existing_paths)} existing output file(s): "
            f"{preview}{suffix}. Pass --overwrite to replace them."
        )

    first_coordinates = _image_coordinates(
        channel_coordinate=layout.channel_coordinates[0],
        position=layout.positions[0],
        z_value=layout.z_values[0],
        fixed_coordinates=layout.fixed_coordinates,
    )
    metadata = dataset.read_metadata(**first_coordinates)
    camera, gain, offset = camera_parameters(metadata)
    voxel_spacing = voxel_size_zyx_um(
        dataset,
        layout=layout,
    )
    typer.echo(
        f"Camera: {camera}; gain: {gain:g} electrons/ADU; offset: {offset:g} ADU"
    )
    typer.echo(
        "Voxel size (Z, Y, X): " + ", ".join(f"{value:g} µm" for value in voxel_spacing)
    )
    if ufish_channel_set and ufish is None:
        ufish = load_default_ufish(gpu_id)
    channel_psfs = (
        generate_channel_psfs(
            layout.channels,
            z_depth=len(selected_z_values),
            voxel_size_zyx_um=voxel_spacing,
        )
        if deconvolve
        else {}
    )

    for position_index, position in enumerate(layout.positions):
        output_position = position_index if position is None else position
        position_dir = output_dir / f"pos{output_position:04d}"
        position_dir.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Processing {position_dir.name}")
        for channel, channel_coordinate in zip(
            layout.channels,
            layout.channel_coordinates,
            strict=True,
        ):
            corrected_stack = read_corrected_stack(
                dataset,
                channel=channel,
                channel_coordinate=channel_coordinate,
                position=position,
                z_values=selected_z_values,
                fixed_coordinates=layout.fixed_coordinates,
                gain=gain,
                offset=offset,
            )
            _write_stack(
                position_dir / f"{channel}.ome.tif",
                corrected_stack,
                voxel_size_zyx_um=voxel_spacing,
                channel_name=channel,
            )
            predictor_input_stack = corrected_stack
            if deconvolve:
                deconvolved_stack = deconvolve_stack(
                    corrected_stack,
                    channel_psfs[channel],
                    gpu_id=gpu_id,
                )
                _write_stack(
                    position_dir / f"{channel}_decon.ome.tif",
                    deconvolved_stack,
                    voxel_size_zyx_um=voxel_spacing,
                    channel_name=channel,
                )
                predictor_input_stack = deconvolved_stack
            if channel in ufish_channel_set:
                if ufish is None:
                    raise RuntimeError("U-FISH was not initialized.")
                _locations, prediction = ufish.predict(
                    predictor_input_stack,
                    axes="zyx",
                    blend_3d=False,
                    batch_size=1,
                )
                prediction_stack = np.asarray(prediction).astype(
                    np.float32,
                    copy=False,
                )
                if prediction_stack.shape != predictor_input_stack.shape:
                    raise ValueError(
                        f"U-FISH returned shape {prediction_stack.shape!r} for channel "
                        f"{channel!r}; expected {predictor_input_stack.shape!r}."
                    )
                _write_stack(
                    position_dir / f"{channel}_ufish.ome.tif",
                    prediction_stack,
                    voxel_size_zyx_um=voxel_spacing,
                    channel_name=channel,
                )


@app.command()
def process(
    dataset_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="NDTiff acquisition directory containing NDTiff.index.",
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Root directory for pos#### output folders.",
        ),
    ] = None,
    ufish_channels: Annotated[
        list[str] | None,
        typer.Option(
            "--ufish-channel",
            help=(
                "Exact metadata channel name to run through U-FISH. Repeat as needed."
            ),
        ),
    ] = None,
    z_start: Annotated[
        int,
        typer.Option(
            "--z-start",
            min=0,
            help="Inclusive zero-based Z-plane offset. Defaults to 0.",
        ),
    ] = 0,
    z_stop: Annotated[
        int,
        typer.Option(
            "--z-stop",
            help="Exclusive zero-based Z-plane offset. Use -1 for the stack end.",
        ),
    ] = -1,
    gpu_id: Annotated[
        int,
        typer.Option(
            "--gpu-id",
            min=0,
            help="CUDA device ID used by U-FISH.",
        ),
    ] = 0,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Replace existing corrected and U-FISH TIFF files.",
        ),
    ] = False,
    deconvolve: Annotated[
        bool,
        typer.Option(
            "--deconvolve",
            help=(
                "Run RLGC for every channel, save *_decon.ome.tif, and use the "
                "deconvolved stack for U-FISH."
            ),
        ),
    ] = False,
    list_channels: Annotated[
        bool,
        typer.Option(
            "--list-channels",
            help="Print channel names parsed from NDTiff metadata and exit.",
        ),
    ] = False,
) -> None:
    """Split an NDTiff acquisition into corrected, per-position TIFF stacks."""
    dataset = _load_dataset(dataset_path)
    try:
        layout = inspect_dataset(dataset)
        if list_channels:
            typer.echo("Channels parsed from NDTiff metadata:")
            for channel in layout.channels:
                typer.echo(f"- {channel}")
            return

        if output_dir is None:
            raise typer.BadParameter(
                "This option is required unless --list-channels is used.",
                param_hint="--output-dir",
            )
        if output_dir == dataset_path:
            raise typer.BadParameter(
                "The output directory must differ from the NDTiff input directory.",
                param_hint="--output-dir",
            )
        if not ufish_channels:
            raise typer.BadParameter(
                "At least one channel is required unless --list-channels is used.",
                param_hint="--ufish-channel",
            )
        missing_channels = sorted(set(ufish_channels) - set(layout.channels))
        if missing_channels:
            raise typer.BadParameter(
                f"Unknown channel(s) {missing_channels!r}; available metadata names "
                f"are {list(layout.channels)!r}.",
                param_hint="--ufish-channel",
            )
        process_dataset(
            dataset,
            output_dir=output_dir,
            ufish_channels=ufish_channels,
            z_start=z_start,
            z_stop=z_stop,
            overwrite=overwrite,
            ufish=None,
            gpu_id=gpu_id,
            deconvolve=deconvolve,
        )
    finally:
        dataset.close()

    typer.echo(f"Finished writing position folders to {output_dir}")


def main() -> None:
    """Run the command-line interface."""
    app(args=_expand_ufish_channel_arguments(sys.argv[1:]))


if __name__ == "__main__":
    main()
