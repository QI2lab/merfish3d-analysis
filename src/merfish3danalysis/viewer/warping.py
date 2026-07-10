"""Local image loading and warp-chain helpers for the viewer."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from merfish3danalysis.utils.decode_warping import (
    compose_optional_warp_transform_zyx_um,
    load_bit_round_transform_zyx_um,
    warp_image_to_reference_frame,
)
from merfish3danalysis.viewer.models import (
    ChannelStack,
    ViewerDisplay,
    WarpChainOptions,
    warp_chain_label,
)


@dataclass(frozen=True)
class _PreparedWarp:
    """Prepared affine and optional SOFIMA data for one local image."""

    transform_zyx_um: np.ndarray
    loaded_flow_field: Any


def _as_zyx(image: Any) -> np.ndarray:
    """
    Convert a loaded image to a 3D zyx NumPy array.

    Parameters
    ----------
    image : Any
        Loaded image, future, or tensorstore array.

    Returns
    -------
    numpy.ndarray
        Image in Z, Y, X order.
    """
    array = np.asarray(_resolve_array(image))
    array = np.squeeze(array)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    if array.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D image, got shape {array.shape}.")
    return array


def _resolve_array(array_like: Any) -> Any:
    """
    Resolve datastore futures or tensorstore slices into array-like data.

    Parameters
    ----------
    array_like : Any
        Array, future, or tensorstore slice.

    Returns
    -------
    Any
        Resolved array-like data.
    """
    if hasattr(array_like, "result"):
        array_like = array_like.result()
    if hasattr(array_like, "read"):
        array_like = array_like.read().result()
    return array_like


def _resolve_loaded_flow_field(loaded_flow_field: Any) -> Any:
    """
    Resolve a datastore-loaded SOFIMA flow field tuple when present.

    Parameters
    ----------
    loaded_flow_field : Any
        SOFIMA flow field tuple or ``None``.

    Returns
    -------
    Any
        Flow field tuple with resolved array data, or ``None``.
    """
    if loaded_flow_field is None:
        return None
    flow_field, attributes = loaded_flow_field
    return _resolve_array(flow_field), attributes


def selected_warp_label(tile: str, local_id: str, options: WarpChainOptions) -> str:
    """
    Return a channel label for a selected local warp-chain display.

    Parameters
    ----------
    tile : str
        Tile identifier.
    local_id : str
        Bit or fiducial round identifier.
    options : WarpChainOptions
        Selected transform-chain components.

    Returns
    -------
    str
        Human-readable channel label.
    """
    chain_label = warp_chain_label(options)
    mode = "native" if chain_label == "native" else f"warped {chain_label}"
    return f"{tile}:{local_id} {mode}"


def _load_bit_emission_wavelength_um(
    datastore: Any,
    *,
    tile: str,
    bit_id: str,
) -> float:
    """
    Load the emission wavelength for one bit.

    Parameters
    ----------
    datastore : Any
        qi2lab datastore-like object.
    tile : str
        Tile identifier.
    bit_id : str
        Readout bit identifier.

    Returns
    -------
    float
        Emission wavelength in microns.
    """
    wavelengths = datastore.load_local_wavelengths_um(tile=tile, bit=bit_id)
    if wavelengths is None:
        raise RuntimeError(f"Missing wavelength metadata for tile={tile} bit={bit_id}.")
    return float(wavelengths[1])


def _native_after_warp_failure(
    image: np.ndarray,
    *,
    label: str,
    warnings: list[str],
    exc: Exception,
) -> np.ndarray:
    """
    Return native image data after recording a viewer warp warning.

    Parameters
    ----------
    image : numpy.ndarray
        Native image.
    label : str
        Channel label used in the warning.
    warnings : list[str]
        Warning list updated in place.
    exc : Exception
        Warp failure exception.

    Returns
    -------
    numpy.ndarray
        Native image as float32.
    """
    warnings.append(f"{label} displayed native; warp failed: {exc}")
    return np.asarray(image, dtype=np.float32)


def _load_sofima_flow_field(
    datastore: Any,
    *,
    tile: str,
    round_id: str | None,
    enabled: bool,
) -> Any:
    """
    Load a SOFIMA flow field only when requested and available.

    Parameters
    ----------
    datastore : Any
        qi2lab datastore-like object.
    tile : str
        Tile identifier.
    round_id : str or None
        Moving fiducial round identifier.
    enabled : bool
        Whether SOFIMA should be used.

    Returns
    -------
    Any
        Resolved flow field tuple, or ``None``.
    """
    if not enabled or round_id is None:
        return None
    return _resolve_loaded_flow_field(
        datastore.load_local_sofima_flow_field(
            tile=tile,
            round=round_id,
            return_future=True,
        )
    )


def _prepare_bit_warp(
    datastore: Any,
    *,
    tile: str,
    bit_id: str,
    options: WarpChainOptions,
) -> _PreparedWarp:
    """
    Prepare selected affine and SOFIMA components for one readout bit.

    Parameters
    ----------
    datastore : Any
        qi2lab datastore-like object.
    tile : str
        Tile identifier.
    bit_id : str
        Readout bit identifier.
    options : WarpChainOptions
        Selected transform-chain components.

    Returns
    -------
    _PreparedWarp
        Prepared affine and optional flow field.
    """
    round_id, stored_round_transform_zyx_um = load_bit_round_transform_zyx_um(
        datastore,
        tile=tile,
        bit_id=bit_id,
    )
    round_transform_zyx_um = (
        stored_round_transform_zyx_um if options.stage_affine else None
    )
    chromatic_transform_zyx_um = None
    if options.chromatic:
        emission_wavelength_um = _load_bit_emission_wavelength_um(
            datastore,
            tile=tile,
            bit_id=bit_id,
        )
        chromatic_transform_zyx_um = datastore.load_chromatic_affine_transform_zyx_um(
            wavelength_um=emission_wavelength_um,
        )
    return _PreparedWarp(
        transform_zyx_um=compose_optional_warp_transform_zyx_um(
            round_transform_zyx_um=round_transform_zyx_um,
            chromatic_transform_zyx_um=chromatic_transform_zyx_um,
        ),
        loaded_flow_field=_load_sofima_flow_field(
            datastore,
            tile=tile,
            round_id=round_id,
            enabled=options.sofima,
        ),
    )


def _fiducial_moving_round(datastore: Any, round_id: str) -> str | None:
    """
    Return the moving round, or None for the reference fiducial round.

    Parameters
    ----------
    datastore : Any
        qi2lab datastore-like object.
    round_id : str
        Fiducial round identifier.

    Returns
    -------
    str or None
        Moving round identifier, or ``None`` for reference round.
    """
    fiducial_round_ids = list(datastore.round_ids or [])
    reference_round_id = fiducial_round_ids[0] if fiducial_round_ids else None
    return round_id if round_id != reference_round_id else None


def _load_fiducial_round_transform(
    datastore: Any,
    *,
    tile: str,
    round_id: str,
    moving_round: str | None,
    enabled: bool,
) -> np.ndarray | None:
    """
    Load a fiducial round transform only when requested and needed.

    Parameters
    ----------
    datastore : Any
        qi2lab datastore-like object.
    tile : str
        Tile identifier.
    round_id : str
        Fiducial round identifier.
    moving_round : str or None
        Moving fiducial round identifier.
    enabled : bool
        Whether the affine transform should be loaded.

    Returns
    -------
    numpy.ndarray or None
        Round transform, or ``None`` when not needed.
    """
    if not enabled or moving_round is None:
        return None
    round_transform_zyx_um = datastore.load_local_round_transform_zyx_um(
        tile=tile,
        round=round_id,
    )
    if round_transform_zyx_um is None:
        raise RuntimeError(
            f"Missing local round transform for tile={tile} round={round_id}."
        )
    return round_transform_zyx_um


def _prepare_fiducial_warp(
    datastore: Any,
    *,
    tile: str,
    round_id: str,
    options: WarpChainOptions,
) -> _PreparedWarp:
    """
    Prepare selected affine and SOFIMA components for one fiducial round.

    Parameters
    ----------
    datastore : Any
        qi2lab datastore-like object.
    tile : str
        Tile identifier.
    round_id : str
        Fiducial round identifier.
    options : WarpChainOptions
        Selected transform-chain components.

    Returns
    -------
    _PreparedWarp
        Prepared affine and optional flow field.
    """
    moving_round = _fiducial_moving_round(datastore, round_id)
    round_transform_zyx_um = _load_fiducial_round_transform(
        datastore,
        tile=tile,
        round_id=round_id,
        moving_round=moving_round,
        enabled=options.stage_affine,
    )
    return _PreparedWarp(
        transform_zyx_um=compose_optional_warp_transform_zyx_um(
            round_transform_zyx_um=round_transform_zyx_um,
            chromatic_transform_zyx_um=None,
        ),
        loaded_flow_field=_load_sofima_flow_field(
            datastore,
            tile=tile,
            round_id=round_id if moving_round is not None else None,
            enabled=options.sofima,
        ),
    )


def _apply_prepared_warp(
    image: np.ndarray,
    prepared_warp: _PreparedWarp,
    *,
    spacing_zyx_um: Any,
    gpu_id: int,
) -> np.ndarray:
    """
    Apply a prepared local viewer warp to one image.

    Parameters
    ----------
    image : numpy.ndarray
        Native image in Z, Y, X order.
    prepared_warp : _PreparedWarp
        Prepared affine and optional flow field.
    spacing_zyx_um : Any
        Z, Y, X voxel spacing in microns.
    gpu_id : int
        CUDA device ID.

    Returns
    -------
    numpy.ndarray
        Warped image.
    """
    return warp_image_to_reference_frame(
        image,
        transform_zyx_um=prepared_warp.transform_zyx_um,
        spacing_zyx_um=spacing_zyx_um,
        loaded_flow_field=prepared_warp.loaded_flow_field,
        gpu_id=gpu_id,
    )


def _warp_bit_for_viewer(
    image: np.ndarray,
    *,
    datastore: Any,
    tile: str,
    bit_id: str,
    options: WarpChainOptions,
    gpu_id: int,
    warnings: list[str],
) -> np.ndarray:
    """
    Warp one bit image for local viewer display.

    Parameters
    ----------
    image : numpy.ndarray
        Native bit image.
    datastore : Any
        qi2lab datastore-like object.
    tile : str
        Tile identifier.
    bit_id : str
        Readout bit identifier.
    options : WarpChainOptions
        Selected transform-chain components.
    gpu_id : int
        CUDA device ID.
    warnings : list[str]
        Warning list updated in place.

    Returns
    -------
    numpy.ndarray
        Native or warped image.
    """
    if options.is_native:
        return np.asarray(image, dtype=np.float32)

    try:
        prepared_warp = _prepare_bit_warp(
            datastore,
            tile=tile,
            bit_id=bit_id,
            options=options,
        )
        return _apply_prepared_warp(
            image,
            prepared_warp,
            spacing_zyx_um=datastore.voxel_size_zyx_um,
            gpu_id=gpu_id,
        )
    except Exception as exc:
        return _native_after_warp_failure(
            image,
            label=f"{tile}:{bit_id}",
            warnings=warnings,
            exc=exc,
        )


def _warp_fiducial_for_viewer(
    image: np.ndarray,
    *,
    datastore: Any,
    tile: str,
    round_id: str,
    options: WarpChainOptions,
    gpu_id: int,
    warnings: list[str],
) -> np.ndarray:
    """
    Warp one fiducial image for local viewer display.

    Parameters
    ----------
    image : numpy.ndarray
        Native fiducial image.
    datastore : Any
        qi2lab datastore-like object.
    tile : str
        Tile identifier.
    round_id : str
        Fiducial round identifier.
    options : WarpChainOptions
        Selected transform-chain components. Chromatic is ignored.
    gpu_id : int
        CUDA device ID.
    warnings : list[str]
        Warning list updated in place.

    Returns
    -------
    numpy.ndarray
        Native or warped image.
    """
    if options.is_native:
        return np.asarray(image, dtype=np.float32)

    try:
        prepared_warp = _prepare_fiducial_warp(
            datastore,
            tile=tile,
            round_id=round_id,
            options=options,
        )
        return _apply_prepared_warp(
            image,
            prepared_warp,
            spacing_zyx_um=datastore.voxel_size_zyx_um,
            gpu_id=gpu_id,
        )
    except Exception as exc:
        return _native_after_warp_failure(
            image,
            label=f"{tile}:{round_id}",
            warnings=warnings,
            exc=exc,
        )


def _append_local_viewer_channel(
    *,
    channels: list[np.ndarray],
    labels: list[str],
    warnings: list[str],
    datastore: Any,
    tile: str,
    local_id: str,
    channel_kind: str,
    image: Any,
    label: str,
    options: WarpChainOptions,
    gpu_id: int,
) -> None:
    """
    Append one selected local viewer image channel.

    Parameters
    ----------
    channels : list[numpy.ndarray]
        Destination image channel list.
    labels : list[str]
        Destination channel label list.
    warnings : list[str]
        Warning list updated in place.
    datastore : Any
        qi2lab datastore-like object.
    tile : str
        Tile identifier.
    local_id : str
        Bit or fiducial round identifier.
    channel_kind : str
        Either ``"fiducial"`` or ``"bit"``.
    image : Any
        Loaded image data.
    label : str
        Display channel label.
    options : WarpChainOptions
        Selected transform-chain components.
    gpu_id : int
        CUDA device ID.
    """
    if image is None:
        return
    image_zyx = _as_zyx(image)
    if channel_kind == "fiducial":
        warped = _warp_fiducial_for_viewer(
            image_zyx,
            datastore=datastore,
            tile=tile,
            round_id=local_id,
            options=options,
            gpu_id=gpu_id,
            warnings=warnings,
        )
    elif channel_kind == "bit":
        warped = _warp_bit_for_viewer(
            image_zyx,
            datastore=datastore,
            tile=tile,
            bit_id=local_id,
            options=options,
            gpu_id=gpu_id,
            warnings=warnings,
        )
    else:
        return
    channels.append(warped)
    labels.append(label)


def _load_fiducial_source_image(
    datastore: Any,
    *,
    tile: str,
    round_id: str,
    source: str,
) -> tuple[Any, str] | None:
    """
    Load one fiducial source image and return its display suffix.

    Parameters
    ----------
    datastore : Any
        qi2lab datastore-like object.
    tile : str
        Tile identifier.
    round_id : str
        Fiducial round identifier.
    source : str
        Fiducial source name.

    Returns
    -------
    tuple[Any, str] or None
        Loaded image and label suffix, or ``None`` for unknown source.
    """
    if source == "corrected":
        return (
            datastore.load_local_corrected_image(
                **{"tile": tile, "round": round_id, "return_future": True}
            ),
            "fiducial corrected",
        )
    if source == "registered":
        return (
            datastore.load_local_registered_image(
                **{"tile": tile, "round": round_id, "return_future": True}
            ),
            "fiducial registered/decon",
        )
    return None


def _load_bit_source_image(
    datastore: Any,
    *,
    tile: str,
    bit_id: str,
    source: str,
) -> tuple[Any, str] | None:
    """
    Load one readout bit source image and return its display suffix.

    Parameters
    ----------
    datastore : Any
        qi2lab datastore-like object.
    tile : str
        Tile identifier.
    bit_id : str
        Readout bit identifier.
    source : str
        Bit source name.

    Returns
    -------
    tuple[Any, str] or None
        Loaded image and label suffix, or ``None`` for unknown source.
    """
    if source == "corrected":
        return (
            datastore.load_local_corrected_image(
                tile=tile, bit=bit_id, return_future=True
            ),
            "corrected",
        )
    if source == "registered":
        return (
            datastore.load_local_registered_image(
                tile=tile, bit=bit_id, return_future=True
            ),
            "registered/decon",
        )
    if source == "feature":
        return (
            datastore.load_local_feature_predictor_image(
                tile=tile, bit=bit_id, return_future=True
            ),
            "feature predictor",
        )
    return None


def _append_source_channel(
    *,
    channels: list[np.ndarray],
    labels: list[str],
    warnings: list[str],
    datastore: Any,
    tile: str,
    local_id: str,
    source: str,
    channel_kind: str,
    options: WarpChainOptions,
    gpu_id: int,
) -> None:
    """
    Load and append one selected fiducial or readout bit source.

    Parameters
    ----------
    channels : list[numpy.ndarray]
        Destination image channel list.
    labels : list[str]
        Destination channel label list.
    warnings : list[str]
        Warning list updated in place.
    datastore : Any
        qi2lab datastore-like object.
    tile : str
        Tile identifier.
    local_id : str
        Bit or fiducial round identifier.
    source : str
        Image source name.
    channel_kind : str
        Either ``"fiducial"`` or ``"bit"``.
    options : WarpChainOptions
        Selected transform-chain components.
    gpu_id : int
        CUDA device ID.
    """
    if channel_kind == "fiducial":
        loaded = _load_fiducial_source_image(
            datastore,
            tile=tile,
            round_id=local_id,
            source=source,
        )
    elif channel_kind == "bit":
        loaded = _load_bit_source_image(
            datastore,
            tile=tile,
            bit_id=local_id,
            source=source,
        )
    else:
        return
    if loaded is None:
        return
    image, suffix = loaded
    _append_local_viewer_channel(
        channels=channels,
        labels=labels,
        warnings=warnings,
        datastore=datastore,
        tile=tile,
        local_id=local_id,
        channel_kind=channel_kind,
        image=image,
        label=f"{selected_warp_label(tile, local_id, options)} {suffix}",
        options=options,
        gpu_id=gpu_id,
    )


def _append_fiducial_channels(
    *,
    channels: list[np.ndarray],
    labels: list[str],
    warnings: list[str],
    datastore: Any,
    tile: str,
    round_ids: list[str],
    sources: list[str],
    options: WarpChainOptions,
    gpu_id: int,
) -> None:
    """
    Append selected fiducial channels.

    Parameters
    ----------
    channels : list[numpy.ndarray]
        Destination image channel list.
    labels : list[str]
        Destination channel label list.
    warnings : list[str]
        Warning list updated in place.
    datastore : Any
        qi2lab datastore-like object.
    tile : str
        Tile identifier.
    round_ids : list[str]
        Selected fiducial round identifiers.
    sources : list[str]
        Selected fiducial source names.
    options : WarpChainOptions
        Selected transform-chain components.
    gpu_id : int
        CUDA device ID.
    """
    for round_id in round_ids:
        for source in sources:
            _append_source_channel(
                channels=channels,
                labels=labels,
                warnings=warnings,
                datastore=datastore,
                tile=tile,
                local_id=round_id,
                source=source,
                channel_kind="fiducial",
                options=options,
                gpu_id=gpu_id,
            )


def _append_bit_channels(
    *,
    channels: list[np.ndarray],
    labels: list[str],
    warnings: list[str],
    datastore: Any,
    tile: str,
    bit_ids: list[str],
    sources: list[str],
    options: WarpChainOptions,
    gpu_id: int,
) -> None:
    """
    Append selected readout bit channels.

    Parameters
    ----------
    channels : list[numpy.ndarray]
        Destination image channel list.
    labels : list[str]
        Destination channel label list.
    warnings : list[str]
        Warning list updated in place.
    datastore : Any
        qi2lab datastore-like object.
    tile : str
        Tile identifier.
    bit_ids : list[str]
        Selected readout bit identifiers.
    sources : list[str]
        Selected bit source names.
    options : WarpChainOptions
        Selected transform-chain components.
    gpu_id : int
        CUDA device ID.
    """
    for bit_id in bit_ids:
        for source in sources:
            _append_source_channel(
                channels=channels,
                labels=labels,
                warnings=warnings,
                datastore=datastore,
                tile=tile,
                local_id=bit_id,
                source=source,
                channel_kind="bit",
                options=options,
                gpu_id=gpu_id,
            )


def _stack_local_channels(
    channels: list[np.ndarray],
    labels: list[str],
) -> ChannelStack:
    """
    Return selected local channels as a validated channel stack.

    Parameters
    ----------
    channels : list[numpy.ndarray]
        Selected image channels.
    labels : list[str]
        Channel labels.

    Returns
    -------
    ChannelStack
        Validated channel stack.
    """
    if not channels:
        raise ValueError("No selected image channels were available to display.")
    shape = channels[0].shape
    if any(channel.shape != shape for channel in channels):
        raise ValueError("Selected image channels do not have matching shapes.")
    return ChannelStack(data=np.stack(channels, axis=0), labels=labels)


def load_local_warped_image_channels(
    datastore: Any,
    tile: str,
    fiducial_round_ids: list[str],
    fiducial_sources: list[str],
    bit_ids: list[str],
    bit_sources: list[str],
    options: WarpChainOptions,
    gpu_id: int = 0,
) -> ViewerDisplay:
    """
    Load selected local images and apply selected warp-chain components.

    Parameters
    ----------
    datastore : Any
        qi2lab datastore-like object.
    tile : str
        Tile identifier.
    fiducial_round_ids : list[str]
        Fiducial rounds selected for display.
    fiducial_sources : list[str]
        Fiducial image sources.
    bit_ids : list[str]
        Readout bits selected for display.
    bit_sources : list[str]
        Bit image sources.
    options : WarpChainOptions
        Selected transform-chain components.
    gpu_id : int, default=0
        CUDA device ID used for interpolation.

    Returns
    -------
    ViewerDisplay
        Channel stack plus warning messages.
    """
    channels: list[np.ndarray] = []
    labels: list[str] = []
    warnings: list[str] = []

    _append_fiducial_channels(
        channels=channels,
        labels=labels,
        warnings=warnings,
        datastore=datastore,
        tile=tile,
        round_ids=fiducial_round_ids,
        sources=fiducial_sources,
        options=options,
        gpu_id=gpu_id,
    )
    _append_bit_channels(
        channels=channels,
        labels=labels,
        warnings=warnings,
        datastore=datastore,
        tile=tile,
        bit_ids=bit_ids,
        sources=bit_sources,
        options=options,
        gpu_id=gpu_id,
    )

    return ViewerDisplay(
        stack=_stack_local_channels(channels, labels),
        warnings=warnings,
    )
