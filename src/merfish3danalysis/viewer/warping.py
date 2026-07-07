"""Local image loading and warp-chain helpers for the viewer."""

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from merfish3danalysis.utils.decode_warping import (
    load_bit_round_transform_zyx_um,
    warp_image_with_sofima_metadata,
)
from merfish3danalysis.viewer.models import (
    ChannelStack,
    ViewerDisplay,
    WarpChainOptions,
    compose_viewer_warp_transform_zyx_um,
    warp_chain_label,
)


def _as_zyx(image: Any) -> np.ndarray:
    """
    Convert a loaded image to a 3D zyx NumPy array.

    Parameters
    ----------
    image : Any
        image for this viewer operation.

    Returns
    -------
    np.ndarray
        Computed viewer result.
    """

    array = np.asarray(image)
    array = np.squeeze(array)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    if array.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D image, got shape {array.shape}.")
    return array


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


def _warp_image_to_reference_for_viewer(
    image: np.ndarray,
    *,
    transform_zyx_um: np.ndarray,
    spacing_zyx_um: Any,
    loaded_flow_field: tuple[np.ndarray, Mapping[str, Any]] | None,
    gpu_id: int,
) -> np.ndarray:
    """
    Warp a local viewer image with optional SOFIMA metadata.

    Parameters
    ----------
    image : numpy.ndarray
        Native Z, Y, X image.
    transform_zyx_um : numpy.ndarray
        Reference-to-moving physical affine.
    spacing_zyx_um : Any
        Reference image spacing in Z, Y, X microns.
    loaded_flow_field : tuple[numpy.ndarray, Mapping[str, Any]] | None
        Optional SOFIMA flow field and attributes.
    gpu_id : int
        CUDA device ID.

    Returns
    -------
    numpy.ndarray
        Native or warped image as float32.
    """

    if loaded_flow_field is None and np.allclose(
        transform_zyx_um,
        np.eye(4, dtype=np.float32),
    ):
        return np.asarray(image, dtype=np.float32)

    if loaded_flow_field is not None:
        sofima_flow_field, flow_attrs = loaded_flow_field
        return warp_image_with_sofima_metadata(
            image,
            transform_zyx_um=transform_zyx_um,
            spacing_zyx_um=spacing_zyx_um,
            sofima_flow_field_xyz_px=sofima_flow_field,
            flow_attrs=flow_attrs,
            gpu_id=gpu_id,
        ).astype(np.float32, copy=False)

    from merfish3danalysis.utils.multiview_registration import (
        warp_array_to_reference_gpu,
    )

    return warp_array_to_reference_gpu(
        image,
        transform_zyx_um=transform_zyx_um,
        spacing_zyx_um=spacing_zyx_um,
        reference_shape=image.shape,
        gpu_id=gpu_id,
    ).astype(np.float32, copy=False)


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
            chromatic_transform_zyx_um = (
                datastore.load_chromatic_affine_transform_zyx_um(
                    wavelength_um=emission_wavelength_um,
                )
            )
        transform_zyx_um = compose_viewer_warp_transform_zyx_um(
            round_transform_zyx_um=round_transform_zyx_um,
            chromatic_transform_zyx_um=chromatic_transform_zyx_um,
        )
        loaded_flow_field = None
        if options.sofima and round_id is not None:
            loaded_flow_field = datastore.load_local_sofima_flow_field(
                tile=tile,
                round=round_id,
                return_future=False,
            )
        return _warp_image_to_reference_for_viewer(
            image,
            transform_zyx_um=transform_zyx_um,
            spacing_zyx_um=datastore.voxel_size_zyx_um,
            loaded_flow_field=loaded_flow_field,
            gpu_id=gpu_id,
        )
    except Exception as exc:
        warnings.append(f"{tile}:{bit_id} displayed native; warp failed: {exc}")
        return np.asarray(image, dtype=np.float32)


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
        round_ids = list(datastore.round_ids or [])
        reference_round_id = round_ids[0] if round_ids else None
        moving_round = round_id if round_id != reference_round_id else None
        round_transform_zyx_um = None
        if options.stage_affine and moving_round is not None:
            round_transform_zyx_um = datastore.load_local_round_transform_zyx_um(
                tile=tile,
                round=round_id,
            )
            if round_transform_zyx_um is None:
                raise RuntimeError(
                    f"Missing local round transform for tile={tile} round={round_id}."
                )
        loaded_flow_field = None
        if options.sofima and moving_round is not None:
            loaded_flow_field = datastore.load_local_sofima_flow_field(
                tile=tile,
                round=round_id,
                return_future=False,
            )
        transform_zyx_um = compose_viewer_warp_transform_zyx_um(
            round_transform_zyx_um=round_transform_zyx_um,
            chromatic_transform_zyx_um=None,
        )
        return _warp_image_to_reference_for_viewer(
            image,
            transform_zyx_um=transform_zyx_um,
            spacing_zyx_um=datastore.voxel_size_zyx_um,
            loaded_flow_field=loaded_flow_field,
            gpu_id=gpu_id,
        )
    except Exception as exc:
        warnings.append(f"{tile}:{round_id} displayed native; warp failed: {exc}")
        return np.asarray(image, dtype=np.float32)


def _append_warped_viewer_channel(
    *,
    channels: list[np.ndarray],
    labels: list[str],
    image: Any,
    label: str,
    warp_image: Callable[[np.ndarray], np.ndarray],
) -> None:
    """
    Append one local viewer image channel after optional warping.

    Parameters
    ----------
    channels : list[numpy.ndarray]
        Mutable list of Z, Y, X display channels.
    labels : list[str]
        Mutable list of channel labels.
    image : Any
        Loaded image, or None when the datastore source is unavailable.
    label : str
        Channel label to append.
    warp_image : Callable[[numpy.ndarray], numpy.ndarray]
        Callable accepting a Z, Y, X image and returning a display image.

    Returns
    -------
    None
        The channels and labels lists are updated in place.
    """

    if image is None:
        return
    channels.append(warp_image(_as_zyx(image)))
    labels.append(label)


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

    for round_id in fiducial_round_ids:

        def warp_fiducial(image: np.ndarray, round_id: str = round_id) -> np.ndarray:
            return _warp_fiducial_for_viewer(
                image,
                datastore=datastore,
                tile=tile,
                round_id=round_id,
                options=options,
                gpu_id=gpu_id,
                warnings=warnings,
            )

        if "corrected" in fiducial_sources:
            _append_warped_viewer_channel(
                channels=channels,
                labels=labels,
                image=datastore.load_local_corrected_image(
                    tile=tile,
                    round=round_id,
                    return_future=False,
                ),
                label=f"{selected_warp_label(tile, round_id, options)} fiducial corrected",
                warp_image=warp_fiducial,
            )
        if "registered" in fiducial_sources:
            _append_warped_viewer_channel(
                channels=channels,
                labels=labels,
                image=datastore.load_local_registered_image(
                    tile=tile,
                    round=round_id,
                    return_future=False,
                ),
                label=f"{selected_warp_label(tile, round_id, options)} fiducial registered/decon",
                warp_image=warp_fiducial,
            )

    for bit_id in bit_ids:

        def warp_bit(image: np.ndarray, bit_id: str = bit_id) -> np.ndarray:
            return _warp_bit_for_viewer(
                image,
                datastore=datastore,
                tile=tile,
                bit_id=bit_id,
                options=options,
                gpu_id=gpu_id,
                warnings=warnings,
            )

        if "corrected" in bit_sources:
            _append_warped_viewer_channel(
                channels=channels,
                labels=labels,
                image=datastore.load_local_corrected_image(
                    tile=tile,
                    bit=bit_id,
                    return_future=False,
                ),
                label=f"{selected_warp_label(tile, bit_id, options)} corrected",
                warp_image=warp_bit,
            )
        if "registered" in bit_sources:
            _append_warped_viewer_channel(
                channels=channels,
                labels=labels,
                image=datastore.load_local_registered_image(
                    tile=tile,
                    bit=bit_id,
                    return_future=False,
                ),
                label=f"{selected_warp_label(tile, bit_id, options)} registered/decon",
                warp_image=warp_bit,
            )
        if "feature" in bit_sources:
            _append_warped_viewer_channel(
                channels=channels,
                labels=labels,
                image=datastore.load_local_feature_predictor_image(
                    tile=tile,
                    bit=bit_id,
                    return_future=False,
                ),
                label=f"{selected_warp_label(tile, bit_id, options)} feature predictor",
                warp_image=warp_bit,
            )

    if not channels:
        raise ValueError("No selected image channels were available to display.")

    shape = channels[0].shape
    if any(channel.shape != shape for channel in channels):
        raise ValueError("Selected image channels do not have matching shapes.")

    return ViewerDisplay(
        stack=ChannelStack(data=np.stack(channels, axis=0), labels=labels),
        warnings=warnings,
    )
