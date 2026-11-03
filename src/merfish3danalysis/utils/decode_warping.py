"""Decode-time image warping helpers.

These helpers keep the transform composition used by pixel decoding in one
place. The actual interpolation remains in
``merfish3danalysis.utils.multiview_registration`` so registration and decoding
share the same GPU warp convention.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_OPTIMIZER_ESTIMATORS = {
    "decoded_rna_on_bit_weighted_centroid_z_translation_yx_affine_graph",
    "decoded_rna_on_bit_weighted_centroid_yx_affine_graph",
}


def optimizer_calibration_is_applicable(calibration: Any) -> bool:
    """Identify corrections produced by the iterative decoder.

    Parameters
    ----------
    calibration : Any
        Calibration metadata to inspect.

    Returns
    -------
    bool
        Whether the metadata identifies an iterative decoder estimator.
    """
    return (
        isinstance(calibration, Mapping)
        and calibration.get("estimator") in _OPTIMIZER_ESTIMATORS
    )


def select_chromatic_affine(
    calibration: Mapping[str, Any],
    *,
    channel_name: str | None = None,
    channel_index: int | None = None,
    wavelength_um: float | None = None,
) -> np.ndarray | None:
    """Select a chromatic affine using the datastore's channel priority.

    Parameters
    ----------
    calibration : Mapping
        Calibration metadata containing channel transforms.
    channel_name : str or None, optional
        Channel name to match first.
    channel_index : int or None, optional
        Channel index to match after the name.
    wavelength_um : float or None, optional
        Emission wavelength in microns to match last.

    Returns
    -------
    numpy.ndarray or None
        Selected affine as float32, or None if no match contains an affine.
    """
    channels = calibration.get("channels", {})
    if not isinstance(channels, Mapping):
        return None

    candidates = []
    if channel_name is not None:
        channel = channels.get(str(channel_name))
        if isinstance(channel, Mapping):
            candidates.append(channel)
    if channel_index is not None:
        for channel in channels.values():
            if isinstance(channel, Mapping) and int(
                channel.get("channel_index", -1)
            ) == int(channel_index):
                candidates.append(channel)
    if wavelength_um is not None:
        wavelength = float(wavelength_um)
        for channel in channels.values():
            if not isinstance(channel, Mapping):
                continue
            stored = channel.get("wavelength_um")
            if stored is not None and np.isclose(float(stored), wavelength):
                candidates.append(channel)

    for channel in candidates:
        affine = channel.get("affine_zyx_um")
        if affine is not None:
            return np.asarray(affine, dtype=np.float32)
    return None


def validate_chromatic_affine(affine: Any) -> np.ndarray | None:
    """Validate a chromatic affine for Scene publication and decoding.

    Parameters
    ----------
    affine : Any
        Candidate homogeneous affine transform.

    Returns
    -------
    numpy.ndarray or None
        Finite, invertible 4-by-4 float32 affine, or None if invalid.
    """
    if affine is None:
        return None
    affine = np.asarray(affine, dtype=np.float32)
    if (
        affine.shape != (4, 4)
        or not np.isfinite(affine).all()
        or not np.allclose(affine[3], [0, 0, 0, 1])
        or abs(np.linalg.det(affine[:3, :3])) < 1e-12
    ):
        return None
    return affine


def load_optimizer_chromatic_affine(
    datastore: Any, *, wavelength_um: float
) -> np.ndarray:
    """Load an applicable iterative optimizer estimate, or identity.

    Parameters
    ----------
    datastore : Any
        Datastore providing chromatic calibration metadata and transforms.
    wavelength_um : float
        Emission wavelength in microns.

    Returns
    -------
    numpy.ndarray
        Validated float32 affine, or identity when no applicable estimate exists.
    """
    identity = np.eye(4, dtype=np.float32)
    load_metadata = getattr(datastore, "load_chromatic_affine_transforms_zyx_um", None)
    if load_metadata is None:
        return identity
    calibration = load_metadata()
    if not optimizer_calibration_is_applicable(calibration):
        return identity
    try:
        affine = validate_chromatic_affine(
            datastore.load_chromatic_affine_transform_zyx_um(
                wavelength_um=wavelength_um
            )
        )
        if affine is None:
            return identity
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return identity
    return affine


def load_bit_round_transform_zyx_um(
    datastore: Any,
    *,
    tile: int | str,
    bit_id: str,
) -> tuple[str | None, np.ndarray]:
    """
    Load the local fiducial-round transform for one readout bit.

    Parameters
    ----------
    datastore : Any
        qi2lab datastore-like object.
    tile : int or str
        Tile index or identifier.
    bit_id : str
        Readout bit identifier.

    Returns
    -------
    tuple[str | None, numpy.ndarray]
        Round identifier and physical Z, Y, X transform. Bits from the
        reference round return ``(None, identity)``.
    """
    round_index = datastore.load_local_round_linker(tile=tile, bit=bit_id) - 1
    if round_index <= 0:
        return None, np.eye(4, dtype=np.float32)

    round_id = datastore.round_ids[round_index]
    round_transform_zyx_um = datastore.load_local_round_transform_zyx_um(
        tile=tile,
        round=round_id,
    )
    if round_transform_zyx_um is None:
        raise RuntimeError(
            f"Missing local round transform for tile={tile} round={round_id}."
        )
    return round_id, np.asarray(round_transform_zyx_um, dtype=np.float32)


def compose_decode_warp_transform_zyx_um(
    *,
    round_transform_zyx_um: np.ndarray,
    chromatic_transform_zyx_um: np.ndarray,
) -> np.ndarray:
    """
    Compose chromatic and fiducial transforms for decode-time image loading.

    Parameters
    ----------
    round_transform_zyx_um : numpy.ndarray
        Physical transform from reference-round Z, Y, X coordinates into the
        bit's native fiducial-round coordinates.
    chromatic_transform_zyx_um : numpy.ndarray
        Chromatic calibration transform mapping the bit wavelength toward the
        reference wavelength in physical Z, Y, X coordinates.

    Returns
    -------
    numpy.ndarray
        Physical transform mapping reference-round output coordinates to the
        native readout image coordinates, suitable for
        :func:`warp_array_to_reference_gpu`.
    """
    return np.linalg.inv(np.asarray(chromatic_transform_zyx_um, dtype=np.float32)) @ (
        np.asarray(round_transform_zyx_um, dtype=np.float32)
    )


def compose_optional_warp_transform_zyx_um(
    *,
    round_transform_zyx_um: np.ndarray | None,
    chromatic_transform_zyx_um: np.ndarray | None,
) -> np.ndarray:
    """
    Compose optional chromatic and round transforms.

    Parameters
    ----------
    round_transform_zyx_um : numpy.ndarray or None
        Physical transform from reference-round Z, Y, X coordinates into the
        moving image coordinates. ``None`` uses identity.
    chromatic_transform_zyx_um : numpy.ndarray or None
        Chromatic calibration transform in physical Z, Y, X coordinates.
        ``None`` uses identity.

    Returns
    -------
    numpy.ndarray
        Physical transform mapping reference-frame output coordinates to the
        native moving image coordinates.
    """
    round_transform = (
        np.eye(4, dtype=np.float32)
        if round_transform_zyx_um is None
        else np.asarray(round_transform_zyx_um, dtype=np.float32)
    )
    chromatic_transform = (
        np.eye(4, dtype=np.float32)
        if chromatic_transform_zyx_um is None
        else np.asarray(chromatic_transform_zyx_um, dtype=np.float32)
    )
    return compose_decode_warp_transform_zyx_um(
        round_transform_zyx_um=round_transform,
        chromatic_transform_zyx_um=chromatic_transform,
    )


def warp_image_to_reference_frame(
    image: np.ndarray,
    *,
    transform_zyx_um: np.ndarray,
    spacing_zyx_um: Sequence[float],
    loaded_flow_field: tuple[np.ndarray, Mapping[str, Any]] | None = None,
    gpu_id: int = 0,
    reference_shape: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Warp an image into the reference frame using affine and optional SOFIMA flow.

    Parameters
    ----------
    image : numpy.ndarray
        Native moving image in Z, Y, X order.
    transform_zyx_um : numpy.ndarray
        Physical transform from reference output coordinates into the moving
        image coordinate system.
    spacing_zyx_um : Sequence[float]
        Reference image voxel spacing in microns, in Z, Y, X order.
    loaded_flow_field : tuple[numpy.ndarray, Mapping[str, Any]] or None
        Optional SOFIMA flow field and metadata returned by the datastore.
    gpu_id : int, default=0
        CUDA device ID used for interpolation.
    reference_shape : Sequence[int] or None, default=None
        Output Z, Y, X shape. Defaults to the flow metadata shape for SOFIMA,
        or the input image shape for affine-only warping.

    Returns
    -------
    numpy.ndarray
        Image sampled in the reference frame as float32.
    """
    if (
        (reference_shape is None or tuple(reference_shape) == image.shape)
        and loaded_flow_field is None
        and np.allclose(
            transform_zyx_um,
            np.eye(4, dtype=np.float32),
        )
    ):
        return np.asarray(image, dtype=np.float32)

    from merfish3danalysis.utils.multiview_registration import (
        warp_array_to_reference_gpu,
    )

    if loaded_flow_field is not None:
        sofima_flow_field, flow_attrs = loaded_flow_field
        if not str(flow_attrs.get("sofima_status", "")).startswith("identity_fallback"):
            return warp_image_with_sofima_metadata(
                image,
                transform_zyx_um=transform_zyx_um,
                spacing_zyx_um=spacing_zyx_um,
                sofima_flow_field_xyz_px=sofima_flow_field,
                flow_attrs=flow_attrs,
                gpu_id=gpu_id,
                reference_shape=reference_shape,
            ).astype(np.float32, copy=False)

    return warp_array_to_reference_gpu(
        image,
        transform_zyx_um=transform_zyx_um,
        spacing_zyx_um=spacing_zyx_um,
        reference_shape=image.shape if reference_shape is None else reference_shape,
        gpu_id=gpu_id,
    ).astype(np.float32, copy=False)


def warp_bit_image_to_reference(
    image: np.ndarray,
    *,
    datastore: Any,
    tile: int | str,
    bit_id: str,
    emission_wavelength_um: float,
    gpu_id: int = 0,
    reference_shape: Sequence[int] | None = None,
    apply_chromatic: bool = True,
) -> np.ndarray:
    """
    Warp one native bit image into the round-1 local reference frame.

    Parameters
    ----------
    image : numpy.ndarray
        Readout image in the bit's native Z, Y, X grid.
    datastore : Any
        qi2lab datastore-like object used to load round, chromatic, and SOFIMA
        metadata.
    tile : int or str
        Tile index or identifier.
    bit_id : str
        Readout bit identifier.
    emission_wavelength_um : float
        Emission wavelength for the bit in microns.
    gpu_id : int, default=0
        CUDA device ID used for interpolation.
    reference_shape : Sequence[int] or None, default=None
        Optional round-1 output Z, Y, X shape.
    apply_chromatic : bool, default=True
        Apply an applicable optimizer estimate. Set False for non-decoding uses.

    Returns
    -------
    numpy.ndarray
        Image sampled in the round-1 local reference frame.
    """
    round_id, round_transform_zyx_um = load_bit_round_transform_zyx_um(
        datastore,
        tile=tile,
        bit_id=bit_id,
    )
    chromatic_transform_zyx_um = (
        load_optimizer_chromatic_affine(datastore, wavelength_um=emission_wavelength_um)
        if apply_chromatic
        else np.eye(4, dtype=np.float32)
    )
    transform_zyx_um = compose_decode_warp_transform_zyx_um(
        round_transform_zyx_um=round_transform_zyx_um,
        chromatic_transform_zyx_um=chromatic_transform_zyx_um,
    )
    spacing_zyx_um = datastore.voxel_size_zyx_um

    loaded_flow_field = None
    if round_id is not None:
        loaded_flow_field = datastore.load_local_sofima_flow_field(
            tile=tile,
            round=round_id,
            return_future=False,
        )

    return warp_image_to_reference_frame(
        image,
        transform_zyx_um=transform_zyx_um,
        spacing_zyx_um=spacing_zyx_um,
        loaded_flow_field=loaded_flow_field,
        gpu_id=gpu_id,
        reference_shape=reference_shape,
    )


def warp_image_with_sofima_metadata(
    image: np.ndarray,
    *,
    transform_zyx_um: np.ndarray,
    spacing_zyx_um: Sequence[float],
    sofima_flow_field_xyz_px: np.ndarray,
    flow_attrs: Mapping[str, Any],
    reference_shape: Sequence[int] | None = None,
    gpu_id: int = 0,
) -> np.ndarray:
    """
    Warp an image using a SOFIMA flow field and datastore flow metadata.

    Parameters
    ----------
    image : numpy.ndarray
        Native moving image in Z, Y, X order.
    transform_zyx_um : numpy.ndarray
        Physical affine transform from reference output coordinates into the
        moving image coordinate system.
    spacing_zyx_um : Sequence[float]
        Reference image voxel spacing in microns, in Z, Y, X order.
    sofima_flow_field_xyz_px : numpy.ndarray
        SOFIMA flow field in X, Y, Z channel order with map axes Z, Y, X.
    flow_attrs : Mapping[str, Any]
        Metadata returned by ``load_local_sofima_flow_field`` or by the SOFIMA
        estimator. This must include ``map_stride_zyx_px`` and
        ``map_box_start_xyz_px``. If ``reference_shape`` is omitted it must also
        include ``reference_shape_zyx_px``.
    reference_shape : Sequence[int] or None, default=None
        Output image shape in Z, Y, X order. When omitted, the shape stored in
        ``flow_attrs`` is used.
    gpu_id : int, default=0
        CUDA device ID used for interpolation.

    Returns
    -------
    numpy.ndarray
        Image sampled in the reference frame as float32.
    """
    from merfish3danalysis.utils.multiview_registration import (
        warp_array_to_reference_with_affine_and_sofima_flow_gpu,
    )

    if reference_shape is None:
        reference_shape = flow_attrs["reference_shape_zyx_px"]
    output_shape = tuple(int(v) for v in reference_shape)
    return warp_array_to_reference_with_affine_and_sofima_flow_gpu(
        image,
        transform_zyx_um=transform_zyx_um,
        spacing_zyx_um=spacing_zyx_um,
        reference_shape=output_shape,
        sofima_flow_field_xyz_px=sofima_flow_field_xyz_px,
        flow_field_stride_zyx_px=flow_attrs["map_stride_zyx_px"],
        flow_field_box_start_xyz_px=flow_attrs["map_box_start_xyz_px"],
        gpu_id=gpu_id,
    ).astype(np.float32, copy=False)


__all__ = [
    "compose_decode_warp_transform_zyx_um",
    "compose_optional_warp_transform_zyx_um",
    "load_bit_round_transform_zyx_um",
    "load_optimizer_chromatic_affine",
    "optimizer_calibration_is_applicable",
    "warp_bit_image_to_reference",
    "warp_image_to_reference_frame",
    "warp_image_with_sofima_metadata",
]
