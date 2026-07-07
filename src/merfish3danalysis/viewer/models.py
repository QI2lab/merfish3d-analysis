"""Data models and coordinate helpers for the datastore viewer."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ChannelStack:
    """Channel-stacked image data and labels for display."""

    data: Any
    labels: list[str]


@dataclass(frozen=True)
class LazyGlobalChannelData:
    """Tensorstore-backed fused image plus in-memory overlay channels."""

    fused_zyx: Any
    overlays: tuple[Any, ...]
    coords: dict[str, Any]

    @property
    def dtype(self) -> np.dtype:
        """Return the fused image dtype."""

        dtype = self.fused_zyx.dtype
        numpy_dtype = getattr(dtype, "numpy_dtype", None)
        if numpy_dtype is not None:
            return np.dtype(numpy_dtype)
        return np.dtype(dtype)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Return channel, Z, Y, X shape."""

        zyx_shape = tuple(int(value) for value in self.fused_zyx.shape)
        return (1 + len(self.overlays), *zyx_shape)

    @property
    def ndim(self) -> int:
        """Return the number of array dimensions."""

        return len(self.shape)

    @property
    def dims(self) -> tuple[str, str, str, str]:
        """Return NDV dimension names."""

        return ("c", "z_um", "y_um", "x_um")

    def with_overlay(self, overlay_zyx: Any) -> "LazyGlobalChannelData":
        """
        Return a new lazy stack with one additional in-memory overlay.

        Parameters
        ----------
        overlay_zyx : Any
            Overlay image or lazy overlay in Z, Y, X order.

        Returns
        -------
        LazyGlobalChannelData
            Updated channel data.
        """

        coords = dict(self.coords)
        coords["c"] = range(self.shape[0] + 1)
        return LazyGlobalChannelData(
            fused_zyx=self.fused_zyx,
            overlays=(*self.overlays, overlay_zyx),
            coords=coords,
        )

    def channel_value_range(self, channel_index: int) -> tuple[float, float] | None:
        """
        Return a cheap display range for an overlay channel when available.

        Parameters
        ----------
        channel_index : int
            Channel index.

        Returns
        -------
        tuple[float, float] or None
            Inclusive display range, or ``None`` when it would require reading data.
        """

        if channel_index == 0:
            return None
        overlay = self.overlays[channel_index - 1]
        max_value = getattr(overlay, "max_value", None)
        if max_value is None:
            return None
        max_value = float(max_value)
        return (1.0, max_value) if max_value > 1.0 else (0.0, 1.0)

    def __getitem__(self, key: Any) -> np.ndarray:
        """
        Read selected channels from tensorstore or in-memory overlays.

        Parameters
        ----------
        key : Any
            NumPy-style channel, Z, Y, X index.

        Returns
        -------
        numpy.ndarray
            Requested image data.
        """

        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (self.ndim - len(key))
        channel_key, zyx_key = key[0], key[1:]
        if isinstance(channel_key, np.integer):
            channel_key = int(channel_key)
        if isinstance(channel_key, int):
            return self._read_channel(channel_key, zyx_key)
        channel_indices = list(range(self.shape[0])[channel_key])
        return np.stack(
            [
                self._read_channel(channel_index, zyx_key)
                for channel_index in channel_indices
            ],
            axis=0,
        )

    def __array_function__(self, *_args: Any, **_kwargs: Any) -> Any:
        """
        Mark the object as NumPy-like without full-array conversion.

        Returns
        -------
        Any
            ``NotImplemented`` so NumPy does not read the full fused zarr.
        """

        return NotImplemented

    def _read_channel(self, channel_index: int, zyx_key: tuple[Any, ...]) -> np.ndarray:
        """
        Read one channel.

        Parameters
        ----------
        channel_index : int
            Channel index.
        zyx_key : tuple[Any, ...]
            Z, Y, X index.

        Returns
        -------
        numpy.ndarray
            Requested channel data.
        """

        if channel_index == 0:
            return self.fused_zyx[zyx_key].read().result()
        overlay = self.overlays[channel_index - 1]
        selected = overlay[zyx_key]
        if hasattr(selected, "read"):
            selected = selected.read().result()
        return np.asarray(selected)


@dataclass(frozen=True)
class GlobalChannelStack:
    """Global channel stack with micron coordinate metadata."""

    stack: ChannelStack
    origin_zyx_um: np.ndarray
    spacing_zyx_um: np.ndarray


@dataclass(frozen=True)
class ProsegRun:
    """One detected Proseg 3D output run."""

    name: str


@dataclass(frozen=True)
class ViewerDisplay:
    """Display payload and warning messages produced by viewer loaders."""

    stack: ChannelStack
    warnings: list[str]


@dataclass(frozen=True)
class ViewerBuildResult:
    """Prepared viewer data produced off the Qt UI thread."""

    stack: ChannelStack
    spacing_zyx_um: np.ndarray
    origin_zyx_um: np.ndarray | None
    context: dict[str, Any]
    status: str
    decoded_spots: Any | None = None
    proseg_run_name: str | None = None
    proseg_transcripts: Any | None = None


@dataclass(frozen=True)
class WarpChainOptions:
    """Viewer-selected local warp-chain components."""

    chromatic: bool = True
    stage_affine: bool = True
    sofima: bool = True

    @property
    def is_native(self) -> bool:
        """
        Return whether all transform components are disabled.

        Returns
        -------
        bool
            True when the viewer should display stored native coordinates.
        """

        return not (self.chromatic or self.stage_affine or self.sofima)


def warp_chain_label(options: WarpChainOptions) -> str:
    """
    Return a compact viewer label for selected warp-chain components.

    Parameters
    ----------
    options : WarpChainOptions
        Selected local viewer transform components.

    Returns
    -------
    str
        Human-readable warp-chain label.
    """

    pieces: list[str] = []
    if options.stage_affine:
        pieces.append("affine")
    if options.chromatic:
        pieces.append("chromatic")
    if options.sofima:
        pieces.append("sofima")
    return "+".join(pieces) if pieces else "native"


def compose_viewer_warp_transform_zyx_um(
    *,
    round_transform_zyx_um: np.ndarray | None,
    chromatic_transform_zyx_um: np.ndarray | None,
) -> np.ndarray:
    """
    Compose optional affine pieces for local viewer display.

    Parameters
    ----------
    round_transform_zyx_um : numpy.ndarray or None
        Physical round/stage affine. ``None`` means identity.
    chromatic_transform_zyx_um : numpy.ndarray or None
        Physical chromatic affine. ``None`` means identity.

    Returns
    -------
    numpy.ndarray
        Physical transform mapping reference-frame output coordinates into
        native moving-image coordinates.
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
    return np.linalg.inv(chromatic_transform) @ round_transform


def stack_with_micron_coords(
    stack: ChannelStack,
    voxel_size_zyx_um: Any,
    origin_zyx_um: Any | None = None,
) -> Any:
    """
    Attach zyx micron coordinates to a channel stack for ndv display.

    Parameters
    ----------
    stack : ChannelStack
        stack for this viewer operation.
    voxel_size_zyx_um : Any
        voxel_size_zyx_um for this viewer operation.
    origin_zyx_um : Any | None
        origin_zyx_um for this viewer operation.

    Returns
    -------
    Any
        Computed viewer result.
    """

    import xarray as xr

    if isinstance(stack.data, LazyGlobalChannelData):
        return stack.data

    data = stack.data.astype(np.float32, copy=False)
    voxel = np.asarray(voxel_size_zyx_um, dtype=np.float32)
    origin = (
        np.zeros(3, dtype=np.float32)
        if origin_zyx_um is None
        else np.asarray(origin_zyx_um, dtype=np.float32)
    )
    if data.ndim != 4 or voxel.shape[0] != 3:
        raise ValueError("Expected channel stack shape (c, z, y, x).")
    if origin.shape[0] != 3:
        raise ValueError("Expected origin shape (3,).")

    return xr.DataArray(
        data,
        dims=("c", "z_um", "y_um", "x_um"),
        coords={
            "c": np.arange(data.shape[0]),
            "z_um": origin[0] + np.arange(data.shape[1], dtype=np.float32) * voxel[0],
            "y_um": origin[1] + np.arange(data.shape[2], dtype=np.float32) * voxel[1],
            "x_um": origin[2] + np.arange(data.shape[3], dtype=np.float32) * voxel[2],
        },
        attrs={"z_spacing_um": float(voxel[0])},
    )
