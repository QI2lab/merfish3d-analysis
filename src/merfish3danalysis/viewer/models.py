"""Data models and coordinate helpers for the datastore viewer."""

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ChannelStack:
    """Channel-stacked image data and labels for display."""

    data: Any
    labels: list[str]


@dataclass(frozen=True)
class LazyGlobalChannelData:
    """Global fused image plus optional image channels."""

    fused_zyx: Any
    image_channels: tuple[Any, ...]
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
        return (1 + len(self.image_channels), *zyx_shape)

    @property
    def ndim(self) -> int:
        """Return the number of array dimensions."""
        return len(self.shape)

    @property
    def dims(self) -> tuple[str, str, str, str]:
        """Return NDV dimension names."""
        return ("c", "z_um", "y_um", "x_um")

    def with_image_channel(self, image_zyx: Any) -> "LazyGlobalChannelData":
        """
        Return a new lazy stack with one additional image channel.

        Parameters
        ----------
        image_zyx : Any
            Image channel in Z, Y, X order.

        Returns
        -------
        LazyGlobalChannelData
            Updated channel data.
        """
        coords = dict(self.coords)
        coords["c"] = range(self.shape[0] + 1)
        return LazyGlobalChannelData(
            fused_zyx=self.fused_zyx,
            image_channels=(*self.image_channels, image_zyx),
            coords=coords,
        )

    def channel_value_range(self, channel_index: int) -> tuple[float, float] | None:
        """
        Return a cheap display range for an image channel when available.

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
        image = self.image_channels[channel_index - 1]
        max_value = getattr(image, "max_value", None)
        if max_value is None:
            return None
        max_value = float(max_value)
        if max_value >= 1.0:
            return (1.0, max_value) if max_value > 1.0 else (0.5, 1.5)
        return (0.0, 1.0)

    def __getitem__(self, key: Any) -> np.ndarray:
        """
        Read selected channels from tensorstore or image channels.

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
        channel_indices = np.arange(self.shape[0], dtype=np.intp)[channel_key]
        return np.stack(
            [
                self._read_channel(int(channel_index), zyx_key)
                for channel_index in channel_indices
            ],
            axis=0,
        )

    def __array_function__(self, *_args: Any, **_kwargs: Any) -> Any:
        """
        Mark the object as NumPy-like without full-array conversion.

        Parameters
        ----------
        *_args : Any
            NumPy protocol positional arguments.
        **_kwargs : Any
            NumPy protocol keyword arguments.

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
            selected = self.fused_zyx[zyx_key]
            if hasattr(selected, "read"):
                selected = selected.read().result()
            return np.asarray(selected)
        image = self.image_channels[channel_index - 1]
        selected = image[zyx_key]
        if hasattr(selected, "read"):
            selected = selected.read().result()
        return np.asarray(selected)


@dataclass(frozen=True)
class GlobalChannelStack:
    """Global channel stack with micron coordinate metadata."""

    stack: ChannelStack
    origin_zyx_um: np.ndarray
    spacing_zyx_um: np.ndarray
    full_shape_zyx: tuple[int, int, int]


@dataclass(frozen=True)
class ViewerDisplay:
    """Display payload and warning messages produced by viewer loaders."""

    stack: ChannelStack
    warnings: list[str]


@dataclass(frozen=True)
class SparsePointLayer:
    """Point geometry rendered as a sparse viewer overlay."""

    index: Any
    selected_genes: tuple[str, ...]
    marker_size: int
    spacing_zyx_um: tuple[float, float, float]
    label: str


@dataclass(frozen=True)
class SparseLineLayer:
    """Polyline geometry rendered as a sparse viewer overlay."""

    polylines: tuple[Any, ...]
    shape: tuple[int, int, int]
    width: int
    color: str
    spacing_zyx_um: tuple[float, float, float]
    label: str
    z_aware: bool = False
    z_polylines_by_index: dict[int, tuple[np.ndarray, ...]] | None = None
    max_polylines_yx: tuple[np.ndarray, ...] = ()


@dataclass(frozen=True)
class SparseOverlayPayload:
    """Sparse point and line layers for the current viewer display."""

    points: tuple[SparsePointLayer, ...] = ()
    lines: tuple[SparseLineLayer, ...] = ()

    @property
    def labels(self) -> list[str]:
        """Return display labels for all sparse layers."""
        return [layer.label for layer in (*self.lines, *self.points)]


@dataclass(frozen=True)
class DisplayContext:
    """Immutable context needed to refresh overlays for a displayed stack."""

    mode: str
    base_stack: ChannelStack
    base_sparse_lines: tuple[SparseLineLayer, ...]
    shape_zyx: tuple[int, int, int]
    spacing_zyx_um: np.ndarray
    origin_zyx_um: np.ndarray | None = None
    full_shape_zyx: tuple[int, int, int] | None = None
    tile: str | None = None
    tile_transform: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    refresh_only: bool = False

    def as_refresh(self) -> "DisplayContext":
        """Return this context marked as an overlay-only refresh."""
        return replace(self, refresh_only=True)


@dataclass(frozen=True)
class ViewerBuildResult:
    """Prepared viewer data produced off the Qt UI thread."""

    stack: ChannelStack
    spacing_zyx_um: np.ndarray
    origin_zyx_um: np.ndarray | None
    context: DisplayContext
    status: str
    sparse_payload: SparseOverlayPayload | None = None


@dataclass(frozen=True)
class ViewerState:
    """Controller selection snapshot used to build display requests."""

    view_mode: str
    tile: str
    fiducial_sources: tuple[str, ...]
    fiducial_rounds: tuple[str, ...]
    bit_ids: tuple[str, ...]
    bit_sources: tuple[str, ...]
    warp_options: "WarpChainOptions"
    gpu_id: int
    include_fused_image: bool
    include_segmentation: bool
    include_cell_boundaries: bool
    include_proseg_boundaries: bool
    include_baysor_boundaries: bool
    transcript_source: str | None
    selected_genes: tuple[str, ...]
    marker_radius: int
    proseg_run_name: str | None
    max_project: bool


@dataclass(frozen=True)
class GlobalDisplayRequest:
    """Controller state needed to build a global display."""

    include_fused_image: bool
    include_segmentation: bool
    include_cell_boundaries: bool
    include_proseg_boundaries: bool
    include_baysor_boundaries: bool
    transcript_source: str | None
    selected_genes: tuple[str, ...]
    marker_radius: int
    proseg_run_name: str | None
    max_project: bool


@dataclass(frozen=True)
class LocalDisplayRequest:
    """Controller state needed to build a local tile display."""

    tile: str
    fiducial_sources: tuple[str, ...]
    fiducial_rounds: tuple[str, ...]
    bit_ids: tuple[str, ...]
    bit_sources: tuple[str, ...]
    warp_options: "WarpChainOptions"
    gpu_id: int
    include_cell_boundaries: bool
    include_proseg_boundaries: bool
    include_baysor_boundaries: bool
    transcript_source: str | None
    selected_genes: tuple[str, ...]
    marker_radius: int
    proseg_run_name: str | None


@dataclass(frozen=True)
class TranscriptRefreshRequest:
    """Controller state needed to rebuild transcript sparse overlays."""

    source: str | None
    selected_genes: tuple[str, ...]
    marker_radius: int
    proseg_run_name: str | None


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


def stack_with_micron_coords(
    stack: ChannelStack,
    voxel_size_zyx_um: Any,
    origin_zyx_um: Any | None = None,
) -> Any:
    """
    Attach Z, Y, X micron coordinates to a channel stack.

    Parameters
    ----------
    stack : ChannelStack
        Channel-stacked image data.
    voxel_size_zyx_um : Any
        Z, Y, X voxel spacing in microns.
    origin_zyx_um : Any | None
        Optional Z, Y, X coordinate origin in microns.

    Returns
    -------
    Any
        NDV-compatible data with micron coordinates.
    """
    import xarray as xr

    if isinstance(stack.data, LazyGlobalChannelData):
        return stack.data

    data = np.asarray(stack.data)
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
