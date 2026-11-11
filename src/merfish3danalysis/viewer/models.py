"""Data models and coordinate helpers for the datastore viewer."""

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from merfish3danalysis.utils.spacing import round_spacing_um


@dataclass(frozen=True)
class ChannelStack:
    """Channel-stacked image data and labels for display.

    Parameters
    ----------
    data : Any
        Image data with channel, Z, Y, X dimensions.
    labels : list[str]
        Display labels in image channel order.
    """

    data: Any
    labels: list[str]


@dataclass(frozen=True)
class LazyGlobalChannelData:
    """Global image channels with physical coordinates and separate dtypes.

    NDV's native TensorStore wrapper uses pixel indices for coordinates, while
    TensorStore stacks require a common dtype for intensity images and labels.

    Parameters
    ----------
    fused_zyx : Any
        Base intensity image in Z, Y, X order, opened lazily or held in memory.
    image_channels : tuple[Any, ...]
        Additional Z, Y, X image channels, retaining their individual dtypes.
    coords : dict[str, Any]
        Dimension names mapped to channel labels or physical coordinate arrays.
    """

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
    def shape(self) -> tuple[int, ...]:
        """Return the channel and image dimensions."""
        zyx_shape = tuple(int(value) for value in self.fused_zyx.shape)
        return (1 + len(self.image_channels), *zyx_shape)

    @property
    def ndim(self) -> int:
        """Return the number of array dimensions."""
        return len(self.shape)

    @property
    def dims(self) -> tuple[str, ...]:
        """Return NDV dimension names."""
        return tuple(self.coords)

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
        """Decline NumPy operations without materializing the full fused image.

        Parameters
        ----------
        *_args : Any
            NumPy protocol positional arguments.
        **_kwargs : Any
            NumPy protocol keyword arguments.
        """
        return NotImplemented

    def _read_channel(self, channel_index: int, zyx_key: tuple[Any, ...]) -> np.ndarray:
        """Read one selected channel in Z, Y, X order.

        Parameters
        ----------
        channel_index : int
            Channel index.
        zyx_key : tuple[Any, ...]
            Z, Y, X index.
        """
        image = (
            self.fused_zyx
            if channel_index == 0
            else self.image_channels[channel_index - 1]
        )
        selected = image[zyx_key]
        if hasattr(selected, "read"):
            selected = selected.read().result()
        return np.asarray(selected)


@dataclass(frozen=True)
class GlobalChannelStack:
    """Global channel stack with micron coordinate metadata.

    Parameters
    ----------
    stack : ChannelStack
        Image channels and their display labels.
    origin_zyx_um : numpy.ndarray
        Global canvas origin in Z, Y, X microns.
    spacing_zyx_um : numpy.ndarray
        Voxel spacing in Z, Y, X microns.
    full_shape_zyx : tuple[int, int, int]
        Unprojected global canvas dimensions in Z, Y, X order.
    """

    stack: ChannelStack
    origin_zyx_um: np.ndarray
    spacing_zyx_um: np.ndarray
    full_shape_zyx: tuple[int, int, int]


@dataclass(frozen=True)
class ViewerDisplay:
    """Display payload and warning messages produced by viewer loaders.

    Parameters
    ----------
    stack : ChannelStack
        Image channels and their display labels.
    warnings : list[str]
        Warnings recorded while loading or warping the requested channels.
    """

    stack: ChannelStack
    warnings: list[str]


@dataclass(frozen=True)
class SparsePointLayer:
    """Point geometry rendered as a sparse viewer overlay.

    Parameters
    ----------
    index : Any
        Transcript coordinates and gene IDs indexed by rounded Z plane.
    selected_genes : tuple[str, ...]
        Gene identifiers selected for display, in color order.
    marker_size : int
        Marker size passed to VisPy, in display pixels.
    spacing_zyx_um : tuple[float, float, float]
        Voxel spacing in Z, Y, X microns.
    label : str
        Display label for the sparse layer.
    """

    index: Any
    selected_genes: tuple[str, ...]
    marker_size: int
    spacing_zyx_um: tuple[float, float, float]
    label: str


@dataclass(frozen=True)
class SparseLineLayer:
    """Polyline geometry rendered as a sparse viewer overlay.

    Parameters
    ----------
    polylines : tuple[Any, ...]
        Y, X vertex arrays, or (z_min, z_max, vertices_yx) tuples for Z-aware lines.
    shape : tuple[int, int, int]
        Canvas dimensions in Z, Y, X order.
    width : int
        Outline width in display pixels.
    color : str
        VisPy-compatible outline color.
    spacing_zyx_um : tuple[float, float, float]
        Voxel spacing in Z, Y, X microns.
    label : str
        Display label for the sparse layer.
    z_aware : bool, default=False
        Whether outlines vary with their Z interval or indexed plane.
    z_polylines_by_index : dict[int, tuple[numpy.ndarray, ...]] | None, default=None
        Y, X outline vertices grouped by integer Z plane, or None if not indexed.
    max_polylines_yx : tuple[numpy.ndarray, ...]
        Y, X outlines for maximum-projection display.
    """

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
    """Sparse point and line layers for the current viewer display.

    Parameters
    ----------
    points : tuple[SparsePointLayer, ...]
        Transcript point layers in display order.
    lines : tuple[SparseLineLayer, ...]
        Cell-boundary line layers in display order.
    """

    points: tuple[SparsePointLayer, ...] = ()
    lines: tuple[SparseLineLayer, ...] = ()

    @property
    def labels(self) -> list[str]:
        """Return display labels for all sparse layers."""
        return [layer.label for layer in (*self.lines, *self.points)]


@dataclass(frozen=True)
class DisplayContext:
    """Immutable context needed to refresh overlays for a displayed stack.

    Parameters
    ----------
    mode : str
        Displayed coordinate frame, local or global.
    base_stack : ChannelStack
        Image stack retained while transcript overlays refresh.
    base_sparse_lines : tuple[SparseLineLayer, ...]
        Boundary layers retained while transcript overlays refresh.
    shape_zyx : tuple[int, int, int]
        Canvas dimensions in Z, Y, X order.
    spacing_zyx_um : numpy.ndarray
        Voxel spacing in Z, Y, X microns.
    origin_zyx_um : numpy.ndarray | None, default=None
        Canvas origin in Z, Y, X microns, or None for local views.
    tile_transform : tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray] | None, default=None
        Tile affine, origin, and spacing in Z, Y, X microns for global overlays.
    refresh_only : bool, default=False
        Whether this result updates overlays without replacing the image stack.
    """

    mode: str
    base_stack: ChannelStack
    base_sparse_lines: tuple[SparseLineLayer, ...]
    shape_zyx: tuple[int, int, int]
    spacing_zyx_um: np.ndarray
    origin_zyx_um: np.ndarray | None = None
    tile_transform: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    refresh_only: bool = False

    def as_refresh(self) -> "DisplayContext":
        """Return this context marked as an overlay-only refresh.

        Returns
        -------
        DisplayContext
            Copy with ``refresh_only`` enabled.
        """
        return replace(self, refresh_only=True)


@dataclass(frozen=True)
class ViewerBuildResult:
    """Prepared viewer data produced off the Qt UI thread.

    Parameters
    ----------
    stack : ChannelStack
        Image channels and their display labels.
    spacing_zyx_um : numpy.ndarray
        Voxel spacing in Z, Y, X microns.
    origin_zyx_um : numpy.ndarray | None
        Canvas origin in Z, Y, X microns, or None for local views.
    context : DisplayContext
        Coordinate and image context retained for later overlay refreshes.
    status : str
        Status text describing the displayed channels and sparse layers.
    sparse_payload : SparseOverlayPayload | None, default=None
        Sparse point and boundary layers, or None when there are none.
    """

    stack: ChannelStack
    spacing_zyx_um: np.ndarray
    origin_zyx_um: np.ndarray | None
    context: DisplayContext
    status: str
    sparse_payload: SparseOverlayPayload | None = None


@dataclass(frozen=True)
class ViewerState:
    """Controller selection snapshot used to build display requests.

    Parameters
    ----------
    view_mode : str
        Selected local-native, local-warped, or global-fused display mode.
    tile : str
        Selected datastore tile identifier.
    fiducial_sources : tuple[str, ...]
        Selected fiducial image sources, corrected or deconvolved.
    fiducial_rounds : tuple[str, ...]
        Selected fiducial round identifiers.
    bit_ids : tuple[str, ...]
        Selected readout bit identifiers.
    bit_sources : tuple[str, ...]
        Selected readout image sources: corrected, deconvolved, or feature predictor.
    warp_options : WarpChainOptions
        Local affine and SOFIMA components selected for display.
    gpu_id : int
        CUDA device index used for local image warping.
    include_fused_image : bool
        Include the fused fiducial intensity image.
    include_segmentation : bool
        Include the Cellpose label image.
    include_cell_boundaries : bool
        Include global Cellpose boundary outlines.
    include_proseg_boundaries : bool
        Include boundary outlines from the selected Proseg run.
    include_baysor_boundaries : bool
        Include Baysor boundary outlines.
    transcript_source : str | None
        Selected datastore, Proseg, or Baysor transcript source, or None.
    selected_genes : tuple[str, ...]
        Gene identifiers selected for display, in color order.
    marker_radius : int
        Transcript marker radius in display pixels.
    proseg_run_name : str | None
        Selected Proseg run name, or None for the default run.
    max_project : bool
        Display the global image and sparse geometry in maximum projection.
    """

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
    """Controller state needed to build a global display.

    Parameters
    ----------
    include_fused_image : bool
        Include the fused fiducial intensity image.
    include_segmentation : bool
        Include the Cellpose label image.
    include_cell_boundaries : bool
        Include global Cellpose boundary outlines.
    include_proseg_boundaries : bool
        Include boundary outlines from the selected Proseg run.
    include_baysor_boundaries : bool
        Include Baysor boundary outlines.
    transcript_source : str | None
        Selected datastore, Proseg, or Baysor transcript source, or None.
    selected_genes : tuple[str, ...]
        Gene identifiers selected for display, in color order.
    marker_radius : int
        Transcript marker radius in display pixels.
    proseg_run_name : str | None
        Selected Proseg run name, or None for the default run.
    max_project : bool
        Display the global image and sparse geometry in maximum projection.
    """

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
    """Controller state needed to build a local tile display.

    Parameters
    ----------
    tile : str
        Selected datastore tile identifier.
    fiducial_sources : tuple[str, ...]
        Selected fiducial image sources, corrected or deconvolved.
    fiducial_rounds : tuple[str, ...]
        Selected fiducial round identifiers.
    bit_ids : tuple[str, ...]
        Selected readout bit identifiers.
    bit_sources : tuple[str, ...]
        Selected readout image sources: corrected, deconvolved, or feature predictor.
    warp_options : WarpChainOptions
        Local affine and SOFIMA components selected for display.
    gpu_id : int
        CUDA device index used for local image warping.
    include_cell_boundaries : bool
        Include global Cellpose boundary outlines.
    include_proseg_boundaries : bool
        Include boundary outlines from the selected Proseg run.
    include_baysor_boundaries : bool
        Include Baysor boundary outlines.
    transcript_source : str | None
        Selected datastore, Proseg, or Baysor transcript source, or None.
    selected_genes : tuple[str, ...]
        Gene identifiers selected for display, in color order.
    marker_radius : int
        Transcript marker radius in display pixels.
    proseg_run_name : str | None
        Selected Proseg run name, or None for the default run.
    """

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
    """Controller state needed to rebuild transcript sparse overlays.

    Parameters
    ----------
    source : str | None
        Selected datastore, Proseg, or Baysor transcript source, or None.
    selected_genes : tuple[str, ...]
        Gene identifiers selected for display, in color order.
    marker_radius : int
        Transcript marker radius in display pixels.
    proseg_run_name : str | None
        Selected Proseg run name, or None for the default run.
    """

    source: str | None
    selected_genes: tuple[str, ...]
    marker_radius: int
    proseg_run_name: str | None


@dataclass(frozen=True)
class WarpChainOptions:
    """Viewer-selected local warp-chain components.

    Parameters
    ----------
    stage_affine : bool, default=True
        Apply the stored local round affine when True.
    sofima : bool, default=True
        Apply the stored residual SOFIMA field when True.
    """

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
        return not (self.stage_affine or self.sofima)


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
    voxel = round_spacing_um(voxel_size_zyx_um).astype(np.float32)
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
