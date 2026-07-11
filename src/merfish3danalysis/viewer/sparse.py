"""Sparse VisPy geometry helpers for datastore viewer overlays."""

from typing import Any

import numpy as np

from merfish3danalysis.viewer.colors import transcript_color_hex
from merfish3danalysis.viewer.models import (
    SparseLineLayer,
    SparseOverlayPayload,
    SparsePointLayer,
)
from merfish3danalysis.viewer.ndv import ndv_canvas_parts


class _LineVisualAccumulator:
    """Collect VisPy line vertices and connectivity arrays."""

    def __init__(self, scale_zyx: np.ndarray) -> None:
        """
        Create an empty accumulator with display scale.

        Parameters
        ----------
        scale_zyx : numpy.ndarray
            Z, Y, X display scale factors.
        """
        self.scale_zyx = scale_zyx
        self.positions: list[np.ndarray] = []
        self.connects: list[np.ndarray] = []
        self.vertex_count = 0

    def append(self, points_yx: np.ndarray, z_value: float, use_3d: bool) -> None:
        """
        Append one polyline at a Z position.

        Parameters
        ----------
        points_yx : numpy.ndarray
            Polyline vertices in Y, X order.
        z_value : float
            Z plane for the polyline.
        use_3d : bool
            Whether to produce 3D positions.
        """
        self.vertex_count = _append_polyline_visual_data(
            self.positions,
            self.connects,
            points_yx,
            z_value,
            use_3d,
            self.scale_zyx,
            self.vertex_count,
        )


def selected_point_data(
    layer: SparsePointLayer,
    z_index: int,
    use_3d: bool,
    project_z: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Return VisPy marker positions and colors for one point layer.

    Parameters
    ----------
    layer : SparsePointLayer
        Point layer to render.
    z_index : int
        Current Z index for 2D rendering.
    use_3d : bool
        Whether to return X, Y, Z positions instead of X, Y positions.
    project_z : bool
        Whether to project all Z planes into one 2D view.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, float]
        Marker positions, RGBA colors, and marker size.
    """
    if not layer.selected_genes or layer.index.coords_sorted.size == 0:
        return _empty_positions(use_3d), np.empty((0, 4), dtype=np.float32), 0.0

    selected_genes = tuple(str(gene) for gene in layer.selected_genes)
    genes = layer.index.genes_sorted.astype(str, copy=False)
    keep = np.isin(genes, selected_genes)
    if not use_3d and not project_z:
        keep &= layer.index.z_sorted == int(z_index)
    if not keep.any():
        return _empty_positions(use_3d), np.empty((0, 4), dtype=np.float32), 0.0

    coords = layer.index.coords_sorted[keep]
    positions = _points_to_vispy_positions(coords, use_3d, layer.spacing_zyx_um)
    colors = _selected_point_colors(genes[keep], selected_genes)
    return positions, colors, float(layer.marker_size)


def _selected_point_colors(
    genes: np.ndarray,
    selected_genes: tuple[str, ...],
) -> np.ndarray:
    """
    Return per-point transcript colors.

    Parameters
    ----------
    genes : numpy.ndarray
        Gene id for each visible point.
    selected_genes : tuple[str, ...]
        Selected genes in display order.

    Returns
    -------
    numpy.ndarray
        RGBA colors for visible points.
    """
    colors = np.empty((genes.shape[0], 4), dtype=np.float32)
    value_count = len(selected_genes)
    for value, gene in enumerate(selected_genes, start=1):
        colors[genes == gene] = _hex_to_rgba(transcript_color_hex(value, value_count))
    return colors


def _line_visual_result(
    positions: list[np.ndarray],
    connects: list[np.ndarray],
    *,
    color: str,
    width: int,
    use_3d: bool,
) -> tuple[np.ndarray, str, int, np.ndarray]:
    """
    Return concatenated line visual data or empty arrays.

    Parameters
    ----------
    positions : list[numpy.ndarray]
        Per-polyline vertex arrays.
    connects : list[numpy.ndarray]
        Per-polyline connectivity arrays.
    color : str
        VisPy line color.
    width : int
        VisPy line width.
    use_3d : bool
        Whether positions are 3D.

    Returns
    -------
    tuple[numpy.ndarray, str, int, numpy.ndarray]
        Vertices, color, width, and connectivity.
    """
    if not positions:
        return (
            _empty_positions(use_3d),
            color,
            width,
            np.empty((0, 2), dtype=np.uint32),
        )
    return (
        np.concatenate(positions, axis=0),
        color,
        width,
        np.concatenate(connects, axis=0),
    )


def selected_line_data(
    layer: SparseLineLayer,
    z_index: int,
    use_3d: bool,
    project_z: bool = False,
) -> tuple[np.ndarray, str, int, np.ndarray]:
    """
    Return VisPy line segment positions for one polyline layer.

    Parameters
    ----------
    layer : SparseLineLayer
        Line layer to render.
    z_index : int
        Current Z index for 2D rendering.
    use_3d : bool
        Whether to return X, Y, Z positions instead of X, Y positions.
    project_z : bool
        Whether to project all Z planes into one 2D view.

    Returns
    -------
    tuple[numpy.ndarray, str, int, numpy.ndarray]
        Vertex positions, color, line width, and explicit segment connectivity.
    """
    accumulator = _LineVisualAccumulator(_display_scale_zyx(layer.spacing_zyx_um))
    if layer.z_aware and not use_3d and project_z:
        _append_projected_z_lines(layer, accumulator, z_index, use_3d)
        return _layer_line_visual_result(layer, accumulator, use_3d)

    if layer.z_aware and not use_3d and layer.z_polylines_by_index is not None:
        _append_z_indexed_lines(layer, accumulator, z_index, use_3d)
        return _layer_line_visual_result(layer, accumulator, use_3d)

    if layer.z_aware:
        _append_z_span_lines(layer, accumulator, z_index, use_3d)
    else:
        _append_repeated_lines(layer, accumulator, z_index, use_3d)
    return _layer_line_visual_result(layer, accumulator, use_3d)


def _layer_line_visual_result(
    layer: SparseLineLayer,
    accumulator: _LineVisualAccumulator,
    use_3d: bool,
) -> tuple[np.ndarray, str, int, np.ndarray]:
    """
    Return accumulated line visual data for a layer.

    Parameters
    ----------
    layer : SparseLineLayer
        Sparse line layer.
    accumulator : _LineVisualAccumulator
        Accumulated line data.
    use_3d : bool
        Whether positions are 3D.

    Returns
    -------
    tuple[numpy.ndarray, str, int, numpy.ndarray]
        Vertices, color, width, and connectivity.
    """
    return _line_visual_result(
        accumulator.positions,
        accumulator.connects,
        color=layer.color,
        width=layer.width,
        use_3d=use_3d,
    )


def _append_projected_z_lines(
    layer: SparseLineLayer,
    accumulator: _LineVisualAccumulator,
    z_index: int,
    use_3d: bool,
) -> None:
    """
    Append max-projected z-aware polylines.

    Parameters
    ----------
    layer : SparseLineLayer
        Sparse line layer.
    accumulator : _LineVisualAccumulator
        Accumulated line data.
    z_index : int
        Current Z index.
    use_3d : bool
        Whether positions are 3D.
    """
    for points_yx in layer.max_polylines_yx:
        accumulator.append(points_yx, float(z_index), use_3d)


def _append_z_indexed_lines(
    layer: SparseLineLayer,
    accumulator: _LineVisualAccumulator,
    z_index: int,
    use_3d: bool,
) -> None:
    """
    Append z-aware polylines already indexed by rounded Z.

    Parameters
    ----------
    layer : SparseLineLayer
        Sparse line layer.
    accumulator : _LineVisualAccumulator
        Accumulated line data.
    z_index : int
        Current Z index.
    use_3d : bool
        Whether positions are 3D.
    """
    if layer.z_polylines_by_index is None:
        return
    for points_yx in layer.z_polylines_by_index.get(int(z_index), ()):
        accumulator.append(points_yx, float(z_index), use_3d)


def _append_z_span_lines(
    layer: SparseLineLayer,
    accumulator: _LineVisualAccumulator,
    z_index: int,
    use_3d: bool,
) -> None:
    """
    Append z-aware polylines represented by Z spans.

    Parameters
    ----------
    layer : SparseLineLayer
        Sparse line layer.
    accumulator : _LineVisualAccumulator
        Accumulated line data.
    z_index : int
        Current Z index.
    use_3d : bool
        Whether positions are 3D.
    """
    for z_min, z_max, points_yx in layer.polylines:
        if not use_3d and not (int(z_min) <= int(z_index) <= int(z_max)):
            continue
        z_value = (float(z_min) + float(z_max)) / 2.0
        accumulator.append(points_yx, z_value, use_3d)


def _append_repeated_lines(
    layer: SparseLineLayer,
    accumulator: _LineVisualAccumulator,
    z_index: int,
    use_3d: bool,
) -> None:
    """
    Append 2D polylines to one slice or every Z plane.

    Parameters
    ----------
    layer : SparseLineLayer
        Sparse line layer.
    accumulator : _LineVisualAccumulator
        Accumulated line data.
    z_index : int
        Current Z index.
    use_3d : bool
        Whether positions are 3D.
    """
    for points_yx in layer.polylines:
        accumulator.append(points_yx, float(z_index), use_3d)


class SparseVispyOverlay:
    """Own sparse VisPy marker and line visuals for one NDV canvas."""

    def __init__(self) -> None:
        """Initialize an empty sparse overlay manager."""
        self._array_viewer: Any | None = None
        self._point_visuals: list[Any] = []
        self._line_visuals: list[Any] = []
        self._point_data_cache: dict[
            tuple[int, int, bool, bool], tuple[np.ndarray, np.ndarray, float]
        ] = {}
        self._line_data_cache: dict[
            tuple[int, int, bool, bool], tuple[np.ndarray, str, int, np.ndarray]
        ] = {}
        self._visual_ndim: int | None = None
        self.payload = SparseOverlayPayload()

    def attach(self, array_viewer: Any) -> None:
        """
        Attach sparse visuals to an NDV ArrayViewer.

        Parameters
        ----------
        array_viewer : Any
            NDV ArrayViewer instance.
        """
        if array_viewer is self._array_viewer:
            return
        self.clear()
        self._array_viewer = array_viewer

    def clear(self) -> None:
        """Remove all sparse visuals from the current canvas."""
        for visual in [
            *self._point_visuals,
            *self._line_visuals,
        ]:
            visual.parent = None
        self._point_visuals.clear()
        self._line_visuals.clear()
        self._point_data_cache.clear()
        self._line_data_cache.clear()
        self._visual_ndim = None
        self.payload = SparseOverlayPayload()
        self._array_viewer = None

    def set_payload(
        self,
        payload: SparseOverlayPayload,
    ) -> None:
        """
        Replace the sparse payload and clear payload-derived caches.

        Parameters
        ----------
        payload : SparseOverlayPayload
            Sparse overlay payload.
        """
        if payload is not self.payload:
            self._point_data_cache.clear()
            self._line_data_cache.clear()
        self.payload = payload

    def update(self, z_index: int, use_3d: bool, project_z: bool = False) -> None:
        """
        Refresh sparse visuals for the current Z index and dimensionality.

        Parameters
        ----------
        z_index : int
            Current Z index for 2D rendering.
        use_3d : bool
            Whether sparse geometry should be rendered in 3D.
        project_z : bool
            Whether sparse geometry should be projected across Z.
        """
        if self._array_viewer is None:
            return
        canvas_controller, view, canvas = ndv_canvas_parts(self._array_viewer)
        if canvas_controller is None or view is None:
            return

        self._sync_visual_mode(canvas_controller, view, use_3d)
        self._sync_visual_counts()
        self._update_point_visuals(view, z_index, use_3d, project_z)
        self._update_line_visuals(view, z_index, use_3d, project_z)

        if canvas is not None:
            canvas.update()

    def _sync_visual_mode(
        self, canvas_controller: Any, view: Any, use_3d: bool
    ) -> None:
        """
        Synchronize VisPy canvas dimensionality and camera state.

        Parameters
        ----------
        canvas_controller : Any
            NDV canvas controller.
        view : Any
            VisPy scene view.
        use_3d : bool
            Whether sparse geometry is 3D.
        """
        ndim = 3 if use_3d else 2
        ndim_changed = self._visual_ndim != ndim
        if self._visual_ndim is not None and ndim_changed:
            self._clear_visuals()
        if ndim_changed:
            with np.errstate(invalid="ignore"):
                canvas_controller.set_ndim(ndim)
            _configure_sparse_camera(view, use_3d)
        self._visual_ndim = ndim

    def _update_point_visuals(
        self,
        view: Any,
        z_index: int,
        use_3d: bool,
        project_z: bool,
    ) -> None:
        """
        Update all transcript point visuals.

        Parameters
        ----------
        view : Any
            VisPy scene view.
        z_index : int
            Current Z index.
        use_3d : bool
            Whether sparse geometry is 3D.
        project_z : bool
            Whether sparse geometry is max-projected.
        """
        for layer_index, (visual, layer) in enumerate(
            zip(self._point_visuals, self.payload.points, strict=True)
        ):
            positions, colors, marker_size = self._cached_point_data(
                layer_index,
                layer,
                z_index,
                use_3d,
                project_z,
            )
            visual.parent = view.scene
            visual.set_data(
                positions,
                face_color=colors,
                edge_color=colors,
                size=marker_size,
                symbol="disc",
                edge_width=0,
            )
            visual.order = 150

    def _update_line_visuals(
        self,
        view: Any,
        z_index: int,
        use_3d: bool,
        project_z: bool,
    ) -> None:
        """
        Update all sparse line visuals.

        Parameters
        ----------
        view : Any
            VisPy scene view.
        z_index : int
            Current Z index.
        use_3d : bool
            Whether sparse geometry is 3D.
        project_z : bool
            Whether sparse geometry is max-projected.
        """
        for layer_index, (visual, layer) in enumerate(
            zip(self._line_visuals, self.payload.lines, strict=True)
        ):
            positions, color, width, connect = self._cached_line_data(
                layer_index,
                layer,
                z_index,
                use_3d,
                project_z,
            )
            visual.parent = view.scene
            visual.set_data(pos=positions, color=color, width=width, connect=connect)
            visual.order = 140

    def _sync_visual_counts(self) -> None:
        """Resize marker and line visuals to match the sparse payload."""
        from vispy import scene

        while len(self._point_visuals) > len(self.payload.points):
            visual = self._point_visuals.pop()
            visual.parent = None
        while len(self._point_visuals) < len(self.payload.points):
            self._point_visuals.append(
                scene.visuals.Markers(
                    scaling="fixed",
                    antialias=1,
                    spherical=True,
                )
            )

        count = len(self.payload.lines)
        while len(self._line_visuals) > count:
            visual = self._line_visuals.pop()
            visual.parent = None
        while len(self._line_visuals) < count:
            self._line_visuals.append(scene.visuals.Line(method="gl", antialias=True))

    def _cached_point_data(
        self,
        layer_index: int,
        layer: SparsePointLayer,
        z_index: int,
        use_3d: bool,
        project_z: bool,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Return cached marker positions and colors for one sparse point layer.

        Parameters
        ----------
        layer_index : int
            Point layer index.
        layer : SparsePointLayer
            Sparse point layer.
        z_index : int
            Current Z index.
        use_3d : bool
            Whether sparse geometry is 3D.
        project_z : bool
            Whether sparse geometry is max-projected.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, float]
            Marker positions, RGBA colors, and marker size.
        """
        cache_z = -1 if use_3d or project_z else int(z_index)
        key = (layer_index, cache_z, use_3d, project_z)
        if key not in self._point_data_cache:
            self._point_data_cache[key] = selected_point_data(
                layer,
                z_index,
                use_3d,
                project_z,
            )
        return self._point_data_cache[key]

    def _cached_line_data(
        self,
        layer_index: int,
        layer: SparseLineLayer,
        z_index: int,
        use_3d: bool,
        project_z: bool,
    ) -> tuple[np.ndarray, str, int, np.ndarray]:
        """
        Return cached line vertices and connectivity for one sparse line layer.

        Parameters
        ----------
        layer_index : int
            Line layer index.
        layer : SparseLineLayer
            Sparse line layer.
        z_index : int
            Current Z index.
        use_3d : bool
            Whether sparse geometry is 3D.
        project_z : bool
            Whether sparse geometry is max-projected.

        Returns
        -------
        tuple[numpy.ndarray, str, int, numpy.ndarray]
            Vertices, color, width, and connectivity.
        """
        cache_z = -1 if use_3d or project_z else int(z_index)
        key = (layer_index, cache_z, use_3d, project_z)
        if key not in self._line_data_cache:
            self._line_data_cache[key] = selected_line_data(
                layer,
                z_index,
                use_3d,
                project_z,
            )
        return self._line_data_cache[key]

    def _clear_visuals(self) -> None:
        """Remove existing visuals while preserving the payload and viewer."""
        for visual in [
            *self._point_visuals,
            *self._line_visuals,
        ]:
            visual.parent = None
        self._point_visuals.clear()
        self._line_visuals.clear()
        self._point_data_cache.clear()
        self._line_data_cache.clear()
        self._visual_ndim = None


def _configure_sparse_camera(view: Any, use_3d: bool) -> None:
    """
    Use orthographic projection for scientific sparse 3D overlays.

    Parameters
    ----------
    view : Any
        VisPy scene view.
    use_3d : bool
        Whether sparse geometry is 3D.
    """
    if not use_3d:
        return
    camera = getattr(view, "camera", None)
    if camera is None:
        return
    if hasattr(camera, "fov"):
        camera.fov = 0.0
    if hasattr(camera, "depth_value"):
        camera.depth_value = 1e6


def _empty_positions(use_3d: bool) -> np.ndarray:
    """
    Return an empty VisPy position array.

    Parameters
    ----------
    use_3d : bool
        Whether the array should have three columns.

    Returns
    -------
    numpy.ndarray
        Empty position array.
    """
    return np.empty((0, 3 if use_3d else 2), dtype=np.float32)


def _points_to_vispy_positions(
    coords_zyx: np.ndarray,
    use_3d: bool,
    spacing_zyx_um: tuple[float, float, float],
) -> np.ndarray:
    """
    Convert Z, Y, X point coordinates to VisPy X, Y or X, Y, Z order.

    Parameters
    ----------
    coords_zyx : numpy.ndarray
        Point coordinates in Z, Y, X order.
    use_3d : bool
        Whether output positions should include Z.
    spacing_zyx_um : tuple[float, float, float]
        Z, Y, X voxel spacing in microns.

    Returns
    -------
    numpy.ndarray
        VisPy marker positions.
    """
    if use_3d:
        return _zyx_to_vispy_positions(coords_zyx, spacing_zyx_um)
    return coords_zyx[:, [2, 1]].astype(np.float32, copy=False)


def _append_polyline_visual_data(
    positions: list[np.ndarray],
    connects: list[np.ndarray],
    points_yx: np.ndarray,
    z_value: float,
    use_3d: bool,
    scale_zyx: np.ndarray,
    vertex_count: int,
) -> int:
    """
    Append one polyline's VisPy vertices and segment connectivity.

    Parameters
    ----------
    positions : list[numpy.ndarray]
        Destination vertex arrays.
    connects : list[numpy.ndarray]
        Destination connectivity arrays.
    points_yx : numpy.ndarray
        Polyline vertices in Y, X order.
    z_value : float
        Z plane for the polyline.
    use_3d : bool
        Whether output positions should include Z.
    scale_zyx : numpy.ndarray
        Z, Y, X display scale factors.
    vertex_count : int
        Current total vertex count.

    Returns
    -------
    int
        Updated total vertex count.
    """
    pos = _polyline_positions(points_yx, z_value, use_3d, scale_zyx)
    if pos.shape[0] < 2:
        return vertex_count
    offset = vertex_count
    connect = np.column_stack(
        [
            np.arange(offset, offset + pos.shape[0] - 1, dtype=np.uint32),
            np.arange(offset + 1, offset + pos.shape[0], dtype=np.uint32),
        ]
    )
    positions.append(pos)
    connects.append(connect)
    return vertex_count + pos.shape[0]


def _polyline_positions(
    points_yx: np.ndarray,
    z_value: float,
    use_3d: bool,
    scale_zyx: np.ndarray,
) -> np.ndarray:
    """
    Convert one Y, X polyline to VisPy vertex positions.

    Parameters
    ----------
    points_yx : numpy.ndarray
        Polyline vertices in Y, X order.
    z_value : float
        Z plane for the polyline.
    use_3d : bool
        Whether output positions should include Z.
    scale_zyx : numpy.ndarray
        Z, Y, X display scale factors.

    Returns
    -------
    numpy.ndarray
        VisPy line vertices.
    """
    points = np.asarray(points_yx, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        return _empty_positions(use_3d)
    if use_3d:
        z_column = np.full(points.shape[0], z_value * scale_zyx[0], dtype=float)
        positions = np.column_stack(
            [
                points[:, 1] * scale_zyx[2],
                points[:, 0] * scale_zyx[1],
                z_column,
            ]
        )
    else:
        positions = points[:, [1, 0]]
    return positions.astype(np.float32, copy=False)


def _zyx_to_vispy_positions(
    coords_zyx: np.ndarray,
    spacing_zyx_um: tuple[float, float, float],
) -> np.ndarray:
    """
    Convert Z, Y, X coordinates to physically scaled VisPy X, Y, Z positions.

    Parameters
    ----------
    coords_zyx : numpy.ndarray
        Coordinates in Z, Y, X order.
    spacing_zyx_um : tuple[float, float, float]
        Z, Y, X voxel spacing in microns.

    Returns
    -------
    numpy.ndarray
        VisPy positions in X, Y, Z order.
    """
    scale_zyx = _display_scale_zyx(spacing_zyx_um)
    positions = np.column_stack(
        [
            coords_zyx[:, 2] * scale_zyx[2],
            coords_zyx[:, 1] * scale_zyx[1],
            coords_zyx[:, 0] * scale_zyx[0],
        ]
    )
    return positions.astype(np.float32, copy=False)


def _display_scale_zyx(spacing_zyx_um: tuple[float, float, float]) -> np.ndarray:
    """
    Return Z, Y, X display scale relative to the X voxel size.

    Parameters
    ----------
    spacing_zyx_um : tuple[float, float, float]
        Z, Y, X voxel spacing in microns.

    Returns
    -------
    numpy.ndarray
        Relative Z, Y, X display scale.
    """
    spacing = np.asarray(spacing_zyx_um, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Expected positive finite Z, Y, X voxel spacing.")
    return spacing / spacing[2]


def _hex_to_rgba(color_hex: str) -> tuple[float, float, float, float]:
    """
    Convert a hex RGB color to RGBA floats.

    Parameters
    ----------
    color_hex : str
        Hex RGB color.

    Returns
    -------
    tuple[float, float, float, float]
        RGBA color with values in [0, 1].
    """
    color = color_hex.lstrip("#")
    return (
        int(color[0:2], 16) / 255.0,
        int(color[2:4], 16) / 255.0,
        int(color[4:6], 16) / 255.0,
        1.0,
    )
