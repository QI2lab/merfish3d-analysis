"""Image loading and sparse geometry helpers for the datastore viewer."""

import ast
import struct
from dataclasses import dataclass
from operator import itemgetter
from typing import Any

import numpy as np

from merfish3danalysis.viewer.models import (
    ChannelStack,
    GlobalChannelStack,
    LazyGlobalChannelData,
)


def _axis_indices(axis_key: Any, axis_size: int) -> tuple[np.ndarray, bool]:
    """
    Return selected axis indices and whether the key was scalar.

    Parameters
    ----------
    axis_key : Any
        NumPy-style axis key.
    axis_size : int
        Axis length.

    Returns
    -------
    tuple[numpy.ndarray, bool]
        Selected indices and scalar-key flag.
    """
    if isinstance(axis_key, np.integer):
        axis_key = int(axis_key)
    indices = np.arange(axis_size, dtype=np.intp)
    if isinstance(axis_key, int):
        return np.asarray([indices[axis_key]], dtype=np.intp), True
    return np.asarray(indices[axis_key], dtype=np.intp), False


@dataclass(frozen=True)
class LazyRepeatedPlaneImage:
    """A 2D image virtually repeated across all Z planes."""

    plane_yx: np.ndarray
    z_size: int

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return Z, Y, X shape."""
        plane_shape = tuple(int(value) for value in self.plane_yx.shape)
        return (self.z_size, *plane_shape)

    @property
    def dtype(self) -> np.dtype:
        """Return the image dtype."""
        dtype = self.plane_yx.dtype
        numpy_dtype = getattr(dtype, "numpy_dtype", None)
        if numpy_dtype is not None:
            return np.dtype(numpy_dtype)
        return np.dtype(dtype)

    @property
    def max_value(self) -> int | None:
        """Return the maximum display value without expanding across Z."""
        if hasattr(self.plane_yx, "read"):
            return None
        if self.plane_yx.size == 0:
            return 0
        return int(np.max(self.plane_yx))

    def __getitem__(self, key: Any) -> np.ndarray:
        """
        Return a requested repeated-plane slice.

        Parameters
        ----------
        key : Any
            Z, Y, X key.

        Returns
        -------
        numpy.ndarray
            Requested image slice.
        """
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (3 - len(key))
        z_indices, z_scalar = _axis_indices(key[0], self.z_size)
        plane = self.plane_yx[key[1], key[2]]
        if hasattr(plane, "read"):
            plane = plane.read().result()
        plane = np.asarray(plane)
        if z_scalar:
            return plane
        return np.broadcast_to(plane, (len(z_indices), *plane.shape)).copy()


@dataclass(frozen=True)
class LazyMaxProjectionImage:
    """A lazy Z-max projection presented as one Z plane."""

    image_zyx: Any

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return projected Z, Y, X shape."""
        image_shape = tuple(int(value) for value in self.image_zyx.shape)
        return (1, *image_shape[1:])

    @property
    def dtype(self) -> np.dtype:
        """Return the image dtype."""
        dtype = self.image_zyx.dtype
        numpy_dtype = getattr(dtype, "numpy_dtype", None)
        if numpy_dtype is not None:
            return np.dtype(numpy_dtype)
        return np.dtype(dtype)

    @property
    def max_value(self) -> int | None:
        """Return no cheap display maximum for lazy projected data."""
        return None

    def __getitem__(self, key: Any) -> np.ndarray:
        """
        Return a requested max-projected slice.

        Parameters
        ----------
        key : Any
            Z, Y, X key.

        Returns
        -------
        numpy.ndarray
            Requested max-projected image slice.
        """
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (3 - len(key))
        _z_indices, z_scalar = _axis_indices(key[0], 1)
        selected = self.image_zyx[(slice(None), key[1], key[2])]
        if hasattr(selected, "read"):
            selected = selected.read().result()
        plane = np.max(np.asarray(selected), axis=0)
        if z_scalar:
            return plane
        return plane[np.newaxis, ...]


@dataclass(frozen=True)
class LazyZeroVolume:
    """A lazy black Z, Y, X canvas."""

    shape: tuple[int, int, int]

    @property
    def dtype(self) -> np.dtype:
        """Return the canvas dtype."""
        return np.dtype(np.uint16)

    def __getitem__(self, key: Any) -> np.ndarray:
        """
        Return a requested zero-filled slice.

        Parameters
        ----------
        key : Any
            Z, Y, X key.

        Returns
        -------
        numpy.ndarray
            Requested zero-filled canvas slice.
        """
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (3 - len(key))
        axis_indices = [_axis_indices(key[axis], self.shape[axis]) for axis in range(3)]
        output_shape = tuple(
            len(indices) for indices, is_scalar in axis_indices if not is_scalar
        )
        return np.broadcast_to(np.asarray(0, dtype=np.uint16), output_shape)


@dataclass(frozen=True)
class _GlobalBaseImage:
    """Selected global base image for NDV display."""

    image_zyx: Any
    shape_zyx: tuple[int, int, int]
    labels: list[str]


@dataclass(frozen=True)
class PointOverlayIndex:
    """Reusable Z-indexed transcript point coordinates and gene ids."""

    coords_sorted: np.ndarray
    genes_sorted: np.ndarray
    z_sorted: np.ndarray
    shape: tuple[int, int, int]

    @classmethod
    def from_points(
        cls,
        coords_zyx: np.ndarray,
        genes: np.ndarray,
        shape: tuple[int, int, int],
    ) -> "PointOverlayIndex":
        """
        Build a reusable point index sorted by rounded Z.

        Parameters
        ----------
        coords_zyx : numpy.ndarray
            Point coordinates in Z, Y, X pixel units.
        genes : numpy.ndarray
            Gene id for each point.
        shape : tuple[int, int, int]
            Target Z, Y, X shape.

        Returns
        -------
        PointOverlayIndex
            Indexed point coordinates.
        """
        coords = np.asarray(coords_zyx, dtype=float)
        gene_ids = np.asarray(genes, dtype=object)
        if coords.size == 0 or gene_ids.size == 0:
            return cls(
                np.empty((0, 3), dtype=float),
                gene_ids[:0],
                np.empty(0, dtype=np.intp),
                shape,
            )
        finite = np.isfinite(coords).all(axis=1)
        coords = coords[finite]
        gene_ids = gene_ids[finite]
        z_int = np.round(coords[:, 0]).astype(np.intp, copy=False)
        order = np.argsort(z_int)
        return cls(coords[order], gene_ids[order], z_int[order], shape)


@dataclass(frozen=True)
class PolylineGeometry:
    """Sparse 2D polyline geometry virtually repeated across Z planes."""

    polylines_yx: tuple[np.ndarray, ...]
    shape: tuple[int, int, int]
    line_thickness: int
    value: int = 1


@dataclass(frozen=True)
class ZPolylineGeometry:
    """Sparse polyline geometry valid over selected Z intervals."""

    polylines: tuple[Any, ...]
    shape: tuple[int, int, int]
    line_thickness: int
    z_polylines_by_index: dict[int, tuple[np.ndarray, ...]] | None = None
    max_polylines_yx: tuple[np.ndarray, ...] = ()
    value: int = 1


@dataclass(frozen=True)
class _LocalBaysorLayer:
    """Processed local geometry for one Baysor cell layer."""

    z_min: float
    z_max: float
    lower_yx: np.ndarray
    closed_yx: np.ndarray
    area: float


def _empty_point_index(shape_zyx: tuple[int, int, int]) -> PointOverlayIndex:
    """
    Return an empty point index for a canvas shape.

    Parameters
    ----------
    shape_zyx : tuple[int, int, int]
        Target Z, Y, X canvas shape.

    Returns
    -------
    PointOverlayIndex
        Empty point index.
    """
    empty_coords = np.empty((0, 3), dtype=float)
    empty_genes = np.empty(0, dtype=object)
    return PointOverlayIndex.from_points(empty_coords, empty_genes, shape_zyx)


def global_datastore_transcript_index(
    datastore_transcripts: Any,
    shape_zyx: tuple[int, int, int],
    origin_zyx_um: Any,
    spacing_zyx_um: Any,
) -> PointOverlayIndex:
    """
    Build a reusable global datastore transcript index.

    Parameters
    ----------
    datastore_transcripts : Any
        Final datastore transcript table.
    shape_zyx : tuple[int, int, int]
        Fused image shape.
    origin_zyx_um : Any
        Fused image origin in Z, Y, X microns.
    spacing_zyx_um : Any
        Fused image spacing in Z, Y, X microns.

    Returns
    -------
    PointOverlayIndex
        Indexed datastore transcript coordinates.
    """
    if datastore_transcripts is None or len(datastore_transcripts) == 0:
        return _empty_point_index(shape_zyx)
    required_columns = {"global_y", "global_x", "gene_id"}
    if not required_columns.issubset(datastore_transcripts.columns):
        return _empty_point_index(shape_zyx)

    origin = np.asarray(origin_zyx_um, dtype=float)
    spacing = np.asarray(spacing_zyx_um, dtype=float)
    coords_yx_um = datastore_transcripts[["global_y", "global_x"]].to_numpy(
        dtype=float,
        copy=False,
    )
    coords_yx = (coords_yx_um - origin[1:]) / spacing[1:]
    if "global_z" in datastore_transcripts.columns:
        coords_z_um = datastore_transcripts["global_z"].to_numpy(
            dtype=float, copy=False
        )
        coords_z = (coords_z_um - origin[0]) / spacing[0]
    else:
        coords_z = np.zeros(coords_yx.shape[0], dtype=float)
    coords_zyx = np.column_stack([coords_z, coords_yx[:, 0], coords_yx[:, 1]])
    genes = (
        datastore_transcripts["gene_id"].astype(str).to_numpy(dtype=object, copy=False)
    )
    return PointOverlayIndex.from_points(coords_zyx, genes, shape_zyx)


def _local_point_index_from_global_zyx_um(
    coords_zyx_um: np.ndarray,
    genes: np.ndarray,
    *,
    shape_zyx: tuple[int, int, int],
    affine_zyx_um: np.ndarray,
    origin_zyx_um: np.ndarray,
    spacing_zyx_um: np.ndarray,
) -> PointOverlayIndex:
    """
    Build a local point index from global Z, Y, X micron coordinates.

    Parameters
    ----------
    coords_zyx_um : numpy.ndarray
        Global point coordinates in Z, Y, X microns.
    genes : numpy.ndarray
        Gene names for each point.
    shape_zyx : tuple[int, int, int]
        Target Z, Y, X canvas shape.
    affine_zyx_um : numpy.ndarray
        Tile-to-global affine.
    origin_zyx_um : numpy.ndarray
        Tile origin in microns.
    spacing_zyx_um : numpy.ndarray
        Tile Z, Y, X voxel spacing in microns.

    Returns
    -------
    PointOverlayIndex
        Local transcript point index.
    """
    if coords_zyx_um.size == 0:
        return _empty_point_index(shape_zyx)
    affine = np.asarray(affine_zyx_um, dtype=float)
    origin = np.asarray(origin_zyx_um, dtype=float)
    spacing = np.asarray(spacing_zyx_um, dtype=float)
    lower_bound, upper_bound = _tile_global_bounds_zyx_um(
        shape_zyx,
        affine_zyx_um=affine,
        origin_zyx_um=origin,
        spacing_zyx_um=spacing,
        radius=0,
    )
    in_tile_bounds = (
        (coords_zyx_um >= lower_bound) & (coords_zyx_um <= upper_bound)
    ).all(axis=1)
    if not in_tile_bounds.any():
        return _empty_point_index(shape_zyx)
    local_coords = _global_zyx_um_to_tile_zyx_px(
        coords_zyx_um[in_tile_bounds],
        affine_zyx_um=affine,
        origin_zyx_um=origin,
        spacing_zyx_um=spacing,
    )
    return PointOverlayIndex.from_points(
        local_coords,
        np.asarray(genes, dtype=object)[in_tile_bounds],
        shape_zyx,
    )


def local_datastore_transcript_index(
    datastore_transcripts: Any,
    shape_zyx: tuple[int, int, int],
    affine_zyx_um: np.ndarray,
    origin_zyx_um: np.ndarray,
    spacing_zyx_um: np.ndarray,
) -> PointOverlayIndex:
    """
    Build local tile transcript coordinates from global datastore coordinates.

    Parameters
    ----------
    datastore_transcripts : Any
        Final datastore transcript table with global coordinates.
    shape_zyx : tuple[int, int, int]
        Output Z, Y, X shape.
    affine_zyx_um : numpy.ndarray
        Tile-to-global affine.
    origin_zyx_um : numpy.ndarray
        Tile physical origin.
    spacing_zyx_um : numpy.ndarray
        Tile spacing.

    Returns
    -------
    PointOverlayIndex
        Indexed local point coordinates.
    """
    if datastore_transcripts is None or len(datastore_transcripts) == 0:
        return _empty_point_index(shape_zyx)
    required_columns = {"global_z", "global_y", "global_x", "gene_id"}
    if not required_columns.issubset(datastore_transcripts.columns):
        return _empty_point_index(shape_zyx)

    coords_zyx = datastore_transcripts[["global_z", "global_y", "global_x"]].to_numpy(
        dtype=float,
        copy=False,
    )
    genes = (
        datastore_transcripts["gene_id"]
        .astype(str)
        .to_numpy(
            dtype=object,
            copy=False,
        )
    )
    return _local_point_index_from_global_zyx_um(
        coords_zyx,
        genes,
        shape_zyx=shape_zyx,
        affine_zyx_um=affine_zyx_um,
        origin_zyx_um=origin_zyx_um,
        spacing_zyx_um=spacing_zyx_um,
    )


def global_transcript_index(
    transcripts: Any,
    shape_zyx: tuple[int, int, int],
    origin_zyx_um: Any,
    spacing_zyx_um: Any,
) -> PointOverlayIndex:
    """
    Build a reusable global transcript index from x, y, z, gene columns.

    Parameters
    ----------
    transcripts : Any
        Transcript table.
    shape_zyx : tuple[int, int, int]
        Fused image shape.
    origin_zyx_um : Any
        Fused image origin in Z, Y, X microns.
    spacing_zyx_um : Any
        Fused image spacing in Z, Y, X microns.

    Returns
    -------
    PointOverlayIndex
        Indexed transcript coordinates.
    """
    if transcripts is None or len(transcripts) == 0:
        return _empty_point_index(shape_zyx)
    required_columns = {"x", "y", "z", "gene"}
    if not required_columns.issubset(transcripts.columns):
        return _empty_point_index(shape_zyx)

    origin = np.asarray(origin_zyx_um, dtype=float)
    spacing = np.asarray(spacing_zyx_um, dtype=float)
    coords_xyz = transcripts[["x", "y", "z"]].to_numpy(dtype=float, copy=False)
    coords_zyx = coords_xyz[:, [2, 1, 0]]
    coords_px = (coords_zyx - origin) / spacing
    genes = transcripts["gene"].astype(str).to_numpy(dtype=object, copy=False)
    return PointOverlayIndex.from_points(coords_px, genes, shape_zyx)


def _global_zyx_um_to_tile_zyx_px(
    global_zyx_um: np.ndarray,
    affine_zyx_um: np.ndarray,
    origin_zyx_um: np.ndarray,
    spacing_zyx_um: np.ndarray,
) -> np.ndarray:
    """
    Transform global Z, Y, X micron coordinates into tile Z, Y, X pixels.

    Parameters
    ----------
    global_zyx_um : numpy.ndarray
        Global coordinates in Z, Y, X microns.
    affine_zyx_um : numpy.ndarray
        Tile-to-global affine.
    origin_zyx_um : numpy.ndarray
        Tile physical origin.
    spacing_zyx_um : numpy.ndarray
        Tile spacing.

    Returns
    -------
    numpy.ndarray
        Local tile coordinates in Z, Y, X pixels.
    """
    inverse_affine = np.linalg.inv(np.asarray(affine_zyx_um, dtype=float))
    homogeneous = np.concatenate(
        [
            np.asarray(global_zyx_um, dtype=float),
            np.ones((global_zyx_um.shape[0], 1), dtype=float),
        ],
        axis=1,
    )
    physical_zyx = (homogeneous @ inverse_affine.T)[:, :3]
    return (physical_zyx - origin_zyx_um) / spacing_zyx_um


def _tile_global_bounds_zyx_um(
    shape_zyx: tuple[int, int, int],
    affine_zyx_um: np.ndarray,
    origin_zyx_um: np.ndarray,
    spacing_zyx_um: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the global coordinate bounds for one local tile canvas.

    Parameters
    ----------
    shape_zyx : tuple[int, int, int]
        Local Z, Y, X canvas shape.
    affine_zyx_um : numpy.ndarray
        Tile-to-global affine.
    origin_zyx_um : numpy.ndarray
        Tile physical origin.
    spacing_zyx_um : numpy.ndarray
        Tile spacing.
    radius : int
        Marker radius in pixels.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Minimum and maximum global Z, Y, X bounds in microns.
    """
    shape = np.asarray(shape_zyx, dtype=float)
    lower_px = np.full(3, -radius, dtype=float)
    upper_px = shape - 1.0 + radius
    corners_px = np.asarray(
        [
            [z, y, x]
            for z in (lower_px[0], upper_px[0])
            for y in (lower_px[1], upper_px[1])
            for x in (lower_px[2], upper_px[2])
        ],
        dtype=float,
    )
    local_physical_zyx = origin_zyx_um + corners_px * spacing_zyx_um
    homogeneous = np.concatenate(
        [
            local_physical_zyx,
            np.ones((local_physical_zyx.shape[0], 1), dtype=float),
        ],
        axis=1,
    )
    global_corners_zyx = (homogeneous @ affine_zyx_um.T)[:, :3]
    return global_corners_zyx.min(axis=0), global_corners_zyx.max(axis=0)


def local_transcript_index(
    transcripts: Any,
    shape_zyx: tuple[int, int, int],
    affine_zyx_um: np.ndarray,
    origin_zyx_um: np.ndarray,
    spacing_zyx_um: np.ndarray,
) -> PointOverlayIndex:
    """
    Build local tile transcript coordinates from x, y, z transcript coordinates.

    Parameters
    ----------
    transcripts : Any
        Transcript table with x, y, z, gene columns.
    shape_zyx : tuple[int, int, int]
        Output Z, Y, X shape.
    affine_zyx_um : numpy.ndarray
        Tile-to-global affine.
    origin_zyx_um : numpy.ndarray
        Tile physical origin.
    spacing_zyx_um : numpy.ndarray
        Tile spacing.

    Returns
    -------
    PointOverlayIndex
        Indexed local point coordinates.
    """
    if transcripts is None or len(transcripts) == 0:
        return _empty_point_index(shape_zyx)
    required_columns = {"x", "y", "z", "gene"}
    if not required_columns.issubset(transcripts.columns):
        return _empty_point_index(shape_zyx)

    coords_xyz = transcripts[["x", "y", "z"]].to_numpy(dtype=float, copy=False)
    coords_zyx = coords_xyz[:, [2, 1, 0]]
    genes = transcripts["gene"].astype(str).to_numpy(dtype=object, copy=False)
    return _local_point_index_from_global_zyx_um(
        coords_zyx,
        genes,
        shape_zyx=shape_zyx,
        affine_zyx_um=affine_zyx_um,
        origin_zyx_um=origin_zyx_um,
        spacing_zyx_um=spacing_zyx_um,
    )


def _parse_baysor_layer(layer: Any) -> tuple[float, float] | None:
    """
    Parse one Baysor Z-layer interval.

    Parameters
    ----------
    layer : Any
        Baysor layer value.

    Returns
    -------
    tuple[float, float] or None
        Z interval in microns.
    """
    if isinstance(layer, str):
        try:
            layer = ast.literal_eval(layer)
        except (SyntaxError, ValueError):
            return None
    values = np.asarray(layer, dtype=float)
    if values.size < 2 or not np.isfinite(values[:2]).all():
        return None
    return float(values[0]), float(values[1])


def _parse_wkb_polygon_xy(geometry: Any) -> np.ndarray | None:
    """
    Parse the exterior ring from a little- or big-endian WKB polygon.

    Parameters
    ----------
    geometry : Any
        WKB polygon bytes.

    Returns
    -------
    numpy.ndarray or None
        Exterior ring in X, Y order.
    """
    if geometry is None:
        return None
    data = bytes(geometry)
    if len(data) < 13:
        return None
    byte_order = data[0]
    endian = "<" if byte_order == 1 else ">" if byte_order == 0 else None
    if endian is None:
        return None
    geom_type = struct.unpack_from(f"{endian}I", data, 1)[0]
    if geom_type != 3:
        return None
    ring_count = struct.unpack_from(f"{endian}I", data, 5)[0]
    if ring_count < 1:
        return None
    offset = 9
    point_count = struct.unpack_from(f"{endian}I", data, offset)[0]
    offset += 4
    if point_count < 2 or len(data) < offset + point_count * 16:
        return None
    coords = np.frombuffer(
        data,
        dtype=np.dtype(f"{endian}f8"),
        count=point_count * 2,
        offset=offset,
    )
    return coords.reshape(point_count, 2).astype(float, copy=False)


def _iter_baysor_boundary_rows(
    boundaries: Any,
) -> list[tuple[str, tuple[float, float], np.ndarray]]:
    """
    Return valid Baysor boundary layer intervals and polygon rings.

    Parameters
    ----------
    boundaries : Any
        Baysor boundary table.

    Returns
    -------
    list[tuple[str, tuple[float, float], numpy.ndarray]]
        Cell id, Z interval, and polygon ring rows.
    """
    if boundaries is None or len(boundaries) == 0:
        return []
    required_columns = {"cell", "layer", "geometry"}
    if not required_columns.issubset(boundaries.columns):
        return []

    rows: list[tuple[str, tuple[float, float], np.ndarray]] = []
    for row in boundaries[["cell", "layer", "geometry"]].itertuples(index=False):
        layer = _parse_baysor_layer(row.layer)
        polygon_xy = _parse_wkb_polygon_xy(row.geometry)
        if layer is None or polygon_xy is None:
            continue
        rows.append((str(row.cell), layer, polygon_xy))
    return rows


def _baysor_rows_by_cell(
    rows: list[tuple[str, tuple[float, float], np.ndarray]],
) -> dict[str, list[tuple[tuple[float, float], np.ndarray]]]:
    """
    Group Baysor boundary rows by cell id.

    Parameters
    ----------
    rows : list[tuple[str, tuple[float, float], numpy.ndarray]]
        Parsed Baysor boundary rows.

    Returns
    -------
    dict[str, list[tuple[tuple[float, float], numpy.ndarray]]]
        Boundary rows grouped by cell id.
    """
    grouped: dict[str, list[tuple[tuple[float, float], np.ndarray]]] = {}
    for cell_id, layer, polygon_xy in rows:
        grouped.setdefault(cell_id, []).append((layer, polygon_xy))
    for cell_rows in grouped.values():
        cell_rows.sort(key=itemgetter(0))
    return grouped


def _close_ring_yx(points_yx: np.ndarray) -> np.ndarray:
    """
    Return a closed Y, X ring without duplicate intermediate endpoints.

    Parameters
    ----------
    points_yx : numpy.ndarray
        Ring vertices in Y, X order.

    Returns
    -------
    numpy.ndarray
        Closed ring in Y, X order.
    """
    points = np.asarray(points_yx, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        return np.empty((0, 2), dtype=float)
    if np.allclose(points[0], points[-1]):
        points = points[:-1]
    if points.shape[0] < 2:
        return np.empty((0, 2), dtype=float)
    return np.vstack([points, points[0]])


def _signed_ring_area_yx(points_yx: np.ndarray) -> float:
    """
    Return the signed area of one open Y, X ring.

    Parameters
    ----------
    points_yx : numpy.ndarray
        Ring vertices in Y, X order.

    Returns
    -------
    float
        Signed ring area.
    """
    if points_yx.shape[0] < 3:
        return 0.0
    y_values = points_yx[:, 0]
    x_values = points_yx[:, 1]
    return float(
        0.5
        * np.sum(x_values * np.roll(y_values, -1) - np.roll(x_values, -1) * y_values)
    )


def _ring_area_tolerance_yx(points_yx: np.ndarray) -> float:
    """
    Return a scale-relative tolerance for detecting degenerate rings.

    Parameters
    ----------
    points_yx : numpy.ndarray
        Ring vertices in Y, X order.

    Returns
    -------
    float
        Degenerate-area tolerance.
    """
    ring = _close_ring_yx(points_yx)
    if ring.shape[0] < 2:
        return 0.0
    coordinate_scale = float(np.abs(ring).max(initial=0.0))
    if coordinate_scale == 0.0:
        coordinate_scale = float(np.linalg.norm(np.diff(ring, axis=0), axis=1).sum())
    return np.finfo(float).eps * ring.shape[0] * coordinate_scale * coordinate_scale


def _closed_valid_ring_yx(points_yx: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Return a closed ring and its valid area, or an empty ring and zero area.

    Parameters
    ----------
    points_yx : numpy.ndarray
        Ring vertices in Y, X order.

    Returns
    -------
    tuple[numpy.ndarray, float]
        Closed ring and absolute area.
    """
    ring = _close_ring_yx(points_yx)
    if ring.shape[0] < 4:
        return np.empty((0, 2), dtype=float), 0.0
    area = abs(_signed_ring_area_yx(ring[:-1]))
    if area <= _ring_area_tolerance_yx(ring):
        return np.empty((0, 2), dtype=float), 0.0
    return ring, area


def _add_z_indexed_polyline(
    z_polylines_by_index: dict[int, list[np.ndarray]],
    z_min: float,
    z_max: float,
    points_yx: np.ndarray,
    z_size: int,
    closed_yx: np.ndarray | None = None,
) -> None:
    """
    Add one Y, X polyline to every integer Z plane it spans.

    Parameters
    ----------
    z_polylines_by_index : dict[int, list[numpy.ndarray]]
        Destination map from Z index to polylines.
    z_min : float
        Lower Z coordinate.
    z_max : float
        Upper Z coordinate.
    points_yx : numpy.ndarray
        Polyline vertices in Y, X order.
    z_size : int
        Number of Z planes.
    closed_yx : numpy.ndarray or None
        Optional precomputed closed ring.
    """
    if closed_yx is None:
        closed_yx, area = _closed_valid_ring_yx(points_yx)
        if area == 0.0:
            return
    lower_index = max(0, int(np.floor(min(z_min, z_max))))
    upper_index = min(z_size - 1, int(np.ceil(max(z_min, z_max))))
    if closed_yx.size == 0:
        return
    for z_index in range(lower_index, upper_index + 1):
        z_polylines_by_index.setdefault(z_index, []).append(closed_yx)


def _freeze_z_indexed_polylines(
    z_polylines_by_index: dict[int, list[np.ndarray]],
) -> dict[int, tuple[np.ndarray, ...]]:
    """
    Return an immutable Z-indexed polyline map.

    Parameters
    ----------
    z_polylines_by_index : dict[int, list[numpy.ndarray]]
        Mutable Z-indexed polyline map.

    Returns
    -------
    dict[int, tuple[numpy.ndarray, ...]]
        Z-indexed polyline map.
    """
    return {
        z_index: tuple(polylines)
        for z_index, polylines in z_polylines_by_index.items()
        if polylines
    }


def global_baysor_boundary_geometry(
    boundaries: Any,
    shape_zyx: tuple[int, int, int],
    origin_zyx_um: Any,
    spacing_zyx_um: Any,
    line_thickness: int = 1,
    source_shape_zyx: tuple[int, int, int] | None = None,
) -> ZPolylineGeometry:
    """
    Create global Baysor 3D boundary geometry.

    Parameters
    ----------
    boundaries : Any
        Baysor boundary table.
    shape_zyx : tuple[int, int, int]
        Global Z, Y, X canvas shape.
    origin_zyx_um : Any
        Global image origin in microns.
    spacing_zyx_um : Any
        Global Z, Y, X voxel spacing in microns.
    line_thickness : int, default=1
        Line thickness in pixels.
    source_shape_zyx : tuple[int, int, int] or None
        Full source Z, Y, X shape used for Z filtering.

    Returns
    -------
    ZPolylineGeometry
        Sparse Baysor boundary geometry.
    """
    origin = np.asarray(origin_zyx_um, dtype=float)
    spacing = np.asarray(spacing_zyx_um, dtype=float)
    z_filter_size = (
        int(source_shape_zyx[0]) if source_shape_zyx is not None else int(shape_zyx[0])
    )
    polylines: list[np.ndarray] = []
    z_polylines_by_index: dict[int, list[np.ndarray]] = {}
    max_polylines: list[np.ndarray] = []
    for cell_rows in _baysor_rows_by_cell(
        _iter_baysor_boundary_rows(boundaries)
    ).values():
        max_area = 0.0
        max_yx: np.ndarray | None = None
        for layer, polygon_xy in cell_rows:
            z_min = (layer[0] - origin[0]) / spacing[0]
            z_max = (layer[1] - origin[0]) / spacing[0]
            if max(z_min, z_max) < 0 or min(z_min, z_max) >= z_filter_size:
                continue
            polygon_yx = polygon_xy[:, ::-1]
            local_yx = (polygon_yx - origin[1:]) / spacing[1:]
            if (
                local_yx[:, 0].max() < 0
                or local_yx[:, 0].min() >= shape_zyx[1]
                or local_yx[:, 1].max() < 0
                or local_yx[:, 1].min() >= shape_zyx[2]
            ):
                continue
            closed_yx, area = _closed_valid_ring_yx(local_yx)
            if area > max_area:
                max_area = area
                max_yx = closed_yx
            if area > 0.0:
                polylines.append((z_min, z_max, closed_yx))
            _add_z_indexed_polyline(
                z_polylines_by_index,
                z_min,
                z_max,
                local_yx,
                z_filter_size,
                closed_yx=closed_yx,
            )
        if max_yx is not None:
            max_polylines.append(max_yx)
    return ZPolylineGeometry(
        tuple(polylines),
        shape_zyx,
        line_thickness,
        _freeze_z_indexed_polylines(z_polylines_by_index),
        tuple(max_polylines),
    )


def _baysor_layer_polygon_zyx(
    layer: tuple[float, float],
    polygon_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return lower and upper global Z, Y, X polygons for one Baysor layer.

    Parameters
    ----------
    layer : tuple[float, float]
        Baysor Z interval in microns.
    polygon_xy : numpy.ndarray
        Polygon ring in X, Y microns.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Lower and upper polygons in Z, Y, X order.
    """
    layer_min, layer_max = min(layer), max(layer)
    lower_polygon_zyx = np.column_stack(
        [
            np.full(polygon_xy.shape[0], layer_min, dtype=float),
            polygon_xy[:, 1],
            polygon_xy[:, 0],
        ]
    )
    upper_polygon_zyx = lower_polygon_zyx.copy()
    upper_polygon_zyx[:, 0] = layer_max
    return lower_polygon_zyx, upper_polygon_zyx


def _zyx_bounds_overlap(
    coords_zyx: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> bool:
    """
    Return whether coordinates overlap an inclusive Z, Y, X bounds box.

    Parameters
    ----------
    coords_zyx : numpy.ndarray
        Coordinates in Z, Y, X order.
    lower_bound : numpy.ndarray
        Lower inclusive Z, Y, X bound.
    upper_bound : numpy.ndarray
        Upper inclusive Z, Y, X bound.

    Returns
    -------
    bool
        True when coordinates overlap the bounds.
    """
    return not (
        coords_zyx[:, 0].max() < lower_bound[0]
        or coords_zyx[:, 0].min() > upper_bound[0]
        or coords_zyx[:, 1].max() < lower_bound[1]
        or coords_zyx[:, 1].min() > upper_bound[1]
        or coords_zyx[:, 2].max() < lower_bound[2]
        or coords_zyx[:, 2].min() > upper_bound[2]
    )


def _zyx_inside_shape(coords_zyx: np.ndarray, shape_zyx: tuple[int, int, int]) -> bool:
    """
    Return whether coordinates overlap a Z, Y, X array shape.

    Parameters
    ----------
    coords_zyx : numpy.ndarray
        Coordinates in Z, Y, X order.
    shape_zyx : tuple[int, int, int]
        Z, Y, X array shape.

    Returns
    -------
    bool
        True when coordinates overlap the shape.
    """
    return not (
        coords_zyx[:, 0].max() < 0
        or coords_zyx[:, 0].min() >= shape_zyx[0]
        or coords_zyx[:, 1].max() < 0
        or coords_zyx[:, 1].min() >= shape_zyx[1]
        or coords_zyx[:, 2].max() < 0
        or coords_zyx[:, 2].min() >= shape_zyx[2]
    )


def _local_baysor_layer(
    layer: tuple[float, float],
    polygon_xy: np.ndarray,
    *,
    affine_zyx_um: np.ndarray,
    origin_zyx_um: np.ndarray,
    spacing_zyx_um: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    shape_zyx: tuple[int, int, int],
) -> _LocalBaysorLayer | None:
    """
    Transform and validate one Baysor layer for a local tile.

    Parameters
    ----------
    layer : tuple[float, float]
        Baysor Z interval in microns.
    polygon_xy : numpy.ndarray
        Polygon ring in X, Y microns.
    affine_zyx_um : numpy.ndarray
        Tile-to-global affine.
    origin_zyx_um : numpy.ndarray
        Tile origin in microns.
    spacing_zyx_um : numpy.ndarray
        Tile Z, Y, X voxel spacing in microns.
    lower_bound : numpy.ndarray
        Lower global Z, Y, X tile bound.
    upper_bound : numpy.ndarray
        Upper global Z, Y, X tile bound.
    shape_zyx : tuple[int, int, int]
        Local Z, Y, X canvas shape.

    Returns
    -------
    _LocalBaysorLayer or None
        Local layer geometry, or ``None`` when outside the tile.
    """
    lower_polygon_zyx, upper_polygon_zyx = _baysor_layer_polygon_zyx(layer, polygon_xy)
    polygon_zyx = np.concatenate([lower_polygon_zyx, upper_polygon_zyx])
    if not _zyx_bounds_overlap(polygon_zyx, lower_bound, upper_bound):
        return None
    lower_local_zyx = _global_zyx_um_to_tile_zyx_px(
        lower_polygon_zyx,
        affine_zyx_um=affine_zyx_um,
        origin_zyx_um=origin_zyx_um,
        spacing_zyx_um=spacing_zyx_um,
    )
    upper_local_zyx = _global_zyx_um_to_tile_zyx_px(
        upper_polygon_zyx,
        affine_zyx_um=affine_zyx_um,
        origin_zyx_um=origin_zyx_um,
        spacing_zyx_um=spacing_zyx_um,
    )
    local_zyx = np.concatenate([lower_local_zyx, upper_local_zyx])
    if not _zyx_inside_shape(local_zyx, shape_zyx):
        return None
    lower_yx = lower_local_zyx[:, 1:]
    closed_yx, area = _closed_valid_ring_yx(lower_yx)
    return _LocalBaysorLayer(
        z_min=float(local_zyx[:, 0].min()),
        z_max=float(local_zyx[:, 0].max()),
        lower_yx=lower_yx,
        closed_yx=closed_yx,
        area=area,
    )


def _append_local_baysor_layer(
    layer_geometry: _LocalBaysorLayer,
    *,
    polylines: list[np.ndarray],
    z_polylines_by_index: dict[int, list[np.ndarray]],
    shape_zyx: tuple[int, int, int],
) -> None:
    """
    Append one processed Baysor layer to local geometry collections.

    Parameters
    ----------
    layer_geometry : _LocalBaysorLayer
        Processed local Baysor layer.
    polylines : list[numpy.ndarray]
        Destination 3D polyline list.
    z_polylines_by_index : dict[int, list[numpy.ndarray]]
        Destination Z-indexed polylines.
    shape_zyx : tuple[int, int, int]
        Local Z, Y, X canvas shape.
    """
    if layer_geometry.area > 0.0:
        polylines.append(
            (
                layer_geometry.z_min,
                layer_geometry.z_max,
                layer_geometry.closed_yx,
            )
        )
    _add_z_indexed_polyline(
        z_polylines_by_index,
        layer_geometry.z_min,
        layer_geometry.z_max,
        layer_geometry.lower_yx,
        shape_zyx[0],
        closed_yx=layer_geometry.closed_yx,
    )


def local_baysor_boundary_geometry(
    boundaries: Any,
    shape_zyx: tuple[int, int, int],
    affine_zyx_um: np.ndarray,
    origin_zyx_um: np.ndarray,
    spacing_zyx_um: np.ndarray,
    line_thickness: int = 1,
) -> ZPolylineGeometry:
    """
    Create local Baysor boundary geometry.

    Parameters
    ----------
    boundaries : Any
        Baysor boundary table.
    shape_zyx : tuple[int, int, int]
        Local Z, Y, X canvas shape.
    affine_zyx_um : numpy.ndarray
        Tile-to-global affine.
    origin_zyx_um : numpy.ndarray
        Tile origin in microns.
    spacing_zyx_um : numpy.ndarray
        Tile Z, Y, X voxel spacing in microns.
    line_thickness : int, default=1
        Line thickness in pixels.

    Returns
    -------
    ZPolylineGeometry
        Sparse local Baysor boundary geometry.
    """
    rows = _iter_baysor_boundary_rows(boundaries)
    if not rows:
        return ZPolylineGeometry((), shape_zyx, line_thickness)

    affine = np.asarray(affine_zyx_um, dtype=float)
    origin = np.asarray(origin_zyx_um, dtype=float)
    spacing = np.asarray(spacing_zyx_um, dtype=float)
    lower_bound, upper_bound = _tile_global_bounds_zyx_um(
        shape_zyx,
        affine_zyx_um=affine,
        origin_zyx_um=origin,
        spacing_zyx_um=spacing,
        radius=line_thickness,
    )
    polylines: list[np.ndarray] = []
    z_polylines_by_index: dict[int, list[np.ndarray]] = {}
    max_polylines: list[np.ndarray] = []
    for cell_rows in _baysor_rows_by_cell(rows).values():
        max_area = 0.0
        max_yx: np.ndarray | None = None
        for layer, polygon_xy in cell_rows:
            layer_geometry = _local_baysor_layer(
                layer,
                polygon_xy,
                affine_zyx_um=affine,
                origin_zyx_um=origin,
                spacing_zyx_um=spacing,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                shape_zyx=shape_zyx,
            )
            if layer_geometry is None:
                continue
            if layer_geometry.area > max_area:
                max_area = layer_geometry.area
                max_yx = layer_geometry.closed_yx
            _append_local_baysor_layer(
                layer_geometry,
                polylines=polylines,
                z_polylines_by_index=z_polylines_by_index,
                shape_zyx=shape_zyx,
            )
        if max_yx is not None:
            max_polylines.append(max_yx)
    return ZPolylineGeometry(
        tuple(polylines),
        shape_zyx,
        line_thickness,
        _freeze_z_indexed_polylines(z_polylines_by_index),
        tuple(max_polylines),
    )


def _global_xy_to_tile_yx(
    global_xy: np.ndarray,
    affine_zyx_um: np.ndarray,
    origin_zyx_um: np.ndarray,
    spacing_zyx_um: np.ndarray,
) -> np.ndarray:
    """
    Transform global xy cell_boundary coordinates into local tile yx pixels.

    Parameters
    ----------
    global_xy : numpy.ndarray
        Cell boundary coordinates in global X, Y microns.
    affine_zyx_um : np.ndarray
        Tile-to-global affine.
    origin_zyx_um : np.ndarray
        Tile physical origin.
    spacing_zyx_um : np.ndarray
        Tile spacing.

    Returns
    -------
    np.ndarray
        Local tile coordinates in Y, X pixels.
    """
    global_zyx = np.zeros((global_xy.shape[0], 3), dtype=float)
    global_zyx[:, 1] = global_xy[:, 1]
    global_zyx[:, 2] = global_xy[:, 0]
    return _global_zyx_um_to_tile_zyx_px(
        global_zyx,
        affine_zyx_um=affine_zyx_um,
        origin_zyx_um=origin_zyx_um,
        spacing_zyx_um=spacing_zyx_um,
    )[:, 1:]


def local_cell_boundary_geometry(
    cell_boundaries: dict[Any, np.ndarray] | None,
    shape_zyx: tuple[int, int, int],
    affine_zyx_um: np.ndarray,
    origin_zyx_um: np.ndarray,
    spacing_zyx_um: np.ndarray,
    line_thickness: int = 1,
) -> PolylineGeometry:
    """
    Create local Cellpose cell boundary geometry.

    Parameters
    ----------
    cell_boundaries : dict[Any, numpy.ndarray] or None
        Cell boundaries in global X, Y coordinates.
    shape_zyx : tuple[int, int, int]
        Local Z, Y, X canvas shape.
    affine_zyx_um : numpy.ndarray
        Tile-to-global affine.
    origin_zyx_um : numpy.ndarray
        Tile origin in microns.
    spacing_zyx_um : numpy.ndarray
        Tile Z, Y, X voxel spacing in microns.
    line_thickness : int, default=1
        Line thickness in pixels.

    Returns
    -------
    PolylineGeometry
        Sparse local cell-boundary geometry.
    """
    if not cell_boundaries:
        return PolylineGeometry((), shape_zyx, line_thickness)

    affine = np.asarray(affine_zyx_um, dtype=float)
    origin = np.asarray(origin_zyx_um, dtype=float)
    spacing = np.asarray(spacing_zyx_um, dtype=float)
    polylines: list[np.ndarray] = []
    for cell_boundary in cell_boundaries.values():
        global_xy = np.asarray(cell_boundary, dtype=float)
        if global_xy.ndim != 2 or global_xy.shape[0] < 2 or global_xy.shape[1] != 2:
            continue
        local_yx = _global_xy_to_tile_yx(
            global_xy,
            affine_zyx_um=affine,
            origin_zyx_um=origin,
            spacing_zyx_um=spacing,
        )
        if (
            local_yx[:, 0].max() < 0
            or local_yx[:, 0].min() >= shape_zyx[1]
            or local_yx[:, 1].max() < 0
            or local_yx[:, 1].min() >= shape_zyx[2]
        ):
            continue
        polylines.append(local_yx)
    return PolylineGeometry(tuple(polylines), shape_zyx, line_thickness)


def global_cell_boundary_geometry_from_source(
    cell_boundaries: dict[Any, np.ndarray] | None,
    shape_zyx: tuple[int, int, int],
    origin_zyx_um: Any,
    spacing_zyx_um: Any,
    line_thickness: int = 1,
) -> PolylineGeometry:
    """
    Create global cell-boundary geometry directly on the fused global canvas.

    Parameters
    ----------
    cell_boundaries : dict[Any, np.ndarray] | None
        Cell boundaries in global X, Y coordinates.
    shape_zyx : tuple[int, int, int]
        Global Z, Y, X canvas shape.
    origin_zyx_um : Any
        Global image origin in microns.
    spacing_zyx_um : Any
        Global Z, Y, X voxel spacing in microns.
    line_thickness : int, default=1
        Cell boundary line thickness in pixels.

    Returns
    -------
    PolylineGeometry
        Sparse global cell-boundary geometry.
    """
    if not cell_boundaries:
        return PolylineGeometry((), shape_zyx, line_thickness)

    origin = np.asarray(origin_zyx_um, dtype=float)
    spacing = np.asarray(spacing_zyx_um, dtype=float)
    polylines: list[np.ndarray] = []
    for cell_boundary in cell_boundaries.values():
        global_xy = np.asarray(cell_boundary, dtype=float)
        if global_xy.ndim != 2 or global_xy.shape[0] < 2 or global_xy.shape[1] != 2:
            continue
        global_yx = global_xy[:, ::-1]
        local_yx = (global_yx - origin[1:]) / spacing[1:]
        if (
            local_yx[:, 0].max() < 0
            or local_yx[:, 0].min() >= shape_zyx[1]
            or local_yx[:, 1].max() < 0
            or local_yx[:, 1].min() >= shape_zyx[2]
        ):
            continue
        polylines.append(local_yx)

    return PolylineGeometry(tuple(polylines), shape_zyx, line_thickness)


def cell_boundary_polylines_for_tile(
    datastore: Any,
    tile: str,
    shape_zyx: tuple[int, int, int],
) -> PolylineGeometry | None:
    """
    Load Cellpose cell boundaries for one tile as sparse local polylines.

    Parameters
    ----------
    datastore : Any
        Datastore object.
    tile : str
        Tile identifier.
    shape_zyx : tuple[int, int, int]
        Local Z, Y, X shape.

    Returns
    -------
    PolylineGeometry or None
        Sparse local cell_boundary geometry.
    """
    cell_boundaries = datastore.load_global_cellpose_roi_zip()
    if not cell_boundaries:
        cell_boundaries = datastore.load_global_cellpose_outlines()
    if cell_boundaries is None:
        return None

    affine, origin, spacing = datastore.load_global_coord_xforms_um(tile=tile)
    if affine is None or origin is None or spacing is None:
        return None

    return local_cell_boundary_geometry(
        cell_boundaries,
        shape_zyx=shape_zyx,
        affine_zyx_um=np.asarray(affine, dtype=float),
        origin_zyx_um=np.asarray(origin, dtype=float),
        spacing_zyx_um=np.asarray(spacing, dtype=float),
        line_thickness=5,
    )


def global_cell_boundary_geometry(
    datastore: Any,
    shape_zyx: tuple[int, int, int],
    origin_zyx_um: Any,
    spacing_zyx_um: Any,
) -> PolylineGeometry | None:
    """
    Load Cellpose cell boundaries on the fused global canvas as sparse geometry.

    Parameters
    ----------
    datastore : Any
        qi2lab datastore-like object.
    shape_zyx : tuple[int, int, int]
        Global Z, Y, X canvas shape.
    origin_zyx_um : Any
        Global image origin in microns.
    spacing_zyx_um : Any
        Global Z, Y, X voxel spacing in microns.

    Returns
    -------
    PolylineGeometry or None
        Sparse global cell-boundary geometry.
    """
    cell_boundaries = datastore.load_global_cellpose_roi_zip()
    if not cell_boundaries:
        cell_boundaries = datastore.load_global_cellpose_outlines()
    if cell_boundaries is None:
        return None

    return global_cell_boundary_geometry_from_source(
        cell_boundaries,
        shape_zyx=shape_zyx,
        origin_zyx_um=origin_zyx_um,
        spacing_zyx_um=spacing_zyx_um,
        line_thickness=5,
    )


def _match_lazy_global_image_shape(
    image: Any,
    shape_zyx: tuple[int, int, int],
    project_z: bool = False,
) -> Any:
    """
    Match a lazy or eager global image to fused Z, Y, X shape.

    Parameters
    ----------
    image : Any
        Image in Y, X or Z, Y, X order.
    shape_zyx : tuple[int, int, int]
        Fused image shape.
    project_z : bool
        Whether to project Z, Y, X image data into one Z plane.

    Returns
    -------
    Any
        Image compatible with the lazy global channel stack.
    """
    image_shape = tuple(int(value) for value in image.shape)
    if image_shape == shape_zyx:
        return image
    if (
        project_z
        and len(image_shape) == 3
        and shape_zyx[0] == 1
        and image_shape[1:] == shape_zyx[1:]
    ):
        return LazyMaxProjectionImage(image)
    if image_shape == shape_zyx[1:]:
        return LazyRepeatedPlaneImage(image, shape_zyx[0])
    if (
        len(image_shape) == 3
        and image_shape[0] == 1
        and image_shape[1:] == shape_zyx[1:]
    ):
        return LazyRepeatedPlaneImage(image[0], shape_zyx[0])
    raise ValueError("Global image shape does not match fused global image.")


def _squeeze_leading_singleton_axes(array: Any, ndim: int) -> Any:
    """
    Remove leading singleton axes without reading lazy image data.

    Parameters
    ----------
    array : Any
        Lazy or eager image array.
    ndim : int
        Required output dimensionality.

    Returns
    -------
    Any
        Image array with leading singleton axes removed.
    """
    squeezed = array
    while len(squeezed.shape) > ndim and int(squeezed.shape[0]) == 1:
        squeezed = squeezed[0]
    if len(squeezed.shape) != ndim:
        raise ValueError(f"Expected image data with {ndim} dimensions.")
    return squeezed


def _global_base_image(
    fused_zyx: Any,
    full_shape_zyx: tuple[int, int, int],
    *,
    include_fused_image: bool,
    max_project_fused_image: bool,
) -> _GlobalBaseImage:
    """
    Return the selected global base image and display labels.

    Parameters
    ----------
    fused_zyx : Any
        Fused global image in Z, Y, X order.
    full_shape_zyx : tuple[int, int, int]
        Full fused image shape.
    include_fused_image : bool
        Whether to use the fused image as the base.
    max_project_fused_image : bool
        Whether to use a lazy Z-max projection.

    Returns
    -------
    _GlobalBaseImage
        Selected base image and labels.
    """
    if include_fused_image and max_project_fused_image:
        return _GlobalBaseImage(
            LazyMaxProjectionImage(fused_zyx),
            (1, *full_shape_zyx[1:]),
            ["global polyDT max projection"],
        )
    if include_fused_image:
        return _GlobalBaseImage(
            fused_zyx,
            full_shape_zyx,
            ["global polyDT fused zarr"],
        )

    shape_zyx = full_shape_zyx
    if max_project_fused_image:
        shape_zyx = (1, *shape_zyx[1:])
    return _GlobalBaseImage(
        LazyZeroVolume(shape_zyx),
        shape_zyx,
        ["global empty canvas"],
    )


def _global_coords(
    shape_zyx: tuple[int, int, int],
    origin_zyx_um: Any,
    spacing_zyx_um: Any,
) -> dict[str, Any]:
    """
    Return NDV coordinate arrays for a global Z, Y, X image.

    Parameters
    ----------
    shape_zyx : tuple[int, int, int]
        Global Z, Y, X canvas shape.
    origin_zyx_um : Any
        Global image origin in microns.
    spacing_zyx_um : Any
        Global Z, Y, X voxel spacing in microns.

    Returns
    -------
    dict[str, Any]
        NDV coordinate arrays.
    """
    origin = np.asarray(origin_zyx_um, dtype=np.float32)
    spacing = np.asarray(spacing_zyx_um, dtype=np.float32)
    return {
        "c": range(1),
        "z_um": origin[0] + np.arange(shape_zyx[0], dtype=np.float32) * spacing[0],
        "y_um": origin[1] + np.arange(shape_zyx[1], dtype=np.float32) * spacing[1],
        "x_um": origin[2] + np.arange(shape_zyx[2], dtype=np.float32) * spacing[2],
    }


def _append_global_segmentation_channel(
    datastore: Any,
    data: LazyGlobalChannelData,
    labels: list[str],
    shape_zyx: tuple[int, int, int],
    *,
    project_z: bool,
) -> LazyGlobalChannelData:
    """
    Append a lazy Cellpose mask channel when it matches the global image.

    Parameters
    ----------
    datastore : Any
        qi2lab datastore-like object.
    data : LazyGlobalChannelData
        Current global channel data.
    labels : list[str]
        Channel labels updated in place.
    shape_zyx : tuple[int, int, int]
        Global Z, Y, X canvas shape.
    project_z : bool
        Whether segmentation should be max-projected.

    Returns
    -------
    LazyGlobalChannelData
        Updated global channel data.
    """
    segmentation = datastore.load_global_cellpose_segmentation_image(return_future=None)
    if segmentation is None:
        return data
    try:
        segmentation_zyx = _match_lazy_global_image_shape(
            segmentation,
            shape_zyx,
            project_z=project_z,
        )
    except ValueError:
        return data
    labels.append("global Cellpose mask")
    return data.with_image_channel(segmentation_zyx)


def load_global_image_channels(
    datastore: Any,
    include_fused_image: bool = True,
    include_segmentation: bool = True,
    max_project_fused_image: bool = False,
) -> GlobalChannelStack:
    """
    Load fused global polyDT image and optional global segmentation image.

    Parameters
    ----------
    datastore : Any
        qi2lab datastore-like object.
    include_fused_image : bool
        Whether to include the fused polyDT image channel.
    include_segmentation : bool
        Whether to include the global Cellpose mask channel.
    max_project_fused_image : bool
        Whether to display fused polyDT as one lazy Z-max projection plane.

    Returns
    -------
    GlobalChannelStack
        Global image stack and coordinate metadata.
    """
    loaded_fiducial = datastore.load_global_fiducial_image(return_future=None)
    if loaded_fiducial is None:
        raise ValueError("No fused global polyDT image was available to display.")
    fused_zyx, _affine_zyx_um, origin_zyx_um, spacing_zyx_um = loaded_fiducial
    fused_zyx = _squeeze_leading_singleton_axes(fused_zyx, 3)
    full_shape_zyx = tuple(int(value) for value in fused_zyx.shape)
    base = _global_base_image(
        fused_zyx,
        full_shape_zyx,
        include_fused_image=include_fused_image,
        max_project_fused_image=max_project_fused_image,
    )
    labels = list(base.labels)
    data = LazyGlobalChannelData(
        fused_zyx=base.image_zyx,
        image_channels=(),
        coords=_global_coords(base.shape_zyx, origin_zyx_um, spacing_zyx_um),
    )

    if include_segmentation:
        data = _append_global_segmentation_channel(
            datastore,
            data,
            labels,
            base.shape_zyx,
            project_z=max_project_fused_image,
        )

    return GlobalChannelStack(
        stack=ChannelStack(data=data, labels=labels),
        origin_zyx_um=np.asarray(origin_zyx_um, dtype=np.float32),
        spacing_zyx_um=np.asarray(spacing_zyx_um, dtype=np.float32),
        full_shape_zyx=full_shape_zyx,
    )
