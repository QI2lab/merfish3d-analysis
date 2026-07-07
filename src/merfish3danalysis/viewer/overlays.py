"""Overlay rasterization helpers for the datastore viewer."""

from typing import Any

import numpy as np
from matplotlib import colormaps

from merfish3danalysis.viewer.models import ChannelStack, GlobalChannelStack, ProsegRun
from merfish3danalysis.viewer.warping import _as_zyx


def _paint_point(
    volume: np.ndarray,
    zyx: np.ndarray,
    radius: int,
    value: float = 1.0,
) -> None:
    """
    Paint one point into a volume.

    Parameters
    ----------
    volume : np.ndarray
        volume for this viewer operation.
    zyx : np.ndarray
        zyx for this viewer operation.
    radius : int
        radius for this viewer operation.
    value : float
        Value painted into the point footprint.

    Returns
    -------
    None
        Computed viewer result.
    """

    z, y, x = np.round(zyx).astype(int)
    z_min = max(0, z - radius)
    z_max = min(volume.shape[0], z + radius + 1)
    y_min = max(0, y - radius)
    y_max = min(volume.shape[1], y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(volume.shape[2], x + radius + 1)
    if z_min < z_max and y_min < y_max and x_min < x_max:
        volume[z_min:z_max, y_min:y_max, x_min:x_max] = value


def _paint_disk_2d(
    image: np.ndarray,
    center_yx: np.ndarray,
    radius: int,
    value: float,
) -> None:
    """
    Paint one circular footprint into a 2D image.

    Parameters
    ----------
    image : numpy.ndarray
        Target Y, X image.
    center_yx : numpy.ndarray
        Center coordinate in Y, X pixel units.
    radius : int
        Disk radius in pixels.
    value : float
        Value painted into the disk.

    Returns
    -------
    None
        Computed viewer result.
    """

    y, x = np.round(center_yx).astype(int)
    y_min = max(0, y - radius)
    y_max = min(image.shape[0], y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(image.shape[1], x + radius + 1)
    if y_min >= y_max or x_min >= x_max:
        return
    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
    mask = (yy - y) ** 2 + (xx - x) ** 2 <= radius**2
    image[y_min:y_max, x_min:x_max][mask] = value


def _paint_points(
    volume: np.ndarray,
    coords_zyx: np.ndarray,
    values: np.ndarray,
    radius: int,
) -> None:
    """
    Paint many points into a volume using vectorized indexed assignment.

    Parameters
    ----------
    volume : numpy.ndarray
        Target Z, Y, X overlay volume.
    coords_zyx : numpy.ndarray
        Point coordinates in Z, Y, X pixel units.
    values : numpy.ndarray
        Display value for each coordinate.
    radius : int
        Paint radius in pixels.

    Returns
    -------
    None
        Computed viewer result.
    """

    if coords_zyx.size == 0 or values.size == 0:
        return

    finite = np.isfinite(coords_zyx).all(axis=1) & np.isfinite(values)
    if not finite.any():
        return

    coords = np.round(coords_zyx[finite]).astype(np.intp, copy=False)
    point_values = values[finite].astype(np.float32, copy=False)
    z_offsets = np.arange(-radius, radius + 1, dtype=np.intp)
    y_offsets, x_offsets = np.nonzero(_circular_footprint(radius))
    y_offsets = y_offsets.astype(np.intp, copy=False) - radius
    x_offsets = x_offsets.astype(np.intp, copy=False) - radius

    for dz in z_offsets:
        z = coords[:, 0] + dz
        z_valid = (z >= 0) & (z < volume.shape[0])
        if not z_valid.any():
            continue
        for dy, dx in zip(y_offsets, x_offsets, strict=True):
            y = coords[:, 1] + dy
            zy_valid = z_valid & (y >= 0) & (y < volume.shape[1])
            if not zy_valid.any():
                continue
            x = coords[:, 2] + dx
            valid = zy_valid & (x >= 0) & (x < volume.shape[2])
            if valid.any():
                volume[z[valid], y[valid], x[valid]] = point_values[valid]


def _circular_footprint(radius: int) -> np.ndarray:
    """
    Return a circular boolean footprint.

    Parameters
    ----------
    radius : int
        Footprint radius in pixels.

    Returns
    -------
    numpy.ndarray
        Circular footprint with shape ``(2 * radius + 1, 2 * radius + 1)``.
    """

    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (yy**2 + xx**2) <= radius**2


def _codeword_values(
    spots: Any, column: str, genes: list[str] | None
) -> dict[str, int]:
    """
    Return stable display values for selected codewords.

    Parameters
    ----------
    spots : Any
        Table containing the codeword column.
    column : str
        Column storing the codeword name.
    genes : list[str] | None
        Optional selected codeword order.

    Returns
    -------
    dict[str, int]
        Codeword name to positive display value.
    """

    if genes is not None:
        ordered_genes = [gene.strip() for gene in genes if gene.strip()]
    else:
        ordered_genes = sorted(spots[column].astype(str).unique())
    return {gene: idx + 1 for idx, gene in enumerate(dict.fromkeys(ordered_genes))}


def codeword_color_hex(value: int, value_count: int) -> str:
    """
    Return the transcript overlay color for one selected codeword value.

    Parameters
    ----------
    value : int
        Positive codeword overlay value.
    value_count : int
        Number of selected codeword values.

    Returns
    -------
    str
        Hex RGB color string.
    """

    if value_count <= 1:
        normalized_value = 1.0
    else:
        normalized_value = (value - 1) / (value_count - 1)
    rgb = colormaps["turbo"](normalized_value)[:3]
    return "#" + "".join(f"{round(channel * 255):02x}" for channel in rgb)


def empty_transcript_overlay(shape_zyx: tuple[int, int, int]) -> np.ndarray:
    """
    Return an empty transparent transcript overlay.

    Parameters
    ----------
    shape_zyx : tuple[int, int, int]
        Output Z, Y, X shape.

    Returns
    -------
    numpy.ndarray
        Overlay initialized to NaN so zero-valued background is not colorized.
    """

    return np.full(shape_zyx, np.nan, dtype=np.float32)


def rasterize_decoded_spots(
    decoded_spots: Any,
    shape_zyx: tuple[int, int, int],
    genes: list[str] | None = None,
    radius: int = 1,
) -> np.ndarray:
    """
    Rasterize decoded spots with tile pixel coordinates into an overlay.

    Parameters
    ----------
    decoded_spots : Any
        decoded_spots for this viewer operation.
    shape_zyx : tuple[int, int, int]
        shape_zyx for this viewer operation.
    genes : list[str] | None
        genes for this viewer operation.
    radius : int
        radius for this viewer operation.

    Returns
    -------
    np.ndarray
        Computed viewer result.
    """

    overlay = empty_transcript_overlay(shape_zyx)
    if decoded_spots is None or len(decoded_spots) == 0:
        return overlay

    required_columns = {"tile_z", "tile_y", "tile_x", "gene_id"}
    if not required_columns.issubset(decoded_spots.columns):
        return overlay

    spots = decoded_spots
    value_by_gene = _codeword_values(spots, "gene_id", genes)
    if genes is not None:
        genes_set = {gene.strip() for gene in genes if gene.strip()}
        spots = spots.loc[spots["gene_id"].astype(str).isin(genes_set)]

    coords = spots[["tile_z", "tile_y", "tile_x"]].to_numpy(dtype=float, copy=False)
    gene_values = (
        spots["gene_id"].astype(str).map(value_by_gene).to_numpy(dtype=np.float32)
    )
    _paint_points(overlay, coords, gene_values, radius)

    return overlay


def rasterize_global_decoded_spots(
    decoded_spots: Any,
    shape_zyx: tuple[int, int, int],
    origin_zyx_um: Any,
    spacing_zyx_um: Any,
    genes: list[str] | None = None,
    radius: int = 1,
) -> np.ndarray:
    """
    Rasterize decoded spots with global micron XY coordinates into an overlay.

    Parameters
    ----------
    decoded_spots : Any
        decoded_spots for this viewer operation.
    shape_zyx : tuple[int, int, int]
        shape_zyx for this viewer operation.
    origin_zyx_um : Any
        origin_zyx_um for this viewer operation.
    spacing_zyx_um : Any
        spacing_zyx_um for this viewer operation.
    genes : list[str] | None
        genes for this viewer operation.
    radius : int
        radius for this viewer operation.

    Returns
    -------
    np.ndarray
        Computed viewer result.
    """

    overlay = empty_transcript_overlay(shape_zyx)
    if decoded_spots is None or len(decoded_spots) == 0:
        return overlay

    required_columns = {"global_y", "global_x", "gene_id"}
    if not required_columns.issubset(decoded_spots.columns):
        return overlay

    spots = decoded_spots
    value_by_gene = _codeword_values(spots, "gene_id", genes)
    if genes is not None:
        genes_set = {gene.strip() for gene in genes if gene.strip()}
        spots = spots.loc[spots["gene_id"].astype(str).isin(genes_set)]

    origin = np.asarray(origin_zyx_um, dtype=float)
    spacing = np.asarray(spacing_zyx_um, dtype=float)
    coords_um = spots[["global_y", "global_x"]].to_numpy(dtype=float, copy=False)
    coords_yx = (coords_um - origin[1:]) / spacing[1:]
    coords_zyx = np.column_stack(
        [np.zeros(coords_yx.shape[0], dtype=float), coords_yx[:, 0], coords_yx[:, 1]]
    )
    gene_values = (
        spots["gene_id"].astype(str).map(value_by_gene).to_numpy(dtype=np.float32)
    )
    _paint_points(overlay, coords_zyx, gene_values, radius)

    return overlay


def rasterize_local_decoded_spots(
    decoded_spots: Any,
    shape_zyx: tuple[int, int, int],
    affine_zyx_um: np.ndarray,
    origin_zyx_um: np.ndarray,
    spacing_zyx_um: np.ndarray,
    genes: list[str] | None = None,
    radius: int = 1,
) -> np.ndarray:
    """
    Rasterize final decoded spots onto a local tile canvas.

    Parameters
    ----------
    decoded_spots : Any
        Final decoded transcript table with global coordinates.
    shape_zyx : tuple[int, int, int]
        Output Z, Y, X shape.
    affine_zyx_um : numpy.ndarray
        Tile-to-global affine.
    origin_zyx_um : numpy.ndarray
        Tile physical origin.
    spacing_zyx_um : numpy.ndarray
        Tile spacing.
    genes : list[str] | None
        Optional gene filter.
    radius : int, default=1
        Paint radius in pixels.

    Returns
    -------
    numpy.ndarray
        Overlay volume.
    """

    overlay = empty_transcript_overlay(shape_zyx)
    if decoded_spots is None or len(decoded_spots) == 0:
        return overlay

    required_columns = {"global_z", "global_y", "global_x", "gene_id"}
    if not required_columns.issubset(decoded_spots.columns):
        return overlay

    spots = decoded_spots
    value_by_gene = _codeword_values(spots, "gene_id", genes)
    if genes is not None:
        genes_set = {gene.strip() for gene in genes if gene.strip()}
        spots = spots.loc[spots["gene_id"].astype(str).isin(genes_set)]

    affine = np.asarray(affine_zyx_um, dtype=float)
    origin = np.asarray(origin_zyx_um, dtype=float)
    spacing = np.asarray(spacing_zyx_um, dtype=float)
    coords_zyx = spots[["global_z", "global_y", "global_x"]].to_numpy(
        dtype=float,
        copy=False,
    )
    lower_bound, upper_bound = _tile_global_bounds_zyx_um(
        shape_zyx,
        affine_zyx_um=affine,
        origin_zyx_um=origin,
        spacing_zyx_um=spacing,
        radius=radius,
    )
    in_tile_bounds = ((coords_zyx >= lower_bound) & (coords_zyx <= upper_bound)).all(
        axis=1
    )
    if not in_tile_bounds.any():
        return overlay
    coords_zyx = coords_zyx[in_tile_bounds]
    spots = spots.loc[in_tile_bounds]
    coords_px = _global_zyx_um_to_tile_zyx_px(
        coords_zyx,
        affine_zyx_um=affine,
        origin_zyx_um=origin,
        spacing_zyx_um=spacing,
    )
    gene_values = (
        spots["gene_id"].astype(str).map(value_by_gene).to_numpy(dtype=np.float32)
    )
    _paint_points(overlay, coords_px, gene_values, radius)
    return overlay


def discover_proseg_runs(datastore: Any) -> list[ProsegRun]:
    """
    Discover available Proseg 3D runs under a datastore.

    Parameters
    ----------
    datastore : Any
        qi2lab datastore-like object.

    Returns
    -------
    list[ProsegRun]
        Detected Proseg runs sorted by display name.
    """

    return [ProsegRun(name=name) for name in datastore.list_proseg_3d_runs()]


def rasterize_global_proseg_transcripts(
    transcripts: Any,
    shape_zyx: tuple[int, int, int],
    origin_zyx_um: Any,
    spacing_zyx_um: Any,
    genes: list[str] | None = None,
    radius: int = 1,
) -> np.ndarray:
    """
    Rasterize Proseg transcript coordinates onto the global fused canvas.

    Parameters
    ----------
    transcripts : Any
        pandas DataFrame with Proseg transcript metadata.
    shape_zyx : tuple[int, int, int]
        Output Z, Y, X shape.
    origin_zyx_um : Any
        Global fused origin in Z, Y, X microns.
    spacing_zyx_um : Any
        Global fused spacing in Z, Y, X microns.
    genes : list[str] | None
        Optional gene filter.
    radius : int, default=1
        Paint radius in pixels.

    Returns
    -------
    numpy.ndarray
        Overlay volume.
    """

    overlay = empty_transcript_overlay(shape_zyx)
    if transcripts is None or len(transcripts) == 0:
        return overlay
    required_columns = {"x", "y", "z", "gene"}
    if not required_columns.issubset(transcripts.columns):
        return overlay

    spots = transcripts
    value_by_gene = _codeword_values(spots, "gene", genes)
    if genes is not None:
        genes_set = {gene.strip() for gene in genes if gene.strip()}
        spots = spots.loc[spots["gene"].astype(str).isin(genes_set)]

    origin = np.asarray(origin_zyx_um, dtype=float)
    spacing = np.asarray(spacing_zyx_um, dtype=float)
    coords_xyz = spots[["x", "y", "z"]].to_numpy(dtype=float, copy=False)
    coords_zyx = coords_xyz[:, [2, 1, 0]]
    coords_px = (coords_zyx - origin) / spacing
    gene_values = (
        spots["gene"].astype(str).map(value_by_gene).to_numpy(dtype=np.float32)
    )
    _paint_points(overlay, coords_px, gene_values, radius)
    return overlay


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


def rasterize_local_proseg_transcripts(
    transcripts: Any,
    shape_zyx: tuple[int, int, int],
    affine_zyx_um: np.ndarray,
    origin_zyx_um: np.ndarray,
    spacing_zyx_um: np.ndarray,
    genes: list[str] | None = None,
    radius: int = 1,
) -> np.ndarray:
    """
    Rasterize Proseg transcripts onto a local tile canvas.

    Parameters
    ----------
    transcripts : Any
        pandas DataFrame with Proseg transcript metadata.
    shape_zyx : tuple[int, int, int]
        Output Z, Y, X shape.
    affine_zyx_um : numpy.ndarray
        Tile-to-global affine.
    origin_zyx_um : numpy.ndarray
        Tile physical origin.
    spacing_zyx_um : numpy.ndarray
        Tile spacing.
    genes : list[str] | None
        Optional gene filter.
    radius : int, default=1
        Paint radius in pixels.

    Returns
    -------
    numpy.ndarray
        Overlay volume.
    """

    overlay = empty_transcript_overlay(shape_zyx)
    if transcripts is None or len(transcripts) == 0:
        return overlay
    required_columns = {"x", "y", "z", "gene"}
    if not required_columns.issubset(transcripts.columns):
        return overlay

    spots = transcripts
    value_by_gene = _codeword_values(spots, "gene", genes)
    if genes is not None:
        genes_set = {gene.strip() for gene in genes if gene.strip()}
        spots = spots.loc[spots["gene"].astype(str).isin(genes_set)]

    affine = np.asarray(affine_zyx_um, dtype=float)
    origin = np.asarray(origin_zyx_um, dtype=float)
    spacing = np.asarray(spacing_zyx_um, dtype=float)
    coords_xyz = spots[["x", "y", "z"]].to_numpy(dtype=float, copy=False)
    coords_zyx = coords_xyz[:, [2, 1, 0]]
    lower_bound, upper_bound = _tile_global_bounds_zyx_um(
        shape_zyx,
        affine_zyx_um=affine,
        origin_zyx_um=origin,
        spacing_zyx_um=spacing,
        radius=radius,
    )
    in_tile_bounds = ((coords_zyx >= lower_bound) & (coords_zyx <= upper_bound)).all(
        axis=1
    )
    if not in_tile_bounds.any():
        return overlay
    coords_zyx = coords_zyx[in_tile_bounds]
    spots = spots.loc[in_tile_bounds]
    coords_px = _global_zyx_um_to_tile_zyx_px(
        coords_zyx,
        affine_zyx_um=affine,
        origin_zyx_um=origin,
        spacing_zyx_um=spacing,
    )
    gene_values = (
        spots["gene"].astype(str).map(value_by_gene).to_numpy(dtype=np.float32)
    )
    _paint_points(overlay, coords_px, gene_values, radius)
    return overlay


def _draw_line_2d(
    image: np.ndarray,
    start_yx: np.ndarray,
    end_yx: np.ndarray,
    thickness: int = 1,
) -> None:
    """
    Draw a line into a 2D image using integer interpolation.

    Parameters
    ----------
    image : np.ndarray
        image for this viewer operation.
    start_yx : np.ndarray
        start_yx for this viewer operation.
    end_yx : np.ndarray
        end_yx for this viewer operation.
    thickness : int, default=1
        Line thickness in pixels.

    Returns
    -------
    None
        Computed viewer result.
    """

    y_min = min(start_yx[0], end_yx[0])
    y_max = max(start_yx[0], end_yx[0])
    x_min = min(start_yx[1], end_yx[1])
    x_max = max(start_yx[1], end_yx[1])
    if y_max < 0 or y_min >= image.shape[0] or x_max < 0 or x_min >= image.shape[1]:
        return

    y0, x0 = np.round(start_yx).astype(int)
    y1, x1 = np.round(end_yx).astype(int)
    steps = int(max(abs(y1 - y0), abs(x1 - x0))) + 1
    if steps <= 0:
        return
    ys = np.round(np.linspace(y0, y1, steps)).astype(int)
    xs = np.round(np.linspace(x0, x1, steps)).astype(int)
    valid = (ys >= 0) & (ys < image.shape[0]) & (xs >= 0) & (xs < image.shape[1])
    radius = max(int(thickness) // 2, 0)
    if radius == 0:
        image[ys[valid], xs[valid]] = 1.0
        return
    for y, x in zip(ys[valid], xs[valid], strict=True):
        _paint_disk_2d(image, np.asarray([y, x], dtype=float), radius, 1.0)


def _global_xy_to_tile_yx(
    global_xy: np.ndarray,
    affine_zyx_um: np.ndarray,
    origin_zyx_um: np.ndarray,
    spacing_zyx_um: np.ndarray,
) -> np.ndarray:
    """
    Transform global xy outline coordinates into local tile yx pixels.

    Parameters
    ----------
    global_xy : np.ndarray
        global_xy for this viewer operation.
    affine_zyx_um : np.ndarray
        affine_zyx_um for this viewer operation.
    origin_zyx_um : np.ndarray
        origin_zyx_um for this viewer operation.
    spacing_zyx_um : np.ndarray
        spacing_zyx_um for this viewer operation.

    Returns
    -------
    np.ndarray
        Computed viewer result.
    """

    inverse_affine = np.linalg.inv(np.asarray(affine_zyx_um, dtype=float))
    output = np.zeros((global_xy.shape[0], 2), dtype=float)
    for idx, xy in enumerate(global_xy):
        global_zyx = np.asarray([0.0, xy[1], xy[0], 1.0], dtype=float)
        physical_zyx = (inverse_affine @ global_zyx)[:3]
        pixel_zyx = (physical_zyx - origin_zyx_um) / spacing_zyx_um
        output[idx, :] = pixel_zyx[1:]
    return output


def rasterize_cell_outlines(
    outlines: dict[Any, np.ndarray] | None,
    shape_zyx: tuple[int, int, int],
    affine_zyx_um: np.ndarray,
    origin_zyx_um: np.ndarray,
    spacing_zyx_um: np.ndarray,
    line_thickness: int = 1,
) -> np.ndarray:
    """
    Rasterize global Cellpose outlines into a selected local tile volume.

    Parameters
    ----------
    outlines : dict[Any, np.ndarray] | None
        outlines for this viewer operation.
    shape_zyx : tuple[int, int, int]
        shape_zyx for this viewer operation.
    affine_zyx_um : np.ndarray
        affine_zyx_um for this viewer operation.
    origin_zyx_um : np.ndarray
        origin_zyx_um for this viewer operation.
    spacing_zyx_um : np.ndarray
        spacing_zyx_um for this viewer operation.
    line_thickness : int, default=1
        Outline line thickness in pixels.

    Returns
    -------
    np.ndarray
        Computed viewer result.
    """

    overlay_2d = np.zeros(shape_zyx[1:], dtype=np.float32)
    if not outlines:
        return np.zeros(shape_zyx, dtype=np.float32)

    for outline in outlines.values():
        global_xy = np.asarray(outline, dtype=float)
        if global_xy.ndim != 2 or global_xy.shape[0] < 2 or global_xy.shape[1] != 2:
            continue
        local_yx = _global_xy_to_tile_yx(
            global_xy,
            affine_zyx_um=np.asarray(affine_zyx_um, dtype=float),
            origin_zyx_um=np.asarray(origin_zyx_um, dtype=float),
            spacing_zyx_um=np.asarray(spacing_zyx_um, dtype=float),
        )
        if (
            local_yx[:, 0].max() < 0
            or local_yx[:, 0].min() >= overlay_2d.shape[0]
            or local_yx[:, 1].max() < 0
            or local_yx[:, 1].min() >= overlay_2d.shape[1]
        ):
            continue
        for idx in range(local_yx.shape[0]):
            _draw_line_2d(
                overlay_2d,
                local_yx[idx - 1],
                local_yx[idx],
                thickness=line_thickness,
            )

    return np.repeat(overlay_2d[np.newaxis, :, :], shape_zyx[0], axis=0)


def rasterize_global_cell_outlines(
    outlines: dict[Any, np.ndarray] | None,
    shape_zyx: tuple[int, int, int],
    origin_zyx_um: Any,
    spacing_zyx_um: Any,
    line_thickness: int = 1,
) -> np.ndarray:
    """
    Rasterize global Cellpose outlines directly onto the fused global canvas.

    Parameters
    ----------
    outlines : dict[Any, np.ndarray] | None
        outlines for this viewer operation.
    shape_zyx : tuple[int, int, int]
        shape_zyx for this viewer operation.
    origin_zyx_um : Any
        origin_zyx_um for this viewer operation.
    spacing_zyx_um : Any
        spacing_zyx_um for this viewer operation.
    line_thickness : int, default=1
        Outline line thickness in pixels.

    Returns
    -------
    np.ndarray
        Computed viewer result.
    """

    overlay_2d = np.zeros(shape_zyx[1:], dtype=np.float32)
    if not outlines:
        return np.zeros(shape_zyx, dtype=np.float32)

    origin = np.asarray(origin_zyx_um, dtype=float)
    spacing = np.asarray(spacing_zyx_um, dtype=float)
    for outline in outlines.values():
        global_xy = np.asarray(outline, dtype=float)
        if global_xy.ndim != 2 or global_xy.shape[0] < 2 or global_xy.shape[1] != 2:
            continue
        global_yx = global_xy[:, ::-1]
        local_yx = (global_yx - origin[1:]) / spacing[1:]
        if (
            local_yx[:, 0].max() < 0
            or local_yx[:, 0].min() >= overlay_2d.shape[0]
            or local_yx[:, 1].max() < 0
            or local_yx[:, 1].min() >= overlay_2d.shape[1]
        ):
            continue
        for idx in range(local_yx.shape[0]):
            _draw_line_2d(
                overlay_2d,
                local_yx[idx - 1],
                local_yx[idx],
                thickness=line_thickness,
            )

    return np.repeat(overlay_2d[np.newaxis, :, :], shape_zyx[0], axis=0)


def cell_outline_overlay_for_tile(
    datastore: Any,
    tile: str,
    shape_zyx: tuple[int, int, int],
) -> np.ndarray | None:
    """
    Load and rasterize Cellpose outlines using existing datastore APIs.

    Parameters
    ----------
    datastore : Any
        datastore for this viewer operation.
    tile : str
        tile for this viewer operation.
    shape_zyx : tuple[int, int, int]
        shape_zyx for this viewer operation.

    Returns
    -------
    np.ndarray | None
        Computed viewer result.
    """

    outlines = datastore.load_global_cellpose_roi_zip()
    if not outlines:
        outlines = datastore.load_global_cellpose_outlines()
    if outlines is None:
        return None

    affine, origin, spacing = datastore.load_global_coord_xforms_um(tile=tile)
    if affine is None or origin is None or spacing is None:
        return None

    return rasterize_cell_outlines(
        outlines,
        shape_zyx=shape_zyx,
        affine_zyx_um=np.asarray(affine, dtype=float),
        origin_zyx_um=np.asarray(origin, dtype=float),
        spacing_zyx_um=np.asarray(spacing, dtype=float),
        line_thickness=5,
    )


def global_cell_outline_overlay(
    datastore: Any,
    shape_zyx: tuple[int, int, int],
    origin_zyx_um: Any,
    spacing_zyx_um: Any,
) -> np.ndarray | None:
    """
    Load and rasterize Cellpose outlines on the fused global canvas.

    Parameters
    ----------
    datastore : Any
        datastore for this viewer operation.
    shape_zyx : tuple[int, int, int]
        shape_zyx for this viewer operation.
    origin_zyx_um : Any
        origin_zyx_um for this viewer operation.
    spacing_zyx_um : Any
        spacing_zyx_um for this viewer operation.

    Returns
    -------
    np.ndarray | None
        Computed viewer result.
    """

    outlines = datastore.load_global_cellpose_roi_zip()
    if not outlines:
        outlines = datastore.load_global_cellpose_outlines()
    if outlines is None:
        return None

    return rasterize_global_cell_outlines(
        outlines,
        shape_zyx=shape_zyx,
        origin_zyx_um=origin_zyx_um,
        spacing_zyx_um=spacing_zyx_um,
        line_thickness=5,
    )


def _match_global_overlay_shape(
    overlay: np.ndarray,
    shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    """
    Convert a 2D or single-plane global overlay to the fused image shape.

    Parameters
    ----------
    overlay : np.ndarray
        overlay for this viewer operation.
    shape_zyx : tuple[int, int, int]
        shape_zyx for this viewer operation.

    Returns
    -------
    np.ndarray
        Computed viewer result.
    """

    overlay_zyx = _as_zyx(overlay)
    if overlay_zyx.shape == shape_zyx:
        return overlay_zyx.astype(np.float32, copy=False)
    if overlay_zyx.shape[0] == 1 and overlay_zyx.shape[1:] == shape_zyx[1:]:
        return np.repeat(overlay_zyx, shape_zyx[0], axis=0).astype(
            np.float32, copy=False
        )
    raise ValueError("Global overlay shape does not match fused global image.")


def load_global_image_channels(
    datastore: Any,
    include_segmentation: bool = True,
) -> GlobalChannelStack:
    """
    Load fused global polyDT image and optional global segmentation image.

    Parameters
    ----------
    datastore : Any
        datastore for this viewer operation.
    include_segmentation : bool
        include_segmentation for this viewer operation.

    Returns
    -------
    GlobalChannelStack
        Computed viewer result.
    """

    loaded = datastore.load_global_fiducial_image(return_future=False)
    if loaded is None:
        raise ValueError("No fused global polyDT image was available to display.")

    fused_image, _affine, origin_zyx_um, spacing_zyx_um = loaded
    fused_zyx = _as_zyx(fused_image)
    fused_projection = np.max(fused_zyx, axis=0, keepdims=True).astype(
        np.float32, copy=False
    )
    channels = [fused_projection]
    labels = ["global polyDT max projection"]

    if include_segmentation:
        segmentation = datastore.load_global_cellpose_segmentation_image(
            return_future=False
        )
        if segmentation is not None:
            segmentation_zyx = _match_global_overlay_shape(
                segmentation, fused_projection.shape
            )
            channels.append(segmentation_zyx)
            labels.append("global Cellpose mask")

    return GlobalChannelStack(
        stack=ChannelStack(data=np.stack(channels, axis=0), labels=labels),
        origin_zyx_um=np.asarray(origin_zyx_um, dtype=np.float32),
        spacing_zyx_um=np.asarray(spacing_zyx_um, dtype=np.float32),
    )


def append_overlay_channel(
    stack: ChannelStack,
    overlay: np.ndarray | None,
    label: str,
) -> ChannelStack:
    """
    Append one overlay channel to an existing channel stack.

    Parameters
    ----------
    stack : ChannelStack
        stack for this viewer operation.
    overlay : np.ndarray | None
        overlay for this viewer operation.
    label : str
        label for this viewer operation.

    Returns
    -------
    ChannelStack
        Computed viewer result.
    """

    if overlay is None:
        return stack
    overlay_zyx = _as_zyx(overlay)
    if overlay_zyx.shape != stack.data.shape[1:]:
        raise ValueError("Overlay shape does not match selected image channels.")
    return ChannelStack(
        data=np.concatenate([stack.data, overlay_zyx[np.newaxis, :, :, :]], axis=0),
        labels=[*stack.labels, label],
    )
