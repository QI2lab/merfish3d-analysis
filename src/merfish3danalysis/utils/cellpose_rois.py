"""Bounded, parallel extraction and vectorized transformation of Cellpose ROIs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import batched
from typing import TYPE_CHECKING

import cv2
import numpy as np
from roifile import ImagejRoi
from scipy.ndimage import find_objects

from merfish3danalysis.utils.spacing import round_spacing_um

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence


def _bounded_map[Input, Output](
    function: Callable[[Input], Output], tasks: Iterable[Input], workers: int
) -> Iterator[Output]:
    """Preserve input order while limiting queued work and result memory."""
    if workers == 1:
        yield from map(function, tasks)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for group in batched(tasks, 2 * workers):
            yield from executor.map(function, group)


def extract_pixel_rois(
    masks: np.ndarray, labels: np.ndarray, *, workers: int = 1
) -> list[ImagejRoi]:
    """Extract Cellpose-compatible contours from label bounding boxes.

    Scan the mask once for bounding boxes, then share it read-only between
    workers. Each contour operation examines only the corresponding cell crop.
    Labels must contain the sorted nonzero mask IDs. Preserve their order,
    longest external contour, and Cellpose's minimum of five contour vertices.
    """
    if not labels.size:
        return []
    if int(labels[-1]) > 4 * len(labels):
        # Avoid a bounding-box table proportional to arbitrarily sparse IDs.
        box_labels = np.searchsorted(labels, masks).astype(np.int32) + 1
        box_labels[masks == 0] = 0
        boxes = find_objects(box_labels, max_label=len(labels))
        del box_labels
    else:
        all_boxes = find_objects(masks, max_label=int(labels[-1]))
        boxes = [all_boxes[int(label) - 1] for label in labels]
    tasks = batched(zip(labels, boxes, strict=True), 128)
    extract = partial(_extract_roi_batch, masks=masks)
    return [roi for batch in _bounded_map(extract, tasks, workers) for roi in batch]


def _extract_roi_batch(
    regions: tuple[tuple[int, tuple[slice, slice]], ...], *, masks: np.ndarray
) -> list[ImagejRoi]:
    rois = []
    for label, (ys, xs) in regions:
        crop = np.asarray(masks[ys, xs] == label, dtype=np.uint8)
        contours = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        contour = max(contours, key=len).reshape(-1, 2)
        if len(contour) > 4:
            contour += (xs.start, ys.start)
            rois.append(ImagejRoi.frompoints(contour, name=str(label)))
    return rois


def _polygon_areas(coordinates: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Vectorized shoelace areas for packed XY polygons with exclusive offsets."""
    # Translate each polygon before products to avoid cancellation at large
    # global origins. Work in float64 on the rounded float32 export coordinates.
    relative = coordinates.astype(np.float64) - np.repeat(
        coordinates[offsets[:-1]].astype(np.float64), np.diff(offsets), axis=0
    )
    following = np.roll(relative, -1, axis=0)
    following[offsets[1:] - 1] = 0  # close each polygon at its translated origin
    cross = relative[:, 0] * following[:, 1] - relative[:, 1] * following[:, 0]
    return np.abs(np.add.reduceat(cross, offsets[:-1])) * 0.5


def _transform_roi_batch(
    rois: Sequence[ImagejRoi],
    *,
    spacing_zyx_um: np.ndarray,
    origin_zyx_um: np.ndarray,
    affine_zyx_um: np.ndarray,
    min_cell_area_um2: float,
) -> list[np.ndarray]:
    coordinates = [roi.coordinates() for roi in rois]
    offsets = np.concatenate(([0], np.cumsum([len(points) for points in coordinates])))
    points_xy = np.concatenate(coordinates).astype(np.float32)
    points_zyx = np.empty((len(points_xy), 3), dtype=np.float32)
    points_zyx[:, 0] = 10
    points_zyx[:, 1:] = points_xy[:, ::-1]
    physical = points_zyx * spacing_zyx_um + origin_zyx_um
    homogeneous = np.ones((len(physical), 4), dtype=physical.dtype)
    homogeneous[:, :3] = physical
    # Keep the original four-row multiplication and float32 rounding semantics,
    # including for sheared transforms and cells exactly at the area cutoff.
    global_xy = (affine_zyx_um @ homogeneous.T).T[:, [2, 1]]
    global_xy = np.round(global_xy, 2).astype(np.float32)
    keep = (
        np.flatnonzero(_polygon_areas(global_xy, offsets) >= min_cell_area_um2)
        if min_cell_area_um2 > 0
        else range(len(rois))
    )
    return [global_xy[offsets[index] : offsets[index + 1]] for index in keep]


def global_rois(
    pixel_rois: Sequence[ImagejRoi],
    spacing_zyx_um: np.ndarray,
    origin_zyx_um: np.ndarray,
    affine_zyx_um: np.ndarray,
    min_cell_area_um2: float,
    *,
    workers: int = 1,
) -> Iterator[ImagejRoi]:
    """Transform and filter bounded batches, yielding retained ROIs in order.

    Array operations run in parallel; the caller can stream ImageJ serialization
    and ZIP writes while subsequent batches are being computed. Rejected cells
    never incur ImageJ ROI construction or serialization.
    """
    transform = partial(
        _transform_roi_batch,
        spacing_zyx_um=round_spacing_um(spacing_zyx_um).astype(
            np.asarray(spacing_zyx_um).dtype
        ),
        origin_zyx_um=np.asarray(origin_zyx_um),
        affine_zyx_um=np.asarray(affine_zyx_um),
        min_cell_area_um2=min_cell_area_um2,
    )
    batches = batched(pixel_rois, 512)
    cell_index = 0
    for batch in _bounded_map(transform, batches, workers):
        for coordinates in batch:
            yield ImagejRoi.frompoints(coordinates, name=f"cell_{cell_index:07d}")
            cell_index += 1
