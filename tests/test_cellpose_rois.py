"""Accuracy and ordering of batched, parallel Cellpose outline processing."""

import numpy as np
import pytest
from cellpose import utils
from roifile import ImagejRoi
from shapely.geometry import Polygon

from merfish3danalysis.cli.qi2lab_microscopes.segment_fiducial import warp_points
from merfish3danalysis.utils.cellpose_rois import (
    _polygon_areas,
    extract_pixel_rois,
    global_rois,
)


@pytest.mark.unit
@pytest.mark.parametrize("workers", [1, 4])
def test_cropped_parallel_contours_match_cellpose(workers):
    masks = np.zeros((200, 200), dtype=np.uint16)
    for i in range(300):
        y, x = divmod(i, 20)
        masks[y * 10 : y * 10 + 7, x * 10 : x * 10 + 7] = i + 1
    masks[2:4, 2:4] = 0  # hole
    masks[180:183, 180:183] = 1  # disconnected component of an existing cell
    masks[190, 190] = 301  # too few contour vertices, as in Cellpose
    labels = np.unique(masks)[1:]
    original_masks = masks.copy()
    expected = utils.outlines_list(masks, multiprocessing=False)

    rois = extract_pixel_rois(masks, labels, workers=workers)

    expected = [
        (str(label), points)
        for label, points in zip(labels, expected, strict=True)
        if len(points)
    ]
    assert [roi.name for roi in rois] == [name for name, _points in expected]
    for roi, (_name, points) in zip(rois, expected, strict=True):
        np.testing.assert_array_equal(roi.coordinates(), points)
    np.testing.assert_array_equal(masks, original_masks)


@pytest.mark.unit
@pytest.mark.parametrize("background", [True, False])
def test_sparse_labels_preserve_original_ids_with_or_without_background(background):
    masks = np.full((10, 10), 2**30, dtype=np.uint32)
    masks[5:] = 7
    if background:
        masks[:, :2] = 0
    rois = extract_pixel_rois(masks, np.array([7, 2**30]), workers=2)
    assert [roi.name for roi in rois] == ["7", str(2**30)]
    assert [Polygon(roi.coordinates()).area for roi in rois] == (
        [28, 28] if background else [36, 36]
    )


@pytest.mark.unit
@pytest.mark.parametrize("workers", [1, 4])
def test_batched_global_transform_matches_original_coordinates_and_filter(workers):
    rng = np.random.default_rng(84)
    pixel_rois = []
    for index in range(1600):
        angles = np.linspace(0, 2 * np.pi, index % 17 + 5, endpoint=False)
        xy = np.column_stack((np.cos(angles), np.sin(angles))) * (5 + index % 30)
        xy += rng.uniform(0, 10000, 2)
        pixel_rois.append(ImagejRoi.frompoints(xy.astype(np.float32), name=str(index)))
    spacing = np.array([1, 0.11, 0.21], dtype=np.float32)
    origin = np.array([0, 40, 30], dtype=np.float32)
    affine = np.array(
        [
            [1, 0.001, 0.002, 12],
            [0.01, 0.95, 0.1, -200],
            [0.03, -0.02, 1.1, 800],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    expected = []
    for roi in pixel_rois:
        xy = roi.coordinates().astype(np.float32)
        zyx = np.column_stack(
            (np.full(len(xy), 10, dtype=np.float32), xy[:, 1], xy[:, 0])
        )
        expected.append(
            np.round(warp_points(zyx, spacing, origin, affine)[:, [2, 1]], 2).astype(
                np.float32
            )
        )
    cutoff = Polygon(expected[625]).area
    expected = [points for points in expected if Polygon(points).area >= cutoff]

    actual = list(
        global_rois(pixel_rois, spacing, origin, affine, cutoff, workers=workers)
    )

    assert 0 < len(actual) == len(expected) < len(pixel_rois)
    assert [roi.name for roi in actual] == [
        f"cell_{i:07d}" for i in range(len(expected))
    ]
    for roi, points in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(roi.coordinates(), points)


@pytest.mark.unit
def test_vectorized_areas_handle_orientation_closure_and_large_origins():
    square = np.array([[0, 0], [3, 0], [3, 3], [0, 3]], dtype=np.float32)
    polygons = [
        square,
        square[::-1],
        np.vstack((square, square[0])),
        square + 1000000,
        np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32),
    ]
    offsets = np.concatenate(([0], np.cumsum([len(points) for points in polygons])))
    areas = _polygon_areas(np.concatenate(polygons), offsets)
    np.testing.assert_array_equal(areas, [9, 9, 9, 9, 0])


@pytest.mark.unit
def test_empty_inputs_produce_no_rois():
    assert (
        extract_pixel_rois(np.zeros((10, 10), dtype=np.uint16), np.array([]), workers=4)
        == []
    )
    assert list(global_rois([], np.ones(3), np.zeros(3), np.eye(4), 1, workers=4)) == []
