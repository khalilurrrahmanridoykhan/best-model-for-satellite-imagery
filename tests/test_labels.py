"""Tests for the polygon -> mask / polygon -> YOLO-box pipeline, using
synthetic geometries -- no dataset download needed to verify this logic
is correct.
"""

import numpy as np
import pytest
from rasterio.transform import Affine
from shapely.geometry import Polygon

from src.labels import (
    format_yolo_label_file,
    polygon_to_yolo_box,
    polygons_to_yolo_boxes,
    rasterize_polygons,
)

# A simple identity-like transform: pixel (col, row) == world (x, y),
# so a polygon's coordinates can be reasoned about directly as pixels.
IDENTITY_TRANSFORM = Affine.identity()

# A transform resembling a *real* georeferenced chip (see
# test_polygon_to_yolo_box_handles_a_real_georeferenced_transform below)
# -- origin at a real-looking longitude/latitude, ~1.2m pixels. Live-
# caught bug this exists to guard against: an identity transform makes
# geographic-space and pixel-space coordinates coincide by construction,
# which silently hid a real unit-mismatch bug that only showed up
# against the actual downloaded SpaceNet data.
GEOREFERENCED_TRANSFORM = Affine(1.0761e-05, 0.0, -115.3075176, 0.0, -1.0761e-05, 36.1282976997)


def test_rasterize_polygons_burns_a_square_into_the_mask():
    square = Polygon([(2, 2), (2, 5), (5, 5), (5, 2)])
    mask = rasterize_polygons([square], height=10, width=10, transform=IDENTITY_TRANSFORM)

    assert mask.shape == (10, 10)
    assert mask.dtype == np.uint8
    assert mask[3, 3] == 1  # inside the square
    assert mask[0, 0] == 0  # outside the square
    assert mask.sum() > 0


def test_rasterize_polygons_handles_an_empty_list():
    """A chip with no buildings is a real, useful training example, not
    an error -- rasterize() itself can't handle an empty shapes list, so
    this must be special-cased rather than crashing."""
    mask = rasterize_polygons([], height=10, width=10, transform=IDENTITY_TRANSFORM)

    assert mask.shape == (10, 10)
    assert mask.sum() == 0


def test_rasterize_polygons_handles_two_separate_buildings():
    a = Polygon([(0, 0), (0, 2), (2, 2), (2, 0)])
    b = Polygon([(6, 6), (6, 8), (8, 8), (8, 6)])
    mask = rasterize_polygons([a, b], height=10, width=10, transform=IDENTITY_TRANSFORM)

    assert mask[1, 1] == 1
    assert mask[7, 7] == 1
    assert mask[4, 4] == 0  # the gap between them


def test_polygon_to_yolo_box_centers_and_normalizes_correctly():
    # A 10x20 box (width x height) with its corner at (10, 10), in a
    # 100x100 image -- center should land at (15, 20)/100.
    box = Polygon([(10, 10), (10, 30), (20, 30), (20, 10)])

    cls, x, y, w, h = polygon_to_yolo_box(box, image_width=100, image_height=100, transform=IDENTITY_TRANSFORM)

    assert cls == 0
    assert x == pytest.approx(0.15)
    assert y == pytest.approx(0.20)
    assert w == pytest.approx(0.10)
    assert h == pytest.approx(0.20)


def test_polygon_to_yolo_box_respects_a_custom_class_id():
    box = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
    cls, *_ = polygon_to_yolo_box(box, image_width=100, image_height=100, transform=IDENTITY_TRANSFORM, class_id=3)
    assert cls == 3


def test_polygon_to_yolo_box_handles_a_real_georeferenced_transform():
    """Live-caught bug: against the actual downloaded SpaceNet data (real
    lon/lat polygons, a real Affine transform), the original
    implementation naively divided geographic-CRS bounds by a pixel
    count and produced box coordinates wildly outside [0, 1]. A polygon
    covering the chip's full extent (as returned by rasterio's own
    transform * (width, height)) must map back to a box spanning
    essentially the whole normalized [0, 1] range.
    """
    width, height = 650, 650
    top_left = GEOREFERENCED_TRANSFORM @ (0, 0)
    bottom_right = GEOREFERENCED_TRANSFORM @ (width, height)
    min_x, max_x = sorted([top_left[0], bottom_right[0]])
    min_y, max_y = sorted([top_left[1], bottom_right[1]])
    full_extent = Polygon([(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)])

    cls, x, y, w, h = polygon_to_yolo_box(full_extent, width, height, transform=GEOREFERENCED_TRANSFORM)

    assert 0.0 <= x <= 1.0
    assert 0.0 <= y <= 1.0
    assert w == pytest.approx(1.0, abs=1e-6)
    assert h == pytest.approx(1.0, abs=1e-6)


def test_polygons_to_yolo_boxes_returns_one_box_per_polygon_in_order():
    a = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
    b = Polygon([(50, 50), (50, 60), (60, 60), (60, 50)])

    boxes = polygons_to_yolo_boxes([a, b], image_width=100, image_height=100, transform=IDENTITY_TRANSFORM)

    assert len(boxes) == 2
    assert boxes[0][1] == pytest.approx(0.05)  # a's x_center
    assert boxes[1][1] == pytest.approx(0.55)  # b's x_center


def test_polygons_to_yolo_boxes_handles_an_empty_list():
    assert polygons_to_yolo_boxes([], image_width=100, image_height=100, transform=IDENTITY_TRANSFORM) == []


def test_format_yolo_label_file_matches_ultralytics_format():
    boxes = [(0, 0.5, 0.5, 0.1, 0.2), (1, 0.25, 0.75, 0.05, 0.05)]

    text = format_yolo_label_file(boxes)

    assert text == "0 0.500000 0.500000 0.100000 0.200000\n1 0.250000 0.750000 0.050000 0.050000"


def test_format_yolo_label_file_handles_no_boxes():
    assert format_yolo_label_file([]) == ""


def test_mask_and_boxes_agree_on_which_buildings_exist():
    """The whole point of this pipeline: both label formats must
    describe the *same* buildings, from the same source polygons."""
    a = Polygon([(10, 10), (10, 20), (20, 20), (20, 10)])
    b = Polygon([(60, 60), (60, 70), (70, 70), (70, 60)])
    polygons = [a, b]

    mask = rasterize_polygons(polygons, height=100, width=100, transform=IDENTITY_TRANSFORM)
    boxes = polygons_to_yolo_boxes(polygons, image_width=100, image_height=100, transform=IDENTITY_TRANSFORM)

    # Every YOLO box's center should land on a masked-in pixel.
    for _, x, y, _, _ in boxes:
        px, py = int(x * 100), int(y * 100)
        assert mask[py, px] == 1
