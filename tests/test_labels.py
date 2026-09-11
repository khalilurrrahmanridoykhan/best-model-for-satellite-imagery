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

    cls, x, y, w, h = polygon_to_yolo_box(box, image_width=100, image_height=100)

    assert cls == 0
    assert x == pytest.approx(0.15)
    assert y == pytest.approx(0.20)
    assert w == pytest.approx(0.10)
    assert h == pytest.approx(0.20)


def test_polygon_to_yolo_box_respects_a_custom_class_id():
    box = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
    cls, *_ = polygon_to_yolo_box(box, image_width=100, image_height=100, class_id=3)
    assert cls == 3


def test_polygons_to_yolo_boxes_returns_one_box_per_polygon_in_order():
    a = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
    b = Polygon([(50, 50), (50, 60), (60, 60), (60, 50)])

    boxes = polygons_to_yolo_boxes([a, b], image_width=100, image_height=100)

    assert len(boxes) == 2
    assert boxes[0][1] == pytest.approx(0.05)  # a's x_center
    assert boxes[1][1] == pytest.approx(0.55)  # b's x_center


def test_polygons_to_yolo_boxes_handles_an_empty_list():
    assert polygons_to_yolo_boxes([], image_width=100, image_height=100) == []


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
    boxes = polygons_to_yolo_boxes(polygons, image_width=100, image_height=100)

    # Every YOLO box's center should land on a masked-in pixel.
    for _, x, y, _, _ in boxes:
        px, py = int(x * 100), int(y * 100)
        assert mask[py, px] == 1
