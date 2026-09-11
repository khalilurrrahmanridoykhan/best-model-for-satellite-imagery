"""Turns one set of building-footprint polygons into the two label formats
this project's two models need -- a pixel mask for U-Net, and normalized
bounding boxes for YOLO. Both come from the exact same source geometries,
which is the whole point of this project: the models are compared on
identical ground truth, not two independently-labeled datasets.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry.base import BaseGeometry


def filter_polygon_geometries(geometries: Iterable[BaseGeometry]) -> list[BaseGeometry]:
    """Keeps only Polygon/MultiPolygon geometries. Live-observed against
    the real downloaded SpaceNet data: labels occasionally include a
    degenerate Point (a building footprint that collapsed to a single
    point -- 1 in 594 geometries across a real 20-chip sample). A Point
    produces a meaningless zero-area YOLO box, and applying it only to
    the YOLO pipeline while leaving it in the U-Net mask would mean the
    two label formats no longer describe the same set of buildings --
    exactly the property this whole project depends on -- so this filter
    is meant to be applied once, before *either* rasterize_polygons() or
    polygons_to_yolo_boxes(), not separately to each.
    """
    return [g for g in geometries if g.geom_type in ("Polygon", "MultiPolygon")]


def rasterize_polygons(
    polygons: Iterable[BaseGeometry],
    height: int,
    width: int,
    transform: Affine,
) -> np.ndarray:
    """Burns a set of polygons into a binary mask on the given pixel grid.
    An empty polygon list still produces a valid all-zero mask (a chip
    with no buildings is a real, useful training example -- not an
    error), matching rasterize()'s own behavior for an empty shapes list.
    """
    geoms = list(polygons)
    if not geoms:
        return np.zeros((height, width), dtype=np.uint8)
    return rasterize(
        [(geom, 1) for geom in geoms],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    )


def polygon_to_yolo_box(
    polygon: BaseGeometry,
    image_width: int,
    image_height: int,
    transform: Affine,
    class_id: int = 0,
) -> tuple[int, float, float, float, float]:
    """One polygon's axis-aligned bounding box, in YOLO's normalized
    (class, x_center, y_center, width, height) format, per Ultralytics'
    label format.

    Live-caught bug this `transform` argument exists to fix: a real
    SpaceNet polygon's coordinates are in the raster's own CRS (lon/lat
    here), not pixel space -- naively dividing polygon.bounds by
    image_width/image_height (as an earlier version of this function
    did) mixes geographic-coordinate units with a pixel-count
    denominator and produces nonsense (values wildly outside [0, 1]).
    `transform` is the same rasterio Affine used to rasterize these same
    polygons into a mask (rasterize_polygons above) -- its inverse maps
    the polygon's geographic bounds into pixel space first, so a YOLO box
    actually lines up with the mask it's meant to match. Checked against
    all four corners of the bounding box, not just two, since a
    rotated/skewed transform could otherwise map the geographic min/max
    corner to something other than the pixel-space min/max corner.
    """
    min_x, min_y, max_x, max_y = polygon.bounds
    inv = ~transform
    corners = [inv @ (min_x, min_y), inv @ (min_x, max_y), inv @ (max_x, min_y), inv @ (max_x, max_y)]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    px_min_x, px_max_x = min(xs), max(xs)
    px_min_y, px_max_y = min(ys), max(ys)

    box_width = px_max_x - px_min_x
    box_height = px_max_y - px_min_y
    x_center = px_min_x + box_width / 2
    y_center = px_min_y + box_height / 2
    return (
        class_id,
        x_center / image_width,
        y_center / image_height,
        box_width / image_width,
        box_height / image_height,
    )


def polygons_to_yolo_boxes(
    polygons: Iterable[BaseGeometry],
    image_width: int,
    image_height: int,
    transform: Affine,
    class_id: int = 0,
) -> list[tuple[int, float, float, float, float]]:
    """polygon_to_yolo_box for every polygon in the chip -- one line per
    building in the resulting YOLO label file, same source geometries
    (and the same transform) as rasterize_polygons() so both label
    formats describe the same buildings, in the same pixel grid."""
    return [polygon_to_yolo_box(p, image_width, image_height, transform, class_id) for p in polygons]


def format_yolo_label_file(boxes: Iterable[tuple[int, float, float, float, float]]) -> str:
    """The exact text Ultralytics expects in one image's .txt label file
    -- one box per line, six-decimal precision (plenty for a normalized
    0-1 coordinate), no trailing blank line."""
    lines = [f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for cls, x, y, w, h in boxes]
    return "\n".join(lines)
