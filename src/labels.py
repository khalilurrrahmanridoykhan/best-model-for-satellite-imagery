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
    class_id: int = 0,
) -> tuple[int, float, float, float, float]:
    """One polygon's axis-aligned bounding box, in YOLO's normalized
    (class, x_center, y_center, width, height) format -- all four
    geometry values in [0, 1], relative to the image's own width/height,
    per Ultralytics' label format.
    """
    min_x, min_y, max_x, max_y = polygon.bounds
    box_width = max_x - min_x
    box_height = max_y - min_y
    x_center = min_x + box_width / 2
    y_center = min_y + box_height / 2
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
    class_id: int = 0,
) -> list[tuple[int, float, float, float, float]]:
    """polygon_to_yolo_box for every polygon in the chip -- one line per
    building in the resulting YOLO label file, same source geometries as
    rasterize_polygons() so both label formats describe the same
    buildings."""
    return [polygon_to_yolo_box(p, image_width, image_height, class_id) for p in polygons]


def format_yolo_label_file(boxes: Iterable[tuple[int, float, float, float, float]]) -> str:
    """The exact text Ultralytics expects in one image's .txt label file
    -- one box per line, six-decimal precision (plenty for a normalized
    0-1 coordinate), no trailing blank line."""
    lines = [f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for cls, x, y, w, h in boxes]
    return "\n".join(lines)
