"""Downloads a subset of SpaceNet Buildings v2 (AOI_2_Vegas) chips -- one
RGB GeoTIFF + one building-footprint GeoJSON per chip -- directly from
the public S3 bucket. No AWS account or credentials needed (anonymous,
unsigned requests to a public bucket).

Downloads individual per-chip objects rather than the official ~25 GB
training tarball for this AOI. SpaceNet also publishes a STAC catalog
(s3://spacenet-dataset/spacenet-stac/SN2_buildings/...) whose per-item
JSON reveals that every chip's image and label are independently
addressable objects under spacenet/SN2_buildings/train/AOI_2_Vegas/ --
verified directly against the real bucket, not assumed -- which is what
makes "a few hundred chips for practice" an actual few-hundred-MB
download instead of an overnight one.

Usage:
    python scripts/download_data.py --n 300
    python scripts/download_data.py --n 300 --out data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "spacenet-dataset"
IMAGE_PREFIX = "spacenet/SN2_buildings/train/AOI_2_Vegas/PS-RGB/"
LABEL_PREFIX = "spacenet/SN2_buildings/train/AOI_2_Vegas/geojson_buildings/"
IMAGE_NAME_TEMPLATE = "SN2_buildings_train_AOI_2_Vegas_PS-RGB_img{n}.tif"
LABEL_NAME_TEMPLATE = "SN2_buildings_train_AOI_2_Vegas_geojson_buildings_img{n}.geojson"


def make_client():
    """Unsigned config -- this is a public bucket; a normal signed boto3
    client would fail with a credentials error for something that needs
    none."""
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_available_chip_ids(client, limit: int | None = None) -> list[int]:
    """Chip ids are not contiguous (some numbers are missing from the
    bucket entirely) -- lists the real image objects that exist rather
    than assuming a clean 1..N range. Stops paginating early once `limit`
    ids are found, so requesting a small practice subset doesn't require
    listing all ~3,850 objects in the AOI first.
    """
    paginator = client.get_paginator("list_objects_v2")
    ids: list[int] = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=IMAGE_PREFIX):
        for obj in page.get("Contents", []):
            name = obj["Key"].rsplit("/", 1)[-1]
            if not name.endswith(".tif"):
                continue
            # "SN2_buildings_train_AOI_2_Vegas_PS-RGB_img123.tif" -> 123
            chip_id = int(name.removesuffix(".tif").rsplit("img", 1)[-1])
            ids.append(chip_id)
            if limit is not None and len(ids) >= limit:
                return sorted(ids)
    return sorted(ids)


def download_chip(client, chip_id: int, out_dir: Path) -> None:
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_key = IMAGE_PREFIX + IMAGE_NAME_TEMPLATE.format(n=chip_id)
    label_key = LABEL_PREFIX + LABEL_NAME_TEMPLATE.format(n=chip_id)

    client.download_file(BUCKET, image_key, str(images_dir / f"img{chip_id}.tif"))
    client.download_file(BUCKET, label_key, str(labels_dir / f"img{chip_id}.geojson"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=300, help="number of chips to download (default: 300)")
    parser.add_argument("--out", type=Path, default=Path("data/raw"), help="output directory (default: data/raw)")
    args = parser.parse_args()

    client = make_client()
    print(f"Listing available chips in s3://{BUCKET}/{IMAGE_PREFIX} ...")
    chip_ids = list_available_chip_ids(client, limit=args.n)
    print(f"Downloading {len(chip_ids)} chips (image + label each) to {args.out} ...")

    for i, chip_id in enumerate(chip_ids, start=1):
        download_chip(client, chip_id, args.out)
        if i % 25 == 0 or i == len(chip_ids):
            print(f"  {i}/{len(chip_ids)}")

    print("Done.")


if __name__ == "__main__":
    main()
