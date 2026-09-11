"""Tests for the chip-id listing/parsing logic in scripts/download_data.py
-- a fake paginator stands in for the real S3 client, so this is
verified without any real network call.
"""

from scripts.download_data import list_available_chip_ids


class _FakePaginator:
    def __init__(self, pages: list[list[str]]):
        self._pages = pages

    def paginate(self, **kwargs):
        for page_keys in self._pages:
            yield {"Contents": [{"Key": k} for k in page_keys]}


class _FakeClient:
    def __init__(self, pages: list[list[str]]):
        self._paginator = _FakePaginator(pages)

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return self._paginator


def _key(chip_id: int) -> str:
    return f"spacenet/SN2_buildings/train/AOI_2_Vegas/PS-RGB/SN2_buildings_train_AOI_2_Vegas_PS-RGB_img{chip_id}.tif"


def test_list_available_chip_ids_parses_ids_from_real_looking_keys():
    client = _FakeClient([[_key(1), _key(10), _key(2)]])
    ids = list_available_chip_ids(client)
    assert ids == [1, 2, 10]  # sorted, not listing order


def test_list_available_chip_ids_handles_non_contiguous_ids():
    """Real SpaceNet chip ids skip numbers (some are missing entirely) --
    must not assume a clean 1..N range."""
    client = _FakeClient([[_key(1), _key(1002), _key(7)]])
    ids = list_available_chip_ids(client)
    assert ids == [1, 7, 1002]


def test_list_available_chip_ids_stops_early_at_the_limit():
    client = _FakeClient([[_key(i) for i in range(1, 6)], [_key(i) for i in range(6, 11)]])
    ids = list_available_chip_ids(client, limit=3)
    assert len(ids) == 3


def test_list_available_chip_ids_spans_multiple_pages():
    client = _FakeClient([[_key(1), _key(2)], [_key(3), _key(4)]])
    ids = list_available_chip_ids(client)
    assert ids == [1, 2, 3, 4]


def test_list_available_chip_ids_ignores_non_tif_keys():
    keys = [_key(1), "spacenet/SN2_buildings/train/AOI_2_Vegas/PS-RGB/.DS_Store"]
    client = _FakeClient([keys])
    ids = list_available_chip_ids(client)
    assert ids == [1]
