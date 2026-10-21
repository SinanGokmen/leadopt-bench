"""Tests for the cached ChEMBL client: pagination, caching, retries.

No network: a fake session answers every request, so these also prove the
pipeline re-runs from cache without touching EBI.
"""

import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leadopt_bench import chembl
from leadopt_bench import config as C


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Serves `pages` in order and counts how often it was called."""

    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = 0
        self.seen_params = []

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        self.seen_params.append(dict(params or {}))
        payload = self.pages[min(self.calls - 1, len(self.pages) - 1)]
        if isinstance(payload, int):
            return FakeResponse({}, status=payload)
        return FakeResponse(payload)


def _page(records, offset, total, has_next, collection="activities"):
    return {
        "page_meta": {
            "limit": C.PAGE_LIMIT,
            "offset": offset,
            "total_count": total,
            "next": "/next" if has_next else None,
        },
        collection: records,
    }


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Never read or write the real data/raw cache during tests."""
    monkeypatch.setattr(C, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(chembl.C, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(chembl.C, "SLEEP_BETWEEN_PAGES", 0)
    yield


def test_paged_returns_every_record_across_pages(monkeypatch):
    a = [{"activity_id": i} for i in range(C.PAGE_LIMIT)]
    b = [{"activity_id": i} for i in range(C.PAGE_LIMIT, C.PAGE_LIMIT + 7)]
    total = len(a) + len(b)
    fake = FakeSession([_page(a, 0, total, True), _page(b, C.PAGE_LIMIT, total, False)])
    monkeypatch.setattr(chembl, "_SESSION", fake)

    got = list(chembl.paged("activity", {"q": 1}, "activities"))
    assert len(got) == total
    assert got[0]["activity_id"] == 0
    assert got[-1]["activity_id"] == total - 1
    assert fake.calls == 2
    assert fake.seen_params[1]["offset"] == C.PAGE_LIMIT


def test_paged_stops_when_next_is_null(monkeypatch):
    fake = FakeSession([_page([{"activity_id": 1}], 0, 1, False)])
    monkeypatch.setattr(chembl, "_SESSION", fake)
    assert len(list(chembl.paged("activity", {"q": 2}, "activities"))) == 1
    assert fake.calls == 1


def test_paged_honors_max_records(monkeypatch):
    recs = [{"activity_id": i} for i in range(C.PAGE_LIMIT)]
    fake = FakeSession([_page(recs, 0, 10_000, True)])
    monkeypatch.setattr(chembl, "_SESSION", fake)
    assert len(list(chembl.paged("activity", {"q": 3}, "activities", max_records=5))) == 5


def test_second_run_is_served_from_disk_cache(monkeypatch):
    fake = FakeSession([_page([{"activity_id": 9}], 0, 1, False)])
    monkeypatch.setattr(chembl, "_SESSION", fake)

    first = list(chembl.paged("activity", {"q": 4}, "activities"))
    second = list(chembl.paged("activity", {"q": 4}, "activities"))
    assert first == second
    assert fake.calls == 1, "cached page must not hit the network again"


def test_refresh_bypasses_the_cache(monkeypatch):
    fake = FakeSession([_page([{"activity_id": 1}], 0, 1, False),
                        _page([{"activity_id": 2}], 0, 1, False)])
    monkeypatch.setattr(chembl, "_SESSION", fake)

    list(chembl.paged("activity", {"q": 5}, "activities"))
    again = list(chembl.paged("activity", {"q": 5}, "activities", refresh=True))
    assert fake.calls == 2
    assert again[0]["activity_id"] == 2


def test_server_error_is_retried_then_succeeds(monkeypatch):
    fake = FakeSession([503, 500, _page([{"activity_id": 7}], 0, 1, False)])
    monkeypatch.setattr(chembl, "_SESSION", fake)
    monkeypatch.setattr(chembl.time, "sleep", lambda *_: None)

    got = list(chembl.paged("activity", {"q": 6}, "activities"))
    assert got == [{"activity_id": 7}]
    assert fake.calls == 3


def test_rate_limit_is_retried(monkeypatch):
    fake = FakeSession([429, _page([{"activity_id": 8}], 0, 1, False)])
    monkeypatch.setattr(chembl, "_SESSION", fake)
    monkeypatch.setattr(chembl.time, "sleep", lambda *_: None)
    assert list(chembl.paged("activity", {"q": 7}, "activities"))[0]["activity_id"] == 8


def test_persistent_failure_raises_rather_than_returning_partial_data(monkeypatch):
    fake = FakeSession([500] * 10)
    monkeypatch.setattr(chembl, "_SESSION", fake)
    monkeypatch.setattr(chembl.time, "sleep", lambda *_: None)
    with pytest.raises(RuntimeError, match="failed after"):
        list(chembl.paged("activity", {"q": 8}, "activities"))
    assert fake.calls == C.RETRIES


def test_failed_request_is_not_cached(monkeypatch):
    """A 500 must not poison the cache with an empty page."""
    fake = FakeSession([500] * 4)
    monkeypatch.setattr(chembl, "_SESSION", fake)
    monkeypatch.setattr(chembl.time, "sleep", lambda *_: None)
    with pytest.raises(RuntimeError):
        list(chembl.paged("activity", {"q": 9}, "activities"))

    good = FakeSession([_page([{"activity_id": 3}], 0, 1, False)])
    monkeypatch.setattr(chembl, "_SESSION", good)
    assert list(chembl.paged("activity", {"q": 9}, "activities"))[0]["activity_id"] == 3


def test_total_count_costs_one_request(monkeypatch):
    fake = FakeSession([_page([], 0, 4242, True)])
    monkeypatch.setattr(chembl, "_SESSION", fake)
    assert chembl.total_count("activity", {"q": 10}) == 4242
    assert fake.calls == 1
    assert fake.seen_params[0]["limit"] == 1


def test_activity_query_filters_to_uncensored_binding_data(monkeypatch):
    fake = FakeSession([_page([], 0, 0, False)])
    monkeypatch.setattr(chembl, "_SESSION", fake)
    chembl.activity_count("CHEMBL301")
    params = fake.seen_params[0]
    assert params["assay_type"] == "B"
    assert params["pchembl_value__isnull"] == "false"
    assert set(params["standard_type__in"].split(",")) == set(C.ACTIVITY_TYPES)


def test_confident_assays_requests_confidence_nine(monkeypatch):
    fake = FakeSession([_page([], 0, 0, False, collection="assays")])
    monkeypatch.setattr(chembl, "_SESSION", fake)
    chembl.confident_assays("CHEMBL301")
    assert fake.seen_params[0]["confidence_score"] == C.CONFIDENCE_SCORE


def test_cache_key_distinguishes_different_queries(monkeypatch):
    fake = FakeSession([_page([{"activity_id": 1}], 0, 1, False),
                        _page([{"activity_id": 2}], 0, 1, False)])
    monkeypatch.setattr(chembl, "_SESSION", fake)
    one = list(chembl.paged("activity", {"target": "A"}, "activities"))
    two = list(chembl.paged("activity", {"target": "B"}, "activities"))
    assert one != two and fake.calls == 2
