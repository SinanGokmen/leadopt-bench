"""Cached read-only client for the ChEMBL REST API.

Every page is written to disk verbatim before being parsed, so re-running any
later step costs nothing and the raw evidence stays auditable.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from typing import Any, Iterator
from urllib.parse import urlencode

import requests

from . import config as C

_SESSION = requests.Session()
_SESSION.headers.update({"Accept": "application/json", "User-Agent": "leadopt_bench/0.1"})


def _cache_path(endpoint: str, params: dict[str, Any]) -> "object":
    key = urlencode(sorted(params.items()))
    digest = hashlib.sha256(key.encode()).hexdigest()[:20]
    d = C.RAW_DIR / endpoint
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{digest}.json.gz"


def _fetch_page(endpoint: str, params: dict[str, Any], refresh: bool = False) -> dict:
    path = _cache_path(endpoint, params)
    if path.exists() and not refresh:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)

    url = f"{C.API_BASE}/{endpoint}.json"
    last_err: Exception | None = None
    for attempt in range(C.RETRIES):
        try:
            r = _SESSION.get(url, params=params, timeout=C.REQUEST_TIMEOUT)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            payload = r.json()
            break
        except Exception as exc:  # network hiccup / throttling
            last_err = exc
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"{endpoint} failed after {C.RETRIES} attempts: {last_err}")

    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return payload


def paged(endpoint: str, params: dict[str, Any], collection: str,
          refresh: bool = False, max_records: int | None = None) -> Iterator[dict]:
    """Yield every record from a paginated endpoint, following page_meta.next."""
    offset = 0
    seen = 0
    while True:
        page_params = dict(params, limit=C.PAGE_LIMIT, offset=offset)
        payload = _fetch_page(endpoint, page_params, refresh=refresh)
        records = payload.get(collection, [])
        for rec in records:
            yield rec
            seen += 1
            if max_records is not None and seen >= max_records:
                return
        meta = payload.get("page_meta") or {}
        if not meta.get("next") or not records:
            return
        offset += C.PAGE_LIMIT
        if offset >= (meta.get("total_count") or 0):
            return
        time.sleep(C.SLEEP_BETWEEN_PAGES)


def total_count(endpoint: str, params: dict[str, Any], refresh: bool = False) -> int:
    """One cheap request that reports how many records a filter would return."""
    payload = _fetch_page(endpoint, dict(params, limit=1, offset=0), refresh=refresh)
    return (payload.get("page_meta") or {}).get("total_count", 0)


# --- typed helpers ------------------------------------------------------------

TARGET_FIELDS = "target_chembl_id,pref_name,target_type,organism"
ASSAY_FIELDS = "assay_chembl_id,document_chembl_id,description,confidence_score,assay_type"
ACTIVITY_FIELDS = (
    "molecule_chembl_id,canonical_smiles,pchembl_value,standard_type,standard_value,"
    "standard_units,standard_relation,assay_chembl_id,document_chembl_id,document_year,"
    "data_validity_comment,potential_duplicate,target_chembl_id,target_pref_name"
)


def target_info(target_chembl_id: str) -> dict:
    payload = _fetch_page("target", {"target_chembl_id": target_chembl_id,
                                     "only": TARGET_FIELDS, "limit": 1, "offset": 0})
    targets = payload.get("targets", [])
    return targets[0] if targets else {}


def confident_assays(target_chembl_id: str) -> list[dict]:
    """Confidence-9 binding assays for one target."""
    params = {
        "target_chembl_id": target_chembl_id,
        "confidence_score": C.CONFIDENCE_SCORE,
        "assay_type": C.ASSAY_TYPE,
        "only": ASSAY_FIELDS,
    }
    return list(paged("assay", params, "assays"))


def activities(target_chembl_id: str) -> list[dict]:
    """Binding IC50/Ki measurements with a pChEMBL value, for one target."""
    params = {
        "target_chembl_id": target_chembl_id,
        "standard_type__in": ",".join(C.ACTIVITY_TYPES),
        "assay_type": C.ASSAY_TYPE,
        "pchembl_value__isnull": "false",
        "only": ACTIVITY_FIELDS,
    }
    return list(paged("activity", params, "activities"))


def activity_count(target_chembl_id: str) -> int:
    params = {
        "target_chembl_id": target_chembl_id,
        "standard_type__in": ",".join(C.ACTIVITY_TYPES),
        "assay_type": C.ASSAY_TYPE,
        "pchembl_value__isnull": "false",
    }
    return total_count("activity", params)
