"""Select the benchmark panel along the contamination axis.

For each target we take two year strata of series:

  old   published long before any current model's training cutoff -- the model
        has almost certainly read these papers
  new   published recently enough to be absent, or nearly absent, from training
        data -- the model has to reason rather than recall

Comparing LLM performance across the two strata is what separates reasoning
from recall. Everything else in the panel (target, series size, activity span)
is held as comparable as the data allows.
"""

from __future__ import annotations

import json

from . import chembl, config as C
from .series import Series, build_series, clean_activities

# The four targets chosen at the end of Task 1.
PANEL_TARGETS = {
    "CHEMBL1163125": "BRD4",      # epigenetic reader
    "CHEMBL325": "HDAC1",         # zinc-dependent metalloenzyme
    "CHEMBL204": "Thrombin",      # serine protease
    "CHEMBL203": "EGFR",          # kinase
}

# Which targets carry which claim, decided after measuring the strata (Task 2).
#
# "exposure_axis" targets have old and new strata that are naturally matched on
# lead potency and available gain, so an old-vs-new difference there is not just
# a difficulty difference. Thrombin and EGFR are not matched -- their post-2024
# series start from an already-potent compound -- so they contribute their old,
# heavily-cited series as an exposure-positive control instead.
TARGET_ROLES = {
    "CHEMBL1163125": "exposure_axis",
    "CHEMBL325": "exposure_axis",
    "CHEMBL204": "exposure_positive_control",
    "CHEMBL203": "exposure_positive_control",
}

OLD_MAX_YEAR = 2015
NEW_MIN_YEAR = 2024

# Quality floor for panel membership, above the bare acceptance criteria:
# a series we hand to an LLM should be a clear optimization campaign.
PANEL_MIN_N = 24
PANEL_MIN_RANGE = 2.2
PANEL_MIN_TANIMOTO = 0.45


def load_target_series(target_chembl_id: str, grouping: str = "assay") -> list[Series]:
    """Rebuild accepted series for one target from the cached ChEMBL pages."""
    name = chembl.target_info(target_chembl_id).get("pref_name") or target_chembl_id
    cleaned = clean_activities(chembl.activities(target_chembl_id))
    ok_assays = {a["assay_chembl_id"] for a in chembl.confident_assays(target_chembl_id)}
    cleaned = [r for r in cleaned if r["assay_chembl_id"] in ok_assays]
    return [s for s in build_series(cleaned, target_chembl_id, name, grouping=grouping)
            if s.passes()]


def _quality_rank(s: Series) -> tuple:
    """Prefer wide activity span, then tight congenericity, then size."""
    return (round(s.log_range, 1), round(s.mean_tanimoto, 2), s.n)


def _stratum(series: list[Series], lo: int | None, hi: int | None) -> list[Series]:
    out = []
    for s in series:
        y = s.summary()["year_min"]
        if y is None:
            continue
        if lo is not None and y < lo:
            continue
        if hi is not None and y > hi:
            continue
        if (s.n >= PANEL_MIN_N and s.log_range >= PANEL_MIN_RANGE
                and s.mean_tanimoto >= PANEL_MIN_TANIMOTO):
            out.append(s)
    return sorted(out, key=_quality_rank, reverse=True)


def build_panel(n_per_stratum: int = 5, grouping: str = "assay") -> dict:
    """Pick up to n_per_stratum old and new series for each panel target."""
    panel: dict = {"grouping": grouping, "old_max_year": OLD_MAX_YEAR,
                   "new_min_year": NEW_MIN_YEAR, "targets": {}}

    for tid, label in PANEL_TARGETS.items():
        series = load_target_series(tid, grouping=grouping)
        old = _stratum(series, None, OLD_MAX_YEAR)[:n_per_stratum]
        new = _stratum(series, NEW_MIN_YEAR, None)[:n_per_stratum]

        for s in old + new:
            s.write()

        panel["targets"][tid] = {
            "label": label,
            "role": TARGET_ROLES.get(tid, "unassigned"),
            "target_pref_name": series[0].target_pref_name if series else tid,
            "n_accepted_total": len(series),
            "old": [s.summary() for s in old],
            "new": [s.summary() for s in new],
        }
        print(f"  {label:<10} accepted={len(series):<4} "
              f"old({'<=' + str(OLD_MAX_YEAR)})={len(old)}  "
              f"new(>={NEW_MIN_YEAR})={len(new)}", flush=True)

    path = C.REPORT_DIR / f"panel_{grouping}.json"
    path.write_text(json.dumps(panel, indent=2))
    print(f"\npanel written: {path}")
    return panel
