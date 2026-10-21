"""Measure, per target, how many usable congeneric series ChEMBL actually yields.

This is the Task-1 decision aid: raw activity counts are misleading, because a
target with 18k measurements can still be too diffuse to give congeneric series.
What matters is series surviving the n / Tanimoto / log-range criteria.
"""

from __future__ import annotations

import json
import statistics

import pandas as pd

from . import chembl, config as C
from .series import build_series, clean_activities

GROUPINGS = ("assay", "document", "cluster")


def survey_target(target_chembl_id: str, groupings=GROUPINGS) -> dict:
    info = chembl.target_info(target_chembl_id)
    name = info.get("pref_name") or target_chembl_id

    raw = chembl.activities(target_chembl_id)
    cleaned = clean_activities(raw)

    # Restrict to confidence-9 binding assays (the activity endpoint cannot filter on this).
    ok_assays = {a["assay_chembl_id"] for a in chembl.confident_assays(target_chembl_id)}
    cleaned = [r for r in cleaned if r["assay_chembl_id"] in ok_assays]

    result = {
        "target_chembl_id": target_chembl_id,
        "target_pref_name": name,
        "n_raw": len(raw),
        "n_clean_conf9": len(cleaned),
        "n_molecules": len({r["molecule_chembl_id"] for r in cleaned}),
        "groupings": {},
    }

    for grouping in groupings:
        candidates = build_series(cleaned, target_chembl_id, name, grouping=grouping)
        passing = [s for s in candidates if s.passes()]
        result["groupings"][grouping] = {
            "n_candidates_n20": len(candidates),
            "n_passing": len(passing),
            "n_passing_temporal": sum(1 for s in passing if s.temporally_splittable),
            "median_n": statistics.median([s.n for s in passing]) if passing else 0,
            "max_n": max([s.n for s in passing], default=0),
            "median_tanimoto": round(statistics.median([s.mean_tanimoto for s in passing]), 3)
                               if passing else 0.0,
            "median_log_range": round(statistics.median([s.log_range for s in passing]), 2)
                                if passing else 0.0,
            "series": [s.summary() for s in passing],
        }
    return result


def survey(target_ids: list[str], out_name: str = "survey") -> pd.DataFrame:
    results = []
    for tid in target_ids:
        print(f"  surveying {tid} ...", flush=True)
        try:
            results.append(survey_target(tid))
        except Exception as exc:
            print(f"    FAILED {tid}: {type(exc).__name__}: {exc}", flush=True)

    (C.REPORT_DIR / f"{out_name}.json").write_text(json.dumps(results, indent=2))

    rows = []
    for r in results:
        row = {
            "target": r["target_chembl_id"],
            "name": r["target_pref_name"][:38],
            "raw": r["n_raw"],
            "clean9": r["n_clean_conf9"],
            "mols": r["n_molecules"],
        }
        for g, gr in r["groupings"].items():
            row[f"{g}_pass"] = gr["n_passing"]
            row[f"{g}_temporal"] = gr["n_passing_temporal"]
            row[f"{g}_medn"] = gr["median_n"]
            row[f"{g}_maxn"] = gr["max_n"]
            row[f"{g}_tan"] = gr["median_tanimoto"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(C.REPORT_DIR / f"{out_name}.csv", index=False)
    return df
