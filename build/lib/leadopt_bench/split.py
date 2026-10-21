"""Split a series into a starting compound and the improved compounds chemists reached.

Two strategies, as the spec asks, and they are deliberately symmetric: each one
defines an *early pool*, draws the lead from it by the same policy, and keeps as
targets every later compound that beats the lead by a real margin.

  activity   early pool = the weakest compounds by potency
  temporal   early pool = the compounds from the series' earliest year(s)

The asymmetry that remains is inherent, not an artifact: an activity split asks
"here is a weak compound, make it potent", while a temporal split asks "here is
where the field stood, now advance it". The second is the harder question, and
whether an LLM's performance differs between them is itself a Phase-1 finding.

Nothing about the targets may reach the LLM -- `Split.prompt_payload()` is the
only thing that is allowed to, and it carries the lead and the target protein.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field

import numpy as np
from rdkit import DataStructs

from . import config as C
from .series import Compound, Series, fingerprints

LEAD_POLICIES = ("representative", "weakest", "most_potent")


@dataclass
class Split:
    series_id: str
    strategy: str                   # "activity" | "temporal"
    target_chembl_id: str
    target_pref_name: str
    standard_type: str
    lead: Compound
    targets: list[Compound] = field(default_factory=list)
    lead_policy: str = "representative"
    early_pool_size: int = 0
    min_gain: float = 1.0
    lead_target_tanimoto: list[float] = field(default_factory=list)
    year_stratum: str = ""          # "old" | "new", carried from the panel

    # --- what the chemists actually achieved -----------------------------
    @property
    def gains(self) -> list[float]:
        return [t.pchembl - self.lead.pchembl for t in self.targets]

    @property
    def max_gain(self) -> float:
        return max(self.gains, default=0.0)

    @property
    def median_gain(self) -> float:
        return statistics.median(self.gains) if self.targets else 0.0

    @property
    def median_lead_target_similarity(self) -> float:
        return (statistics.median(self.lead_target_tanimoto)
                if self.lead_target_tanimoto else 0.0)

    def is_usable(self) -> bool:
        """A split is only a question worth asking if someone actually answered it."""
        return len(self.targets) >= 3 and self.max_gain >= self.min_gain

    # --- the only data the LLM may see -----------------------------------
    def prompt_payload(self) -> dict:
        return {
            "series_id": self.series_id,
            "strategy": self.strategy,
            "target_protein": self.target_pref_name,
            "target_chembl_id": self.target_chembl_id,
            "assay_endpoint": self.standard_type,
            "lead_smiles": self.lead.smiles,
            "lead_pchembl": self.lead.pchembl,
        }

    def summary(self) -> dict:
        return {
            "series_id": self.series_id,
            "strategy": self.strategy,
            "year_stratum": self.year_stratum,
            "target_chembl_id": self.target_chembl_id,
            "target_pref_name": self.target_pref_name,
            "standard_type": self.standard_type,
            "lead_molecule": self.lead.molecule_chembl_id,
            "lead_pchembl": round(self.lead.pchembl, 2),
            "lead_year": self.lead.document_year,
            "lead_policy": self.lead_policy,
            "early_pool_size": self.early_pool_size,
            "n_targets": len(self.targets),
            "max_gain": round(self.max_gain, 2),
            "median_gain": round(self.median_gain, 2),
            "median_lead_target_tanimoto": round(self.median_lead_target_similarity, 3),
            "target_years": sorted({t.document_year for t in self.targets if t.document_year}),
            "usable": self.is_usable(),
        }

    def to_dict(self) -> dict:
        return {**self.summary(),
                "lead": asdict(self.lead),
                "targets": [asdict(t) for t in self.targets],
                "lead_target_tanimoto": [round(x, 3) for x in self.lead_target_tanimoto]}

    def write(self):
        d = C.DATA_DIR / "splits"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{self.series_id}__{self.strategy}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


# --- lead selection -----------------------------------------------------------

def _pick_lead(pool: list[Compound], series: Series, policy: str) -> Compound:
    """Choose the starting compound out of the early pool.

    "representative" picks the pool member most similar to the series as a whole.
    A chemist does not start optimizing from a structural oddity, and the weakest
    compound in a series is often exactly that -- a probe that broke the SAR.
    """
    if policy == "weakest":
        return min(pool, key=lambda c: c.pchembl)
    if policy == "most_potent":
        return max(pool, key=lambda c: c.pchembl)
    if policy != "representative":
        raise ValueError(f"unknown lead policy: {policy}")

    all_fps = fingerprints([c.smiles for c in series.compounds])
    pool_ids = {c.molecule_chembl_id for c in pool}
    best, best_score = pool[0], -1.0
    for comp, fp in zip(series.compounds, all_fps):
        if comp.molecule_chembl_id not in pool_ids:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, all_fps)
        score = (sum(sims) - 1.0) / max(len(sims) - 1, 1)   # exclude self-similarity
        if score > best_score:
            best, best_score = comp, score
    return best


def _lead_target_similarity(lead: Compound, targets: list[Compound]) -> list[float]:
    fps = fingerprints([lead.smiles] + [t.smiles for t in targets])
    return list(DataStructs.BulkTanimotoSimilarity(fps[0], fps[1:]))


def _assemble(series: Series, early: list[Compound], later: list[Compound],
              strategy: str, policy: str, min_gain: float,
              year_stratum: str = "", min_targets: int = 3) -> Split | None:
    if not early or not later:
        return None

    def targets_for(lead: Compound) -> list[Compound]:
        return [c for c in later
                if c.pchembl - lead.pchembl >= min_gain
                and c.molecule_chembl_id != lead.molecule_chembl_id]

    # Pick the lead only from pool members that leave enough headroom to be a
    # question at all. Without this, a centrally-placed lead can sit so close to
    # the top of its series that nothing clears min_gain, and the series is lost
    # even though a slightly weaker sibling would have posed a fine problem.
    viable = [c for c in early if len(targets_for(c)) >= min_targets]
    lead = _pick_lead(viable or early, series, policy)
    targets = targets_for(lead)
    if not targets:
        return None
    return Split(
        series_id=series.series_id,
        strategy=strategy,
        target_chembl_id=series.target_chembl_id,
        target_pref_name=series.target_pref_name,
        standard_type=series.standard_type,
        lead=lead,
        targets=sorted(targets, key=lambda c: c.pchembl, reverse=True),
        lead_policy=policy,
        early_pool_size=len(early),
        min_gain=min_gain,
        lead_target_tanimoto=_lead_target_similarity(lead, targets),
        year_stratum=year_stratum,
    )


# --- the two strategies -------------------------------------------------------

def activity_split(series: Series, lead_quantile: float = 0.25,
                   target_quantile: float = 0.75, min_gain: float = 1.0,
                   lead_policy: str = "representative",
                   year_stratum: str = "") -> Split | None:
    """Lead from the weak tail, targets from the potent tail.

    Quantile-relative, so the lead's absolute potency follows whatever the
    series happens to span. Kept as a sensitivity control; `window_split` is
    the primary strategy because this one confounds era with difficulty.
    """
    vals = np.array([c.pchembl for c in series.compounds], dtype=float)
    lo = float(np.quantile(vals, lead_quantile))
    hi = float(np.quantile(vals, target_quantile))
    early = [c for c in series.compounds if c.pchembl <= lo]
    later = [c for c in series.compounds if c.pchembl >= hi]
    return _assemble(series, early, later, "activity", lead_policy, min_gain, year_stratum)


def window_split(series: Series, lead_window: tuple[float, float] = (5.0, 7.0),
                 min_gain: float = 1.0, lead_policy: str = "representative",
                 year_stratum: str = "") -> Split | None:
    """Lead drawn from a fixed absolute potency window; targets beat it by min_gain.

    Anchoring the lead to an absolute potency is what makes series from different
    eras pose the same question. Under the quantile split, pre-2015 campaigns
    handed the model a micromolar starting point with ~3.3 log of headroom while
    post-2024 campaigns handed it an already-potent one with ~2.4 -- so an era
    effect could not be told apart from a difficulty effect.
    """
    lo, hi = lead_window
    early = [c for c in series.compounds if lo <= c.pchembl <= hi]
    return _assemble(series, early, list(series.compounds), "window",
                     lead_policy, min_gain, year_stratum)


def temporal_split(series: Series, min_early_fraction: float = 0.25,
                   min_gain: float = 1.0, lead_policy: str = "representative",
                   year_stratum: str = "") -> Split | None:
    """Lead from the earliest year(s), targets from later years.

    Returns None when the series sits in a single publication year, which is
    always the case for assay- and document-grouped series. That is a property
    of ChEMBL, not a bug: one assay carries one document, and one document one
    year. Only cluster-grouped series can be split this way.
    """
    years = sorted({c.document_year for c in series.compounds if c.document_year})
    if len(years) < 2:
        return None

    # Grow the early window year by year until it holds a usable fraction of the
    # series, so that the lead is drawn from a real cohort and not a single paper
    # that happens to contribute one compound.
    early: list[Compound] = []
    cutoff = years[0]
    for y in years[:-1]:
        cutoff = y
        early = [c for c in series.compounds
                 if c.document_year and c.document_year <= cutoff]
        if len(early) >= max(3, int(min_early_fraction * series.n)):
            break
    later = [c for c in series.compounds
             if c.document_year and c.document_year > cutoff]
    return _assemble(series, early, later, "temporal", lead_policy, min_gain, year_stratum)
