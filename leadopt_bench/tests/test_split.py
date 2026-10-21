"""Tests for early/late splitting -- including that the answer cannot leak."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leadopt_bench.series import build_series
from leadopt_bench.split import activity_split, temporal_split, window_split
from leadopt_bench.tests.test_series import _activity, _congeneric_rows


def _series(n=28, years=None):
    rows = _congeneric_rows(n=n)
    if years:
        for i, r in enumerate(rows):
            r["document_year"] = years[i % len(years)]
    return build_series(rows, "CHEMBL301", "CDK2", grouping="assay")[0]


def _multiyear_series(n=40):
    """A cluster-grouped series spanning 2008 and 2018, improving over time."""
    rows = _congeneric_rows(n=n)
    for i, r in enumerate(rows):
        early = i < n // 2
        r["document_year"] = 2008 if early else 2018
        r["document_chembl_id"] = "D_EARLY" if early else "D_LATE"
        r["assay_chembl_id"] = "A_EARLY" if early else "A_LATE"
        # later work is more potent: 5.0-6.0 early, 7.0-8.5 late
        r["pchembl"] = (5.0 + i / n) if early else (7.0 + 1.5 * (i - n // 2) / (n / 2))
    return build_series(rows, "CHEMBL301", "CDK2", grouping="cluster")[0]


# --- activity split -----------------------------------------------------------

def test_activity_split_puts_lead_below_every_target():
    sp = activity_split(_series())
    assert sp is not None
    assert all(t.pchembl > sp.lead.pchembl for t in sp.targets)


def test_activity_split_enforces_the_minimum_gain():
    sp = activity_split(_series(), min_gain=1.0)
    assert all(g >= 1.0 for g in sp.gains)


def test_activity_split_targets_are_sorted_most_potent_first():
    sp = activity_split(_series())
    assert [t.pchembl for t in sp.targets] == sorted(
        [t.pchembl for t in sp.targets], reverse=True)


def test_activity_split_returns_none_on_a_flat_series():
    s = _series()
    for c in s.compounds:
        c.pchembl = 7.0
    assert activity_split(s) is None


def test_lead_is_never_also_a_target():
    sp = activity_split(_series())
    ids = {t.molecule_chembl_id for t in sp.targets}
    assert sp.lead.molecule_chembl_id not in ids


@pytest.mark.parametrize("policy,expect", [("weakest", min), ("most_potent", max)])
def test_explicit_lead_policies_select_the_extreme_of_the_pool(policy, expect):
    s = _series()
    sp = activity_split(s, lead_policy=policy)
    import numpy as np
    vals = [c.pchembl for c in s.compounds]
    lo = float(np.quantile(vals, 0.25))
    pool = [c.pchembl for c in s.compounds if c.pchembl <= lo]
    assert sp.lead.pchembl == pytest.approx(expect(pool))


def test_representative_lead_comes_from_the_weak_pool():
    s = _series()
    import numpy as np
    lo = float(np.quantile([c.pchembl for c in s.compounds], 0.25))
    sp = activity_split(s, lead_policy="representative")
    assert sp.lead.pchembl <= lo


def test_unknown_lead_policy_is_rejected():
    with pytest.raises(ValueError):
        activity_split(_series(), lead_policy="vibes")


# --- window split (primary strategy) ------------------------------------------

def test_window_split_lead_sits_inside_the_window():
    sp = window_split(_series(), lead_window=(5.5, 6.5))
    assert sp is not None
    assert 5.5 <= sp.lead.pchembl <= 6.5


def test_window_split_targets_all_beat_the_lead_by_min_gain():
    sp = window_split(_series(), lead_window=(5.0, 7.0), min_gain=1.0)
    assert sp.targets
    assert all(t.pchembl - sp.lead.pchembl >= 1.0 for t in sp.targets)


def test_window_split_returns_none_when_no_compound_is_in_the_window():
    s = _series()
    for c in s.compounds:
        c.pchembl = 9.5                       # whole series above the window
    assert window_split(s, lead_window=(5.0, 7.0)) is None


def test_window_split_returns_none_when_nothing_beats_the_lead():
    s = _series()
    for c in s.compounds:
        c.pchembl = 6.0                       # all inside the window, no headroom
    assert window_split(s, lead_window=(5.0, 7.0), min_gain=1.0) is None


def test_window_split_anchors_lead_potency_across_different_series_spans():
    """The point of the window: two series with very different spans must still
    hand the model a comparably potent starting compound."""
    wide, narrow = _series(), _series()
    for i, c in enumerate(wide.compounds):
        c.pchembl = 4.0 + 5.5 * i / (wide.n - 1)      # 4.0 -> 9.5
    for i, c in enumerate(narrow.compounds):
        c.pchembl = 5.8 + 2.4 * i / (narrow.n - 1)    # 5.8 -> 8.2

    w_win = window_split(wide, lead_window=(5.0, 7.0))
    n_win = window_split(narrow, lead_window=(5.0, 7.0))
    assert abs(w_win.lead.pchembl - n_win.lead.pchembl) < 2.0

    # the quantile split, by contrast, drifts with the series span
    w_q = activity_split(wide)
    n_q = activity_split(narrow)
    assert abs(w_q.lead.pchembl - n_q.lead.pchembl) > abs(
        w_win.lead.pchembl - n_win.lead.pchembl)


def test_window_strategy_is_labelled_distinctly():
    assert window_split(_series()).strategy == "window"
    assert activity_split(_series()).strategy == "activity"


# --- temporal split -----------------------------------------------------------

def test_temporal_split_is_impossible_within_a_single_year():
    """Not a bug: one ChEMBL assay carries one document, and one document one year."""
    assert temporal_split(_series(years=[2012])) is None


def test_temporal_split_works_across_years():
    sp = temporal_split(_multiyear_series())
    assert sp is not None
    assert sp.lead.document_year == 2008
    assert all(t.document_year == 2018 for t in sp.targets)


def test_temporal_lead_precedes_every_target_in_time():
    sp = temporal_split(_multiyear_series())
    assert all((t.document_year or 0) > (sp.lead.document_year or 0) for t in sp.targets)


def test_temporal_split_reports_the_real_gain():
    sp = temporal_split(_multiyear_series())
    assert sp.max_gain >= 1.0
    assert sp.median_gain > 0


# --- usability and leakage ----------------------------------------------------

def test_usable_requires_several_answered_targets():
    sp = activity_split(_series())
    assert sp.is_usable()
    sp.targets = sp.targets[:1]
    assert not sp.is_usable()


def test_prompt_payload_hides_the_targets():
    """The whole experiment collapses if the answer reaches the model."""
    sp = activity_split(_series())
    payload = sp.prompt_payload()
    blob = __import__("json").dumps(payload)
    for t in sp.targets:
        assert t.smiles not in blob
        assert t.molecule_chembl_id not in blob
    assert "targets" not in payload
    assert payload["lead_smiles"] == sp.lead.smiles


def test_prompt_payload_carries_what_the_task_needs():
    payload = activity_split(_series()).prompt_payload()
    assert payload["target_protein"]
    assert payload["assay_endpoint"] in ("IC50", "Ki")
    assert payload["lead_smiles"]


def test_split_roundtrips_through_json():
    import json
    sp = activity_split(_series())
    payload = json.loads(json.dumps(sp.to_dict()))
    assert payload["n_targets"] == len(sp.targets)
    assert len(payload["targets"]) == len(sp.targets)
    assert payload["lead"]["smiles"] == sp.lead.smiles


def test_lead_target_similarity_is_recorded_for_every_target():
    sp = activity_split(_series())
    assert len(sp.lead_target_tanimoto) == len(sp.targets)
    assert all(0.0 <= x <= 1.0 for x in sp.lead_target_tanimoto)
