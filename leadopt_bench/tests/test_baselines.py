"""Tests for the control arms.

The decisive one: the oracle must score as perfect recovery. If it does not,
the scoring is broken and every other number in the project is suspect.
"""

import json
import sys
from pathlib import Path

import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leadopt_bench import baselines as B
from leadopt_bench import llm, metrics as M
from leadopt_bench.series import build_series
from leadopt_bench.split import window_split
from leadopt_bench.tests.test_series import _congeneric_rows

LEAD = "c1ccc2ncnc(Nc3ccc(F)cc3)c2c1"


@pytest.fixture(autouse=True)
def isolated_responses(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "RESPONSE_DIR", tmp_path / "responses")
    yield


# --- molecule surgery ---------------------------------------------------------

def test_attachment_cores_finds_cut_points():
    cores = B.attachment_cores(LEAD)
    assert cores
    for core, sub in cores:
        assert "[*:1]" in core and "[*:1]" in sub


def test_attachment_cores_is_empty_for_garbage():
    assert B.attachment_cores("not a molecule") == []


def test_graft_rebuilds_a_valid_molecule():
    core, _ = B.attachment_cores(LEAD)[0]
    out = B.graft(core, B.SUBSTITUENTS["chloro"])
    assert out is not None
    assert Chem.MolFromSmiles(out) is not None


def test_graft_returns_none_on_bad_input():
    assert B.graft("nonsense[*:1]", B.SUBSTITUENTS["chloro"]) is None


def test_every_catalogued_substituent_can_be_grafted():
    core, _ = B.attachment_cores(LEAD)[0]
    for name, smi in B.SUBSTITUENTS.items():
        out = B.graft(core, smi)
        assert out is not None, f"{name} failed to graft"
        assert Chem.MolFromSmiles(out) is not None, f"{name} grafted to invalid mol"


# --- fixtures for the provider tests ------------------------------------------

def _split_on_disk(tmp_path, monkeypatch):
    """A real series + split written where the contextual baselines look."""
    monkeypatch.setattr(M.C, "SERIES_DIR", tmp_path / "series")
    monkeypatch.setattr(B.C, "SERIES_DIR", tmp_path / "series")
    monkeypatch.setattr(B.C, "DATA_DIR", tmp_path)
    (tmp_path / "series").mkdir(parents=True, exist_ok=True)
    (tmp_path / "splits").mkdir(parents=True, exist_ok=True)

    series = build_series(_congeneric_rows(n=28), "CHEMBL301", "CDK2", grouping="assay")[0]
    (tmp_path / "series" / f"{series.series_id}.json").write_text(
        json.dumps(series.to_dict()))

    sp = window_split(series, lead_window=(5.0, 7.0), year_stratum="old")
    assert sp is not None and sp.is_usable()
    (tmp_path / "splits" / f"{sp.series_id}__window.json").write_text(
        json.dumps(sp.to_dict()))
    return json.loads(json.dumps(sp.to_dict())), sp


# --- generic and random -------------------------------------------------------

def test_generic_baseline_produces_valid_molecules():
    payload = {"series_id": "s", "strategy": "window", "target_pref_name": "CDK2",
               "target_chembl_id": "CHEMBL301", "standard_type": "IC50",
               "lead": {"smiles": LEAD, "molecule_chembl_id": "L"},
               "lead_pchembl": 6.0, "targets": [], "usable": True}
    run = llm.query_split(payload, B.GenericProvider(n=5))
    assert not run.error
    assert run.suggestions
    for s in run.suggestions:
        assert Chem.MolFromSmiles(s.smiles) is not None


def test_generic_baseline_never_echoes_the_lead():
    payload = {"series_id": "s", "strategy": "window", "target_pref_name": "CDK2",
               "target_chembl_id": "CHEMBL301", "standard_type": "IC50",
               "lead": {"smiles": LEAD, "molecule_chembl_id": "L"},
               "lead_pchembl": 6.0, "targets": [], "usable": True}
    run = llm.query_split(payload, B.GenericProvider(n=5))
    canon_lead = Chem.MolToSmiles(Chem.MolFromSmiles(LEAD))
    for s in run.suggestions:
        assert Chem.MolToSmiles(Chem.MolFromSmiles(s.smiles)) != canon_lead


def test_random_baseline_is_reproducible_for_a_given_seed():
    payload = {"series_id": "s", "strategy": "window", "target_pref_name": "CDK2",
               "target_chembl_id": "CHEMBL301", "standard_type": "IC50",
               "lead": {"smiles": LEAD, "molecule_chembl_id": "L"},
               "lead_pchembl": 6.0, "targets": [], "usable": True}
    a = llm.query_split(payload, B.RandomAnalogProvider(n=5, seed=7))
    b = llm.query_split(payload, B.RandomAnalogProvider(n=5, seed=7), refresh=True)
    assert [s.smiles for s in a.suggestions] == [s.smiles for s in b.suggestions]


def test_random_baseline_differs_between_seeds():
    payload = {"series_id": "s", "strategy": "window", "target_pref_name": "CDK2",
               "target_chembl_id": "CHEMBL301", "standard_type": "IC50",
               "lead": {"smiles": LEAD, "molecule_chembl_id": "L"},
               "lead_pchembl": 6.0, "targets": [], "usable": True}
    a = llm.query_split(payload, B.RandomAnalogProvider(n=5, seed=1))
    b = llm.query_split(payload, B.RandomAnalogProvider(n=5, seed=999), refresh=True)
    assert [s.smiles for s in a.suggestions] != [s.smiles for s in b.suggestions]


def test_random_baseline_does_not_repeat_itself():
    payload = {"series_id": "s", "strategy": "window", "target_pref_name": "CDK2",
               "target_chembl_id": "CHEMBL301", "standard_type": "IC50",
               "lead": {"smiles": LEAD, "molecule_chembl_id": "L"},
               "lead_pchembl": 6.0, "targets": [], "usable": True}
    run = llm.query_split(payload, B.RandomAnalogProvider(n=5))
    smis = [s.smiles for s in run.suggestions]
    assert len(smis) == len(set(smis))


# --- the oracle: the metric's own sanity check --------------------------------

def test_oracle_scores_as_perfect_recovery(tmp_path, monkeypatch):
    """If the metrics cannot score the truth as recovery, they are broken."""
    split, sp = _split_on_disk(tmp_path, monkeypatch)
    run = llm.query_split(split, B.OracleProvider(n=5))
    assert run.suggestions, "oracle must find its split by lead SMILES"

    sc = M.score_run(run.to_dict(), split)
    assert sc.validity_rate == 1.0
    assert sc.recovery_rate_exact == 1.0
    assert sc.recovery_rate_close == 1.0
    assert all(s.category == "recovered_target" for s in sc.suggestions)
    assert sc.any_recovery


def test_oracle_direction_precision_is_perfect(tmp_path, monkeypatch):
    """The truth cannot point in the wrong direction, by construction."""
    split, sp = _split_on_disk(tmp_path, monkeypatch)
    run = llm.query_split(split, B.OracleProvider(n=5))
    sc = M.score_run(run.to_dict(), split)
    dps = [s.direction_precision for s in sc.suggestions
           if s.direction_precision is not None]
    if dps:
        assert min(dps) == pytest.approx(1.0)
        assert not any(s.direction_misses for s in sc.suggestions)


def test_oracle_returns_nothing_without_a_bound_split():
    prov = B.OracleProvider(n=5)
    text, _, _ = prov.complete("sys", "no lead here", "m", 100, "medium", {})
    assert json.loads(text)["suggestions"] == []


def test_two_splits_sharing_a_lead_each_get_their_own_targets(tmp_path, monkeypatch):
    """Regression: the oracle used to be indexed by lead SMILES, and the same
    compound is the chosen lead in more than one assay series of a protein, so
    one split's answers silently replaced the other's."""
    split, sp = _split_on_disk(tmp_path, monkeypatch)
    twin = json.loads(json.dumps(split))
    twin["series_id"] = "TWIN_IC50"
    twin["targets"] = list(reversed(twin["targets"]))

    prov = B.OracleProvider(n=3)
    a = llm.query_split(split, prov)
    b = llm.query_split(twin, prov)

    assert [s.smiles for s in a.suggestions] == [t["smiles"] for t in split["targets"][:3]]
    assert [s.smiles for s in b.suggestions] == [t["smiles"] for t in twin["targets"][:3]]


def test_real_providers_have_no_context_hook():
    """The bind_split hook must exist only on the deliberate cheats."""
    assert not hasattr(llm.MockProvider(), "bind_split")
    assert not hasattr(B.GenericProvider(), "bind_split")
    assert not hasattr(B.RandomAnalogProvider(), "bind_split")
    assert hasattr(B.OracleProvider(), "bind_split")
    assert hasattr(B.SeriesNeighbourProvider(), "bind_split")


# --- the neighbour control ----------------------------------------------------

def test_neighbour_baseline_never_returns_a_real_target(tmp_path, monkeypatch):
    """It must encode target familiarity only -- not the answer."""
    split, sp = _split_on_disk(tmp_path, monkeypatch)
    run = llm.query_split(split, B.SeriesNeighbourProvider(n=5))
    target_smiles = {t["smiles"] for t in split["targets"]}
    for s in run.suggestions:
        assert s.smiles not in target_smiles
        assert s.smiles != split["lead"]["smiles"]


def test_neighbour_baseline_is_empty_when_the_target_has_one_series(tmp_path, monkeypatch):
    """With no other series to borrow from there is nothing to propose, which
    is the honest answer rather than a silent fallback to the same series."""
    split, sp = _split_on_disk(tmp_path, monkeypatch)
    run = llm.query_split(split, B.SeriesNeighbourProvider(n=5))
    assert run.suggestions == []


# --- registry -----------------------------------------------------------------

def test_registry_builds_every_baseline():
    for name in B.BASELINES:
        assert isinstance(B.get_baseline(name), llm.Provider)


def test_registry_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown baseline"):
        B.get_baseline("wishful-thinking")


def test_baselines_report_distinct_provider_names():
    names = {B.get_baseline(n).name for n in B.BASELINES}
    assert len(names) == len(B.BASELINES)
