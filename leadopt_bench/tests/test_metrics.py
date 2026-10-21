"""Tests for the scoring axes.

The decisive test here is that the mock provider -- which bolts a single atom
onto the lead and writes a filler rationale -- does NOT come out looking like
good medicinal chemistry. A scoring bug that rewards it would otherwise be
invisible until it had already produced a "finding".
"""

import json
import sys
from pathlib import Path

import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leadopt_bench import config as C
from leadopt_bench import metrics as M
from leadopt_bench.split import window_split
from leadopt_bench.tests.test_series import _congeneric_rows
from leadopt_bench.series import build_series

LEAD = "c1ccc2ncnc(Nc3ccc(F)cc3)c2c1"


# --- axis 1: validity ---------------------------------------------------------

def test_valid_molecule_passes():
    mol, reason = M.validate(LEAD)
    assert mol is not None and reason == ""


@pytest.mark.parametrize("smiles,expect", [
    ("", "empty"),
    ("   ", "empty"),
    ("not a molecule", "unparseable"),
    ("c1ccccc1(", "unparseable"),
    ("C(C)(C)(C)(C)C", "unparseable"),       # pentavalent carbon fails sanitization
    ("CCO", "too_small"),
    ("CC(=O)Oc1ccccc1C(=O)O.CCN", "multiple_fragments"),
])
def test_invalid_structures_are_rejected_with_a_reason(smiles, expect):
    mol, reason = M.validate(smiles)
    assert mol is None
    assert reason.startswith(expect)


def test_absurdly_large_molecule_is_rejected():
    mol, reason = M.validate("C" * 300)
    assert mol is None and "mw_out_of_range" in reason


# --- axis 2: similarity floor -------------------------------------------------

def test_similarity_floor_is_the_least_similar_real_target():
    split = {"lead_target_tanimoto": [0.9, 0.5, 0.62, 0.77]}
    assert M.similarity_floor(split, quantile=0.0) == pytest.approx(0.5)


def test_similarity_floor_falls_back_when_too_few_targets():
    assert M.similarity_floor({"lead_target_tanimoto": [0.9]}) == M.DEFAULT_SIMILARITY_FLOOR
    assert M.similarity_floor({}) == M.DEFAULT_SIMILARITY_FLOOR


def test_similarity_floor_quantile_is_monotonic():
    split = {"lead_target_tanimoto": [0.3, 0.4, 0.5, 0.6, 0.7]}
    assert (M.similarity_floor(split, 0.0)
            <= M.similarity_floor(split, 0.5)
            <= M.similarity_floor(split, 1.0))


def test_one_outlier_target_cannot_collapse_the_floor():
    """Regression: a BRD4 series whose least-similar target sat at 0.076 made
    the scaffold-violation axis vacuous -- almost anything passed."""
    split = {"lead_target_tanimoto": [0.076, 0.59, 0.61, 0.65, 0.70]}
    floor = M.similarity_floor(split, quantile=0.0)
    assert floor == pytest.approx(M.SIMILARITY_FLOOR_MIN)
    assert floor > 0.2


def test_the_clamp_never_raises_a_healthy_floor():
    split = {"lead_target_tanimoto": [0.55, 0.60, 0.70, 0.80]}
    assert M.similarity_floor(split, quantile=0.0) == pytest.approx(0.55)


def test_clamp_stays_below_the_series_congenericity_criterion():
    """The floor must remain more generous than the bar a series had to clear."""
    assert M.SIMILARITY_FLOOR_MIN < C.MIN_MEAN_TANIMOTO


def test_tanimoto_is_one_for_identical_molecules():
    m = Chem.MolFromSmiles(LEAD)
    assert M.tanimoto(m, m) == pytest.approx(1.0)


# --- axis 4: functional group deltas -----------------------------------------

def test_fg_delta_detects_an_added_halogen():
    lead = Chem.MolFromSmiles("c1ccc2ncnc(Nc3ccccc3)c2c1")
    mod = Chem.MolFromSmiles("c1ccc2ncnc(Nc3ccc(Cl)cc3)c2c1")
    assert M.fg_delta(lead, mod) == {"chlorine": 1}


def test_fg_delta_detects_a_removed_group():
    lead = Chem.MolFromSmiles("c1ccc2ncnc(Nc3ccc(Cl)cc3)c2c1")
    mod = Chem.MolFromSmiles("c1ccc2ncnc(Nc3ccccc3)c2c1")
    assert M.fg_delta(lead, mod) == {"chlorine": -1}


def test_fg_delta_is_empty_for_the_same_molecule():
    m = Chem.MolFromSmiles(LEAD)
    assert M.fg_delta(m, m) == {}


def test_fg_catalogue_detects_hydroxamic_acid():
    """HDAC1 series live or die on the zinc-binding group, so this one matters."""
    mol = Chem.MolFromSmiles("ONC(=O)CCCCCCc1ccccc1")
    assert M.fg_counts(mol).get("hydroxamic_acid", 0) >= 1


def test_fg_catalogue_detects_amidine_for_protease_series():
    mol = Chem.MolFromSmiles("NC(=N)c1ccccc1")
    c = M.fg_counts(mol)
    assert c.get("amidine", 0) >= 1


def test_series_improving_deltas_collects_every_real_change():
    lead = Chem.MolFromSmiles("c1ccc2ncnc(Nc3ccccc3)c2c1")
    targets = [Chem.MolFromSmiles("c1ccc2ncnc(Nc3ccc(Cl)cc3)c2c1"),
               Chem.MolFromSmiles("c1ccc2ncnc(Nc3ccc(C#N)cc3)c2c1")]
    got = M.series_improving_deltas(lead, targets)
    assert ("chlorine", "+") in got
    assert ("nitrile", "+") in got
    assert ("fluorine", "+") not in got


# --- MMP ----------------------------------------------------------------------

def test_mmp_finds_the_substituent_swap():
    lead = Chem.MolFromSmiles("c1ccc2ncnc(Nc3ccc(F)cc3)c2c1")
    mod = Chem.MolFromSmiles("c1ccc2ncnc(Nc3ccc(Cl)cc3)c2c1")
    tr = M.mmp_transforms(lead, mod)
    assert tr, "a single-atom swap must yield a transformation"
    assert any("F" in a and "Cl" in b for a, b in tr)


def test_mmp_is_empty_for_unrelated_molecules():
    lead = Chem.MolFromSmiles(LEAD)
    other = Chem.MolFromSmiles("CC12CCC3CCCCC3C1CCC2O")
    assert M.mmp_transforms(lead, other) == set()


# --- scoring: the categories --------------------------------------------------

def _real_split():
    s = build_series(_congeneric_rows(n=28), "CHEMBL301", "CDK2", grouping="assay")[0]
    sp = window_split(s, lead_window=(5.0, 7.0), year_stratum="old")
    assert sp is not None and sp.is_usable()
    return json.loads(json.dumps(sp.to_dict())), sp, s


def _run_with(smiles_list, **over):
    sugg = [{"smiles": s, "modification": "m", "rationale": "r",
             "expected_fold_improvement": 2, "confidence": "medium", "index": i}
            for i, s in enumerate(smiles_list)]
    return {"suggestions": sugg, "model": "test", "prompt_variant": "implicit",
            "effort": "medium", "run_index": 0, "n_requested": len(sugg),
            "year_stratum": "old", **over}


def test_invalid_suggestion_is_categorised_invalid():
    split, sp, _ = _real_split()
    sc = M.score_run(_run_with(["this is not a molecule"]), split)
    assert sc.suggestions[0].category == "invalid"
    assert sc.validity_rate == 0.0


def test_echoing_the_lead_back_is_not_an_answer():
    split, sp, _ = _real_split()
    sc = M.score_run(_run_with([sp.lead.smiles]), split)
    assert sc.suggestions[0].category == "unchanged"


def test_repeating_a_suggestion_is_flagged_duplicate():
    split, sp, _ = _real_split()
    novel = sp.lead.smiles + "C"
    sc = M.score_run(_run_with([novel, novel]), split)
    cats = [s.category for s in sc.suggestions]
    assert cats[1] == "duplicate"
    assert cats[0] != "duplicate"


def test_proposing_a_real_improved_compound_is_recovered_target():
    split, sp, _ = _real_split()
    sc = M.score_run(_run_with([sp.targets[0].smiles]), split)
    s = sc.suggestions[0]
    assert s.category == "recovered_target"
    assert s.exact_match_target == sp.targets[0].molecule_chembl_id
    assert sc.recovery_rate_exact == 1.0
    assert sc.any_recovery


def test_recovery_requires_moving_toward_the_target_not_just_sitting_near_the_lead():
    """Regression: bolting one atom onto the lead used to score as recovery,
    because a one-atom change sits ~0.9 from every close analog chemists made.
    The suggestion never approached the answer -- it just never left the lead."""
    split, sp, _ = _real_split()
    stuck = sp.lead.smiles + "C"          # trivially near the lead
    mol, _ = M.validate(stuck)
    assert mol is not None, "fixture must build a parseable molecule"

    sc = M.score_run(_run_with([stuck]), split)
    s = sc.suggestions[0]
    if s.category == "recovered_target":
        assert s.approach_gain is not None and s.approach_gain > 0, \
            "recovery must require positive approach gain"


def test_approach_gain_is_positive_for_a_real_target():
    split, sp, _ = _real_split()
    sc = M.score_run(_run_with([sp.targets[0].smiles]), split)
    s = sc.suggestions[0]
    assert s.category == "recovered_target"
    assert s.approach_gain is not None and s.approach_gain > 0


def test_unrelated_molecule_is_a_scaffold_violation():
    split, sp, _ = _real_split()
    sc = M.score_run(_run_with(["CC12CCC3CCCCC3C1CCC2O"]), split)
    s = sc.suggestions[0]
    assert s.valid
    assert s.category == "scaffold_violation"
    assert s.tanimoto_to_lead < s.similarity_floor
    assert sc.scaffold_compliance_rate == 0.0


def test_close_analog_absent_from_the_series_is_not_penalised():
    """The spec's trap: the LLM may propose something better than anything in
    the series. A close analog the chemists never made must not be scored as a
    failure of any kind."""
    split, sp, _ = _real_split()
    # Derive a one-atom analog OF THIS LEAD, swapping a methyl for iodine --
    # iodine is the one halogen the fixture series never uses.
    lead = Chem.MolFromSmiles(sp.lead.smiles)
    prods = Chem.ReplaceSubstructs(lead, Chem.MolFromSmarts("[CX4H3]"),
                                   Chem.MolFromSmiles("I"), replaceAll=False)
    novel = Chem.MolToSmiles(prods[0])
    series_smiles = {c["smiles"] for c in split["targets"]}
    assert novel not in series_smiles

    sc = M.score_run(_run_with([novel]), split)
    s = sc.suggestions[0]
    assert s.valid
    assert s.category not in ("invalid", "unchanged", "scaffold_violation")
    assert s.tanimoto_to_lead >= s.similarity_floor


def test_direction_precision_rewards_a_real_improving_change():
    split, sp, series = _real_split()
    # use the exact FG change that a real target carries
    target = sp.targets[0]
    sc = M.score_run(_run_with([target.smiles]), split)
    s = sc.suggestions[0]
    if s.fg_delta:
        assert s.direction_precision == pytest.approx(1.0)
        assert not s.direction_misses


def test_direction_precision_is_none_when_nothing_changed_in_the_catalogue():
    split, sp, _ = _real_split()
    sc = M.score_run(_run_with([sp.lead.smiles]), split)
    assert sc.suggestions[0].direction_precision is None


# --- description vs structure -------------------------------------------------

def test_named_structures_prefers_the_longest_name():
    found = M.named_structures("the hydroxamic acid chelates zinc")
    assert "hydroxamic acid" in found
    # "acid" alone must not also be reported from the same span
    assert "carboxylic acid" not in found


def test_named_structures_finds_ring_systems():
    found = M.named_structures("replace the quinoline with a pyridine ring")
    assert "quinoline" in found and "pyridine" in found


def test_named_structures_is_empty_for_vague_text():
    assert M.named_structures("improve binding affinity somehow") == []


def test_claim_about_a_group_present_in_the_lead_is_grounded():
    lead = Chem.MolFromSmiles("ONC(=O)CCCCCCc1ccccc1")   # has a hydroxamic acid
    sug = Chem.MolFromSmiles("ONC(=O)CCCCCCc1ccc(F)cc1")
    assert M.ungrounded_claims(lead, sug, "extend the hydroxamic acid linker") == []


def test_claim_about_a_group_added_in_the_proposal_is_grounded():
    lead = Chem.MolFromSmiles("c1ccc2ncnc(Nc3ccccc3)c2c1")
    sug = Chem.MolFromSmiles("c1ccc2ncnc(Nc3ccc(N4CCOCC4)cc3)c2c1")   # gains morpholine
    assert M.ungrounded_claims(lead, sug, "append a morpholine") == []


def test_claim_about_a_ring_in_neither_molecule_is_ungrounded():
    """Regression from a real gemma3:12b response: asked to improve an
    isoxazole-urea HDAC1 inhibitor, it described edits to a 'piperidine' and a
    'quinoline', neither of which is in the compound, while actually editing
    the isoxazole. The rationale was not wrong chemistry -- it was about a
    different molecule."""
    lead = Chem.MolFromSmiles(
        "CC(C)(C)c1cc(NC(=O)Nc2ccc(Oc3ccnc(-c4cnn(CCCCCCCCCCC(=O)NO)c4)c3)cc2)no1")
    sug = Chem.MolFromSmiles(
        "CC(C)(C)c1cc(NC(=O)Nc2ccc(Oc3ccnc(-c4cnn(CCCCCCCCCCC(=O)NO)c4)c3)cc2)c(O)n1")
    bad = M.ungrounded_claims(
        lead, sug, "Introduction of a hydroxyl group at the 6-position of the "
                   "quinoline ring. The quinoline ring system is lipophilic.")
    assert "quinoline" in bad


def test_ungrounded_rate_is_reported_per_run():
    split, sp, _ = _real_split()
    run = _run_with([sp.targets[0].smiles])
    run["suggestions"][0]["modification"] = "trim the piperazine"
    run["suggestions"][0]["rationale"] = "the piperazine clashes with the pocket"
    sc = M.score_run(run, split)
    s = sc.suggestions[0]
    if "piperazine" not in M.named_structures(sp.lead.smiles):
        assert s.describes_other_molecule
        assert sc.ungrounded_rate == 1.0


def test_a_grounded_description_does_not_raise_the_rate():
    split, sp, _ = _real_split()
    run = _run_with([sp.targets[0].smiles])
    run["suggestions"][0]["modification"] = "add a substituent"
    run["suggestions"][0]["rationale"] = "fills the pocket better"
    sc = M.score_run(run, split)
    assert sc.ungrounded_rate == 0.0


# --- trade-off blindness (spec Task 5: "optimized one property") --------------

def test_an_all_additive_response_is_visible_as_such():
    split, sp, _ = _real_split()
    lead = Chem.MolFromSmiles(sp.lead.smiles)
    grown = [Chem.MolToSmiles(Chem.MolFromSmiles(sp.lead.smiles + "C" * n))
             for n in (1, 2, 3)]
    sc = M.score_run(_run_with(grown), split)
    valid = [s for s in sc.suggestions if s.valid]
    if valid:
        assert sc.fraction_additive == 1.0
        assert sc.mean_mw_delta > 0
        assert not sc.any_suggestion_trims


def test_a_response_that_trims_the_molecule_is_recognised():
    split, sp, _ = _real_split()
    lead = Chem.MolFromSmiles(sp.lead.smiles)
    trimmed = Chem.ReplaceSubstructs(lead, Chem.MolFromSmarts("[CX4H3]"),
                                     Chem.MolFromSmiles("[H]"), replaceAll=False)
    smaller = Chem.MolToSmiles(Chem.RemoveHs(trimmed[0]))
    sc = M.score_run(_run_with([smaller]), split)
    if sc.suggestions[0].valid and sc.suggestions[0].heavy_atom_delta < 0:
        assert sc.any_suggestion_trims
        assert sc.fraction_additive == 0.0


def test_mw_delta_is_zero_for_the_lead_itself():
    split, sp, _ = _real_split()
    sc = M.score_run(_run_with([sp.lead.smiles]), split)
    assert sc.suggestions[0].mw_delta == pytest.approx(0.0, abs=1e-6)


# --- the mock floor -----------------------------------------------------------

def test_mock_provider_output_is_flagged_generic_not_skilful():
    """MockProvider appends one atom and writes filler. If the metrics call that
    good lead optimization, the metrics are wrong."""
    from leadopt_bench import llm
    split, sp, _ = _real_split()
    run = llm.query_split(split, llm.MockProvider(n=5))
    sc = M.score_run(run.to_dict(), split)

    scored = [s for s in sc.suggestions if s.valid]
    assert scored, "the mock should at least produce parseable molecules"
    assert sc.generic_rate > 0.5, "single-atom additions must be flagged generic"
    assert sc.recovery_rate_exact == 0.0, "the mock cannot recover real compounds"


def test_mock_run_scores_without_crashing_on_every_axis():
    from leadopt_bench import llm
    split, sp, _ = _real_split()
    run = llm.query_split(split, llm.MockProvider(n=5))
    sc = M.score_run(run.to_dict(), split)
    d = sc.summary()
    for key in ("validity_rate", "scaffold_compliance_rate", "recovery_exact",
                "recovery_close", "generic_rate"):
        assert 0.0 <= d[key] <= 1.0
    assert sum(sc.counts.values()) == len(sc.suggestions)


# --- evaluating an arbitrary molecule by hand ---------------------------------

@pytest.fixture
def split_on_disk(tmp_path, monkeypatch):
    """A real split written where evaluate_smiles looks for it."""
    monkeypatch.setattr(M.C, "DATA_DIR", tmp_path)
    monkeypatch.setattr(M.C, "SERIES_DIR", tmp_path / "series")
    (tmp_path / "splits").mkdir(parents=True, exist_ok=True)
    (tmp_path / "series").mkdir(parents=True, exist_ok=True)

    series = build_series(_congeneric_rows(n=28), "CHEMBL301", "CDK2", grouping="assay")[0]
    (tmp_path / "series" / f"{series.series_id}.json").write_text(
        json.dumps(series.to_dict()))
    sp = window_split(series, lead_window=(5.0, 7.0), year_stratum="old")
    (tmp_path / "splits" / f"{sp.series_id}__window.json").write_text(
        json.dumps(sp.to_dict()))
    return sp


def test_evaluate_smiles_recognises_a_real_improved_compound(split_on_disk):
    sp = split_on_disk
    best = max(sp.targets, key=lambda t: t.pchembl)
    sc = M.evaluate_smiles(best.smiles, sp.series_id)
    assert sc.category == "recovered_target"
    assert sc.matched_series_molecule == best.molecule_chembl_id
    assert sc.matched_series_pchembl == pytest.approx(best.pchembl)
    assert "KİMYACILAR BUNU YAPTI" in M.explain(sc)


def test_evaluate_smiles_rejects_an_unrelated_molecule(split_on_disk):
    sc = M.evaluate_smiles("CC(=O)Oc1ccccc1C(=O)O", split_on_disk.series_id)
    assert sc.category == "scaffold_violation"
    assert "iskeleti terk" in M.explain(sc)


def test_evaluate_smiles_reports_invalid_input(split_on_disk):
    sc = M.evaluate_smiles("not a molecule", split_on_disk.series_id)
    assert not sc.valid
    assert "GEÇERSİZ" in M.explain(sc)


def test_evaluate_smiles_labels_an_untried_analog_as_the_experts_call(split_on_disk):
    """A molecule nobody made must not be scored good or bad."""
    sp = split_on_disk
    lead = Chem.MolFromSmiles(sp.lead.smiles)
    prods = Chem.ReplaceSubstructs(lead, Chem.MolFromSmarts("[CX4H3]"),
                                   Chem.MolFromSmiles("I"), replaceAll=False)
    sc = M.evaluate_smiles(Chem.MolToSmiles(prods[0]), sp.series_id)
    assert sc.category not in ("invalid", "unchanged")
    if sc.category == "novel_plausible":
        assert "karar uzmanın" in M.explain(sc)


def test_evaluate_smiles_checks_a_supplied_rationale(split_on_disk):
    sp = split_on_disk
    best = max(sp.targets, key=lambda t: t.pchembl)
    sc = M.evaluate_smiles(best.smiles, sp.series_id,
                           modification="trim the piperazine",
                           rationale="the piperazine clashes with the pocket")
    if "piperazine" not in M.named_structures(sp.lead.smiles):
        assert "piperazine" in sc.ungrounded_claims
        assert "GEREKÇE TUTARSIZ" in M.explain(sc)


def test_evaluate_smiles_errors_clearly_on_an_unknown_series(split_on_disk):
    with pytest.raises(FileNotFoundError, match="no window split"):
        M.evaluate_smiles("CCO", "CHEMBL_NOT_A_SERIES")


def test_available_series_lists_what_can_be_tested_against(split_on_disk):
    rows = M.available_series()
    assert rows
    row = rows[0]
    for key in ("series_id", "target", "endpoint", "lead_smiles",
                "lead_pchembl", "n_targets", "max_gain"):
        assert key in row


# --- roll-ups and persistence -------------------------------------------------

def test_counts_cover_every_suggestion_exactly_once():
    split, sp, _ = _real_split()
    sc = M.score_run(_run_with([
        "garbage!!",
        sp.lead.smiles,
        sp.targets[0].smiles,
        "CC12CCC3CCCCC3C1CCC2O",
    ]), split)
    assert sum(sc.counts.values()) == 4
    assert all(c in M.CATEGORIES for c in (s.category for s in sc.suggestions))


def test_error_response_scores_as_empty_not_crash():
    split, sp, _ = _real_split()
    run = _run_with([])
    run["error"] = "RuntimeError: upstream exploded"
    sc = M.score_run(run, split)
    assert sc.n_returned == 0
    assert sc.error
    assert sc.validity_rate == 0.0


def test_score_roundtrips_through_json():
    split, sp, _ = _real_split()
    sc = M.score_run(_run_with([sp.targets[0].smiles]), split)
    payload = json.loads(json.dumps(sc.to_dict()))
    assert payload["series_id"] == sc.series_id
    assert len(payload["suggestions"]) == 1


def test_stratum_is_carried_through_for_the_exposure_analysis():
    split, sp, _ = _real_split()
    sc = M.score_run(_run_with([sp.targets[0].smiles], year_stratum="new"), split)
    assert sc.year_stratum == "new"
