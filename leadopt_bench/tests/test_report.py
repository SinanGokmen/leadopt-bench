"""Tests for the colour-coded verdict rendering."""

import json
import sys
from pathlib import Path

import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leadopt_bench import metrics as M
from leadopt_bench import report as R
from leadopt_bench.series import build_series
from leadopt_bench.split import window_split
from leadopt_bench.tests.test_series import _congeneric_rows


@pytest.fixture
def split_on_disk(tmp_path, monkeypatch):
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


# --- colours ------------------------------------------------------------------

def test_every_category_has_a_colour_and_label():
    for cat in M.CATEGORIES:
        assert R.verdict_colour(cat) != "", cat
        assert R.verdict_label(cat) != "", cat


def test_unknown_category_degrades_gracefully():
    assert R.verdict_colour("something_new") == R.NEUTRAL
    assert R.verdict_label("something_new") == "something_new"


def test_settled_and_unsettled_verdicts_differ_in_colour():
    """The one distinction the colour must carry."""
    made = R.verdict_colour("recovered_target")
    unknown = R.verdict_colour("novel_plausible")
    broken = R.verdict_colour("invalid")
    assert len({made, unknown, broken}) == 3


def test_invalid_and_scaffold_violation_share_the_warning_colour():
    assert R.verdict_colour("invalid") == R.verdict_colour("scaffold_violation")


# --- html ---------------------------------------------------------------------

def test_verdict_html_of_a_real_improvement_shows_the_measured_value(split_on_disk):
    sp = split_on_disk
    best = max(sp.targets, key=lambda t: t.pchembl)
    sc = M.evaluate_smiles(best.smiles, sp.series_id)
    html = R.verdict_html(sc)
    assert "CHEMISTS MADE THIS" in html
    assert best.molecule_chembl_id in html
    assert str(best.pchembl) in html
    assert R.verdict_colour("recovered_target") in html


def test_verdict_html_of_an_untried_analog_says_the_chemist_decides(split_on_disk):
    sp = split_on_disk
    lead = Chem.MolFromSmiles(sp.lead.smiles)
    prods = Chem.ReplaceSubstructs(lead, Chem.MolFromSmarts("[CX4H3]"),
                                   Chem.MolFromSmiles("I"), replaceAll=False)
    sc = M.evaluate_smiles(Chem.MolToSmiles(prods[0]), sp.series_id)
    html = R.verdict_html(sc)
    if sc.category == "novel_plausible":
        assert "medicinal chemist has to judge" in html


def test_verdict_html_of_garbage_reports_the_reason(split_on_disk):
    sc = M.evaluate_smiles("not a molecule", split_on_disk.series_id)
    html = R.verdict_html(sc)
    assert "not a valid molecule" in html
    assert sc.invalid_reason in html


def test_verdict_html_surfaces_an_inconsistent_rationale(split_on_disk):
    sp = split_on_disk
    best = max(sp.targets, key=lambda t: t.pchembl)
    sc = M.evaluate_smiles(best.smiles, sp.series_id,
                           modification="trim the piperazine",
                           rationale="the piperazine clashes with the pocket")
    html = R.verdict_html(sc)
    if sc.ungrounded_claims:
        assert "Rationale inconsistent" in html
        assert "piperazine" in html


def test_verdict_html_includes_series_context_when_given(split_on_disk):
    sp = split_on_disk
    sc = M.evaluate_smiles(sp.targets[0].smiles, sp.series_id)
    series = next(s for s in M.available_series() if s["series_id"] == sp.series_id)
    html = R.verdict_html(sc, series)
    assert series["target"] in html
    assert str(series["lead_pchembl"]) in html


def test_verdict_html_is_a_single_closed_block(split_on_disk):
    sc = M.evaluate_smiles(split_on_disk.targets[0].smiles, split_on_disk.series_id)
    html = R.verdict_html(sc)
    assert html.startswith("<div")
    assert html.count("<div") == html.count("</div>")
    assert html.count("<table") == html.count("</table>")


def test_direction_accuracy_is_shown_with_its_null_floor(split_on_disk):
    """The metric is meaningless without it, so the floor travels with the number."""
    sp = split_on_disk
    sc = M.evaluate_smiles(sp.targets[0].smiles, sp.series_id)
    if sc.direction_precision is not None:
        html = R.verdict_html(sc)
        assert "null floor" in html


# --- legend -------------------------------------------------------------------

def test_legend_covers_every_category():
    html = R.legend_html()
    for cat in M.CATEGORIES:
        assert cat in html


def test_legend_says_the_blue_case_is_not_scored():
    html = R.legend_html()
    assert "Not scored" in html
    assert "only a chemist can judge" in html


def test_legend_warns_colour_is_not_a_quality_judgement():
    assert "not whether a molecule is good" in R.legend_html()


# --- text fallback ------------------------------------------------------------

def test_text_verdict_carries_the_same_verdict(split_on_disk):
    sc = M.evaluate_smiles(split_on_disk.targets[0].smiles, split_on_disk.series_id)
    txt = R.text_verdict(sc)
    assert "CHEMISTS MADE THIS" in txt
    assert "<div" not in txt
