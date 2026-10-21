"""Tests for the cleaning and congenericity logic that every later metric rests on."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leadopt_bench import config as C
from leadopt_bench.series import (
    Compound,
    build_series,
    clean_activities,
    fingerprints,
    mean_pairwise_tanimoto,
    murcko_scaffold,
    standardize,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
IMATINIB = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"


# --- standardization ----------------------------------------------------------

def test_standardize_strips_salt_counterion():
    """A hydrochloride salt must reduce to its neutral parent."""
    assert standardize("CCN(CC)CCNc1ccccc1.Cl") == standardize("CCN(CC)CCNc1ccccc1")


def test_standardize_neutralizes_charge():
    assert standardize("c1ccccc1C(=O)[O-]") == standardize("c1ccccc1C(=O)O")


def test_standardize_rejects_unparseable():
    assert standardize("this is not a smiles") is None
    assert standardize("c1ccccc1(") is None


def test_standardize_rejects_out_of_range_mass():
    assert standardize("CC") is None                     # too small to be a lead
    assert standardize("C" * 200) is None                # far above MW_MAX


def test_standardize_is_canonical_and_idempotent():
    once = standardize(ASPIRIN)
    assert once is not None
    assert standardize(once) == once


# --- activity cleaning --------------------------------------------------------

def _activity(**over):
    base = {
        "molecule_chembl_id": "CHEMBL1",
        "canonical_smiles": IMATINIB,
        "pchembl_value": "7.5",
        "standard_type": "IC50",
        "standard_relation": "=",
        "standard_units": "nM",
        "assay_chembl_id": "CHEMBL_A1",
        "document_chembl_id": "CHEMBL_D1",
        "document_year": 2010,
        "data_validity_comment": None,
        "potential_duplicate": 0,
    }
    base.update(over)
    return base


def test_clean_keeps_a_well_formed_measurement():
    out = clean_activities([_activity()])
    assert len(out) == 1
    assert out[0]["pchembl"] == 7.5
    assert out[0]["std_smiles"] == standardize(IMATINIB)


@pytest.mark.parametrize("override", [
    {"standard_relation": ">"},                      # censored: ">10 uM" is not a potency
    {"standard_relation": "<"},
    {"data_validity_comment": "Outside typical range"},
    {"potential_duplicate": 1},
    {"standard_units": "ug.mL-1"},                   # unit we cannot convert without MW context
    {"pchembl_value": None},
    {"canonical_smiles": "nonsense((("},
])
def test_clean_drops_untrustworthy_measurements(override):
    assert clean_activities([_activity(**override)]) == []


def test_clean_collapses_nothing_by_itself():
    """Cleaning is row-wise; de-duplication happens at series assembly."""
    rows = clean_activities([_activity(), _activity(pchembl_value="8.0")])
    assert len(rows) == 2


# --- congenericity ------------------------------------------------------------

def test_mean_pairwise_tanimoto_of_identical_molecules_is_one():
    fps = fingerprints([ASPIRIN, ASPIRIN, ASPIRIN])
    assert mean_pairwise_tanimoto(fps) == pytest.approx(1.0)


def test_mean_pairwise_tanimoto_separates_congeneric_from_diverse():
    congeneric = fingerprints([
        "c1ccc2ncnc(Nc3ccccc3)c2c1",
        "c1ccc2ncnc(Nc3ccc(F)cc3)c2c1",
        "c1ccc2ncnc(Nc3ccc(Cl)cc3)c2c1",
        "c1ccc2ncnc(Nc3ccc(C)cc3)c2c1",
    ])
    diverse = fingerprints([ASPIRIN, IMATINIB, "CCO", "C1CCCCC1C(=O)NC2CCNCC2"])
    assert mean_pairwise_tanimoto(congeneric) > C.MIN_MEAN_TANIMOTO
    assert mean_pairwise_tanimoto(diverse) < C.MIN_MEAN_TANIMOTO


def test_mean_pairwise_tanimoto_handles_degenerate_input():
    assert mean_pairwise_tanimoto(fingerprints([ASPIRIN])) == 0.0
    assert mean_pairwise_tanimoto([]) == 0.0


def test_murcko_scaffold_ignores_substituents_but_keeps_rings():
    a = murcko_scaffold("c1ccc2ncnc(Nc3ccc(F)cc3)c2c1")
    b = murcko_scaffold("c1ccc2ncnc(Nc3ccc(Cl)cc3)c2c1")
    assert a == b and a != ""
    assert murcko_scaffold(ASPIRIN) != a


# --- series assembly ----------------------------------------------------------

def _congeneric_rows(n=25, assay="CHEMBL_A1", year=2010, doc="CHEMBL_D1"):
    """n quinazoline analogs spanning 3 log units of potency."""
    tails = ["F", "Cl", "Br", "C", "CC", "OC", "N", "O", "C(F)(F)F", "C#N"]
    rows = []
    for i in range(n):
        tail = tails[i % len(tails)]
        smi = f"c1ccc2ncnc(Nc3ccc({tail})cc3{'C' * (i // len(tails))})c2c1"
        rows.append(_activity(
            molecule_chembl_id=f"CHEMBL_M{i}",
            canonical_smiles=smi,
            pchembl_value=str(5.0 + 3.0 * i / (n - 1)),
            assay_chembl_id=assay,
            document_chembl_id=doc,
            document_year=year,
        ))
    return clean_activities(rows)


def test_build_series_accepts_a_congeneric_potency_ladder():
    series = build_series(_congeneric_rows(), "CHEMBL301", "CDK2", grouping="assay")
    assert len(series) == 1
    s = series[0]
    assert s.passes()
    assert s.n >= C.MIN_COMPOUNDS
    assert s.log_range >= C.MIN_LOG_RANGE
    assert s.mean_tanimoto > C.MIN_MEAN_TANIMOTO


def test_series_compounds_are_sorted_weakest_first():
    s = build_series(_congeneric_rows(), "CHEMBL301", "CDK2", grouping="assay")[0]
    potencies = [c.pchembl for c in s.compounds]
    assert potencies == sorted(potencies)


def test_build_series_rejects_too_few_compounds():
    rows = _congeneric_rows(n=C.MIN_COMPOUNDS - 1)
    assert build_series(rows, "CHEMBL301", "CDK2", grouping="assay") == []


def test_build_series_rejects_flat_activity_range():
    rows = _congeneric_rows()
    for r in rows:
        r["pchembl"] = 7.0
    s = build_series(rows, "CHEMBL301", "CDK2", grouping="assay")[0]
    assert not s.passes()          # congeneric, but nothing was optimized
    assert s.log_range == 0.0


def test_repeated_measurements_collapse_to_one_median_compound():
    rows = _congeneric_rows()
    dup = dict(rows[0])
    dup["pchembl"] = rows[0]["pchembl"] + 2.0
    s = build_series(rows + [dup], "CHEMBL301", "CDK2", grouping="assay")[0]
    assert s.n == len(rows)        # no new compound appeared
    target = next(c for c in s.compounds if c.molecule_chembl_id == rows[0]["molecule_chembl_id"])
    assert target.n_measurements == 2
    assert target.pchembl == pytest.approx(rows[0]["pchembl"] + 1.0)


def test_single_year_series_is_not_temporally_splittable():
    s = build_series(_congeneric_rows(year=2010), "CHEMBL301", "CDK2", grouping="assay")[0]
    assert not s.temporally_splittable
    assert s.year_span == 0


def test_series_spanning_two_documents_is_temporally_splittable():
    rows = (_congeneric_rows(n=22, assay="A1", doc="D1", year=2004)
            + _congeneric_rows(n=22, assay="A2", doc="D2", year=2012))
    for i, r in enumerate(rows):                      # keep molecule ids distinct
        r["molecule_chembl_id"] = f"CHEMBL_X{i}"
    clusters = build_series(rows, "CHEMBL301", "CDK2", grouping="cluster")
    assert clusters, "scaffold bucket should survive clustering"
    assert any(s.temporally_splittable and s.year_span == 8 for s in clusters)


def test_grouping_by_assay_separates_two_experiments():
    rows = (_congeneric_rows(n=21, assay="A1", doc="D1")
            + _congeneric_rows(n=21, assay="A2", doc="D2"))
    for i, r in enumerate(rows):
        r["molecule_chembl_id"] = f"CHEMBL_Y{i}"
    assert len(build_series(rows, "CHEMBL301", "CDK2", grouping="assay")) == 2
    assert len(build_series(rows, "CHEMBL301", "CDK2", grouping="document")) == 2


def _peripheral_ring_variants(n=24, year_split=True):
    """Constant core, varied *cyclic* solubilizing group -- so every compound has
    a different Murcko scaffold even though it is one congeneric series.
    This is the real shape of the CDK2 series CHEMBL5735220 (134 compounds,
    32 distinct Murcko frameworks)."""
    # Rings that differ in size or heteroatom, so each really is a distinct
    # Murcko framework (plain substituents like -OH or -F get stripped away).
    rings = ["N1CCCCC1", "N1CCOCC1", "N1CCCC1", "N1CCN(C)CC1",
             "N1CCCCCC1", "N1CCC1", "N1CCSCC1", "N1CCNCC1"]
    rows = []
    for i in range(n):
        ring = rings[i % len(rings)]
        smi = f"Cc1nc2ccc(-c3nc(Nc4ccc5c(n4)CCN(CC{ring})C5)ncc3F)cc2n1C{'C' * (i // len(rings))}"
        rows.append(_activity(
            molecule_chembl_id=f"CHEMBL_R{i}",
            canonical_smiles=smi,
            pchembl_value=str(5.5 + 2.5 * i / (n - 1)),
            assay_chembl_id="A1" if i % 2 else "A2",
            document_chembl_id="D1" if i % 2 else "D2",
            document_year=(2015 if i % 2 else 2021) if year_split else 2015,
        ))
    return clean_activities(rows)


def test_cluster_mode_keeps_a_series_together_when_peripheral_rings_vary():
    """Regression: bucketing by Murcko scaffold shattered this into singletons."""
    rows = _peripheral_ring_variants(n=24)
    assert len({murcko_scaffold(r["std_smiles"]) for r in rows}) > 5, \
        "fixture must actually vary the scaffold"

    clusters = [s for s in build_series(rows, "CHEMBL301", "CDK2", grouping="cluster")
                if s.passes()]
    assert clusters, "congeneric series must survive clustering"
    assert max(s.n for s in clusters) >= 20


def test_cluster_mode_series_can_span_years_for_a_temporal_split():
    rows = _peripheral_ring_variants(n=24, year_split=True)
    clusters = [s for s in build_series(rows, "CHEMBL301", "CDK2", grouping="cluster")
                if s.passes()]
    assert any(s.temporally_splittable for s in clusters)
    assert any(s.year_span == 6 for s in clusters)


def test_cluster_series_records_a_dominant_scaffold():
    rows = _congeneric_rows(n=25)
    clusters = build_series(rows, "CHEMBL301", "CDK2", grouping="cluster")
    assert clusters and clusters[0].scaffold_smiles != ""


def test_butina_handles_single_and_empty_input():
    from leadopt_bench.series import butina_clusters
    assert butina_clusters(fingerprints([ASPIRIN])) == [[0]]
    assert butina_clusters([]) == [[]]


def test_butina_separates_two_distinct_chemotypes():
    from leadopt_bench.series import butina_clusters
    quinazolines = ["c1ccc2ncnc(Nc3ccc(%s)cc3)c2c1" % t for t in ["F", "Cl", "Br", "C", "OC"]]
    steroids = ["CC12CCC3CCCCC3C1CCC2%s" % t for t in ["O", "C", "N", "F", "Cl"]]
    clusters = butina_clusters(fingerprints(quinazolines + steroids))
    assert len(clusters) >= 2
    # no cluster may mix a quinazoline (idx<5) with a steroid (idx>=5)
    for c in clusters:
        assert not ({i for i in c if i < 5} and {i for i in c if i >= 5})


def test_unknown_grouping_is_rejected():
    with pytest.raises(ValueError):
        build_series([], "CHEMBL301", "CDK2", grouping="nonsense")


def test_series_roundtrips_through_json():
    import json
    s = build_series(_congeneric_rows(), "CHEMBL301", "CDK2", grouping="assay")[0]
    payload = json.loads(json.dumps(s.to_dict()))
    assert payload["n_compounds"] == s.n
    assert len(payload["compounds"]) == s.n
    assert Compound(**payload["compounds"][0]).pchembl == s.compounds[0].pchembl
