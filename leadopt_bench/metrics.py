"""Score LLM suggestions against what the chemists actually did.

Five axes, as the spec asks, plus the categorisation that keeps recovery from
being read as a verdict on its own.

The two thresholds that could have been arbitrary are instead calibrated from
the series itself:

* the similarity floor is the *least* similar thing the real chemists made from
  this lead. A suggestion only counts as "abandoned the scaffold" if it strays
  further than the real campaign ever did.
* "potency-increasing modification" is not a curated belief about chemistry --
  it is read off the lead-to-target functional-group deltas in that same series.

What is deliberately NOT here: any potency, ADMET or synthesizability predictor.
A suggestion absent from the series is labelled `novel_plausible`, never scored
as good or bad -- we have no ground truth for it, and inventing one with a model
is the reward-hacking failure this whole project is a reaction to.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdMMPA

from . import config as C
from .series import MW_MAX, MW_MIN, _FPGEN, murcko_scaffold

RDLogger.DisableLog("rdApp.*")

EXACT_TANIMOTO = 0.99      # canonical-SMILES identity is the real test; this backs it up
CLOSE_TANIMOTO = 0.85      # "the same idea as a real compound, one atom off"
# A close match must also have moved TOWARD that target relative to the lead.
# Without this, bolting any atom onto the lead scores as recovery, because a
# one-atom change sits ~0.9 from every close analog the chemists made -- the
# suggestion never approached the answer, it just never left the lead.
MIN_APPROACH_GAIN = 0.0
DEFAULT_SIMILARITY_FLOOR = 0.40   # only used when a series has too few targets to calibrate
# Lower bound on the per-series floor. A handful of "congeneric" series contain
# one target that is nearly unrelated to the lead (BRD4 bromodomain series span
# several chemotypes), and taking the raw minimum there drops the floor to 0.08,
# which makes the scaffold-violation axis vacuous for that series. 0.25 is
# deliberately well below the 0.40 mean-pairwise criterion a series had to meet
# to be accepted at all, so it stays generous; measured over the 32-split panel
# it excludes 1.1% of real targets and catches an unrelated probe in 32/32
# series, where the raw minimum left 4 series degenerate.
SIMILARITY_FLOOR_MIN = 0.25


# --- functional group catalogue ----------------------------------------------
# Common medicinal-chemistry substituents, at the level a chemist would name a
# change ("added a fluorine", "swapped phenyl for pyridyl"). Kept small and
# legible on purpose: this is the vocabulary the direction metric is expressed
# in, so it has to be auditable by hand.

FG_SMARTS: dict[str, str] = {
    "fluorine": "[F]",
    "chlorine": "[Cl]",
    "bromine": "[Br]",
    "iodine": "[I]",
    "trifluoromethyl": "[CX4]([F])([F])[F]",
    "nitrile": "[NX1]#[CX2]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "sulfonamide": "[SX4](=[OX1])(=[OX1])[NX3]",
    "sulfone": "[SX4](=[OX1])(=[OX1])([#6])[#6]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "ester": "[CX3](=O)[OX2H0][#6]",
    "amide": "[NX3][CX3](=[OX1])[#6]",
    "urea": "[NX3][CX3](=[OX1])[NX3]",
    "carbamate": "[NX3][CX3](=[OX1])[OX2][#6]",
    "hydroxamic_acid": "[CX3](=[OX1])[NX3][OX2H1]",
    "primary_amine": "[NX3;H2;!$(NC=O)]",
    "secondary_amine": "[NX3;H1;!$(NC=O)]",
    "tertiary_amine": "[NX3;H0;!$(NC=O);!$(N=*);!$(n)]",
    "guanidine": "[NX3][CX3](=[NX2])[NX3]",
    "amidine": "[CX3](=[NX2])[NX3]",
    "hydroxyl": "[OX2H][#6]",
    "ether": "[OD2]([#6])[#6]",
    "thioether": "[SD2]([#6])[#6]",
    "methyl": "[CX4H3]",
    "cyclopropyl": "[CH2]1[CH2][CH1]1",
    "phenyl": "c1ccccc1",
    "pyridine": "n1ccccc1",
    "pyrimidine": "n1cnccc1",
    "imidazole": "c1cnc[nH]1",
    "triazole": "c1nncn1",
    "tetrazole": "c1nnn[nH]1",
    "thiophene": "s1cccc1",
    "furan": "o1cccc1",
    "piperidine": "C1CCNCC1",
    "piperazine": "C1CNCCN1",
    "morpholine": "C1COCCN1",
    "boronic_acid": "[BX3]([OX2H])[OX2H]",
}

_FG_PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in FG_SMARTS.items()}
_FG_PATTERNS = {k: v for k, v in _FG_PATTERNS.items() if v is not None}

# Substituents so small and universal that adding one is advice you could give
# about any molecule. Used for the "generic suggestion" flag, never for scoring.
GENERIC_GROUPS = {"fluorine", "chlorine", "methyl", "hydroxyl", "bromine"}


# --- vocabulary for checking a stated modification against the structure ------
# Ring systems and groups a chemist names when describing a change. This exists
# because a local 12B model, asked to improve an isoxazole-urea HDAC1 inhibitor,
# described edits to a "piperidine" and a "quinoline" -- neither of which is in
# the molecule -- while actually editing the isoxazole. The rationale was not
# merely wrong chemistry; it was about a different compound. That is measurable:
# a group the text names should exist in the lead or in the proposal.

NAMED_STRUCTURES: dict[str, str] = {
    # rings
    "benzene": "c1ccccc1", "phenyl": "c1ccccc1",
    "pyridine": "n1ccccc1", "pyridyl": "n1ccccc1",
    "pyrimidine": "n1cnccc1",
    "pyrazine": "n1ccncc1",
    "pyrrole": "[nH]1cccc1",
    "pyrazole": "c1cn[nH]c1",
    "imidazole": "c1cnc[nH]1",
    "triazole": "c1nncn1",
    "tetrazole": "c1nnn[nH]1",
    "isoxazole": "c1ccno1",
    "oxazole": "c1ccno1",
    "isothiazole": "c1ccns1",
    "thiazole": "c1cscn1",
    "thiophene": "s1cccc1",
    "furan": "o1cccc1",
    "quinoline": "c1ccc2ncccc2c1",
    "quinazoline": "c1ccc2ncncc2c1",
    "indole": "c1ccc2[nH]ccc2c1",
    "benzimidazole": "c1ccc2[nH]cnc2c1",
    "naphthalene": "c1ccc2ccccc2c1",
    "piperidine": "C1CCNCC1", "piperidinyl": "C1CCNCC1",
    "piperazine": "C1CNCCN1", "piperazinyl": "C1CNCCN1",
    "morpholine": "C1COCCN1", "morpholino": "C1COCCN1",
    "pyrrolidine": "C1CCNC1",
    "azetidine": "C1CNC1",
    "cyclopropyl": "[CH2]1[CH2][CH1]1",
    "cyclohexyl": "C1CCCCC1",
    # groups
    "hydroxamic acid": "[CX3](=[OX1])[NX3][OX2H1]",
    "hydroxamate": "[CX3](=[OX1])[NX3][OX2H1]",
    "sulfonamide": "[SX4](=[OX1])(=[OX1])[NX3]",
    "sulfone": "[SX4](=[OX1])(=[OX1])([#6])[#6]",
    "nitrile": "[NX1]#[CX2]", "cyano": "[NX1]#[CX2]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "urea": "[NX3][CX3](=[OX1])[NX3]",
    "carbamate": "[NX3][CX3](=[OX1])[OX2][#6]",
    "amide": "[NX3][CX3](=[OX1])[#6]",
    "ester": "[CX3](=O)[OX2H0][#6]",
    "carboxylic acid": "[CX3](=O)[OX2H1]",
    "guanidine": "[NX3][CX3](=[NX2])[NX3]",
    "amidine": "[CX3](=[NX2])[NX3]",
    "trifluoromethyl": "[CX4]([F])([F])[F]",
    "boronic acid": "[BX3]([OX2H])[OX2H]",
    "methoxy": "[CH3][OX2][#6]",
    "hydroxyl": "[OX2H][#6]",
    "methyl": "[CX4H3]",
}

_NAMED_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in NAMED_STRUCTURES.items()}
_NAMED_PATTERNS = {k: v for k, v in _NAMED_PATTERNS.items() if v is not None}

# Longest names first, so "hydroxamic acid" is matched before "acid"-like substrings
_NAMED_ORDER = sorted(_NAMED_PATTERNS, key=len, reverse=True)


# --- records ------------------------------------------------------------------

CATEGORIES = (
    "invalid",                # does not parse or sanitize
    "unchanged",              # identical to the lead: not an answer
    "duplicate",              # same molecule proposed twice in one response
    "scaffold_violation",     # valid, but strays further than the real campaign did
    "recovered_target",       # a compound chemists made that WAS more potent
    "recovered_nonimprover",  # a compound chemists made that was NOT more potent
    "novel_plausible",        # valid, close enough, absent from the series
)


@dataclass
class SuggestionScore:
    index: int
    smiles: str
    canonical: str = ""
    modification: str = ""
    rationale: str = ""
    confidence: str = ""
    expected_fold_improvement: float | None = None

    category: str = ""
    invalid_reason: str = ""

    # axis 1-2
    valid: bool = False
    tanimoto_to_lead: float = 0.0
    similarity_floor: float = 0.0
    same_murcko_scaffold: bool = False

    # axis 3
    exact_match_target: str = ""
    best_target_tanimoto: float = 0.0
    best_series_tanimoto: float = 0.0
    matched_series_molecule: str = ""
    matched_series_pchembl: float | None = None
    # how much closer to the matched target the suggestion got than the lead was.
    # Positive means the suggestion actually moved toward the real answer.
    approach_gain: float | None = None

    # axis 4
    fg_delta: dict = field(default_factory=dict)
    direction_hits: list = field(default_factory=list)
    direction_misses: list = field(default_factory=list)
    direction_precision: float | None = None

    # axis 5 support
    mmp_transforms: list = field(default_factory=list)
    mmp_matches_series: bool = False
    generic_flag: bool = False
    heavy_atom_delta: int = 0
    mw_delta: float = 0.0

    # does the stated modification correspond to the molecule at all?
    named_in_text: list = field(default_factory=list)
    ungrounded_claims: list = field(default_factory=list)
    describes_other_molecule: bool = False


@dataclass
class RunScore:
    series_id: str
    strategy: str
    year_stratum: str
    target_pref_name: str
    model: str
    prompt_variant: str
    effort: str
    run_index: int
    lead_smiles: str
    lead_pchembl: float
    similarity_floor: float
    n_requested: int
    n_returned: int
    error: str = ""
    suggestions: list[SuggestionScore] = field(default_factory=list)

    # roll-ups
    @property
    def counts(self) -> dict:
        c = Counter(s.category for s in self.suggestions)
        return {k: c.get(k, 0) for k in CATEGORIES}

    @property
    def validity_rate(self) -> float:
        return _safe_div(sum(1 for s in self.suggestions if s.valid), len(self.suggestions))

    @property
    def scaffold_compliance_rate(self) -> float:
        valid = [s for s in self.suggestions if s.valid]
        return _safe_div(sum(1 for s in valid if s.category != "scaffold_violation"), len(valid))

    @property
    def recovery_rate_exact(self) -> float:
        return _safe_div(sum(1 for s in self.suggestions if s.exact_match_target),
                         len(self.suggestions))

    @property
    def recovery_rate_close(self) -> float:
        return _safe_div(sum(1 for s in self.suggestions
                             if s.category == "recovered_target"), len(self.suggestions))

    @property
    def any_recovery(self) -> bool:
        return any(s.category == "recovered_target" for s in self.suggestions)

    @property
    def generic_rate(self) -> float:
        return _safe_div(sum(1 for s in self.suggestions if s.generic_flag),
                         len(self.suggestions))

    @property
    def ungrounded_rate(self) -> float:
        """Share of valid suggestions whose description names a structure that
        is in neither the lead nor the proposal -- the model describing a
        compound it was never given."""
        valid = [s for s in self.suggestions if s.valid]
        return _safe_div(sum(1 for s in valid if s.describes_other_molecule), len(valid))

    @property
    def fraction_additive(self) -> float:
        """Share of valid suggestions that only add atoms.

        Real lead optimization fights property creep: potency is bought, and
        chemists trade it against size and lipophilicity. A response where every
        proposal inflates the molecule is showing the spec's "optimized one
        property, never saw the trade-offs" failure mode. Measured by counting
        atoms -- no ADMET model is involved.
        """
        valid = [s for s in self.suggestions if s.valid]
        return _safe_div(sum(1 for s in valid if s.heavy_atom_delta > 0), len(valid))

    @property
    def mean_mw_delta(self) -> float | None:
        valid = [s for s in self.suggestions if s.valid]
        return statistics.mean([s.mw_delta for s in valid]) if valid else None

    @property
    def any_suggestion_trims(self) -> bool:
        """Did the model ever propose making the molecule smaller?"""
        return any(s.heavy_atom_delta < 0 for s in self.suggestions if s.valid)

    @property
    def mean_direction_precision(self) -> float | None:
        vals = [s.direction_precision for s in self.suggestions
                if s.direction_precision is not None]
        return statistics.mean(vals) if vals else None

    def summary(self) -> dict:
        return {
            "series_id": self.series_id,
            "strategy": self.strategy,
            "year_stratum": self.year_stratum,
            "target_pref_name": self.target_pref_name,
            "model": self.model,
            "prompt_variant": self.prompt_variant,
            "effort": self.effort,
            "run_index": self.run_index,
            "n_requested": self.n_requested,
            "n_returned": self.n_returned,
            "error": self.error,
            "similarity_floor": round(self.similarity_floor, 3),
            "validity_rate": round(self.validity_rate, 3),
            "scaffold_compliance_rate": round(self.scaffold_compliance_rate, 3),
            "recovery_exact": round(self.recovery_rate_exact, 3),
            "recovery_close": round(self.recovery_rate_close, 3),
            "any_recovery": self.any_recovery,
            "generic_rate": round(self.generic_rate, 3),
            "ungrounded_rate": round(self.ungrounded_rate, 3),
            "fraction_additive": round(self.fraction_additive, 3),
            "mean_mw_delta": (round(self.mean_mw_delta, 1)
                              if self.mean_mw_delta is not None else None),
            "any_suggestion_trims": self.any_suggestion_trims,
            "mean_direction_precision": (round(self.mean_direction_precision, 3)
                                         if self.mean_direction_precision is not None else None),
            **self.counts,
        }

    def to_dict(self) -> dict:
        return {**self.summary(), "lead_smiles": self.lead_smiles,
                "lead_pchembl": self.lead_pchembl,
                "suggestions": [asdict(s) for s in self.suggestions]}


def _safe_div(a: int, b: int) -> float:
    return a / b if b else 0.0


# --- axis 1: validity ---------------------------------------------------------

def validate(smiles: str):
    """Return (mol, reason). mol is None when the structure is unusable."""
    if not smiles or not smiles.strip():
        return None, "empty"
    mol = Chem.MolFromSmiles(smiles)          # MolFromSmiles sanitizes
    if mol is None:
        return None, "unparseable"
    if mol.GetNumHeavyAtoms() < 6:
        return None, "too_small"
    mw = Descriptors.MolWt(mol)
    if not (MW_MIN <= mw <= MW_MAX * 1.5):    # a little slack: a proposal may add bulk
        return None, f"mw_out_of_range({mw:.0f})"
    if len(Chem.GetMolFrags(mol)) > 1:
        return None, "multiple_fragments"
    return mol, ""


# --- axis 2: similarity -------------------------------------------------------

def _fp(mol):
    return _FPGEN.GetFingerprint(mol)


def tanimoto(mol_a, mol_b) -> float:
    return float(DataStructs.TanimotoSimilarity(_fp(mol_a), _fp(mol_b)))


def similarity_floor(split: dict, quantile: float = 0.0,
                     clamp: float = SIMILARITY_FLOOR_MIN) -> float:
    """How far from the lead the real campaign actually went.

    quantile 0.0 uses the least similar real target: a suggestion is only a
    scaffold violation if it strays further than anything the chemists
    themselves made. `clamp` keeps a single outlier target from collapsing the
    floor to near zero -- see SIMILARITY_FLOOR_MIN.
    """
    sims = split.get("lead_target_tanimoto") or []
    if len(sims) < 3:
        return max(clamp, DEFAULT_SIMILARITY_FLOOR)
    ordered = sorted(sims)
    if quantile <= 0:
        value = float(ordered[0])
    else:
        idx = min(int(quantile * (len(ordered) - 1)), len(ordered) - 1)
        value = float(ordered[idx])
    return max(clamp, value)


# --- axis 4: functional-group direction --------------------------------------

def fg_counts(mol) -> Counter:
    c = Counter()
    for name, patt in _FG_PATTERNS.items():
        n = len(mol.GetSubstructMatches(patt, uniquify=True))
        if n:
            c[name] = n
    return c


def fg_delta(lead_mol, other_mol) -> dict[str, int]:
    """Functional groups gained (+) or lost (-) going from lead to other."""
    a, b = fg_counts(lead_mol), fg_counts(other_mol)
    out = {}
    for name in set(a) | set(b):
        d = b.get(name, 0) - a.get(name, 0)
        if d:
            out[name] = d
    return out


def _delta_pairs(delta: dict[str, int]) -> set[tuple[str, str]]:
    return {(name, "+" if d > 0 else "-") for name, d in delta.items()}


def named_structures(text: str) -> list[str]:
    """Structure names a description mentions, longest match first."""
    low = (text or "").lower()
    found, consumed = [], []
    for name in _NAMED_ORDER:
        idx = low.find(name)
        if idx < 0:
            continue
        # skip if this span was already claimed by a longer name
        if any(a <= idx < b for a, b in consumed):
            continue
        consumed.append((idx, idx + len(name)))
        found.append(name)
    return found


def ungrounded_claims(lead_mol, sug_mol, text: str) -> list[str]:
    """Structures the description names that appear in neither molecule.

    Conservative on purpose: a name is grounded if it is in the lead (so it
    could be modified or removed) or in the proposal (so it could have been
    added). Only a name present in neither is counted -- that is the model
    describing a compound it was not given.
    """
    out = []
    for name in named_structures(text):
        patt = _NAMED_PATTERNS[name]
        if lead_mol.HasSubstructMatch(patt):
            continue
        if sug_mol is not None and sug_mol.HasSubstructMatch(patt):
            continue
        out.append(name)
    return out


def series_improving_deltas(lead_mol, target_mols) -> set[tuple[str, str]]:
    """Every functional-group change that accompanied a real potency gain."""
    out: set[tuple[str, str]] = set()
    for t in target_mols:
        out |= _delta_pairs(fg_delta(lead_mol, t))
    return out


# --- MMP transformations ------------------------------------------------------

def mmp_transforms(lead_mol, other_mol, max_cuts: int = 1) -> set[tuple[str, str]]:
    """Single-cut transformations shared between two molecules.

    Returns {(lead_fragment, other_fragment)} for every common core. Empty when
    the two molecules share no core at this cut depth, which is itself a signal.
    """
    def frags(mol) -> dict[str, str]:
        out = {}
        for _, frag in rdMMPA.FragmentMol(mol, maxCuts=max_cuts, resultsAsMols=False):
            parts = frag.split(".")
            if len(parts) != 2:
                continue
            # the larger piece is the core, the smaller the substituent
            a, b = sorted(parts, key=lambda s: Chem.MolFromSmiles(s).GetNumHeavyAtoms()
                          if Chem.MolFromSmiles(s) else 0)
            out[b] = a
        return out

    try:
        lf, of = frags(lead_mol), frags(other_mol)
    except Exception:
        return set()
    return {(lf[core], of[core]) for core in set(lf) & set(of) if lf[core] != of[core]}


# --- scoring ------------------------------------------------------------------

def load_series(series_id: str) -> dict | None:
    path = C.SERIES_DIR / f"{series_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


def score_run(run: dict, split: dict, floor_quantile: float = 0.0) -> RunScore:
    """Score one LLM response against one split.

    `run` is the dict written by LLMRun.to_dict(); `split` by Split.write().
    """
    lead_smiles = split["lead"]["smiles"]
    lead_mol = Chem.MolFromSmiles(lead_smiles)
    lead_canon = Chem.MolToSmiles(lead_mol)
    lead_scaffold = murcko_scaffold(lead_smiles)
    floor = similarity_floor(split, floor_quantile)

    target_rows = split["targets"]
    target_mols, target_ids, target_canon = [], [], {}
    for t in target_rows:
        m = Chem.MolFromSmiles(t["smiles"])
        if m is None:
            continue
        target_mols.append(m)
        target_ids.append(t["molecule_chembl_id"])
        target_canon[Chem.MolToSmiles(m)] = t

    # the rest of the series: compounds chemists made that did NOT clear the gain bar
    series = load_series(split["series_id"]) or {}
    improved_ids = set(target_ids) | {split["lead"]["molecule_chembl_id"]}
    other_rows = [c for c in series.get("compounds", [])
                  if c["molecule_chembl_id"] not in improved_ids]
    other_mols, other_canon = [], {}
    for c in other_rows:
        m = Chem.MolFromSmiles(c["smiles"])
        if m is None:
            continue
        other_mols.append(m)
        other_canon[Chem.MolToSmiles(m)] = c

    # how similar the lead already is to each real compound: the bar a
    # suggestion must clear before it counts as having recovered that compound
    lead_fp = _fp(lead_mol)
    target_fps = [_fp(m) for m in target_mols]
    other_fps = [_fp(m) for m in other_mols]
    lead_to_target = (DataStructs.BulkTanimotoSimilarity(lead_fp, target_fps)
                      if target_fps else [])
    lead_to_other = (DataStructs.BulkTanimotoSimilarity(lead_fp, other_fps)
                     if other_fps else [])

    improving = series_improving_deltas(lead_mol, target_mols)
    series_mmp: set[tuple[str, str]] = set()
    for t in target_mols:
        series_mmp |= mmp_transforms(lead_mol, t)

    score = RunScore(
        series_id=split["series_id"], strategy=split["strategy"],
        year_stratum=run.get("year_stratum", split.get("year_stratum", "")),
        target_pref_name=split["target_pref_name"],
        model=run.get("model", ""), prompt_variant=run.get("prompt_variant", ""),
        effort=run.get("effort", ""), run_index=run.get("run_index", 0),
        lead_smiles=lead_smiles, lead_pchembl=split["lead_pchembl"],
        similarity_floor=floor,
        n_requested=run.get("n_requested", 0),
        n_returned=len(run.get("suggestions", [])),
        error=run.get("error", ""),
    )

    seen_canonical: set[str] = set()
    for item in run.get("suggestions", []):
        s = SuggestionScore(
            index=item.get("index", 0),
            smiles=item.get("smiles", ""),
            modification=item.get("modification", ""),
            rationale=item.get("rationale", ""),
            confidence=item.get("confidence", ""),
            expected_fold_improvement=item.get("expected_fold_improvement"),
            similarity_floor=floor,
        )
        mol, reason = validate(s.smiles)
        if mol is None:
            s.category, s.invalid_reason = "invalid", reason
            score.suggestions.append(s)
            continue

        s.valid = True
        s.canonical = Chem.MolToSmiles(mol)
        s.tanimoto_to_lead = tanimoto(lead_mol, mol)
        s.same_murcko_scaffold = (murcko_scaffold(s.canonical) == lead_scaffold)
        s.heavy_atom_delta = mol.GetNumHeavyAtoms() - lead_mol.GetNumHeavyAtoms()
        s.mw_delta = Descriptors.MolWt(mol) - Descriptors.MolWt(lead_mol)
        s.fg_delta = fg_delta(lead_mol, mol)

        pairs = _delta_pairs(s.fg_delta)
        if pairs:
            hits = sorted(f"{n}{d}" for n, d in pairs & improving)
            misses = sorted(f"{n}{d}" for n, d in pairs - improving)
            s.direction_hits, s.direction_misses = hits, misses
            s.direction_precision = len(pairs & improving) / len(pairs)

        s.mmp_transforms = sorted(f"{a}>>{b}" for a, b in mmp_transforms(lead_mol, mol))
        s.mmp_matches_series = bool(mmp_transforms(lead_mol, mol) & series_mmp)

        s.generic_flag = (abs(s.heavy_atom_delta) <= 1
                          and bool(pairs)
                          and all(n in GENERIC_GROUPS for n, _ in pairs))

        claim_text = f"{s.modification} {s.rationale}"
        s.named_in_text = named_structures(claim_text)
        s.ungrounded_claims = ungrounded_claims(lead_mol, mol, claim_text)
        s.describes_other_molecule = bool(s.ungrounded_claims)

        # recovery, against improved compounds and then the rest of the series
        close_i = None
        if target_mols:
            sims = DataStructs.BulkTanimotoSimilarity(_fp(mol), target_fps)
            best_i = max(range(len(sims)), key=lambda i: sims[i])
            s.best_target_tanimoto = float(sims[best_i])
            # a close match must be close AND closer than the lead already was
            candidates = [i for i in range(len(sims))
                          if sims[i] >= CLOSE_TANIMOTO
                          and sims[i] - lead_to_target[i] > MIN_APPROACH_GAIN]
            if candidates:
                close_i = max(candidates, key=lambda i: sims[i] - lead_to_target[i])
                s.approach_gain = float(sims[close_i] - lead_to_target[close_i])
            else:
                s.approach_gain = float(sims[best_i] - lead_to_target[best_i])
        other_i = None
        if other_mols:
            sims_o = DataStructs.BulkTanimotoSimilarity(_fp(mol), other_fps)
            s.best_series_tanimoto = float(max(sims_o))
            # same approach-gain guard as for targets: proposing something that
            # merely sits near the lead is not recovering a real compound
            cand_o = [i for i in range(len(sims_o))
                      if sims_o[i] >= CLOSE_TANIMOTO
                      and sims_o[i] - lead_to_other[i] > MIN_APPROACH_GAIN]
            if cand_o:
                other_i = max(cand_o, key=lambda i: sims_o[i] - lead_to_other[i])

        if s.canonical in target_canon:
            s.exact_match_target = target_canon[s.canonical]["molecule_chembl_id"]
            s.matched_series_molecule = s.exact_match_target
            s.matched_series_pchembl = target_canon[s.canonical]["pchembl"]

        # categorise, most specific first
        if s.canonical == lead_canon:
            s.category = "unchanged"
        elif s.canonical in seen_canonical:
            s.category = "duplicate"
        elif s.exact_match_target or close_i is not None:
            s.category = "recovered_target"
            if not s.matched_series_molecule and close_i is not None:
                s.matched_series_molecule = target_ids[close_i]
                s.matched_series_pchembl = target_rows[close_i]["pchembl"]
        elif s.canonical in other_canon or other_i is not None:
            s.category = "recovered_nonimprover"
            row = other_canon.get(s.canonical)
            if row:
                s.matched_series_molecule = row["molecule_chembl_id"]
                s.matched_series_pchembl = row["pchembl"]
            elif other_i is not None:
                s.matched_series_molecule = other_rows[other_i]["molecule_chembl_id"]
                s.matched_series_pchembl = other_rows[other_i]["pchembl"]
        elif s.tanimoto_to_lead < floor:
            s.category = "scaffold_violation"
        else:
            s.category = "novel_plausible"

        seen_canonical.add(s.canonical)
        score.suggestions.append(s)

    return score


def evaluate_smiles(smiles: str, series_id: str, strategy: str = "window",
                    modification: str = "", rationale: str = "") -> SuggestionScore:
    """Score one molecule -- yours, a chemist's, anyone's -- against a real series.

    A molecule cannot be judged in the abstract: "is this good?" has no referent.
    What this answers is "is this a good optimization of *that* lead?", which the
    published series does settle. Give it a SMILES and a series id and it reports
    whether the structure is valid, whether it stayed on the scaffold, whether
    chemists in fact made it (or a close analog) and what they measured, and
    whether its change-type matches the ones that raised potency in that series.

    Pass `modification`/`rationale` too and the stated reasoning is checked
    against the structure as well.
    """
    path = C.DATA_DIR / "splits" / f"{series_id}__{strategy}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no {strategy} split for {series_id}; list available ones with "
            f"leadopt_bench.metrics.available_series()")
    split = json.loads(path.read_text())
    run = {
        "suggestions": [{"smiles": smiles, "modification": modification,
                         "rationale": rationale, "index": 0,
                         "expected_fold_improvement": None, "confidence": ""}],
        "model": "manual", "prompt_variant": "manual", "effort": "-",
        "run_index": 0, "n_requested": 1,
        "year_stratum": split.get("year_stratum", ""),
    }
    return score_run(run, split).suggestions[0]


def available_series(strategy: str = "window") -> list[dict]:
    """Every series you can test a molecule against, with its lead."""
    out = []
    for p in sorted((C.DATA_DIR / "splits").glob(f"*__{strategy}.json")):
        s = json.loads(p.read_text())
        if not s.get("usable"):
            continue
        out.append({
            "series_id": s["series_id"],
            "target": s["target_pref_name"],
            "endpoint": s["standard_type"],
            "lead_smiles": s["lead"]["smiles"],
            "lead_pchembl": s["lead_pchembl"],
            "n_targets": s["n_targets"],
            "max_gain": s["max_gain"],
            "year": s.get("lead_year"),
        })
    return out


def explain(score: SuggestionScore) -> str:
    """Plain-language verdict for one evaluated molecule."""
    lines = []
    if not score.valid:
        return f"GEÇERSİZ: {score.invalid_reason} -- bu bir molekül değil."
    lines.append(f"geçerli molekül, lead'e Tanimoto {score.tanimoto_to_lead:.3f} "
                 f"(seri tabanı {score.similarity_floor:.3f})")
    if score.same_murcko_scaffold:
        lines.append("aynı Murcko iskeleti")
    verdicts = {
        "unchanged": "lead'in kendisi -- değişiklik yok",
        "duplicate": "bu molekül zaten önerilmişti",
        "scaffold_violation": "iskeleti terk etmiş: gerçek kampanyanın gittiğinden "
                              "daha uzak, bu artık optimizasyon değil",
        "recovered_target": "KİMYACILAR BUNU YAPTI ve potensi arttı",
        "recovered_nonimprover": "kimyacılar bunu (ya da çok benzerini) yaptı, "
                                 "ama potens artmadı",
        "novel_plausible": "seride yok -- kimse yapmamış, referansımız yok, "
                           "karar uzmanın",
    }
    lines.append(verdicts.get(score.category, score.category))
    if score.matched_series_molecule:
        pch = score.matched_series_pchembl
        lines.append(f"eşleşen gerçek bileşik: {score.matched_series_molecule}"
                     + (f", ölçülmüş p-değeri {pch}" if pch is not None else ""))
    if score.direction_precision is not None:
        lines.append(f"değişiklik tipi isabeti {score.direction_precision:.2f}"
                     + (f" (seride işe yarayanlar: {', '.join(score.direction_hits)})"
                        if score.direction_hits else "")
                     + (f" (seride görülmeyenler: {', '.join(score.direction_misses)})"
                        if score.direction_misses else ""))
    if score.ungrounded_claims:
        lines.append("GEREKÇE TUTARSIZ -- molekülde olmayan yapılardan bahsediyor: "
                     + ", ".join(score.ungrounded_claims))
    if score.heavy_atom_delta:
        lines.append(f"boyut değişimi {score.heavy_atom_delta:+d} ağır atom, "
                     f"{score.mw_delta:+.0f} Da")
    return "\n  ".join(lines)


def score_all(model: str, variant: str = "implicit", effort: str = "medium",
              strategy: str = "window", run_index: int = 0,
              floor_quantile: float = 0.0) -> list[RunScore]:
    """Score every cached response for one condition."""
    from .llm import PROMPT_VERSION, RESPONSE_DIR

    d = RESPONSE_DIR / model.replace("/", "_") / f"{PROMPT_VERSION}_{variant}_{effort}"
    out = []
    for path in sorted(d.glob(f"*__{strategy}__r{run_index}.json")):
        run = json.loads(path.read_text())
        split_path = C.DATA_DIR / "splits" / f"{run['series_id']}__{run['strategy']}.json"
        if not split_path.exists():
            continue
        out.append(score_run(run, json.loads(split_path.read_text()), floor_quantile))
    return out


def write_scores(scores: list[RunScore], name: str) -> dict:
    """Persist per-suggestion detail and a flat table for analysis."""
    detail = C.REPORT_DIR / f"scores_{name}.json"
    detail.write_text(json.dumps([s.to_dict() for s in scores], indent=2))

    import pandas as pd
    df = pd.DataFrame([s.summary() for s in scores])
    df.to_csv(C.REPORT_DIR / f"scores_{name}.csv", index=False)

    # one row per suggestion, for the manual rationale review the spec asks for
    rows = []
    for sc in scores:
        for s in sc.suggestions:
            rows.append({
                "series_id": sc.series_id, "year_stratum": sc.year_stratum,
                "target": sc.target_pref_name, "model": sc.model,
                "index": s.index, "category": s.category,
                "tanimoto_to_lead": round(s.tanimoto_to_lead, 3),
                "similarity_floor": round(s.similarity_floor, 3),
                "best_target_tanimoto": round(s.best_target_tanimoto, 3),
                "approach_gain": (round(s.approach_gain, 3)
                                  if s.approach_gain is not None else None),
                "direction_precision": s.direction_precision,
                "direction_hits": ";".join(s.direction_hits),
                "direction_misses": ";".join(s.direction_misses),
                "generic": s.generic_flag,
                "describes_other_molecule": s.describes_other_molecule,
                "ungrounded_claims": ";".join(s.ungrounded_claims),
                "named_in_text": ";".join(s.named_in_text),
                "heavy_atom_delta": s.heavy_atom_delta,
                "mw_delta": round(s.mw_delta, 1),
                "matched_molecule": s.matched_series_molecule,
                "matched_pchembl": s.matched_series_pchembl,
                "modification": s.modification,
                "rationale": s.rationale,
                "confidence": s.confidence,
                "expected_fold": s.expected_fold_improvement,
                "smiles": s.canonical or s.smiles,
                "invalid_reason": s.invalid_reason,
            })
    sug = pd.DataFrame(rows)
    sug.to_csv(C.REPORT_DIR / f"suggestions_{name}.csv", index=False)
    return {"runs": len(scores), "suggestions": len(rows),
            "detail": str(detail), "table": str(C.REPORT_DIR / f'scores_{name}.csv')}
