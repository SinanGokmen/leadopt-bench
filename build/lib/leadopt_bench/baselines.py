"""Null and oracle control arms, so an LLM number means something.

Any single score is uninterpretable on its own. "The model recovered 12% of the
real improvements" is only a finding once you know what random chemistry, rote
generic advice, and perfect knowledge score on the same 32 splits.

The ladder, weakest to strongest:

  generic    the same five substituents on every lead, forever. This is advice
             you could give about any molecule without reading it.
  random     a random medicinal-chemistry substituent at a random attachment
             point. Chemically plausible, entirely uninformed.
  neighbour  compounds taken from *other* series against the same protein.
             Tests whether merely knowing the target's chemotypes scores well --
             if an LLM cannot beat this, it contributed no reasoning.
  oracle     the real improved compounds. Upper bound; proves the metrics can
             actually reach 1.0 rather than being capped by a scoring bug.

`oracle` and `neighbour` are given data no real model is given. They are
deliberate cheats, labelled as such, and must never be reported as models.

Every baseline builds molecules through RDKit fragmentation and re-assembly,
so unlike the naive MockProvider its output is chemically valid. That matters:
it separates "the metric caught an invalid structure" from "the metric caught
bad chemistry".
"""

from __future__ import annotations

import json
import random
import re

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMMPA

from . import config as C
from .llm import DEFAULT_N_SUGGESTIONS, Provider

RDLogger.DisableLog("rdApp.*")

# Attachment-point substituents a medicinal chemist actually reaches for.
SUBSTITUENTS: dict[str, str] = {
    "fluoro": "F[*:1]",
    "chloro": "Cl[*:1]",
    "bromo": "Br[*:1]",
    "methyl": "C[*:1]",
    "ethyl": "CC[*:1]",
    "isopropyl": "CC(C)[*:1]",
    "cyclopropyl": "C1CC1[*:1]",
    "hydroxyl": "O[*:1]",
    "methoxy": "CO[*:1]",
    "amino": "N[*:1]",
    "dimethylamino": "CN(C)[*:1]",
    "nitrile": "N#C[*:1]",
    "trifluoromethyl": "FC(F)(F)[*:1]",
    "methylsulfonyl": "CS(=O)(=O)[*:1]",
    "sulfonamide": "NS(=O)(=O)[*:1]",
    "carboxamide": "NC(=O)[*:1]",
    "carboxylic_acid": "OC(=O)[*:1]",
    "morpholine": "C1COCCN1[*:1]",
    "piperazine": "C1CN(CCN1)[*:1]",
    "phenyl": "c1ccccc1[*:1]",
    "pyridin_3_yl": "c1cccnc1[*:1]",
    "imidazol_1_yl": "c1cncn1[*:1]",
}

# The five things you can say about any molecule without having looked at it.
GENERIC_CHOICES = ["fluoro", "chloro", "methyl", "hydroxyl", "trifluoromethyl"]


# --- molecule surgery ---------------------------------------------------------

def attachment_cores(smiles: str, max_cores: int = 40) -> list[tuple[str, str]]:
    """Single-cut (core, removed_substituent) pairs, both carrying [*:1]."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    out, seen = [], set()
    try:
        frags = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False)
    except Exception:
        return []
    for _, frag in frags:
        parts = frag.split(".")
        if len(parts) != 2:
            continue
        sized = []
        for p in parts:
            m = Chem.MolFromSmiles(p)
            sized.append((m.GetNumHeavyAtoms() if m else 0, p))
        sized.sort()
        sub, core = sized[0][1], sized[1][1]
        if core in seen:
            continue
        seen.add(core)
        out.append((core, sub))
        if len(out) >= max_cores:
            break
    return out


def graft(core_smiles: str, substituent_smiles: str) -> str | None:
    """Join a [*:1] core to a [*:1] substituent. None if the result is unusable."""
    core = Chem.MolFromSmiles(core_smiles)
    sub = Chem.MolFromSmiles(substituent_smiles)
    if core is None or sub is None:
        return None
    try:
        prod = Chem.molzip(core, sub)
        Chem.SanitizeMol(prod)
    except Exception:
        return None
    smi = Chem.MolToSmiles(prod)
    return smi if Chem.MolFromSmiles(smi) is not None else None


def _lead_from_prompt(user: str) -> str:
    m = re.search(r"Starting compound \(SMILES\): (\S+)", user)
    return m.group(1) if m else ""


def _payload(items: list[dict]) -> str:
    return json.dumps({"suggestions": items})


def _suggestion(smiles: str, modification: str, rationale: str,
                fold: float = 3.0, confidence: str = "medium") -> dict:
    return {"smiles": smiles, "modification": modification, "rationale": rationale,
            "expected_fold_improvement": fold, "confidence": confidence}


# --- baselines ----------------------------------------------------------------

class GenericProvider(Provider):
    """Rote advice: the same five substituents on every lead, forever."""

    name = "baseline_generic"
    DEFAULT_MODEL = "baseline-generic"

    def __init__(self, n: int = DEFAULT_N_SUGGESTIONS):
        self.n = n
        self.calls = 0

    def complete(self, system, user, model, max_tokens, effort, schema):
        self.calls += 1
        lead = _lead_from_prompt(user)
        cores = attachment_cores(lead)
        out = []
        for name in GENERIC_CHOICES:
            if len(out) >= self.n:
                break
            for core, _sub in cores:
                smi = graft(core, SUBSTITUENTS[name])
                if smi and smi != lead:
                    out.append(_suggestion(
                        smi, f"add {name}",
                        f"Adding a {name} group should improve potency by "
                        f"strengthening interactions in the binding site."))
                    break
        return _payload(out), {"input_tokens": 0, "output_tokens": 0}, "end_turn"


class RandomAnalogProvider(Provider):
    """A random medicinal-chemistry substituent at a random attachment point."""

    name = "baseline_random"
    DEFAULT_MODEL = "baseline-random"

    def __init__(self, n: int = DEFAULT_N_SUGGESTIONS, seed: int = 20261016):
        self.n = n
        self.seed = seed
        self.calls = 0

    def complete(self, system, user, model, max_tokens, effort, schema):
        self.calls += 1
        lead = _lead_from_prompt(user)
        # seeded per lead, so the whole run is reproducible
        rng = random.Random(f"{self.seed}:{lead}")
        cores = attachment_cores(lead)
        names = list(SUBSTITUENTS)
        out, tried = [], set()
        for _ in range(self.n * 40):
            if len(out) >= self.n or not cores:
                break
            core, _sub = rng.choice(cores)
            name = rng.choice(names)
            if (core, name) in tried:
                continue
            tried.add((core, name))
            smi = graft(core, SUBSTITUENTS[name])
            if smi and smi != lead and smi not in {s["smiles"] for s in out}:
                out.append(_suggestion(
                    smi, f"replace substituent with {name}",
                    f"Substituting {name} at this position explores the "
                    f"structure-activity relationship around the core."))
        return _payload(out), {"input_tokens": 0, "output_tokens": 0}, "end_turn"


class _ContextualProvider(Provider):
    """Base for baselines that are handed data no real model receives.

    `query_split` passes the split in through `bind_split`. An earlier version
    looked the split up by lead SMILES instead, which was wrong: the same
    compound is measured in several assays of the same protein, so it can be the
    chosen lead for two different series, and one silently overwrote the other.
    The oracle then answered with another series' compounds and scored itself
    0.988 instead of 1.0 -- a control arm quietly grading the wrong question.
    """

    def __init__(self, n: int = DEFAULT_N_SUGGESTIONS, strategy: str = "window"):
        self.n = n
        self.calls = 0
        self.strategy = strategy
        self._bound: dict | None = None

    def bind_split(self, split: dict) -> None:
        self._bound = split


class OracleProvider(_ContextualProvider):
    """Returns the real improved compounds. A deliberate cheat: upper bound only.

    Its purpose is to prove the metrics can reach 1.0. If the oracle does not
    score as perfect recovery, the scoring is broken, not the chemistry.
    """

    name = "oracle"
    DEFAULT_MODEL = "oracle-truth"

    def complete(self, system, user, model, max_tokens, effort, schema):
        self.calls += 1
        split = self._bound
        if not split:
            return _payload([]), {}, "end_turn"
        out = []
        for t in split["targets"][:self.n]:
            out.append(_suggestion(
                t["smiles"], "the compound the chemists actually made",
                "Known improved analog from the published series.",
                fold=10 ** (t["pchembl"] - split["lead_pchembl"]), confidence="high"))
        return _payload(out), {"input_tokens": 0, "output_tokens": 0}, "end_turn"


class SeriesNeighbourProvider(_ContextualProvider):
    """Potent compounds from OTHER series against the same protein.

    This is the sharp control. It encodes "I know which chemotypes work on this
    target" and nothing else -- no reasoning about the lead in front of it. An
    LLM that cannot beat this has contributed no lead-optimization reasoning,
    only target familiarity.
    """

    name = "baseline_neighbour"
    DEFAULT_MODEL = "baseline-neighbour"

    def __init__(self, n: int = DEFAULT_N_SUGGESTIONS, strategy: str = "window"):
        super().__init__(n=n, strategy=strategy)
        self._series_cache: dict[str, list[dict]] = {}

    def _pool(self, split: dict) -> list[dict]:
        tid = split.get("target_chembl_id", "")
        if tid not in self._series_cache:
            pool = []
            for p in sorted(C.SERIES_DIR.glob("*.json")):
                s = json.loads(p.read_text())
                if s.get("target_chembl_id") == tid:
                    for c in s.get("compounds", []):
                        pool.append({**c, "_series": s.get("series_id", p.stem)})
            self._series_cache[tid] = pool
        return self._series_cache[tid]

    def complete(self, system, user, model, max_tokens, effort, schema):
        self.calls += 1
        split = self._bound
        if not split:
            return _payload([]), {}, "end_turn"

        own = split["series_id"]
        banned = {t["smiles"] for t in split["targets"]} | {split["lead"]["smiles"]}
        pool = [c for c in self._pool(split)
                if c.get("_series") != own and c["smiles"] not in banned]
        pool.sort(key=lambda c: -c["pchembl"])

        out, seen = [], set()
        for c in pool:
            if len(out) >= self.n:
                break
            if c["smiles"] in seen:
                continue
            seen.add(c["smiles"])
            out.append(_suggestion(
                c["smiles"], "a potent compound known for this target",
                f"This chemotype reaches p{split['standard_type']} "
                f"{c['pchembl']:.1f} against the same protein.",
                confidence="high"))
        return _payload(out), {"input_tokens": 0, "output_tokens": 0}, "end_turn"


BASELINES = {
    "generic": GenericProvider,
    "random": RandomAnalogProvider,
    "neighbour": SeriesNeighbourProvider,
    "oracle": OracleProvider,
}


def get_baseline(name: str, **kwargs) -> Provider:
    if name not in BASELINES:
        raise ValueError(f"unknown baseline: {name}; have {sorted(BASELINES)}")
    return BASELINES[name](**kwargs)
