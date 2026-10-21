"""Turn raw ChEMBL activities into congeneric SAR series.

Three grouping modes, because "series" is not a single thing in ChEMBL:

  assay     one series per assay_chembl_id -- the spec's "same experiment".
            Tightest comparability; every member measured under one protocol.
            A single assay carries one publication year, so a *temporal* split
            is impossible inside it.
  document  one series per (document, standard_type) -- a paper's SAR table,
            possibly spanning a few assays. Also single-year.
  cluster   one series per congeneric chemical cluster within (target,
            standard_type), pooled across documents and years. The only mode
            where a temporal split has any meaning, at the cost of mixing
            protocols between papers.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Iterable

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from rdkit import DataStructs

from . import config as C

RDLogger.DisableLog("rdApp.*")

_FPGEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=C.FP_RADIUS, fpSize=C.FP_BITS
)
_LARGEST = rdMolStandardize.LargestFragmentChooser()

MW_MIN, MW_MAX = 150.0, 900.0

# Butina needs the full condensed distance matrix: n*(n-1)/2 float32 values.
# 8000 compounds is 128 MB, which is fine; past that we fall back to bucketing
# by Murcko scaffold first and clustering inside each bucket.
CLUSTER_POOL_MAX = 8000


# --- compound records ---------------------------------------------------------

@dataclass
class Compound:
    molecule_chembl_id: str
    smiles: str                 # standardized parent
    pchembl: float
    standard_type: str
    assay_chembl_id: str
    document_chembl_id: str
    document_year: int | None
    n_measurements: int = 1


def standardize(smiles: str) -> str | None:
    """Largest fragment, uncharged, canonical. None if unusable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _LARGEST.choose(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
    except Exception:
        return None
    if mol is None or mol.GetNumHeavyAtoms() < 8:
        return None
    mw = Descriptors.MolWt(mol)
    if not (MW_MIN <= mw <= MW_MAX):
        return None
    return Chem.MolToSmiles(mol)


def clean_activities(raw: Iterable[dict]) -> list[dict]:
    """Keep only uncensored, flag-free, parseable measurements."""
    out = []
    for a in raw:
        if a.get("data_validity_comment"):
            continue                              # ChEMBL itself distrusts it
        if a.get("potential_duplicate"):
            continue
        if a.get("standard_relation") != "=":
            continue                              # ">10000" is not a potency
        if a.get("standard_units") != "nM":
            continue
        pch = a.get("pchembl_value")
        smi = a.get("canonical_smiles")
        if not pch or not smi:
            continue
        std = standardize(smi)
        if std is None:
            continue
        out.append({**a, "pchembl": float(pch), "std_smiles": std})
    return out


def _collapse(records: list[dict]) -> list[Compound]:
    """One row per molecule; median pChEMBL when a molecule repeats."""
    by_mol: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_mol[r["molecule_chembl_id"]].append(r)
    comps = []
    for mol_id, rows in by_mol.items():
        vals = [r["pchembl"] for r in rows]
        years = [r["document_year"] for r in rows if r.get("document_year")]
        comps.append(Compound(
            molecule_chembl_id=mol_id,
            smiles=rows[0]["std_smiles"],
            pchembl=float(statistics.median(vals)),
            standard_type=rows[0]["standard_type"],
            assay_chembl_id=rows[0]["assay_chembl_id"],
            document_chembl_id=rows[0]["document_chembl_id"],
            document_year=min(years) if years else None,
            n_measurements=len(rows),
        ))
    return comps


# --- congenericity ------------------------------------------------------------

def fingerprints(smiles: list[str]):
    return [_FPGEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


def mean_pairwise_tanimoto(fps) -> float:
    n = len(fps)
    if n < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        total += float(sum(sims))
        count += len(sims)
    return total / count if count else 0.0


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko framework: ring systems plus the linkers joining them."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return ""
    return Chem.MolToSmiles(scaf) if scaf is not None else ""


def butina_clusters(fps, cutoff: float = 0.6) -> list[list[int]]:
    """Cluster by Tanimoto distance; cutoff 0.6 == similarity 0.4.

    The condensed distance matrix is held in a preallocated float32 array
    rather than a Python list, which is what makes thousand-ligand pools
    tractable (a list of 32M floats would cost well over a gigabyte).
    """
    n = len(fps)
    if n < 2:
        return [list(range(n))]
    dists = np.empty(n * (n - 1) // 2, dtype=np.float32)
    pos = 0
    for i in range(1, n):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]),
                          dtype=np.float32)
        dists[pos:pos + sims.size] = 1.0 - sims
        pos += sims.size
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    return [list(c) for c in clusters]


# --- series -------------------------------------------------------------------

@dataclass
class Series:
    series_id: str
    grouping: str
    target_chembl_id: str
    target_pref_name: str
    standard_type: str
    compounds: list[Compound] = field(default_factory=list)
    mean_tanimoto: float = 0.0
    document_chembl_ids: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    scaffold_smiles: str = ""

    @property
    def n(self) -> int:
        return len(self.compounds)

    @property
    def log_range(self) -> float:
        vals = [c.pchembl for c in self.compounds]
        return max(vals) - min(vals) if vals else 0.0

    @property
    def n_documents(self) -> int:
        return len(set(self.document_chembl_ids))

    @property
    def year_span(self) -> int:
        ys = [y for y in self.years if y]
        return (max(ys) - min(ys)) if ys else 0

    @property
    def temporally_splittable(self) -> bool:
        """At least two distinct publication years to order compounds by."""
        return len({y for y in self.years if y}) >= 2

    def passes(self) -> bool:
        return (self.n >= C.MIN_COMPOUNDS
                and self.mean_tanimoto > C.MIN_MEAN_TANIMOTO
                and self.log_range >= C.MIN_LOG_RANGE)

    def summary(self) -> dict:
        return {
            "series_id": self.series_id,
            "grouping": self.grouping,
            "target_chembl_id": self.target_chembl_id,
            "target_pref_name": self.target_pref_name,
            "standard_type": self.standard_type,
            "n_compounds": self.n,
            "mean_tanimoto": round(self.mean_tanimoto, 3),
            "pchembl_min": round(min(c.pchembl for c in self.compounds), 2),
            "pchembl_max": round(max(c.pchembl for c in self.compounds), 2),
            "log_range": round(self.log_range, 2),
            "n_documents": self.n_documents,
            "year_min": min((y for y in self.years if y), default=None),
            "year_max": max((y for y in self.years if y), default=None),
            "year_span": self.year_span,
            "temporally_splittable": self.temporally_splittable,
            "scaffold_smiles": self.scaffold_smiles,
        }

    def to_dict(self) -> dict:
        return {**self.summary(),
                "compounds": [asdict(c) for c in self.compounds]}

    def write(self) -> "object":
        path = C.SERIES_DIR / f"{self.series_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def _dominant_scaffold(comps: list[Compound]) -> str:
    """The most common Murcko framework in a cluster, for human inspection."""
    counts: dict[str, int] = defaultdict(int)
    for c in comps:
        scaf = murcko_scaffold(c.smiles)
        if scaf:
            counts[scaf] += 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _make_series(series_id: str, grouping: str, target_id: str, target_name: str,
                 std_type: str, comps: list[Compound]) -> Series:
    fps = fingerprints([c.smiles for c in comps])
    return Series(
        series_id=series_id,
        grouping=grouping,
        target_chembl_id=target_id,
        target_pref_name=target_name,
        standard_type=std_type,
        compounds=sorted(comps, key=lambda c: c.pchembl),
        mean_tanimoto=mean_pairwise_tanimoto(fps),
        document_chembl_ids=[c.document_chembl_id for c in comps],
        years=[c.document_year for c in comps],
    )


def build_series(cleaned: list[dict], target_id: str, target_name: str,
                 grouping: str = "assay") -> list[Series]:
    """Group cleaned activities into candidate series (unfiltered)."""
    if grouping == "assay":
        keyfn = lambda r: (r["assay_chembl_id"], r["standard_type"])
    elif grouping == "document":
        keyfn = lambda r: (r["document_chembl_id"], r["standard_type"])
    elif grouping == "cluster":
        keyfn = lambda r: (target_id, r["standard_type"])
    else:
        raise ValueError(f"unknown grouping: {grouping}")

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in cleaned:
        buckets[keyfn(r)].append(r)

    out: list[Series] = []
    for key, rows in buckets.items():
        comps = _collapse(rows)
        if len(comps) < C.MIN_COMPOUNDS:
            continue
        std_type = key[1]
        if grouping != "cluster":
            out.append(_make_series(f"{key[0]}_{std_type}", grouping,
                                    target_id, target_name, std_type, comps))
            continue
        # cluster mode: carve the target-wide pool into congeneric clusters by
        # fingerprint similarity. Clustering the whole pool at once is what lets
        # a series span several papers -- and therefore several years, which is
        # the only way a temporal split becomes possible. Bucketing by Murcko
        # scaffold first would be cheaper but splits real series apart whenever
        # a peripheral ring is varied, so it is only the large-pool fallback.
        if len(comps) <= CLUSTER_POOL_MAX:
            buckets_to_cluster = [comps]
        else:
            by_scaffold: dict[str, list[Compound]] = defaultdict(list)
            for comp in comps:
                by_scaffold[murcko_scaffold(comp.smiles)].append(comp)
            buckets_to_cluster = [b for scaf, b in by_scaffold.items()
                                  if scaf and len(b) >= C.MIN_COMPOUNDS]

        serial = 0
        for bucket in buckets_to_cluster:
            fps = fingerprints([c.smiles for c in bucket])
            for idxs in butina_clusters(fps):
                if len(idxs) < C.MIN_COMPOUNDS:
                    continue
                sub = [bucket[j] for j in idxs]
                s = _make_series(f"{target_id}_{std_type}_c{serial:03d}", grouping,
                                 target_id, target_name, std_type, sub)
                s.scaffold_smiles = _dominant_scaffold(sub)
                out.append(s)
                serial += 1
    return out
