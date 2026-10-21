# leadopt_bench — design notes and methodological decisions

The data and scoring layer for measuring LLM lead-optimization reasoning against
**published experimental SAR data**. No prediction-model oracle, no docking, no
affinity predictor in this phase — the right answer is already in ChEMBL.

This file records *why* things are built the way they are, including the mistakes
that were found and corrected. The short overview lives in the
[top-level README](../README.md).

## Status

| Task | Status |
|---|---|
| 1 — Data collection infrastructure | Done; 14 targets surveyed, 716 series |
| 2 — Early/late splitting | Done; 32 primary + 33 sensitivity splits |
| 3 — LLM evaluation loop | Done; Gemma 3 12B run on all 32 splits |
| 4 — Comparison metrics | Done; calibrated against the control ladder, 3 definition flaws fixed |
| 5 — Failure analysis | Distribution measured for the 12B arm; expert review still needed |

**What is missing:** a frontier-model arm. The provider interface supports it and
a full 32-split run costs well under a dollar, but no commercial API key was
available here, so only local open-weight models were run.

## The panel (result of Tasks 1–2)

Four targets, two year strata, `assay` grouping:

| Target | Role | old (≤2015) | new (2024+) |
|---|---|---|---|
| BRD4 | exposure axis | 5 | 5 |
| HDAC1 | exposure axis | 5 | 4 |
| Thrombin | exposure-positive control | 5 | (2, excluded from the claim) |
| EGFR | exposure-positive control | 5 | (1, excluded from the claim) |

**Primary strategy `window`**: the lead is drawn from a fixed pChEMBL window
[5.0, 7.0] with a required gain of ≥1.0 log. The `activity` strategy (bottom
quartile) is also written to disk as a sensitivity control.

### What the exposure axis does and does not claim

We do **not** claim that post-2024 series are absent from training data — current
models' knowledge cutoffs cover those years. What the axis measures is **exposure
redundancy**: a 2013 EGFR paper has been seen again and again through thousands of
citations, reviews and textbooks; a 2024 BRD4 paper probably once.

When the axis was first built it was confounded with difficulty: older campaigns
start in the micromolar range and gain ~3.3 log, while modern ones start from an
already-potent compound and gain ~2.4 (lead potency p=0.047). The fixed potency
window fixed this — across the exposure-axis targets, lead potency now gives
p=0.51, gain p=0.18, target count p=0.87. Almost all of the confound lived in
thrombin and EGFR, which is why those two contribute only their old series, as a
control.

### Lead selection

The **representative** compound from inside the window: the one with the highest
Tanimoto similarity to the series as a whole. A chemist does not start optimizing
from the strangest member of a series, and the weakest compound in a series is
often exactly that — a probe that broke the SAR.

Selection is also restricted to members that **leave enough headroom**: a centrally
placed lead can sit so close to the top of its series that nothing clears
`min_gain`, and the series is wasted.

## Modules

- `config.py` — paths and series acceptance criteria (single source of truth)
- `chembl.py` — ChEMBL REST client; every page is written to disk as `.json.gz`, so
  a re-run never touches the network
- `series.py` — cleaning, standardization, congeneric series assembly
- `survey.py` — measures **how many usable series** each target actually yields
- `panel.py` — splits the four targets into year strata and selects the panel
- `split.py` — early/late splitting: `window` (primary), `activity`, `temporal`
- `llm.py` — provider-agnostic evaluation loop; schema-constrained JSON output,
  on-disk cache per condition, `MockProvider` for a network-free dry run
- `metrics.py` — five scoring axes, categorisation, and `evaluate_smiles` for
  checking an arbitrary molecule by hand
- `baselines.py` — null and oracle control arms (ladder below)

## Metrics (Task 4)

Every suggestion falls into exactly one category: `invalid`, `unchanged`,
`duplicate`, `scaffold_violation`, `recovered_target`, `recovered_nonimprover`,
`novel_plausible`. Alongside these are orthogonal flags: `generic_flag`,
`direction_precision`, `approach_gain`, `mw_delta`, `ungrounded_claims`.

**There is no potency, ADMET or synthesizability predictor anywhere.** A suggestion
absent from the series is labelled `novel_plausible` and **not scored** good or bad
— we have no ground truth for it, and inventing one with a model would be exactly
the reward-hacking this project is a reaction to.

Two thresholds are calibrated from the series rather than chosen by hand:

- the **similarity floor** = the *least* similar compound real chemists made from
  this lead
- a **"potency-increasing modification"** = the lead→target functional-group deltas
  observed in that same series

## First real LLM arm: Gemma 3 12B

Google DeepMind Gemma 3 12B (Q4_K_M), served locally with Ollama, temperature 0,
schema-constrained JSON. 32 splits × 5 suggestions = **160 suggestions**, 61.7
minutes, zero cost.

| Arm | valid | scaffold | recovery | **direction** | ungrounded | = lead | invalid |
|---|---|---|---|---|---|---|---|
| **Gemma 3 12B** | 0.681 | 0.825 | **0.000** | **0.491** | **0.334** | **0.094** | **0.319** |
| mock | 0.594 | 0.594 | 0.006 | 0.526 | 0.006 | 0.000 | 0.406 |
| generic | 1.000 | 1.000 | 0.019 | 0.607 | 0.038 | 0.000 | 0.000 |
| random | 1.000 | 0.938 | 0.013 | 0.533 | 0.013 | 0.000 | 0.000 |
| neighbour | 1.000 | 0.019 | 0.000 | 0.507 | 0.000 | 0.000 | 0.000 |
| oracle | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |

### What is significant and what is not

This distinction matters as much as the finding itself.

**NOT significant — indistinguishable from the null arms:**

| Comparison | Gemma | Null | p |
|---|---|---|---|
| direction vs `mock` | 0.491 | 0.526 | 0.693 |
| direction vs `generic` | 0.491 | 0.607 | 0.166 |
| direction vs `random` | 0.491 | 0.533 | 0.495 |
| recovery vs `generic` | 0/160 | 3/160 | 0.248 |
| recovery vs `random` | 0/160 | 2/160 | 0.498 |

**Significant — and in the wrong direction:** validity 109/160 vs 160/160,
**p = 7.6 × 10⁻¹⁸**.

So the correct statement is not "worse than random" but **"indistinguishable from
random"**: none of the 160 suggestions recovered a real improved compound, and the
change types it chose overlapped with the potency-raising ones no better than
picking a substituent at random. The only axis on which it separates statistically
is validity, and there it is clearly behind.

### Failure distribution (Task 5)

| Category | n | % |
|---|---|---|
| invalid molecule | 51 | 31.9% |
| absent from the series but plausible (`novel_plausible`) | 89 | 55.6% |
| returned the lead unchanged | 15 | 9.4% |
| scaffold violation | 3 | 1.9% |
| in the series, but not an improver | 1 | 0.6% |
| duplicate | 1 | 0.6% |
| **recovered a real improvement** | **0** | **0.0%** |

**41.3% of suggestions are not even answers** (31.9% invalid + 9.4% the lead
returned verbatim).

In **33.4% of valid suggestions the rationale cites a structure present in neither
the lead nor the proposal** — against 0–3.8% in the null arms. The most frequently
invented: nitro (24), sulfonamide (6), piperazine (5), piperidine (4), hydroxamate (4).

After the deterministic filters, **106 of 160 (66%) can be discarded
automatically**, leaving **54** for an expert to judge.

**A hypothesis that was tested and rejected:** the nitro hallucination looked like it
might come from misreading the `no1` fragment (an isoxazole ring closure) in the
SMILES as a nitro group. It does not: the hallucination appears in 17 splits, none
of whose leads contain a nitro group, and only one of which has an N–O ring. So it
is not a representation confusion — the model emits generic medicinal-chemistry
vocabulary without reference to the molecule in front of it.

**Scope caveat:** this is a 12B open-weight model. The finding is about the 12B
scale; claiming "LLMs cannot do lead optimization" would require a frontier model.

### An inconclusive arm: ChemLLM 7B

A chemistry-specialized model (AI4Chem, InternLM-2 base) was run on the same 32
splits to test whether Gemma's validity failures come from scale or from general
rather than chemistry-specific training. The result was degenerate: 10 of 32
responses unparseable, only 27 of 160 requested suggestions delivered, and **zero
valid molecules** among those 27.

This is **not reported as a finding about ChemLLM**, because two variables were
uncontrolled:

1. AI4Chem recommend a bespoke system prompt, which was not used — every arm gets
   the same prompt so the comparison stays clean.
2. The only 7B GGUF available was a third-party 4-bit quantization whose fidelity
   could not be verified (the file's metadata does match `internlm2 / 7.74B`).

Zero valid molecules in 27 attempts is extreme enough that "the model cannot do it"
cannot be separated from "the setup is broken". Resolving it needs a run with the
authors' own system prompt, and ideally a verified quantization.

Two configuration problems found along the way are worth recording: the GGUF
carries no chat template and neither does the official AI4Chem repository, so Ollama
had been passing the raw prompt with no role markers at all; and setting
`repeat_penalty` to 1.0 (to protect long SMILES chains such as `CCCCCCCCCCC`) sent
the model into a repetition loop that ran for 345 seconds. The authors' recommended
1.5 avoids the loop.

### The control ladder (`baselines.py`)

A single score cannot be interpreted. "The model recovered 12% of the real
improvements" only becomes a finding once you know what random chemistry, rote
advice and perfect knowledge score on the **same 32 splits**. Weakest to strongest:

| Arm | What it does | valid | scaffold | exact rec. | close rec. | generic | **direction** |
|---|---|---|---|---|---|---|---|
| `mock` | bolts an atom onto the SMILES string | 0.594 | 0.594 | 0.000 | 0.006 | 0.444 | 0.526 |
| `generic` | the same five substituents on every lead | 1.000 | 1.000 | 0.000 | 0.019 | 0.350 | 0.607 |
| `random` | a random but valid analog | 1.000 | 0.938 | 0.000 | 0.013 | 0.025 | 0.533 |
| `neighbour` | potent compounds from *other* series of the same protein | 1.000 | **0.019** | 0.000 | **0.000** | 0.000 | 0.507 |
| `oracle` | the real improved compounds (a deliberate cheat) | 1.000 | 1.000 | **1.000** | **1.000** | 0.079 | **1.000** |

All of them answer 5 suggestions × 32 splits = 160. `oracle` and `neighbour` receive
the split through `bind_split`, data no real model is given — deliberate cheats,
labelled as such, never to be reported as models.

The ladder establishes four things:

1. **The metrics reach their ceiling.** The oracle scores exactly 1.000 on every
   axis, so no scoring bug is capping the measurement.
2. **Direction accuracy has a floor of 0.53–0.61.** The most important calibration
   result. A model scoring 0.6 has demonstrated nothing; only values approaching
   1.0 mean anything. This is why the `generic` arm is permanent.
3. **Recovery is highly specific.** Every null arm is ≤1.9%. Random valid chemistry
   almost never hits a real improved compound, so recovery is a hard, meaningful
   metric. The bar an LLM has to clear is 0.019.
4. **The similarity constraint does real work.** The `neighbour` arm proposes
   genuinely potent compounds against the right protein, and 98.1% are flagged as
   scaffold violations. Recovery cannot be gamed by dumping known actives.

### The mock floor (calibrating the metrics)

`MockProvider` appends a single atom to the lead and writes a filler rationale. The
metrics must not mistake that for good chemistry — over 32 splits and 160
suggestions:

| Metric | Mock floor |
|---|---|
| exact recovery | **0.000** |
| close recovery | **0.006** |
| generic rate | 0.444 |
| **direction accuracy** | **0.526** |

The last row is the critical one: a model that only ever adds F/Cl/methyl/OH scores
**0.53** on direction accuracy, because halogen scanning genuinely is a
potency-raising change. **Direction accuracy must never be read in absolute terms**,
which is why the mock arm stays as a permanent condition.

(The mock's 40.6% `invalid` rate is a mock artifact — it does naive string surgery
on SMILES. Not a finding, but it does prove the validity axis works.)

### Three definition flaws that were found and fixed

**1. The similarity floor collapsed on a single outlier target.** The raw minimum
dropped to 0.076 in four series (BRD4 bromodomain series span several chemotypes),
which made the scaffold-violation axis vacuous for them. It is now clamped at 0.25 —
deliberately below the 0.40 mean-pairwise criterion a series must already meet.
Measured: this excludes 1.1% of real targets and catches an unrelated probe molecule
in 32/32 series, where the raw minimum left 4 series degenerate.

**2. Recovery rewarded "standing next to the lead".** Adding one atom to the lead
gives ~0.9 similarity to every close analog chemists made, so a suggestion earned
credit without ever *approaching* the answer. Recovery now requires a positive
`approach_gain`: the suggestion must be closer to that compound than the lead
already was. Spurious `recovered_nonimprover` matches fell from 6 to 1.

**3. Direction accuracy has a high floor** (0.526 above) and cannot be reported as
an absolute value.

### Three design decisions about the LLM loop

**The rationale is a separate field from the molecule.** `rationale` is mandatory in
the schema. A suggestion can be a modification chemists really made while the stated
reason is invented; Phase 1 exists precisely to tell those apart, so the two are
never collapsed into one block of prose.

**Output is enforced by schema, not extracted from prose.** A regex over free text
fails in ways that *correlate* with model quality — the parsing error would quietly
become part of the result. An unparseable response is recorded as an error, not
rescued.

**No `temperature`.** Current Claude models reject the parameter (400); depth is set
with `output_config.effort`, and effort is recorded with every response as part of
the experimental condition. Local models do accept temperature, and it is pinned to
0 there for reproducibility.

The prompt has two variants: `implicit` (the similarity constraint is not stated, so
that "invented a new molecule instead of optimizing" remains a measurable failure
mode) and `explicit` (scaffold retention is requested outright). The primary
measurement uses `implicit`.

## Data path

```
ChEMBL activities (binding, IC50/Ki, pChEMBL present)
  → restrict to confidence_score 9 assays
  → row cleaning     (drop censored / flagged / unconvertible measurements)
  → standardization  (largest fragment, uncharge, 150<MW<900, canonical)
  → one row per molecule (median pChEMBL across repeats)
  → grouping         (assay | document | cluster)
  → acceptance       (n≥20, mean pairwise Tanimoto>0.40, log range≥2.0)
```

Every step writes to disk; every step is re-runnable.

### Why there are three groupings

"Series" is not one thing in ChEMBL:

| Mode | Definition | Comparability | Temporal split |
|---|---|---|---|
| `assay` | one `assay_chembl_id` | highest: a single protocol | **impossible** (one year) |
| `document` | one paper's SAR table | high | **impossible** (one year) |
| `cluster` | Murcko scaffold + Butina, across papers | lower: protocols mix | possible |

This directly affects Task 2: a single assay carries a single publication year, so a
**temporal split is only meaningful in `cluster` mode** — at the cost of mixing
protocols. The `survey` output carries a `temporally_splittable` field per series.

`cluster` mode clusters the whole pool with Butina, holding the condensed distance
matrix in a preallocated `float32` numpy array (8000 compounds ≈ 128 MB). It does
**not** bucket by scaffold first, because real series are damaged by that: CDK2's
`CHEMBL5735220` series has 134 compounds but **32 distinct Murcko scaffolds** — the
core is constant and what varies is the ring of the solubilizing amine
(piperidine/morpholine/azetidine…). Bucketing by scaffold shattered that series into
singletons. Scaffold bucketing survives only as the fallback for pools larger than
8000.

### Why the two splits are different experiments

Applied to the same cluster-grouped series, the activity split and the temporal split
pick the same starting compound in only **1 of 14** cases, and the Jaccard overlap of
their target sets is **0.42**. Median achievable gain is 3.15 log for the activity
split versus 2.06 for the temporal one. Collapsing them into a single metric would be
wrong, and 7 of 20 temporally-splittable series yield no temporal split at all —
later work in those series did not raise potency, presumably optimizing selectivity
or PK instead.

## What cleaning discards

| Filter | Reason |
|---|---|
| `standard_relation != "="` | `>10 µM` is not a potency, it is a lower bound |
| `data_validity_comment` set | ChEMBL itself distrusts that measurement |
| `potential_duplicate` | a second record of the same measurement inflates the series |
| `standard_units != "nM"` | µg/mL cannot be converted without MW context |
| RDKit cannot parse | unusable structure |
| MW < 150 or > 900 | a fragment, or past the size lead optimization deals with |

## Usage

```bash
# survey targets (writes data/reports/survey_phase1.{json,csv})
python3 -c "from leadopt_bench.survey import survey; survey(['CHEMBL301'])"

# evaluate one molecule against a known series
python3 -c "from leadopt_bench import metrics as M; \
  print(M.explain(M.evaluate_smiles('CCO...', 'CHEMBL6106211_IC50')))"

# tests
python3 -m pytest leadopt_bench/tests/ -q
```

## Tests

180 tests, no network access (fake session and providers). Coverage includes:
salt/charge standardization; discarding censored and flagged measurements;
separating congeneric from diverse sets; collapsing repeat measurements to a median;
a series whose peripheral ring varies not being shattered by clustering (a
regression test for the flaw above); pagination; cache hits; the cache not being
poisoned after a failure; split ordering and thresholds; **no target compound leaking
into the prompt**; cache separation per condition; provider failures being recorded
rather than raised; invalid structures rejected with a reason; a single outlier target
not collapsing the similarity floor (regression); recovery requiring approach
(regression); **the mock arm not being mistaken for good chemistry**; a close analog
absent from the series not being penalised; and a wall-clock deadline on local model
calls (regression — a run once sat on a single split for eight hours because
`requests`' timeout is an idle timer, not a total one).

On top of the test suite, every dry run audits all 32 **real** prompts for leakage
(currently: 0 leaks).
