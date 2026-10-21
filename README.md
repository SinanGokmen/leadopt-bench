# leadopt-bench

A pipeline that measures **LLM lead-optimization reasoning** retrospectively, against
published experimental SAR data.

The idea: hand a model an **early** compound from a known optimization series, ask it to
improve the compound, then compare its proposals with the **later** analogs medicinal
chemists actually reached. The answer key is not a prediction model — it is measured
activity from ChEMBL.

**Deliberately absent in this phase:** ADMET predictors, docking, affinity oracles.
A suggestion that is absent from the series is labelled but **not scored** — there is no
ground truth for it, and inventing one with a model would reproduce the reward-hacking
problem this project is a reaction to.

---

## Try a molecule right now

**Nothing to install:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SinanGokmen/leadopt-bench/blob/main/try_a_molecule.ipynb)

The notebook runs in the browser. Run the setup cell once; after that every cell is
self-contained and can be run on its own, in any order. Two of them take your input:

* **Option A** — paste any SMILES
* **Option B** — paste a ChEMBL compound id instead

Each takes **one input**. The campaign to compare against is picked automatically —
whichever of the 32 starting compounds your molecule most resembles — and the cell
prints which one it chose. If the molecule resembles none of them, it says so rather
than forcing a meaningless comparison.

Either way you get a colour-coded verdict, the two molecules drawn side by side, and
the option to browse what an LLM proposed on the same series.

Or locally:

```bash
git clone https://github.com/SinanGokmen/leadopt-bench.git
cd leadopt-bench
python3 -m pip install -e .
```

> Use `python3 -m pip`, not bare `pip`. On many machines they are different
> interpreters, and a bare `pip` can install into a Python you are not running.

The 32 series, their splits and the scored model responses are committed, so this
works offline — nothing is downloaded and no API key is needed.

Now check your own idea — or a medicinal chemist's — against the historical record:

```python
from leadopt_bench import metrics as M

# which series you can test against
for s in M.available_series()[:5]:
    print(s["series_id"], s["target"], "lead p=", s["lead_pchembl"])

# evaluate a molecule
sc = M.evaluate_smiles(
    "CC(C)(C)c1cc(NC(=O)Nc2ccc(Oc3ccnc(-c4cnn(CCCCCCC(=O)NO)c4)c3)cc2)no1",
    "CHEMBL6106211_IC50",
    modification="shorten the acyl chain",
    rationale="the hydroxamic acid chelates zinc; chain length should match the channel",
)
print(M.explain(sc))
```

Real output of that call — chemists did make this compound, so the tool reports what
they measured:

```
valid molecule, Tanimoto 1.000 to the lead (series floor 0.663)
  same Murcko scaffold
  CHEMISTS MADE THIS and it was more potent
  matched real compound: CHEMBL6149363, measured p-value 7.51
  size change -4 heavy atoms, -56 Da
```

And for an analog nobody ever made:

```
valid molecule, Tanimoto 0.867 to the lead (series floor 0.663)
  same Murcko scaffold
  absent from the series -- nobody made it, we have no ground truth, this one is the expert's call
  change-type accuracy 0.00 (changes never seen in this series: amide-, carboxylic_acid+, hydroxamic_acid-, hydroxyl+)
  size change -1 heavy atoms, -15 Da
```

That last line is a useful warning: in HDAC inhibitors the hydroxamic acid is the
zinc-binding group, and this series never explored removing it.

> Note the Tanimoto of 1.000 in the first example. Morgan fingerprints are circular, so
> they do not distinguish chain-length changes beyond their radius. The tool reports the
> size change separately, so it is not blind — but the scaffold check itself is
> insensitive to chain length.

---

## How it works

### The problem it is built around

Ask a language model to improve a drug candidate and you get a confident answer. The
hard part is knowing whether the answer is any good.

The field's usual solution is to score the proposals with a *prediction model* — an
ADMET or affinity predictor. That creates a circular problem: the generator learns to
satisfy the predictor, the benchmark uses the same predictor, and nobody can see the
difference between real chemistry and gaming the scorer.

This pipeline avoids the circle by going backwards in time. Medicinal chemists have
already run thousands of optimization campaigns and published the results. In a
finished campaign **the answer is known and it was measured, not predicted**. So:
hide the ending, hand the model the beginning, and compare.

### Building one question

A question comes from a real published campaign. The pipeline pulls activity data
from ChEMBL and keeps only series that are genuine optimization campaigns:

- a single protein target, direct and unambiguous (ChEMBL confidence score 9)
- a single assay, so the potencies are comparable to each other
- at least 20 compounds, mean pairwise Tanimoto > 0.40 — a congeneric series, not an
  unrelated pile of hits
- at least 2 log units of activity range, so something was actually optimized

Take a worked example, `CHEMBL6106211_IC50` — an HDAC1 series from a 2025 paper, 31
compounds spanning pIC50 **5.31 to 8.47**. Every compound shares the same
architecture: a cap group, an aryl ether, a pyridine–pyrazole core, an alkyl linker,
and a hydroxamic acid as the zinc-binding group.

The pipeline picks a **starting compound** from the weak end — here one at pIC50 5.46,
carrying a bulky *tert*-butyl-isoxazole cap and an eleven-carbon linker. The remaining
26 compounds that beat it by at least 1 log unit become the **answer key**.

What the chemists actually did, visible in that key: they **dropped the bulky isoxazole
cap** for a simple urea and **tuned the linker down to about seven carbons**. The best
compound in the series reaches pIC50 8.47 — a thousand-fold gain.

### What the model is given

Only this:

```
Target protein: Histone deacetylase 1
Assay endpoint: IC50
Starting compound (SMILES): CC(C)(C)c1cc(NC(=O)Nc2ccc(Oc3ccnc(-c4cnn(CCCCCCCCCCC(=O)NO)c4)c3)cc2)no1
Measured potency of the starting compound: pIC50 = 5.46

Propose 5 distinct structural modifications ... explain the structure-activity
reasoning for why you expect it to help.
```

No answer compounds, no count of them, no hint of how much gain is available. Every
run audits all 32 real prompts for leakage.

The rationale is a **separate required field** from the molecule, because a suggestion
can be a change chemists really made while the stated reason is invented — and telling
those two apart is the entire point.

### Grading an answer

Three checks, none of which asks a model for an opinion:

| Check | How it is decided |
|---|---|
**Is it a valid molecule?** | RDKit parsing and sanitization, single fragment, sensible mass |
**Was the scaffold kept?** | Morgan fingerprint Tanimoto to the starting compound |
**Is the reasoning about this molecule?** | every ring and functional group the text names is looked for in the starting compound **or** in the proposal |

The two thresholds that could have been arbitrary are instead **read off the series
itself**:

- the similarity floor is the *least similar* compound the real chemists made from that
  lead. A proposal only counts as abandoning the scaffold if it strays further than the
  real campaign ever did.
- "a modification that raises potency" is not a curated belief about chemistry — it is
  the set of functional-group changes observed between the lead and the real improved
  compounds in that same series.

The third check is the mechanizable part of what a chemist does when they glance at a
suggestion and immediately see it is about the wrong compound. In this HDAC1 series,
Gemma 3 12B proposed edits to a "piperidine" and a "quinoline" — neither is present
anywhere in the molecule — while actually deleting an oxygen from the isoxazole ring.
It never mentioned the hydroxamic acid, which is the one group an HDAC chemist would
look at first.

### What it refuses to do

If a proposal is a valid, on-scaffold molecule that simply never appears in the series,
it is labelled `novel_plausible` and **left unscored**. Nobody made that compound, so
there is no measurement, and inventing one with a predictor would rebuild the circular
problem above. Those are the suggestions a human has to judge — and a tool that
pretends otherwise is worse than one that admits the gap.

### Why there are control arms

A single number is uninterpretable. "The model recovered 12% of the real improvements"
only becomes meaningful once you know what *rote advice* and *random chemistry* score
on the same 32 questions. So five reference arms answer the identical questions,
including an **oracle** that is handed the real answers and a **generic** arm that
proposes the same five substituents to every molecule, forever.

That calibration changed how the results must be read: the generic arm scores **0.607**
on change-type accuracy, because halogen scanning genuinely does raise potency. Any
model scoring around 0.6 there has demonstrated nothing at all.

## What it found

**Gemma 3 12B** (local, free), 32 splits × 5 suggestions = 160 suggestions:

| Metric | Gemma 3 12B | generic advice | random analog | oracle |
|---|---|---|---|---|
valid molecule | 0.681 | 1.000 | 1.000 | 1.000 |
recovery | **0.000** | 0.019 | 0.013 | 1.000 |
change-type accuracy | **0.491** | 0.607 | 0.533 | 1.000 |
ungrounded rationale | **0.334** | 0.038 | 0.013 | 0.000 |

**The correct statement is not "worse than random" but "indistinguishable from
random":** change-type accuracy does not separate from any null arm
(p = 0.17–0.78), and neither does recovery (0/160 vs 3/160, p = 0.25). The only axis
where it differs significantly is validity — and there it is behind
(p = 7.6 × 10⁻¹⁸).

How the failures break down: 51 invalid molecules, 15 copies of the lead, 3 scaffold
violations, 35 with ungrounded rationales. **106 of 160 suggestions (66%) can be
filtered automatically**, leaving 54 for an expert to judge.

33% of its rationales cite substructures that are **not in the molecule** — a
piperidine that isn't there, a quinoline that isn't there, a fabricated nitro group in
17 of the 32 splits.

> Scope caveat: this is a 12B open-weight model, so the finding is about the 12B scale.
> A frontier-model arm is wired up in the code but was not run here.

---

## The control ladder

A single score is uninterpretable. Five arms answer the same 32 questions:

| Arm | What it does |
|---|---|
`generic` | the same five substituents on every lead — advice you could give any molecule |
`random` | a random but chemically valid analog |
`neighbour` | potent compounds from *other* series against the same protein |
`oracle` | the real answers (a deliberate cheat, to check the ceiling) |
`mock` | bolts an atom onto the SMILES string (a parsing floor) |

Two of these matter most:

- **`oracle` scores exactly 1.000** → the metrics reach their ceiling; no scoring bug
  is capping the measurement.
- **`neighbour` scores 98% scaffold violations** → the similarity constraint does real
  work; recovery cannot be gamed by dumping known actives.

And one calibration warning: the `generic` arm, which only ever adds fluorine, chlorine
or a methyl, scores **0.607** on change-type accuracy — because halogen scanning really
does raise potency. **That metric cannot be read in absolute terms**, only against its
null floor.

---

## The pipeline

```
ChEMBL activities (binding, IC50/Ki, pChEMBL present)
  → restrict to confidence_score 9 assays
  → row cleaning     (drop censored, flagged, unconvertible measurements)
  → standardization  (largest fragment, uncharge, 150<MW<900, canonical)
  → one row per molecule (median pChEMBL across repeats)
  → congeneric series (n≥20, mean pairwise Tanimoto>0.40, log range≥2.0)
  → early/late split (lead from a fixed potency window, gain≥1 log)
  → ask the LLM      (only lead + protein + endpoint + measured potency)
  → three checks + categorisation
```

14 targets surveyed → **716 series**. Panel: BRD4, HDAC1, thrombin, EGFR → **32 splits**.

Every ChEMBL page is written to disk gzipped, so a re-run never touches the network.

### Three checks — facts, not judgments

None of these asks a model for an opinion; all three are verifiable:

1. **Is it a valid molecule?** RDKit parsing and sanitization, single fragment, mass range.
2. **Was the scaffold kept?** Morgan Tanimoto, with the threshold **calibrated from the
   series itself**: the least similar compound real chemists made from that lead
   (clamped at 0.25).
3. **Is the rationale consistent?** A 40-entry SMARTS catalogue of rings and groups;
   every structure the text names is looked for in the lead **or** in the proposal. If
   it is in neither, the model is describing a compound it was never given.

Check 3 is the mechanizable part of what a domain expert does when they glance at a
suggestion and immediately see it is nonsense.

---

## Install and run

```bash
python3 -m pip install -e ".[test]"   # editable, so the committed data stays findable

# tests (no network access required)
python3 -m pytest -q
```

### Re-running a model over the whole panel

Every response is cached on disk per condition, so re-running costs nothing and
only missing splits are queried.

```bash
# a local open-weight model — needs ollama running and the model pulled:
#   ollama pull gemma3:12b
python3 -c "from leadopt_bench import llm; \
  llm.run_batch(llm.OllamaProvider(), model='gemma3:12b')"

# a large open-weight model through OpenRouter (free tier covers 32 requests)
export OPENROUTER_API_KEY=sk-or-...
python3 -c "from leadopt_bench import llm; \
  llm.run_batch(llm.get_provider('openrouter'), \
                model='nvidia/nemotron-3-ultra-550b-a55b:free')"

# a frontier model
export ANTHROPIC_API_KEY=sk-ant-...
python3 -c "from leadopt_bench import llm; \
  llm.run_batch(llm.get_provider('anthropic'), model='claude-opus-5-5')"

# no credentials at all — exercise the pipeline with the mock provider
python3 -c "from leadopt_bench import llm; llm.run_batch(llm.MockProvider())"
```

Then score, which writes a per-run table and a per-suggestion table with every
rationale, ready for expert review:

```bash
python3 -c "from leadopt_bench import metrics as M; \
  print(M.write_scores(M.score_all('gemma3:12b'), 'report'))"
# -> leadopt_bench/data/reports/scores_report.csv
#    leadopt_bench/data/reports/suggestions_report.csv
```

Providers share one interface: `ollama` (local), `openrouter`, `anthropic`, `mock`.
Switching models is a single parameter.

### Running the control arms

```bash
python3 -c "
from leadopt_bench import baselines as B, llm, metrics as M
for name in ('generic', 'random', 'neighbour', 'oracle'):
    p = B.get_baseline(name)
    llm.run_batch(p, strategy='window')
    M.write_scores(M.score_all(p.DEFAULT_MODEL), f'baseline_{name}')
"
```

### Rebuilding the panel from ChEMBL

Only needed to change targets or criteria — the committed cache already covers the
current panel.

```bash
python3 -c "from leadopt_bench.survey import survey; \
  survey(['CHEMBL301', 'CHEMBL204'])"         # how many usable series per target
python3 -c "from leadopt_bench.panel import build_panel; build_panel()"
```

Thresholds live in one place, `leadopt_bench/config.py`: `MIN_COMPOUNDS`,
`MIN_MEAN_TANIMOTO`, `MIN_LOG_RANGE`, `CONFIDENCE_SCORE`.

## Modules

| File | What it does |
|---|---|
`chembl.py` | cached ChEMBL REST client |
`series.py` | cleaning, standardization, congeneric series assembly |
`survey.py` | measures how many usable series each target yields |
`panel.py` | splits targets into year strata and selects the panel |
`split.py` | early/late splitting (`window`, `activity`, `temporal`) |
`llm.py` | provider interface, prompt, schema-constrained output, caching |
`metrics.py` | five scoring axes, three checks, `evaluate_smiles` |
`baselines.py` | null and oracle control arms |

Detailed design notes and methodological decisions:
[leadopt_bench/README.md](leadopt_bench/README.md)

## Results in the repository

Every scored run is committed, so the results can be read without running anything —
GitHub renders these as sortable tables in the browser:

| File | What it holds |
|---|---|
[`suggestions_final_gemma3_12b.csv`](leadopt_bench/data/reports/suggestions_final_gemma3_12b.csv) | all 160 LLM suggestions: molecule, modification, rationale, category, flags |
[`scores_final_gemma3_12b.csv`](leadopt_bench/data/reports/scores_final_gemma3_12b.csv) | per-split roll-up for the same run |
[`suggestions_final_oracle-truth.csv`](leadopt_bench/data/reports/suggestions_final_oracle-truth.csv) | the oracle arm — the real answers, for comparison |
[`suggestions_final_baseline-generic.csv`](leadopt_bench/data/reports/suggestions_final_baseline-generic.csv) | rote generic advice, the null floor |
[`survey_phase1.csv`](leadopt_bench/data/reports/survey_phase1.csv) | all 14 targets surveyed, with usable-series counts |
[`task2_manifest.json`](leadopt_bench/data/reports/task2_manifest.json) | the 32 splits: lead, gain, target count, stratum |

`suggestions_*.csv` is the file to open first — one row per suggestion, with the
model's own reasoning in a column, which is what an expert review would work from.

## Tests

208 tests, no network access (fake session and providers). The most important ones:

- **no target compound leaks into the prompt** — all 32 real prompts are audited on
  every run
- **the oracle arm scores perfect recovery** — if it does not, the scoring is broken
- a congeneric series is not shattered by clustering
- a single outlier target cannot collapse the similarity floor
- recovery requires actually moving toward the target
- the mock arm is not mistaken for good chemistry

## License

MIT
