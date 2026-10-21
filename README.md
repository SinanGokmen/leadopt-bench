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

Check your own idea — or a medicinal chemist's — against the historical record:

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
pip install -r requirements.txt

# tests (no network access required)
python3 -m pytest leadopt_bench/tests/ -q

# a full run with a local model (ollama)
python3 -c "from leadopt_bench import llm; \
  llm.run_batch(llm.OllamaProvider(), model='gemma3:12b')"

# scoring
python3 -c "from leadopt_bench import metrics as M; \
  M.write_scores(M.score_all('gemma3:12b'), 'report')"
```

Providers share one interface: `ollama` (local), `openrouter`, `anthropic`, `mock`.
Switching models is a single parameter.

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

## Tests

180 tests, no network access (fake session and providers). The most important ones:

- **no target compound leaks into the prompt** — all 32 real prompts are audited on
  every run
- **the oracle arm scores perfect recovery** — if it does not, the scoring is broken
- a congeneric series is not shattered by clustering
- a single outlier target cannot collapse the similarity floor
- recovery requires actually moving toward the target
- the mock arm is not mistaken for good chemistry

## License

MIT
