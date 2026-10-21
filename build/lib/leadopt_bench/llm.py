"""Ask an LLM to improve a lead compound, and record what it said and why.

Design constraints that matter for the experiment:

* The rationale is a separate field from the molecule. A suggestion can be a
  modification chemists really made while the stated reason is invented, and
  Phase 1 exists to tell those apart -- so they are never collapsed into prose.
* Structured output is enforced by schema, not parsed out of free text. A
  regex over prose fails in ways that correlate with model quality, which
  would quietly become part of the result.
* No target compound, count, or achievable gain ever enters the prompt. The
  only input is `Split.prompt_payload()`.
* Providers are swappable and every response is cached on disk by condition,
  so a re-run costs nothing and a model can be added without touching metrics.

Note on sampling: current Claude models reject `temperature`; depth is set with
`output_config.effort` instead. Effort is part of the experimental condition and
is recorded with every response.
"""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from . import config as C

PROMPT_VERSION = "v1"
RESPONSE_DIR = C.DATA_DIR / "responses"
DEFAULT_N_SUGGESTIONS = 5

# --- the prompt ---------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a medicinal chemist working on a lead optimization campaign. "
    "You propose specific structural modifications to a compound and explain "
    "the structure-activity reasoning behind each one. You are precise about "
    "chemistry: every structure you propose must be a valid, synthetically "
    "reasonable molecule, and every rationale must be one you would defend in "
    "a project meeting."
)

TASK_IMPLICIT = (
    "Propose {n} distinct structural modifications to this compound that you "
    "expect to increase its potency against the stated target.\n\n"
    "For each proposal give the complete SMILES of the modified molecule, name "
    "the change you made, and explain the structure-activity reasoning for why "
    "you expect it to help."
)

TASK_EXPLICIT = TASK_IMPLICIT + (
    "\n\nThis is lead optimization, not scaffold hopping: retain the core "
    "scaffold of the starting compound and propose close analogs."
)

PROMPT_VARIANTS = {"implicit": TASK_IMPLICIT, "explicit": TASK_EXPLICIT}

# Every arm gets the same system prompt, because changing it per model would
# confound the comparison. The exception is a deliberate robustness check: some
# chemistry-tuned models ship a bespoke system prompt and may be fragile without
# it, so their own wording can be run as a separate, separately-cached condition
# -- never as a substitute for the shared one.
SYSTEM_VARIANTS: dict[str, str] = {"default": SYSTEM_PROMPT}


def build_user_prompt(payload: dict, n: int = DEFAULT_N_SUGGESTIONS,
                      variant: str = "implicit") -> str:
    """Assemble the task text. `payload` must come from Split.prompt_payload()."""
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant: {variant}")
    task = PROMPT_VARIANTS[variant].format(n=n)
    return (
        f"Target protein: {payload['target_protein']} "
        f"(ChEMBL {payload['target_chembl_id']})\n"
        f"Assay endpoint: {payload['assay_endpoint']}\n\n"
        f"Starting compound (SMILES): {payload['lead_smiles']}\n"
        f"Measured potency of the starting compound: "
        f"p{payload['assay_endpoint']} = {payload['lead_pchembl']}\n\n"
        f"{task}"
    )


SUGGESTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "string",
                        "description": "Complete SMILES of the modified molecule.",
                    },
                    "modification": {
                        "type": "string",
                        "description": "Short name of the structural change, "
                                       "e.g. 'replace phenyl with 3-pyridyl'.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "The structure-activity reasoning for why "
                                       "this change should increase potency.",
                    },
                    "expected_fold_improvement": {
                        "type": "number",
                        "description": "Expected potency gain as a fold change "
                                       "(1 = no change, 10 = ten-fold more potent).",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                },
                "required": ["smiles", "modification", "rationale",
                             "expected_fold_improvement", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["suggestions"],
    "additionalProperties": False,
}


# --- records ------------------------------------------------------------------

@dataclass
class Suggestion:
    smiles: str
    modification: str
    rationale: str
    expected_fold_improvement: float | None = None
    confidence: str = ""
    index: int = 0


@dataclass
class LLMRun:
    series_id: str
    strategy: str
    year_stratum: str
    target_pref_name: str
    provider: str
    model: str
    prompt_version: str
    prompt_variant: str
    effort: str
    n_requested: int
    system_prompt: str
    user_prompt: str
    # defaulted so responses cached before this field existed still deserialize
    system_variant: str = "default"
    suggestions: list[Suggestion] = field(default_factory=list)
    raw_text: str = ""
    usage: dict = field(default_factory=dict)
    stop_reason: str = ""
    error: str = ""
    elapsed_s: float = 0.0
    run_index: int = 0
    timestamp: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["n_suggestions"] = len(self.suggestions)
        return d


# --- providers ----------------------------------------------------------------

class Provider(ABC):
    name: str = "abstract"

    @abstractmethod
    def complete(self, system: str, user: str, model: str, max_tokens: int,
                 effort: str, schema: dict) -> tuple[str, dict, str]:
        """Return (text, usage, stop_reason)."""


class AnthropicProvider(Provider):
    name = "anthropic"
    DEFAULT_MODEL = "claude-opus-5-5"

    def __init__(self, client=None):
        import anthropic
        self._anthropic = anthropic
        self.client = client or anthropic.Anthropic()

    def complete(self, system: str, user: str, model: str, max_tokens: int,
                 effort: str, schema: dict) -> tuple[str, dict, str]:
        resp = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort,
                           "format": {"type": "json_schema", "schema": schema}},
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        usage = {
            "input_tokens": getattr(resp.usage, "input_tokens", None),
            "output_tokens": getattr(resp.usage, "output_tokens", None),
            "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", None),
        }
        return text, usage, resp.stop_reason or ""


class OpenRouterProvider(Provider):
    """Open-weight models through OpenRouter's OpenAI-compatible endpoint.

    This is the practical route to a large open model without local hardware.
    The strongest open weights (Kimi K3, GLM-5.2, DeepSeek V4) need a data
    centre; a 16 GB laptop tops out around 14B, which is barely a step up from
    Gemma 3 12B. OpenRouter's free tier, by contrast, serves models in the
    hundreds of billions of parameters at no cost, and a full panel run is 32
    requests -- inside the free daily allowance at the time of writing.

    Free-tier limits and the model list change often; check the live catalogue
    before a run rather than trusting the IDs below. Free tiers may also train
    on submitted data, which is harmless here only because every prompt is
    built from published ChEMBL records.
    """

    name = "openrouter"
    DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
    ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

    # Large open-weight models seen on the free tier. Verify before relying on them.
    FREE_MODELS = (
        "nvidia/nemotron-3-ultra-550b-a55b:free",      # 550B total / 55B active
        "thinkingmachines/inkling:free",               # 975B / 41B
        "dots-studio/dots-3-note-preview:free",        # 280B / 16B
        "nvidia/nemotron-3-super-120b-a12b:free",      # 120B / 12B
    )

    def __init__(self, api_key: str | None = None, endpoint: str | None = None,
                 temperature: float = 0.0, timeout: int = 600):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. A free account at openrouter.ai "
                "issues a key without payment details.")
        self.endpoint = endpoint or self.ENDPOINT
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, system: str, user: str, model: str, max_tokens: int,
                 effort: str, schema: dict) -> tuple[str, dict, str]:
        import requests

        body = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": min(max_tokens, 8192),
            "temperature": self.temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "suggestions", "strict": True,
                                "schema": schema},
            },
        }
        r = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json=body, timeout=self.timeout)
        r.raise_for_status()
        payload = r.json()
        if "error" in payload and not payload.get("choices"):
            raise RuntimeError(f"openrouter: {payload['error']}")
        choice = payload["choices"][0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = {
            "input_tokens": (payload.get("usage") or {}).get("prompt_tokens"),
            "output_tokens": (payload.get("usage") or {}).get("completion_tokens"),
            "served_by": payload.get("provider"),
        }
        return text, usage, choice.get("finish_reason") or ""


class OllamaProvider(Provider):
    """A locally served open-weight model, through ollama's HTTP API.

    Free to run, so it is the natural first real arm: the premise under test is
    that these systems fail on hard lead optimization, and a small local model
    will fail legibly, populating the failure taxonomy at no cost.

    What it cannot do is carry the headline claim. A 12B model's failure is
    evidence about 12B models; "LLMs cannot do lead optimization" needs a
    frontier model. Treat this as the low-capability end of the curve.

    Unlike current Claude models, local models do accept `temperature`, and it
    is pinned to 0 here so a re-run reproduces the same suggestions.
    """

    name = "ollama"
    DEFAULT_MODEL = "gemma3:12b"
    ENDPOINT = "http://localhost:11434/api/chat"

    def __init__(self, endpoint: str | None = None, num_ctx: int = 8192,
                 temperature: float = 0.0, timeout: int = 300,
                 num_predict: int = 4096):
        self.endpoint = endpoint or self.ENDPOINT
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout          # hard wall-clock deadline, enforced below
        self.num_predict = num_predict

    def complete(self, system: str, user: str, model: str, max_tokens: int,
                 effort: str, schema: dict) -> tuple[str, dict, str]:
        import requests

        # Streamed, with a wall-clock deadline checked between chunks.
        # A non-streaming call relied on requests' socket timeout, which is an
        # idle timer, not a total one: a run against ChemLLM-7B sat on a single
        # split for eight hours because the connection never went quiet long
        # enough to trip it. A deadline has to be enforced client-side.
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": True,
            "format": schema,          # ollama enforces the JSON schema
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": min(max_tokens, self.num_predict),
            },
        }
        deadline = time.time() + self.timeout
        parts: list[str] = []
        usage: dict = {}
        done_reason = ""

        with requests.post(self.endpoint, json=body, stream=True,
                           timeout=(10, 120)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if time.time() > deadline:
                    raise TimeoutError(
                        f"ollama exceeded its {self.timeout}s deadline "
                        f"after {len(''.join(parts))} chars")
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise RuntimeError(f"ollama: {chunk['error']}")
                piece = (chunk.get("message") or {}).get("content", "")
                if piece:
                    parts.append(piece)
                if chunk.get("done"):
                    usage = {
                        "input_tokens": chunk.get("prompt_eval_count"),
                        "output_tokens": chunk.get("eval_count"),
                        "total_duration_s": round(
                            (chunk.get("total_duration") or 0) / 1e9, 1),
                    }
                    done_reason = chunk.get("done_reason") or ""
                    break
        return "".join(parts), usage, done_reason


class MockProvider(Provider):
    """Deterministic stand-in: lets the pipeline and metrics be exercised with
    no network, no key and no spend. It echoes the lead back with trivial
    decorations, which is also a useful floor for the metrics -- a scoring bug
    that credits this as good reasoning is visible immediately."""

    name = "mock"
    DEFAULT_MODEL = "mock-echo"

    def __init__(self, n: int = DEFAULT_N_SUGGESTIONS):
        self.n = n
        self.calls = 0

    def complete(self, system: str, user: str, model: str, max_tokens: int,
                 effort: str, schema: dict) -> tuple[str, dict, str]:
        self.calls += 1
        m = re.search(r"Starting compound \(SMILES\): (\S+)", user)
        lead = m.group(1) if m else "c1ccccc1"
        tweaks = [("F", "add fluorine"), ("Cl", "add chlorine"), ("C", "add methyl"),
                  ("O", "add hydroxyl"), ("N", "add amine")]
        out = []
        for i in range(self.n):
            halo, label = tweaks[i % len(tweaks)]
            out.append({
                "smiles": lead + halo,
                "modification": label,
                "rationale": f"Generic placeholder rationale {i}: "
                             f"{label} should improve binding.",
                "expected_fold_improvement": float(2 + i),
                "confidence": "medium",
            })
        return json.dumps({"suggestions": out}), {"input_tokens": 0, "output_tokens": 0}, "end_turn"


def get_provider(name: str, **kwargs) -> Provider:
    if name == "anthropic":
        return AnthropicProvider(**kwargs)
    if name == "ollama":
        return OllamaProvider(**kwargs)
    if name == "openrouter":
        return OpenRouterProvider(**kwargs)
    if name == "mock":
        return MockProvider(**kwargs)
    raise ValueError(f"unknown provider: {name}")


# --- parsing ------------------------------------------------------------------

def parse_suggestions(text: str) -> list[Suggestion]:
    """Read the schema-constrained JSON. Tolerates a code fence, nothing more --
    silently rescuing malformed output would hide a real failure mode."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
        cleaned = re.sub(r"\n```\s*$", "", cleaned)
    payload = json.loads(cleaned)
    items = payload["suggestions"] if isinstance(payload, dict) else payload
    out = []
    for i, item in enumerate(items):
        out.append(Suggestion(
            smiles=str(item.get("smiles", "")).strip(),
            modification=str(item.get("modification", "")).strip(),
            rationale=str(item.get("rationale", "")).strip(),
            expected_fold_improvement=item.get("expected_fold_improvement"),
            confidence=str(item.get("confidence", "")).strip(),
            index=i,
        ))
    return out


# --- the loop -----------------------------------------------------------------

def _run_path(model: str, variant: str, effort: str, series_id: str,
              strategy: str, run_index: int, system_variant: str = "default"):
    tag = f"{PROMPT_VERSION}_{variant}_{effort}"
    if system_variant != "default":
        tag += f"_sys-{system_variant}"
    d = RESPONSE_DIR / model.replace("/", "_") / tag
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{series_id}__{strategy}__r{run_index}.json"


def query_split(split: dict, provider: Provider, model: str | None = None,
                n: int = DEFAULT_N_SUGGESTIONS, variant: str = "implicit",
                effort: str = "medium", max_tokens: int = 16000,
                run_index: int = 0, refresh: bool = False,
                system_variant: str = "default") -> LLMRun:
    """Ask one model about one split. Cached on disk by condition.

    `split` is the dict written by Split.write(); only its prompt-facing fields
    are read, so a leak would have to be introduced in Split itself.
    """
    model = model or getattr(provider, "DEFAULT_MODEL", "unknown")
    series_id, strategy = split["series_id"], split["strategy"]
    if system_variant not in SYSTEM_VARIANTS:
        raise ValueError(f"unknown system variant: {system_variant}")
    system = SYSTEM_VARIANTS[system_variant]
    path = _run_path(model, variant, effort, series_id, strategy, run_index,
                     system_variant=system_variant)
    if path.exists() and not refresh:
        raw = json.loads(path.read_text())
        raw["suggestions"] = [Suggestion(**s) for s in raw.pop("suggestions", [])]
        raw.pop("n_suggestions", None)
        return LLMRun(**raw)

    payload = {
        "series_id": series_id,
        "strategy": strategy,
        "target_protein": split["target_pref_name"],
        "target_chembl_id": split.get("target_chembl_id", ""),
        "assay_endpoint": split["standard_type"],
        "lead_smiles": split["lead"]["smiles"],
        "lead_pchembl": split["lead_pchembl"],
    }
    user = build_user_prompt(payload, n=n, variant=variant)

    # Control arms (oracle, neighbour) are deliberately handed the split, which
    # no real model receives. The hook is opt-in: a provider without
    # `bind_split` -- every real one -- sees nothing but the prompt above.
    if hasattr(provider, "bind_split"):
        provider.bind_split(split)

    run = LLMRun(
        series_id=series_id, strategy=strategy,
        year_stratum=split.get("year_stratum", ""),
        target_pref_name=split["target_pref_name"],
        provider=provider.name, model=model,
        prompt_version=PROMPT_VERSION, prompt_variant=variant,
        system_variant=system_variant, effort=effort,
        n_requested=n, system_prompt=system, user_prompt=user,
        run_index=run_index,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    t0 = time.time()
    try:
        text, usage, stop = provider.complete(
            system, user, model, max_tokens, effort, SUGGESTION_SCHEMA)
        run.raw_text, run.usage, run.stop_reason = text, usage, stop
        run.suggestions = parse_suggestions(text)
    except Exception as exc:
        run.error = f"{type(exc).__name__}: {exc}"
    run.elapsed_s = round(time.time() - t0, 2)

    path.write_text(json.dumps(run.to_dict(), indent=2))
    return run


def load_splits(strategy: str = "window") -> list[dict]:
    """Every split of one strategy that Task 2 wrote to disk."""
    d = C.DATA_DIR / "splits"
    out = []
    for p in sorted(d.glob(f"*__{strategy}.json")):
        s = json.loads(p.read_text())
        if s.get("usable"):
            out.append(s)
    return out


def run_batch(provider: Provider, model: str | None = None,
              strategy: str = "window", **kwargs) -> list[LLMRun]:
    runs = []
    splits = load_splits(strategy)
    for i, split in enumerate(splits, 1):
        run = query_split(split, provider, model=model, **kwargs)
        flag = "ERR" if run.error else f"{len(run.suggestions)} sug"
        print(f"  [{i}/{len(splits)}] {split['series_id']:<26}{flag}", flush=True)
        runs.append(run)
    return runs
