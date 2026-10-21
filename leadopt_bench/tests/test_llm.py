"""Tests for the LLM evaluation loop. No network: a fake client answers.

The leakage tests are the important ones here -- if a target compound reaches
the prompt, every downstream number is meaningless and nothing else would fail.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leadopt_bench import llm
from leadopt_bench.split import window_split
from leadopt_bench.tests.test_split import _series


@pytest.fixture(autouse=True)
def isolated_responses(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "RESPONSE_DIR", tmp_path / "responses")
    yield


@pytest.fixture
def split_dict():
    sp = window_split(_series(), lead_window=(5.0, 7.0), year_stratum="old")
    assert sp is not None and sp.is_usable()
    return json.loads(json.dumps(sp.to_dict())), sp


# --- prompt construction ------------------------------------------------------

def test_prompt_contains_what_the_chemist_would_know(split_dict):
    d, sp = split_dict
    payload = sp.prompt_payload()
    text = llm.build_user_prompt(payload)
    assert sp.lead.smiles in text
    assert sp.target_pref_name in text
    assert str(sp.lead.pchembl) in text


def test_prompt_never_contains_a_target_compound(split_dict):
    d, sp = split_dict
    text = llm.build_user_prompt(sp.prompt_payload())
    for t in sp.targets:
        assert t.smiles not in text
        assert t.molecule_chembl_id not in text


def test_prompt_hides_how_much_gain_is_achievable(split_dict):
    """Telling the model '3.2 log units are available' would be a hint."""
    d, sp = split_dict
    text = llm.build_user_prompt(sp.prompt_payload())
    assert str(round(sp.max_gain, 2)) not in text
    assert f"{len(sp.targets)} compounds" not in text


def test_explicit_variant_adds_the_scaffold_instruction(split_dict):
    d, sp = split_dict
    implicit = llm.build_user_prompt(sp.prompt_payload(), variant="implicit")
    explicit = llm.build_user_prompt(sp.prompt_payload(), variant="explicit")
    assert "scaffold" not in implicit.lower()
    assert "scaffold" in explicit.lower()


def test_requested_count_appears_in_the_task(split_dict):
    d, sp = split_dict
    assert "7 distinct" in llm.build_user_prompt(sp.prompt_payload(), n=7)


def test_unknown_variant_is_rejected(split_dict):
    d, sp = split_dict
    with pytest.raises(ValueError):
        llm.build_user_prompt(sp.prompt_payload(), variant="nope")


# --- parsing ------------------------------------------------------------------

def test_parse_reads_the_schema_shape():
    text = json.dumps({"suggestions": [
        {"smiles": "CCO", "modification": "add ethyl", "rationale": "because",
         "expected_fold_improvement": 3, "confidence": "high"}]})
    out = llm.parse_suggestions(text)
    assert len(out) == 1
    assert out[0].smiles == "CCO"
    assert out[0].expected_fold_improvement == 3
    assert out[0].index == 0


def test_parse_tolerates_a_code_fence():
    inner = json.dumps({"suggestions": [
        {"smiles": "CCO", "modification": "m", "rationale": "r",
         "expected_fold_improvement": 1, "confidence": "low"}]})
    assert llm.parse_suggestions(f"```json\n{inner}\n```")[0].smiles == "CCO"


def test_parse_accepts_a_bare_list():
    text = json.dumps([{"smiles": "CCO", "modification": "m", "rationale": "r"}])
    assert len(llm.parse_suggestions(text)) == 1


def test_parse_raises_on_malformed_output():
    """A parse failure is a finding, not something to paper over."""
    with pytest.raises(Exception):
        llm.parse_suggestions("I would add a methyl group, probably.")


def test_parse_indexes_suggestions_in_order():
    text = json.dumps({"suggestions": [
        {"smiles": f"C{'C' * i}O", "modification": "m", "rationale": "r"}
        for i in range(4)]})
    assert [s.index for s in llm.parse_suggestions(text)] == [0, 1, 2, 3]


# --- the loop -----------------------------------------------------------------

def test_query_split_records_suggestions_and_the_exact_prompt(split_dict):
    d, sp = split_dict
    run = llm.query_split(d, llm.MockProvider(n=5))
    assert len(run.suggestions) == 5
    assert not run.error
    assert run.user_prompt and sp.lead.smiles in run.user_prompt
    assert run.system_prompt == llm.SYSTEM_PROMPT
    assert run.prompt_version == llm.PROMPT_VERSION


def test_query_split_carries_the_stratum_for_the_exposure_analysis(split_dict):
    d, _ = split_dict
    assert llm.query_split(d, llm.MockProvider()).year_stratum == "old"


def test_second_query_is_served_from_cache(split_dict):
    d, _ = split_dict
    p = llm.MockProvider()
    llm.query_split(d, p)
    llm.query_split(d, p)
    assert p.calls == 1, "a cached condition must not be re-queried"


def test_refresh_requeries(split_dict):
    d, _ = split_dict
    p = llm.MockProvider()
    llm.query_split(d, p)
    llm.query_split(d, p, refresh=True)
    assert p.calls == 2


def test_different_conditions_do_not_share_a_cache_entry(split_dict):
    d, _ = split_dict
    p = llm.MockProvider()
    llm.query_split(d, p, variant="implicit")
    llm.query_split(d, p, variant="explicit")
    llm.query_split(d, p, effort="high")
    llm.query_split(d, p, run_index=1)
    assert p.calls == 4


def test_cached_run_roundtrips_without_loss(split_dict):
    d, _ = split_dict
    first = llm.query_split(d, llm.MockProvider())
    again = llm.query_split(d, llm.MockProvider())
    assert [s.smiles for s in again.suggestions] == [s.smiles for s in first.suggestions]
    assert again.user_prompt == first.user_prompt


def test_provider_failure_is_recorded_not_raised(split_dict):
    d, _ = split_dict

    class Broken(llm.Provider):
        name = "broken"
        DEFAULT_MODEL = "broken-1"

        def complete(self, *a, **k):
            raise RuntimeError("upstream exploded")

    run = llm.query_split(d, Broken())
    assert run.error.startswith("RuntimeError")
    assert run.suggestions == []


def test_unparseable_response_is_recorded_as_an_error(split_dict):
    d, _ = split_dict

    class Prose(llm.Provider):
        name = "prose"
        DEFAULT_MODEL = "prose-1"

        def complete(self, *a, **k):
            return "Add a methyl group.", {}, "end_turn"

    run = llm.query_split(d, Prose())
    assert run.error
    assert run.raw_text == "Add a methyl group."


def test_get_provider_resolves_known_names_and_rejects_others():
    assert isinstance(llm.get_provider("mock"), llm.MockProvider)
    with pytest.raises(ValueError):
        llm.get_provider("gpt-whatever")


def test_schema_requires_a_rationale_per_suggestion():
    """The rationale must be structurally mandatory, not merely requested."""
    item = llm.SUGGESTION_SCHEMA["properties"]["suggestions"]["items"]
    assert "rationale" in item["required"]
    assert "smiles" in item["required"]
    assert item["additionalProperties"] is False


def test_openrouter_requires_a_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        llm.OpenRouterProvider()


def test_openrouter_reads_the_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert llm.OpenRouterProvider().api_key == "sk-or-test"


def test_openrouter_sends_schema_and_pinned_temperature(monkeypatch, split_dict):
    """Open models do accept temperature; it is pinned to 0 for reproducibility."""
    d, _ = split_dict
    captured = {}
    content = json.dumps({"suggestions": [
        {"smiles": "CCO", "modification": "m", "rationale": "r",
         "expected_fold_improvement": 2, "confidence": "low"}]})

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": content},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 22},
                    "provider": "TestProvider"}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update({"url": url, "headers": headers, "body": json})
        return FakeResp()

    import requests
    monkeypatch.setattr(requests, "post", fake_post)

    prov = llm.OpenRouterProvider(api_key="sk-or-test")
    run = llm.query_split(d, prov, model="nvidia/nemotron-3-ultra-550b-a55b:free")

    assert captured["headers"]["Authorization"] == "Bearer sk-or-test"
    assert captured["body"]["temperature"] == 0.0
    fmt = captured["body"]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["schema"] == llm.SUGGESTION_SCHEMA
    assert len(run.suggestions) == 1
    assert run.usage["input_tokens"] == 11
    assert run.usage["served_by"] == "TestProvider"


def test_openrouter_surfaces_an_api_error(monkeypatch, split_dict):
    d, _ = split_dict

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"error": {"message": "rate limited", "code": 429}}

    import requests
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: FakeResp())
    run = llm.query_split(d, llm.OpenRouterProvider(api_key="k"))
    assert "rate limited" in run.error


class FakeStream:
    """Minimal stand-in for a streamed ollama response."""

    def __init__(self, lines, on_iter=None):
        self._lines = lines
        self._on_iter = on_iter

    def raise_for_status(self):
        pass

    def iter_lines(self):
        for ln in self._lines:
            if self._on_iter:
                self._on_iter()
            yield ln

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ollama_chunks(content, done_extra=None):
    mid = json.dumps({"message": {"content": content}})
    done = {"message": {"content": ""}, "done": True, "done_reason": "stop",
            "prompt_eval_count": 5, "eval_count": 7,
            "total_duration": 2_000_000_000}
    done.update(done_extra or {})
    return [mid.encode(), json.dumps(done).encode()]


def test_ollama_pins_temperature_and_streams_with_schema(monkeypatch, split_dict):
    d, _ = split_dict
    captured = {}
    content = json.dumps({"suggestions": [
        {"smiles": "CCO", "modification": "m", "rationale": "r",
         "expected_fold_improvement": 2, "confidence": "low"}]})

    def fake_post(url, json=None, stream=None, timeout=None):
        captured.update({"url": url, "body": json, "stream": stream})
        return FakeStream(_ollama_chunks(content))

    import requests
    monkeypatch.setattr(requests, "post", fake_post)

    run = llm.query_split(d, llm.OllamaProvider(), model="gemma3:12b")
    assert captured["body"]["options"]["temperature"] == 0.0
    assert captured["body"]["format"] == llm.SUGGESTION_SCHEMA
    assert captured["body"]["stream"] is True
    assert captured["stream"] is True
    assert run.usage["output_tokens"] == 7
    assert run.usage["total_duration_s"] == 2.0
    assert len(run.suggestions) == 1


def test_ollama_reassembles_a_chunked_response(monkeypatch, split_dict):
    d, _ = split_dict
    full = json.dumps({"suggestions": [
        {"smiles": "CCO", "modification": "m", "rationale": "r",
         "expected_fold_improvement": 2, "confidence": "low"}]})
    lines = [json.dumps({"message": {"content": c}}).encode() for c in
             (full[:10], full[10:25], full[25:])]
    lines.append(json.dumps({"message": {"content": ""}, "done": True,
                             "done_reason": "stop"}).encode())

    import requests
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: FakeStream(lines))
    run = llm.query_split(d, llm.OllamaProvider())
    assert not run.error
    assert len(run.suggestions) == 1


def test_ollama_enforces_a_wall_clock_deadline(monkeypatch, split_dict):
    """Regression: a run sat on one split for eight hours. requests' timeout is
    an idle timer, so a connection that keeps trickling never trips it."""
    d, _ = split_dict
    clock = {"t": 1000.0}

    def advance():
        clock["t"] += 50.0          # each chunk "takes" 50s

    monkeypatch.setattr(llm.time, "time", lambda: clock["t"])
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeStream(
        [json.dumps({"message": {"content": "x"}}).encode()] * 50,
        on_iter=advance))

    run = llm.query_split(d, llm.OllamaProvider(timeout=120))
    assert "TimeoutError" in run.error
    assert "deadline" in run.error


def test_ollama_surfaces_a_streamed_error(monkeypatch, split_dict):
    d, _ = split_dict
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeStream(
        [json.dumps({"error": "model not found"}).encode()]))
    run = llm.query_split(d, llm.OllamaProvider())
    assert "model not found" in run.error


def test_ollama_caps_num_predict(monkeypatch, split_dict):
    d, _ = split_dict
    captured = {}

    def fake_post(url, json=None, stream=None, timeout=None):
        captured.update(json)
        return FakeStream(_ollama_chunks(json_module.dumps({"suggestions": []})))

    import json as json_module
    import requests
    monkeypatch.setattr(requests, "post", fake_post)
    llm.query_split(d, llm.OllamaProvider(num_predict=1024), max_tokens=16000)
    assert captured["options"]["num_predict"] == 1024


def test_get_provider_knows_every_backend():
    import os as _os
    assert isinstance(llm.get_provider("mock"), llm.MockProvider)
    assert isinstance(llm.get_provider("ollama"), llm.OllamaProvider)
    _os.environ["OPENROUTER_API_KEY"] = "k"
    try:
        assert isinstance(llm.get_provider("openrouter"), llm.OpenRouterProvider)
    finally:
        _os.environ.pop("OPENROUTER_API_KEY", None)


def test_anthropic_provider_sends_no_temperature(monkeypatch, split_dict):
    """Current Claude models reject `temperature`; effort is the only knob."""
    d, _ = split_dict
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)

            class R:
                content = [type("B", (), {"type": "text", "text": json.dumps(
                    {"suggestions": [{"smiles": "CCO", "modification": "m",
                                      "rationale": "r",
                                      "expected_fold_improvement": 2,
                                      "confidence": "low"}]})})()]
                usage = type("U", (), {"input_tokens": 10, "output_tokens": 20,
                                       "cache_read_input_tokens": 0})()
                stop_reason = "end_turn"
            return R()

    class FakeClient:
        messages = FakeMessages()

    prov = llm.AnthropicProvider(client=FakeClient())
    run = llm.query_split(d, prov, model="claude-opus-5-5", effort="high")

    assert "temperature" not in captured
    assert captured["output_config"]["effort"] == "high"
    assert captured["output_config"]["format"]["type"] == "json_schema"
    assert captured["model"] == "claude-opus-5-5"
    assert len(run.suggestions) == 1
    assert run.usage["input_tokens"] == 10
