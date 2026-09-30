"""The provider chain in services/llm.py: Ollama → Groq → template text.

Providers are replaced with stubs, so these run offline and never spend a Groq
request. What is pinned is the routing: who is asked, in what order, what gets
credited, and that a failure anywhere lands on text rather than an exception.
"""

import pytest

from services import llm


@pytest.fixture
def chain(monkeypatch):
    """Both providers stubbed. Tests flip `state` to take either one down."""
    state = {"ollama": True, "groq": True, "calls": []}

    def ollama_status(model=None, fresh=False):
        return ({"ok": True, "model": model or llm.DEFAULT_MODEL} if state["ollama"] else
                {"ok": False, "model": model or llm.DEFAULT_MODEL, "reason": "ConnectError: refused"})

    def ollama_chat(system, user, model, stream):
        state["calls"].append("ollama")
        return iter(["from ", "ollama"]) if stream else "from ollama"

    def groq_chat(system, user, stream):
        state["calls"].append("groq")
        if not state["groq"]:
            raise llm.LLMUnavailable("HTTP 429 from Groq")
        return iter(["from ", "groq"]) if stream else "from groq"

    monkeypatch.setattr(llm, "PROVIDER_ORDER", ["ollama", "groq"])
    monkeypatch.setattr(llm, "_ollama_status", ollama_status)
    monkeypatch.setattr(llm, "_ollama_chat", ollama_chat)
    monkeypatch.setattr(llm, "_groq_chat", groq_chat)
    return state


FACTS = {"display_name": "Blue Jay", "scientific_name": "Cyanocitta cristata",
         "size_cm": "25-30 cm", "field_marks": ["Blue crest"], "habitat": "Oak woods"}
RESULT = {"species": {"display_name": "Blue Jay"}, "info": FACTS,
          "confidence": 0.93, "confidence_band": "high"}


def test_ollama_answers_first_when_it_is_up(chain):
    out = llm.narrate(RESULT)
    assert out["generated"] is True and out["text"] == "from ollama"
    assert out["model"] == llm.DEFAULT_MODEL
    assert chain["calls"] == ["ollama"]


def test_groq_answers_when_ollama_is_down(chain):
    chain["ollama"] = False
    out = llm.narrate(RESULT)
    assert out["generated"] is True and out["text"] == "from groq"
    assert "Groq" in out["model"]
    assert chain["calls"] == ["groq"]


def test_template_text_when_every_provider_fails(chain):
    chain["ollama"] = False
    chain["groq"] = False
    narration = llm.narrate(RESULT)
    assert narration["generated"] is False and "Blue Jay" in narration["text"]

    answer = llm.chat("what does it eat?", "Species: Blue Jay\nDiet: acorns")
    assert answer["generated"] is False
    assert "field guide says" in answer["text"] and "acorns" in answer["text"]

    comparison = llm.compare({**FACTS, "folder": "a"},
                             {**FACTS, "folder": "b", "display_name": "Steller Jay"})
    assert comparison["generated"] is False and comparison["summary"]


def test_stream_switches_to_groq_before_the_first_token(chain):
    chain["ollama"] = False
    assert "".join(llm.chat("hi", "ctx", stream=True)) == "from groq"


def test_stream_moves_on_when_ollama_breaks_before_its_first_token(chain, monkeypatch):
    def broken(system, user, model, stream):
        def gen():
            raise ConnectionError("daemon went away")
            yield  # pragma: no cover
        return gen()
    monkeypatch.setattr(llm, "_ollama_chat", broken)
    assert "".join(llm.chat("hi", "ctx", stream=True)) == "from groq"


def test_stream_raises_only_when_every_provider_fails(chain):
    chain["ollama"] = False
    chain["groq"] = False
    with pytest.raises(llm.LLMUnavailable):
        list(llm.chat("hi", "ctx", stream=True))


def test_groq_is_skipped_without_a_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(llm.LLMUnavailable, match="GROQ_API_KEY"):
        llm._groq_chat("s", "u", stream=False)
    assert llm._groq_status()["ok"] is False


def test_available_keeps_the_old_fields(chain, monkeypatch):
    monkeypatch.setattr(llm, "_groq_status", lambda fresh=False: {"ok": False, "model": "m", "reason": "no key"})
    status = llm.available()
    assert status["ok"] is True and status["provider"] == "ollama"
    assert status["model"] == llm.DEFAULT_MODEL
    assert set(status["providers"]) == {"ollama", "groq"}


def test_reasoning_models_are_asked_for_prose_only(monkeypatch):
    """gpt-oss must not return its reasoning, and gets token headroom for it."""
    sent = {}

    class Res:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": "A Blue Jay."}}]}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.update(json)
        return Res()

    import httpx
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(llm, "GROQ_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setattr(httpx, "post", fake_post)
    assert llm._groq_chat("s", "u", stream=False) == "A Blue Jay."
    assert sent["include_reasoning"] is False and sent["reasoning_effort"] == "low"
    assert sent["max_tokens"] > llm.OPTIONS["num_predict"]
