"""LLM services — narration, grounded chat, species comparison.

Three levels, tried in order, so a feature never fails outright:

  1. **Ollama**, on localhost — the default qwen3:8b is the same model that
     generated data/species_kb.json, so the prose in the app and the prose in the
     knowledge base come from one voice. No key, no network egress.
  2. **Groq**, when GROQ_API_KEY is set — the hosted fallback for deployments
     where no Ollama daemon runs. Same prompts, same facts-only grounding.
  3. **Template text** built from the knowledge base, when neither answers.

The order comes from BIRD_LLM_ORDER (default "ollama,groq"). Ollama's
availability is probed with a short timeout and cached briefly, so a server
without the daemon skips straight to Groq instead of waiting out a timeout.

**Facts come from the KB, prose comes from the model.** Sizes, IUCN status,
field marks and scientific names are passed in and echoed, never invented. The
model's job is to make a paragraph out of them read like a field guide. Every
response is tagged with `generated` so the UI can label machine-written text —
the same honesty rule the knowledge base already follows.
"""

import json
import os
import time

DEFAULT_MODEL = os.environ.get("BIRD_LLM_MODEL", "qwen3:8b")
TIMEOUT_SEC = float(os.environ.get("BIRD_LLM_TIMEOUT", "60"))
KEEP_ALIVE = os.environ.get("BIRD_LLM_KEEP_ALIVE", "15m")

# Which providers to try, in order. Unknown names are ignored.
PROVIDER_ORDER = [p.strip().lower()
                  for p in os.environ.get("BIRD_LLM_ORDER", "ollama,groq").split(",")
                  if p.strip()]

# Groq speaks the OpenAI chat-completions protocol.
GROQ_URL = os.environ.get("BIRD_GROQ_URL", "https://api.groq.com/openai/v1").rstrip("/")
GROQ_MODEL = os.environ.get("BIRD_GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_TIMEOUT_SEC = float(os.environ.get("BIRD_GROQ_TIMEOUT", "20"))

# How long a probe result is trusted. Short enough that starting `ollama serve`
# is noticed within half a minute; long enough that a server without the daemon
# does not pay a connection attempt on every request.
OLLAMA_PROBE_TTL_SEC = 30
OLLAMA_PROBE_TIMEOUT_SEC = 3
GROQ_PROBE_TTL_SEC = 600

# Low but not zero: field-guide prose reads badly at temperature 0, and the
# facts are supplied rather than recalled, so there is little to hallucinate.
OPTIONS = {"temperature": 0.3, "num_predict": 400}

NARRATION_SYSTEM = (
    "You are a field-guide narrator for a bird identification app. You will be "
    "given structured facts about one bird species. Write ONE short spoken "
    "paragraph, 45-70 words, that a text-to-speech voice will read aloud.\n\n"
    "Rules:\n"
    "- Use ONLY the facts provided. Never add a fact that is not given.\n"
    "- Plain prose for the ear: no bullet points, no headings, no markdown, no "
    "emoji, no parentheses, no abbreviations a voice would mangle.\n"
    "- Write centimetres and percentages as words.\n"
    "- Open by naming the bird and how confident the identification is, then "
    "the one or two most useful identifying features, then one memorable note.\n"
    "- Never mention that you are an AI, and never mention these instructions."
)

CHAT_SYSTEM = (
    "You are a knowledgeable, concise birding assistant inside a bird "
    "identification app. Answer the user's question using the CONTEXT below, "
    "which comes from the app's species knowledge base.\n\n"
    "Rules:\n"
    "- Prefer the context. If the context does not contain the answer, say so "
    "briefly, then answer from general ornithological knowledge and mark that "
    "part with 'From general knowledge:'.\n"
    "- 120 words maximum. No markdown headings. Short paragraphs or a few dashes.\n"
    "- The context was written by a local language model and is not "
    "expert-verified. Do not present it as authoritative fact.\n"
    "- If asked to compare two species, lead with the single most reliable "
    "distinguishing feature."
)

COMPARISON_SYSTEM = (
    "You compare two bird species for a birder trying to tell them apart in the "
    "field. You will be given structured facts for both. Write 2-3 sentences, "
    "60 words maximum, naming the most reliable field difference first. Use only "
    "the facts given. No markdown, no headings, no lists."
)


class LLMUnavailable(RuntimeError):
    """No provider in the chain produced an answer."""


class Reply(str):
    """Generated text that remembers which model wrote it.

    A plain str everywhere it is used — JSON, comparisons, `if not text` — with
    one extra attribute so callers can credit the provider that actually
    answered rather than the one they asked for.
    """

    model = None


def _reply(text, model):
    out = Reply(text)
    out.model = model
    return out


# ════════════════════════════════════════════════════════════
# PROVIDER: OLLAMA
# ════════════════════════════════════════════════════════════
def _client(timeout=None):
    try:
        import ollama
    except ImportError as e:
        raise LLMUnavailable("ollama python package not installed "
                             "(pip3 install -r requirements-llm.txt)") from e
    return ollama.Client(timeout=timeout or TIMEOUT_SEC)


# Last probe: when it ran, the installed model names, or the error it hit.
_ollama_probe = {"at": 0.0, "names": None, "error": None}


def _probe_ollama(fresh=False):
    """(names, error) from Ollama's model list, cached for OLLAMA_PROBE_TTL_SEC."""
    now = time.monotonic()
    if not fresh and _ollama_probe["at"] and now - _ollama_probe["at"] < OLLAMA_PROBE_TTL_SEC:
        return _ollama_probe["names"], _ollama_probe["error"]
    names, error = None, None
    try:
        listing = _client(timeout=OLLAMA_PROBE_TIMEOUT_SEC).list()
        names = []
        for m in getattr(listing, "models", None) or listing.get("models", []):
            name = getattr(m, "model", None) or (m.get("model") if isinstance(m, dict) else None)
            if name:
                names.append(name)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    _ollama_probe.update(at=now, names=names, error=error)
    return names, error


def _forget_ollama():
    _ollama_probe["at"] = 0.0


def _has_model(names, model):
    # Ollama reports "qwen3:8b"; accept a bare "qwen3" request against it.
    return model in names or any(n.split(":")[0] == model.split(":")[0] for n in names)


def _ollama_status(model=None, fresh=False):
    model = model or DEFAULT_MODEL
    names, error = _probe_ollama(fresh)
    if error:
        return {"ok": False, "model": model, "reason": error}
    if _has_model(names, model):
        return {"ok": True, "model": model, "installed": names}
    return {"ok": False, "model": model,
            "reason": f"model not pulled (have: {', '.join(names) or 'none'})",
            "installed": names}


def _ollama_chat(system, user, model, stream):
    resp = _client().chat(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        # qwen3 emits <think> blocks by default. They are slow and useless here.
        think=False,
        options=OPTIONS,
        # Hold the weights in memory between calls. An 8B model costs ~8s to
        # load, which is most of the latency on a cold narration request and
        # would otherwise be paid again on every follow-up question.
        keep_alive=KEEP_ALIVE,
        stream=stream,
    )
    if stream:
        return (chunk["message"]["content"] for chunk in resp)
    return (resp["message"]["content"] or "").strip()


# ════════════════════════════════════════════════════════════
# PROVIDER: GROQ
# ════════════════════════════════════════════════════════════
def _groq_key():
    # Read per call, so a key added to the environment after import is honoured.
    return os.environ.get("GROQ_API_KEY", "").strip()


_groq_probe = {"at": 0.0, "status": None}


def _groq_status(fresh=False):
    """Is Groq usable? Key present, and — checked at most every ten minutes —
    accepted. Never raises and never includes the key in a reason."""
    if not _groq_key():
        return {"ok": False, "model": GROQ_MODEL, "reason": "GROQ_API_KEY not set"}
    now = time.monotonic()
    if not fresh and _groq_probe["status"] and now - _groq_probe["at"] < GROQ_PROBE_TTL_SEC:
        return _groq_probe["status"]
    try:
        import httpx
        res = httpx.get(f"{GROQ_URL}/models", timeout=5,
                        headers={"Authorization": f"Bearer {_groq_key()}"})
        if res.status_code != 200:
            status = {"ok": False, "model": GROQ_MODEL, "reason": _groq_error(res)}
        elif GROQ_MODEL not in {m.get("id") for m in res.json().get("data", [])}:
            status = {"ok": False, "model": GROQ_MODEL,
                      "reason": f"{GROQ_MODEL} is not available to this Groq key "
                                f"(set BIRD_GROQ_MODEL)"}
        else:
            status = {"ok": True, "model": GROQ_MODEL}
    except Exception as e:
        status = {"ok": False, "model": GROQ_MODEL, "reason": f"{type(e).__name__}"}
    _groq_probe.update(at=now, status=status)
    return status


def _groq_reasoning(model):
    """Request extras for reasoning models, keyed by model family.

    These models think before they answer and the thinking counts against
    max_tokens. Kept to a minimum and never returned — the reply must be the
    prose alone, exactly as with Ollama's think=False. The token headroom keeps
    a long think from starving the answer."""
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low", "include_reasoning": False}, 200
    if model.startswith("qwen/"):
        return {"reasoning_effort": "none"}, 0
    return {}, 0


def _groq_error(res):
    """A short, key-free description of a failed Groq response."""
    try:
        detail = res.json().get("error", {}).get("message", "")
    except Exception:
        detail = ""
    return f"HTTP {res.status_code} from Groq" + (f": {detail[:120]}" if detail else "")


def _groq_chat(system, user, stream):
    key = _groq_key()
    if not key:
        raise LLMUnavailable("GROQ_API_KEY not set")
    import httpx

    extras, headroom = _groq_reasoning(GROQ_MODEL)
    body = {"model": GROQ_MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": OPTIONS["temperature"],
            "max_tokens": OPTIONS["num_predict"] + headroom,
            "stream": stream,
            **extras}
    headers = {"Authorization": f"Bearer {key}"}
    url = f"{GROQ_URL}/chat/completions"

    if not stream:
        res = httpx.post(url, json=body, headers=headers, timeout=GROQ_TIMEOUT_SEC)
        if res.status_code != 200:
            raise LLMUnavailable(_groq_error(res))
        return (res.json()["choices"][0]["message"]["content"] or "").strip()

    def pieces():
        with httpx.stream("POST", url, json=body, headers=headers,
                          timeout=GROQ_TIMEOUT_SEC) as res:
            if res.status_code != 200:
                res.read()
                raise LLMUnavailable(_groq_error(res))
            for line in res.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                delta = (json.loads(data).get("choices") or [{}])[0].get("delta") or {}
                if delta.get("content"):
                    yield delta["content"]
    return pieces()


# ════════════════════════════════════════════════════════════
# THE CHAIN
# ════════════════════════════════════════════════════════════
def _groq_label():
    return f"{GROQ_MODEL} via Groq"


def _first_then_rest(pieces):
    """Pull the first non-empty piece now, so a provider that cannot answer
    fails here — before anything reaches the user — and the chain can move on."""
    it = iter(pieces)
    for first in it:
        if first:
            break
    else:
        raise LLMUnavailable("empty response")

    def stream():
        yield first
        yield from it
    return stream()


def _chat(system, user, model=None, stream=False):
    """Ask each provider in turn; the first to answer wins.

    Returns a Reply (a str carrying `.model`), or with `stream=True` a generator
    of text pieces. Raises LLMUnavailable when every provider fails, which is
    the signal for the caller's template text. `model` applies to Ollama only —
    the Groq model is fixed by configuration, not by whoever sent the request.
    """
    failures = []
    for provider in PROVIDER_ORDER:
        try:
            if provider == "ollama":
                status = _ollama_status(model)
                if not status["ok"]:
                    raise LLMUnavailable(status["reason"])
                label = model or DEFAULT_MODEL
                out = _ollama_chat(system, user, label, stream)
            elif provider == "groq":
                label = _groq_label()
                out = _groq_chat(system, user, stream)
            else:
                continue
            if stream:
                return _first_then_rest(out)
            if not out:
                raise LLMUnavailable("empty response")
            return _reply(out, label)
        except Exception as e:
            if provider == "ollama":
                _forget_ollama()
            failures.append(f"{provider}: {type(e).__name__}: {e}")
    raise LLMUnavailable("; ".join(failures) or "no language model configured")


def available(model=None):
    """Which provider would answer right now? Never raises.

    `ok`, `model`, `reason` and `installed` keep their old meaning for the first
    provider that is up; `provider` names it, and `providers` has the detail for
    each link in the chain.
    """
    providers = {}
    for provider in PROVIDER_ORDER:
        if provider == "ollama":
            providers["ollama"] = _ollama_status(model, fresh=True)
        elif provider == "groq":
            groq = _groq_status()
            providers["groq"] = {**groq, "model": _groq_label()} if groq["ok"] else groq

    active = next((name for name, st in providers.items() if st["ok"]), None)
    if active:
        out = {**providers[active], "provider": active}
    else:
        reasons = "; ".join(f"{n}: {st.get('reason')}" for n, st in providers.items())
        out = {"ok": False, "provider": None,
               "model": (providers.get("ollama") or {}).get("model") or model or DEFAULT_MODEL,
               "reason": reasons or "no language model configured"}
    if "installed" in (providers.get("ollama") or {}):
        out["installed"] = providers["ollama"]["installed"]
    out["providers"] = providers
    return out


# ════════════════════════════════════════════════════════════
# FACT SHEETS — what the model is allowed to know
# ════════════════════════════════════════════════════════════
def _facts(info, name=None, confidence=None, band=None):
    """Flatten a species_detail dict into labelled lines for the prompt."""
    lines = [f"Species: {name or info.get('display_name')}"]
    if confidence is not None:
        pct = f"{confidence * 100:.0f} percent"
        lines.append(f"Identification confidence: {pct} ({band or ''} confidence)".strip())
    for label, key in (("Scientific name", "scientific_name"),
                       ("Family", "family"),
                       ("Order", "order"),
                       ("Size", "size_cm"),
                       ("Diet", "diet"),
                       ("Habitat", "habitat"),
                       ("Range", "range_description"),
                       ("Migration", "migration"),
                       ("Conservation status", "conservation_status"),
                       ("Notable fact", "fun_fact")):
        value = info.get(key)
        if value:
            lines.append(f"{label}: {value}")
    marks = info.get("field_marks") or []
    if marks:
        lines.append("Field marks: " + "; ".join(marks[:4]))
    similar = info.get("similar_species") or []
    if similar:
        lines.append("Similar species: " + "; ".join(
            f"{s['name']} — {s['how_to_distinguish']}" for s in similar[:2]))
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# NARRATION
# ════════════════════════════════════════════════════════════
def _template_narration(info, name, confidence, band):
    """Deterministic fallback narration. Reads acceptably; just less fluid."""
    parts = []
    if confidence is not None:
        qualifier = {"high": "This is almost certainly",
                     "moderate": "This looks like",
                     "low": "This might be"}.get(band, "This looks like")
        parts.append(f"{qualifier} a {name}, at "
                     f"{confidence * 100:.0f} percent confidence.")
    else:
        parts.append(f"This is a {name}.")
    if info.get("size_cm"):
        parts.append(f"It measures about {info['size_cm']}.")
    marks = info.get("field_marks") or []
    if marks:
        parts.append(f"Look for {marks[0].rstrip('.').lower()}.")
    if info.get("habitat"):
        parts.append(f"You would find it in {info['habitat'].rstrip('.').lower()}.")
    if info.get("fun_fact"):
        parts.append(info["fun_fact"])
    return " ".join(parts)


def narrate(result, model=None):
    """One spoken paragraph for a bird_core image or audio result.

    This is what makes the voice feature worth having. Reading the on-screen
    report aloud produces "SPECIES colon Blue Jay, FIELD MARKS colon bullet…";
    a written paragraph sounds like a person who knows birds.
    """
    answer = result.get("answer") or {}
    if answer.get("identified_by") == "verifier" and answer.get("display_name"):
        return _narrate_verified(answer, model)

    info = result.get("info") or {}
    species = result.get("species") or {}
    name = species.get("display_name") or species.get("common_name") or "this bird"
    confidence = result.get("confidence")
    band = result.get("confidence_band")

    gate = result.get("openset") or {}
    if gate.get("enabled") and not gate.get("is_bird", True):
        return {
            "text": ("I do not think this is one of the two hundred species I know. "
                     "The ranking on screen is only what the model would say if "
                     "forced to choose. Try a clearer photo of a single bird."),
            "generated": False,
            "reason": "open-set rejection",
        }

    if not info:
        return {"text": f"This appears to be a {name}, but I have no notes on it.",
                "generated": False, "reason": "no knowledge-base record"}

    fallback = _template_narration(info, name, confidence, band)
    try:
        text = _chat(NARRATION_SYSTEM, _facts(info, name, confidence, band), model)
        if not text:
            raise LLMUnavailable("empty response")
        return {"text": text, "generated": True,
                "model": getattr(text, "model", None) or model or DEFAULT_MODEL,
                "kb_source": (result.get("provenance") or {}).get("kb_source")}
    except Exception as e:
        return {"text": fallback, "generated": False,
                "reason": f"{type(e).__name__}: {e}"}


def _narrate_verified(answer, model=None):
    """Narrate the bird the open-vocabulary verifier named.

    The classifier's guess is wrong in this case by definition, so none of its
    fields are used — no percentage either, since the verifier's similarity is
    not a probability. A species inside the 200 still gets its knowledge-base
    notes; one outside has none, and the narration says so plainly rather than
    letting the model recall facts it cannot be checked against.
    """
    name = answer["display_name"]
    sci = answer.get("scientific_name")
    info = answer.get("info") or {}
    named = f"{name}, {sci}" if sci else name

    if not info:
        return {"text": (f"This is a {named}, confirmed by a second identification model. "
                         "It is outside the two hundred species in my field guide, so I "
                         "do not have notes on it yet."),
                "generated": False, "reason": "outside the knowledge base"}

    fallback = (f"This is a {named}, confirmed by a second identification model. "
                + _template_narration(info, name, None, None).split(".", 1)[-1].strip())
    try:
        text = _chat(NARRATION_SYSTEM, _facts(info, name), model)
        if not text:
            raise LLMUnavailable("empty response")
        return {"text": text, "generated": True,
                "model": getattr(text, "model", None) or model or DEFAULT_MODEL}
    except Exception as e:
        return {"text": fallback.strip(), "generated": False,
                "reason": f"{type(e).__name__}: {e}"}


# ════════════════════════════════════════════════════════════
# GROUNDED CHAT
# ════════════════════════════════════════════════════════════
# Retrieval here is lexical, not vector. For 200 short records that is not a
# compromise, it is the better tool: the chat almost always already knows which
# species is on screen, so the primary "retrieval" is a dict lookup, and
# cross-species questions are answered well by scoring shared terms. It also
# avoids pulling an embedding model and a vector store into the runtime for a
# corpus that fits in memory. chromadb stays in requirements-llm.txt for the
# offline KB work, not for this path.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "of",
    "and", "or", "to", "in", "on", "for", "with", "how", "what", "which",
    "why", "when", "where", "who", "can", "you", "i", "it", "its", "this",
    "that", "these", "those", "bird", "birds", "species", "tell", "me",
    "about", "between", "difference", "differences", "from", "they", "them",
    "my", "have", "has", "be", "there", "their", "at", "by", "as", "not",
}


def _tokens(text):
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
    return {t for t in cleaned.split() if len(t) > 2 and t not in _STOPWORDS}


def retrieve(question, catalogue, kb, limit=3):
    """Pick the KB records most likely to answer `question`.

    `catalogue` is bird_core.species_catalogue() output; `kb` is the raw record
    dict. Name matches dominate deliberately — "what does a Blue Jay eat" should
    retrieve the Blue Jay, not three records that happen to mention acorns.
    """
    q = _tokens(question)
    if not q:
        return []
    scored = []
    for row in catalogue:
        folder = row["folder"]
        name_tokens = _tokens(row["display_name"] or "")
        sci_tokens = _tokens(row["scientific_name"] or "")
        score = 4.0 * len(q & name_tokens) + 3.0 * len(q & sci_tokens)
        # A full multi-word name match is a near-certain hit.
        if name_tokens and name_tokens <= q:
            score += 6.0
        record = kb.get(folder) or {}
        body = " ".join(str(v) for k, v in record.items()
                        if not k.startswith("_") and isinstance(v, str))
        score += 0.6 * len(q & _tokens(body))
        if score > 0:
            scored.append((score, folder))
    scored.sort(reverse=True)
    return [folder for _, folder in scored[:limit]]


def build_context(folders, detail_fn):
    """Render retrieved species into the CONTEXT block for the prompt."""
    blocks = []
    for folder in folders:
        info = detail_fn(folder)
        blocks.append(_facts(info, info["display_name"]))
    return "\n\n---\n\n".join(blocks)


def template_answer(context):
    """The third level for chat: the field-guide entry itself, plainly introduced."""
    first = (context or "").split("\n\n---\n\n")[0].strip()
    if not first:
        return ("The language model isn't available right now, and the field guide "
                "has no entry that matches this question. Try naming the species.")
    return ("The language model isn't available right now, so here's what the "
            "field guide says:\n\n" + first)


def chat(question, context, history=None, model=None, stream=False):
    """Answer a birding question against retrieved context.

    With `stream=True` this returns a generator of text chunks so the dashboard
    can render tokens as they arrive. Providers are tried before the first chunk
    is yielded, so the generator raises LLMUnavailable only when every provider
    failed; the caller then shows `template_answer(context)`.
    """
    convo = ""
    for turn in (history or [])[-4:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        convo += f"{role}: {turn.get('content', '')}\n"

    prompt = (f"CONTEXT:\n{context or '(no matching species in the knowledge base)'}\n\n"
              + (f"CONVERSATION SO FAR:\n{convo}\n" if convo else "")
              + f"QUESTION: {question}")

    if stream:
        def generate():
            yield from _chat(CHAT_SYSTEM, prompt, model, stream=True)
        return generate()

    try:
        text = _chat(CHAT_SYSTEM, prompt, model)
        return {"text": text, "generated": True,
                "model": getattr(text, "model", None) or model or DEFAULT_MODEL}
    except Exception as e:
        return {"text": template_answer(context),
                "generated": False, "reason": f"{type(e).__name__}: {e}"}


# ════════════════════════════════════════════════════════════
# COMPARISON
# ════════════════════════════════════════════════════════════
# Numbers are assembled from the KB and the prose summary is generated. The
# model never gets to invent a size or an IUCN status, because those are the
# facts a birder would actually act on.
COMPARE_FIELDS = [
    ("Scientific name",     "scientific_name"),
    ("Family",              "family"),
    ("Order",               "order"),
    ("Size",                "size_cm"),
    ("Diet",                "diet"),
    ("Habitat",             "habitat"),
    ("Range",               "range_description"),
    ("Migration",           "migration"),
    ("Conservation status", "conservation_status"),
]


def compare(info_a, info_b, model=None):
    """Structured side-by-side of two species, plus a generated summary."""
    rows = []
    for label, key in COMPARE_FIELDS:
        a, b = info_a.get(key), info_b.get(key)
        if a or b:
            rows.append({"label": label, "a": a, "b": b,
                         "differs": (a or "").strip().lower() != (b or "").strip().lower()})

    # A look-alike note that names the other species is the most useful line
    # available, so surface it explicitly rather than leaving it in the table.
    def cross_note(src, other_name):
        for s in src.get("similar_species") or []:
            if other_name.lower() in (s.get("name") or "").lower():
                return s.get("how_to_distinguish")
        return None

    note_a = cross_note(info_a, info_b["display_name"])
    note_b = cross_note(info_b, info_a["display_name"])

    payload = {
        "a": {"folder": info_a["folder"], "display_name": info_a["display_name"],
              "field_marks": info_a.get("field_marks") or []},
        "b": {"folder": info_b["folder"], "display_name": info_b["display_name"],
              "field_marks": info_b.get("field_marks") or []},
        "rows": rows,
        "known_look_alike": bool(note_a or note_b),
        "how_to_distinguish": note_a or note_b,
    }

    prompt = (f"SPECIES A:\n{_facts(info_a, info_a['display_name'])}\n\n"
              f"SPECIES B:\n{_facts(info_b, info_b['display_name'])}\n\n"
              f"How does a birder tell {info_a['display_name']} from "
              f"{info_b['display_name']} in the field?")
    try:
        payload["summary"] = _chat(COMPARISON_SYSTEM, prompt, model)
        payload["generated"] = True
    except Exception as e:
        # The table is the substance; the summary is a convenience.
        differing = [r["label"].lower() for r in rows if r["differs"]][:3]
        payload["summary"] = (
            note_a or note_b or
            (f"The knowledge base has no direct look-alike note for this pair. "
             f"They differ in {', '.join(differing)}." if differing else
             "The knowledge base has little to separate these two.")
        )
        payload["generated"] = False
        payload["reason"] = f"{type(e).__name__}: {e}"
    return payload
