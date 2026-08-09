"""Local LLM services — narration, grounded chat, species comparison.

Runs against Ollama on localhost. No API key, no network egress, no cloud
provider; the default model is the same qwen3:8b that generated
data/species_kb.json, so the prose in the app and the prose in the knowledge
base come from one voice.

Two rules shape everything here:

**Ollama is optional.** Every function degrades to deterministic template text
built from the knowledge base if the daemon is down, the model is missing or a
call times out. A birding app that cannot identify a bird because an LLM is not
running would be a worse app than the one this replaces.

**Facts come from the KB, prose comes from the model.** Sizes, IUCN status,
field marks and scientific names are passed in and echoed, never invented. The
model's job is to make a paragraph out of them read like a field guide. Every
response is tagged with `generated` so the UI can label machine-written text —
the same honesty rule the knowledge base already follows.
"""

import os

DEFAULT_MODEL = os.environ.get("BIRD_LLM_MODEL", "qwen3:8b")
TIMEOUT_SEC = float(os.environ.get("BIRD_LLM_TIMEOUT", "60"))
KEEP_ALIVE = os.environ.get("BIRD_LLM_KEEP_ALIVE", "15m")

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
    """Ollama is not reachable, or the model is not pulled."""


def _client():
    try:
        import ollama
    except ImportError as e:
        raise LLMUnavailable("ollama python package not installed "
                             "(pip3 install -r requirements-llm.txt)") from e
    return ollama.Client(timeout=TIMEOUT_SEC)


def available(model=None):
    """Is Ollama up with the requested model pulled? Never raises."""
    model = model or DEFAULT_MODEL
    try:
        listing = _client().list()
    except Exception as e:
        return {"ok": False, "model": model, "reason": f"{type(e).__name__}: {e}"}

    names = []
    for m in getattr(listing, "models", None) or listing.get("models", []):
        name = getattr(m, "model", None) or (m.get("model") if isinstance(m, dict) else None)
        if name:
            names.append(name)
    # Ollama reports "qwen3:8b"; accept a bare "qwen3" request against it.
    if model in names or any(n.split(":")[0] == model.split(":")[0] for n in names):
        return {"ok": True, "model": model, "installed": names}
    return {"ok": False, "model": model,
            "reason": f"model not pulled (have: {', '.join(names) or 'none'})",
            "installed": names}


def _chat(system, user, model=None, stream=False):
    resp = _client().chat(
        model=model or DEFAULT_MODEL,
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
        return resp
    return (resp["message"]["content"] or "").strip()


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
                "model": model or DEFAULT_MODEL,
                "kb_source": (result.get("provenance") or {}).get("kb_source")}
    except Exception as e:
        return {"text": fallback, "generated": False,
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


def chat(question, context, history=None, model=None, stream=False):
    """Answer a birding question against retrieved context.

    With `stream=True` this returns a generator of text chunks so the dashboard
    can render tokens as they arrive; the caller is responsible for catching
    LLMUnavailable, since a generator cannot fall back after it has started.
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
            for chunk in _chat(CHAT_SYSTEM, prompt, model, stream=True):
                piece = chunk["message"]["content"]
                if piece:
                    yield piece
        return generate()

    try:
        return {"text": _chat(CHAT_SYSTEM, prompt, model), "generated": True,
                "model": model or DEFAULT_MODEL}
    except Exception as e:
        return {"text": ("I cannot reach the local language model, so I can only "
                         "show you the knowledge-base entry rather than answer in "
                         "my own words. Start it with `ollama serve`."),
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
