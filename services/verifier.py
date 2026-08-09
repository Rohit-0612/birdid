"""Open-vocabulary bird identification — names birds outside the trained 200.

The CUB classifier can only ever answer with one of its 200 classes. The open-set
gate in bird_core can say "that is not one of my 200", but not *what it is*, which
is the question the user actually has. This module answers it.

BioCLIP (Stevens et al. 2024) is a CLIP model trained on Tree-of-Life imagery for
species identification. Because CLIP scores an image against *text*, the class list
is not baked into the weights: we supply candidate species names and it ranks them.
That makes the vocabulary a data file rather than a retraining job.

Candidates come from birdnetlib's BirdNET_GLOBAL_6K_V2.4 label file, which is
already on disk — 6,522 entries, filtered to 6,423 birds (see NON_BIRD_PATTERN).

Everything runs locally. The only network access in this project is the one-time
~400 MB weight download from HuggingFace; inference afterwards is offline.

Structured like services/llm.py: the dependency is optional, absence degrades to
`ran: false` rather than an error, and every answer carries its provenance.
"""

import hashlib
import json
import os
import re
import threading

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_DIR, "data", "verifier")
THRESHOLD_PATH = os.path.join(CACHE_DIR, "threshold.json")

MODEL_ID = os.environ.get("BIRD_VERIFIER_MODEL", "hf-hub:imageomics/bioclip")

# Prompt template. Three variants were measured on the labelled open-set birds and
# scored identically (6/10 top-1), so the most informative one is used: it gives
# BioCLIP both names, which helps where a scientific name is obscure.
PROMPT = "a photo of {sci} with common name {common}."

# BirdNET is a *sound* model, and its 6.5k labels include the non-bird things a
# microphone picks up: 92 North American frogs, toads, crickets and katydids, a few
# mammals, and 7 noise classes (Engine, Siren, Fireworks…). Left in, they get
# matched: a Bald Eagle photo was ranked as "Honey Bee" before this filter.
#
# Filtering on common names rather than genera is deliberate — the common names are
# descriptive and consistent, while a genus blocklist silently rots as labels change.
NON_BIRD_PATTERN = re.compile(
    r"""\b(toad|frog|treefrog|peeper|bullfrog|spadefoot|salamander|newt
        |cricket|katydid|cicada|grasshopper|conehead|grig|trig|locust
        |squirrel|chipmunk|bat|coyote|wolf|deer|bear
        |dog|engine|environmental|fireworks|gun|noise|siren)\b""",
    re.IGNORECASE | re.VERBOSE,
)

# Real birds whose names contain one of the words above. Without this, seventeen
# frogmouths and eight batises would be thrown out with the frogs and the bats.
BIRD_EXCEPTIONS = re.compile(r"\b(frogmouth|batis|bateleur)\b", re.IGNORECASE)

_lock = threading.Lock()
_state = {"model": None, "preprocess": None, "tokenizer": None,
          "labels": None, "text_embeddings": None, "device": None}


class VerifierUnavailable(RuntimeError):
    """open_clip/timm not installed, or the weights could not be loaded."""


# ════════════════════════════════════════════════════════════
# CANDIDATE SPECIES
# ════════════════════════════════════════════════════════════
def labels_path():
    """Locate BirdNET's label file inside the installed package."""
    import birdnetlib
    return os.path.join(os.path.dirname(birdnetlib.__file__),
                        "models", "analyzer", "BirdNET_GLOBAL_6K_V2.4_Labels.txt")


def load_labels():
    """Candidate species as [(scientific, common)], birds only.

    Note the coverage limit this inherits: BirdNET only lists species it can hear,
    so non-vocal and captive-exotic birds are absent — Struthio camelus (Common
    Ostrich) is not in the file at all. The verifier can never name a species that
    is not a candidate, so eval_verifier.py reports those cases separately from
    genuine model errors.
    """
    path = labels_path()
    with open(path) as f:
        raw = [line for line in f.read().splitlines() if line.strip()]

    pairs = []
    for line in raw:
        if "_" not in line:
            continue
        sci, common = line.split("_", 1)
        if NON_BIRD_PATTERN.search(common) and not BIRD_EXCEPTIONS.search(common):
            continue
        pairs.append((sci, common))
    return pairs


def labels_sha256():
    with open(labels_path(), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ════════════════════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════════════════════
def _cache_key():
    digest = hashlib.sha256(
        f"{MODEL_ID}|{labels_sha256()}|{PROMPT}".encode()).hexdigest()[:16]
    return digest


def _text_cache_path():
    return os.path.join(CACHE_DIR, f"text_embeddings_{_cache_key()}.npz")


def _load(progress=False):
    """Load model, tokenizer and the cached text embeddings. Idempotent.

    Deliberately lazy: this costs a few seconds and ~600 MB of RSS, and the API
    must still start promptly when nothing triggers verification.
    """
    if _state["model"] is not None:
        return _state

    with _lock:
        if _state["model"] is not None:
            return _state
        try:
            import open_clip
            import torch
        except ImportError as e:
            raise VerifierUnavailable(
                "open_clip_torch not installed "
                "(pip3 install -r requirements-verify.txt)") from e

        import bird_core  # reuse the already-chosen device

        if progress:
            print(f"Loading {MODEL_ID}…")
        model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID)
        tokenizer = open_clip.get_tokenizer(MODEL_ID)
        device = bird_core.device
        model = model.to(device).eval()

        labels = load_labels()
        embeddings = _text_embeddings(model, tokenizer, labels, device, progress)

        _state.update(model=model, preprocess=preprocess, tokenizer=tokenizer,
                      labels=labels, text_embeddings=embeddings, device=device)
        return _state


def _text_embeddings(model, tokenizer, labels, device, progress=False):
    """Encode every candidate name once, then cache to disk.

    Encoding 6,423 prompts takes ~35s. Doing it per request would make the feature
    unusable; doing it per process start would make the API slow to boot. The cache
    is keyed on model + label-file hash + prompt template, so changing any of them
    invalidates it rather than silently mixing embeddings.
    """
    import torch

    path = _text_cache_path()
    if os.path.exists(path):
        try:
            blob = np.load(path)
            if blob["embeddings"].shape[0] == len(labels):
                return torch.from_numpy(blob["embeddings"]).to(device)
        except (OSError, KeyError, ValueError):
            pass  # regenerate below

    if progress:
        print(f"Encoding {len(labels)} species names (one-time, ~35s)…")
    chunks = []
    with torch.no_grad():
        for i in range(0, len(labels), 512):
            prompts = [PROMPT.format(sci=s, common=c) for s, c in labels[i:i + 512]]
            emb = model.encode_text(tokenizer(prompts).to(device))
            chunks.append(emb / emb.norm(dim=-1, keepdim=True))
    embeddings = torch.cat(chunks)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, embeddings=embeddings.cpu().numpy().astype(np.float32))
    return embeddings


def available():
    """Can verification run? Never raises — mirrors llm.available()."""
    try:
        import open_clip  # noqa: F401
        import timm  # noqa: F401
    except ImportError as e:
        return {"ok": False, "model": MODEL_ID,
                "reason": f"{e.name} not installed (pip3 install -r requirements-verify.txt)"}

    try:
        n_labels = len(load_labels())
    except Exception as e:
        return {"ok": False, "model": MODEL_ID, "reason": f"label file unreadable: {e}"}

    return {
        "ok": True,
        "model": MODEL_ID,
        "species": n_labels,
        "loaded": _state["model"] is not None,
        "text_cache": os.path.exists(_text_cache_path()),
        "threshold": load_threshold(),
    }


def load_threshold():
    """Acceptance threshold fitted by scripts/eval_verifier.py, or None."""
    if not os.path.exists(THRESHOLD_PATH):
        return None
    try:
        with open(THRESHOLD_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def warm(progress=True):
    """Preload the model and text cache. Safe to call at startup or from a script."""
    try:
        _load(progress=progress)
        return True
    except VerifierUnavailable:
        return False


# ════════════════════════════════════════════════════════════
# IDENTIFICATION
# ════════════════════════════════════════════════════════════
def encode_images(pil_images, batch_size=16):
    """Normalised image embeddings for a list of PIL images."""
    import torch

    state = _load()
    model, preprocess, device = state["model"], state["preprocess"], state["device"]
    out = []
    with torch.no_grad():
        for i in range(0, len(pil_images), batch_size):
            batch = torch.stack([preprocess(img.convert("RGB"))
                                 for img in pil_images[i:i + batch_size]]).to(device)
            feats = model.encode_image(batch)
            out.append(feats / feats.norm(dim=-1, keepdim=True))
    return torch.cat(out)


def rank(image_embeddings, topk=5):
    """Rank candidate species for pre-encoded image embeddings.

    Returns a list (one per image) of lists of
    {scientific_name, common_name, similarity, score, margin}.

    `similarity` is the raw cosine; `score` is a softmax over all candidates using
    the model's learned logit scale, which is the number worth thresholding on.
    `margin` is top-1 minus top-2 similarity — a thin margin means the model is
    torn between two species, which similarity alone does not reveal.
    """
    import torch

    state = _load()
    labels, text = state["labels"], state["text_embeddings"]
    logit_scale = state["model"].logit_scale.exp().detach()

    sims = image_embeddings @ text.T
    probs = (logit_scale * sims).softmax(dim=-1)
    top = sims.topk(min(topk, len(labels)), dim=-1)

    results = []
    for row in range(sims.shape[0]):
        idxs = top.indices[row].tolist()
        row_sims = top.values[row].tolist()
        margin = row_sims[0] - row_sims[1] if len(row_sims) > 1 else float("nan")
        results.append([
            {
                "scientific_name": labels[i][0],
                "common_name": labels[i][1],
                "similarity": float(s),
                "score": float(probs[row, i]),
                "margin": float(margin) if rank_i == 0 else None,
            }
            for rank_i, (i, s) in enumerate(zip(idxs, row_sims))
        ])
    return results


def identify(image, topk=5):
    """Identify one image against the full candidate species list.

    Returns a dict with `top` (ranked candidates), `best`, and `confident` — the
    last decided by the fitted threshold, or left None when no threshold exists so
    that an unfitted install never silently asserts confidence it has not earned.
    """
    from PIL import Image

    pil = image if hasattr(image, "convert") else Image.open(image)
    ranked = rank(encode_images([pil]), topk=topk)[0]
    best = ranked[0]

    threshold = load_threshold()
    confident = None
    if threshold:
        confident = (best["score"] >= threshold.get("min_score", 0.0)
                     and best["margin"] >= threshold.get("min_margin", 0.0))

    # Map back into the trained 200 by common name. Deliberately not by scientific
    # name: the knowledge base's scientific names are LLM-generated and some are
    # wrong (Cactus Wren is listed as "Campylhyla carolinensis", which is not a
    # real binomial), whereas match_cub_species handles CUB's abbreviated and
    # misspelled folder names.
    import bird_core
    cub_folder = bird_core.match_cub_species(best["common_name"])

    return {
        "ran": True,
        "model": MODEL_ID,
        "species_considered": len(_state["labels"]),
        "top": ranked,
        "best": best,
        "confident": confident,
        "in_cub_200": cub_folder is not None,
        "cub_folder": cub_folder,
        "threshold": threshold,
    }
