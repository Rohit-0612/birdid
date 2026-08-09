"""Bird identification core — model, knowledge base, open-set gate.

Every public function here returns a **dict**, never formatted text. That is
the whole point of this module: the old code returned ASCII blobs, which meant
the React frontend had to regex-scrape them (`parseBirdResult`) and threw away
most of what the knowledge base knows. Structured output lets the dashboard,
the voice narrator, the chat grounding and the life list all read the same
payload.

Rendering lives in bird_text.py, the Gradio UI in bird_complete_local.py, the
HTTP surface in api.py. Importing this module loads the checkpoint but starts
no server and has no other side effects.
"""

import os

# Must be set before torch is imported: a few ops still have no MPS kernel and
# need to fall back to CPU instead of raising.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import tempfile
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

from bird_data import (
    display_name,
    HABITAT_MAP,
    MIGRATION_MAP,
    SIMILAR_SPECIES,
)

try:
    from birdnetlib import Recording
    from birdnetlib.analyzer import Analyzer
    BIRDNET_ANALYZER = Analyzer()
    BIRDNET_AVAILABLE = True
    print("✅ BirdNET loaded successfully")
except ImportError:
    BIRDNET_AVAILABLE = False
    BIRDNET_ANALYZER = None
    print("⚠️  birdnetlib not installed. Run: pip3 install birdnetlib")


# ════════════════════════════════════════════════════════════
# PATHS — resolved relative to this file, overridable by env var
# so the project runs from any checkout without editing source.
# ════════════════════════════════════════════════════════════
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_DIR = os.environ.get("BIRD_TEST_DIR", os.path.join(PROJECT_DIR, "birds_split", "test"))
SAVE_DIR = os.environ.get("BIRD_SAVE_DIR", os.path.join(PROJECT_DIR, "evaluation"))
SPECIES_KB_PATH = os.path.join(PROJECT_DIR, "data", "species_kb.json")
OPENSET_THRESHOLD_PATH = os.path.join(PROJECT_DIR, "data", "openset", "threshold.json")

# Prefer the slim inference checkpoint (79 MB); fall back to the full training
# checkpoint (235 MB) which additionally carries optimizer state.
_MODEL_CANDIDATES = [
    os.environ.get("BIRD_MODEL_PATH"),
    os.path.join(PROJECT_DIR, "best_bird_model_official.pth"),
    os.path.join(PROJECT_DIR, "best_bird_model_inference.pth"),
    os.path.join(PROJECT_DIR, "best_bird_model.pth"),
]
MODEL_PATH = next((p for p in _MODEL_CANDIDATES if p and os.path.exists(p)), None)
if MODEL_PATH is None:
    raise FileNotFoundError(
        "No model checkpoint found. Expected best_bird_model_inference.pth or "
        f"best_bird_model.pth in {PROJECT_DIR}, or set BIRD_MODEL_PATH."
    )

os.makedirs(SAVE_DIR, exist_ok=True)


def pick_device():
    """CUDA if present, else Apple-Silicon MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = pick_device()
DEVICE_LABEL = {"cuda": "CUDA", "mps": "Apple MPS", "cpu": "CPU"}[device.type]
print(f"\n🚀 Running on: {device}")

# ── Load model ───────────────────────────────────────────────
print(f"Loading model from {os.path.basename(MODEL_PATH)} "
      f"({os.path.getsize(MODEL_PATH) / 2**20:.0f} MB)...")
checkpoint  = torch.load(MODEL_PATH, map_location=device)
CLASS_NAMES = checkpoint["class_names"]
NUM_SPECIES = len(CLASS_NAMES)

model = models.efficientnet_v2_s(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_SPECIES)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()
print(f"✅ Model loaded — {NUM_SPECIES} species")

# ── Provenance gate ──────────────────────────────────────────
# Checkpoints produced by train.py record the sha256 of the split manifests
# they were trained against, so any accuracy claim can be traced to an exact
# image list. The original checkpoint has no such record, and it shows:
# measured top-1 was 94.6% on official CUB train, 94.2% on official test, and
# 95.2% on official test images that were never in birds_split/train — images
# it supposedly never saw scoring HIGHER than ones it did, against a realistic
# ~86–88% ceiling for this architecture. It has effectively seen all of CUB.
#
# Any evaluation of such a checkpoint is meaningless, so say so loudly rather
# than rendering a confident-looking 94%.
MODEL_IS_AUDITED = "split_manifest_sha256" in checkpoint
CHECKPOINT_VAL_ACC = checkpoint.get("val_acc")

CONTAMINATION_WARNING = (
    "⚠️  UNTRUSTWORTHY RESULT — this checkpoint has no recorded training split.\n"
    "    Measured evidence says it saw essentially the whole CUB dataset:\n"
    "    official-test images never held out still score ~95%, above the\n"
    "    ~86–88% realistic ceiling for this architecture.\n"
    "    Whatever number appears below is inflated by memorisation.\n"
    "    Retrain with train.py (official split, manifest hashed) for a real one.\n"
    + "─" * 55 + "\n"
)

# Same fact, one sentence, no box drawing — for the dashboard's honesty panel.
CONTAMINATION_SUMMARY = (
    "This checkpoint has no recorded training split and measured evidence says "
    "it saw essentially all of CUB, so its accuracy is inflated by memorisation. "
    "Species predictions are still useful; the confidence number is not a "
    "calibrated probability of being right."
)

if not MODEL_IS_AUDITED:
    print("⚠️  Loaded checkpoint has no split manifest — its accuracy numbers "
          "are not trustworthy (see CONTAMINATION_WARNING)")

# ── Transforms ───────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

raw_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
])

norm_transform = transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
)

# ── Coverage guard ───────────────────────────────────────────
# A key that is not one of the model's classes is always a bug: it can never be
# looked up, so it is dead data. Fail hard on that. A *missing* key is a known,
# tracked gap, so it only warns.
_unknown_keys = (set(HABITAT_MAP) | set(MIGRATION_MAP)
                 | set(SIMILAR_SPECIES)) - set(CLASS_NAMES)
assert not _unknown_keys, (
    f"{len(_unknown_keys)} lookup key(s) match no model class and can never "
    f"be reached: {sorted(_unknown_keys)}"
)

print(f"✅ Curated species data — habitat {len(HABITAT_MAP)}/{NUM_SPECIES}, "
      f"migration {len(MIGRATION_MAP)}/{NUM_SPECIES}, "
      f"look-alikes {len(SIMILAR_SPECIES)}/{NUM_SPECIES}")


# ════════════════════════════════════════════════════════════
# LLM KNOWLEDGE BASE (optional overlay)
# ════════════════════════════════════════════════════════════
# data/species_kb.json is generated locally by scripts/build_species_kb.py
# against an Ollama model. It is machine-written and NOT expert-verified, so it
# is layered *underneath* the curated tables rather than over them:
#
#   curated text wins wherever it exists
#   the LLM fills only genuine gaps (15 species had no notes at all)
#   extra fields (scientific name, size, diet, field marks) are additive
#
# Every value carries its provenance so the UI can label what came from where.
# An unlabelled LLM fact in a portfolio project is a liability; a labelled one
# with a measured error rate is a feature.
SPECIES_KB = {}
KB_SOURCE_MODEL = None

if os.path.exists(SPECIES_KB_PATH):
    try:
        with open(SPECIES_KB_PATH) as _f:
            _raw = json.load(_f)
        _valid = set(CLASS_NAMES)
        SPECIES_KB = {k: v for k, v in _raw.items() if k in _valid}
        if SPECIES_KB:
            KB_SOURCE_MODEL = next(iter(SPECIES_KB.values())).get("_source", "unknown")
        _skipped = len(_raw) - len(SPECIES_KB)
        print(f"🤖 LLM knowledge base — {len(SPECIES_KB)}/{NUM_SPECIES} records "
              f"from {KB_SOURCE_MODEL}"
              + (f" ({_skipped} ignored: no matching class)" if _skipped else ""))
    except (json.JSONDecodeError, OSError) as e:
        print(f"⚠️  Could not read {SPECIES_KB_PATH}: {e}")
else:
    print("ℹ️  No LLM knowledge base yet — run scripts/build_species_kb.py")

_still_missing = [f for f in CLASS_NAMES
                  if f not in HABITAT_MAP and f not in SPECIES_KB]
if _still_missing:
    print(f"⚠️  {len(_still_missing)} species still have no notes from any source")

LLM_DISCLAIMER = (
    f"ℹ️  Some notes above were generated locally by {KB_SOURCE_MODEL} and are "
    "not expert-verified."
) if SPECIES_KB else None


def species_info(folder):
    """Merge curated + LLM data for one species, tracking where each came from.

    Returns a dict with habitat/migration/similar plus a `used_llm` flag so the
    caller can attach the provenance footnote.
    """
    kb = SPECIES_KB.get(folder, {})
    used_llm = False

    habitat = HABITAT_MAP.get(folder)
    if not habitat and kb.get("habitat"):
        habitat, used_llm = kb["habitat"], True

    migration = MIGRATION_MAP.get(folder)
    if not migration and kb.get("migration"):
        migration, used_llm = kb["migration"], True

    similar = SIMILAR_SPECIES.get(folder)
    if not similar and kb.get("similar_species"):
        first = kb["similar_species"][0]
        similar = (first.get("name", ""), first.get("how_to_distinguish", ""))
        used_llm = True

    # Extras exist only in the KB, so they are always machine-written.
    extras = {k: kb[k] for k in
              ("scientific_name", "size_cm", "diet", "field_marks",
               "conservation_status", "fun_fact")
              if kb.get(k)}
    if extras:
        used_llm = True

    return {
        "habitat":   habitat or "Location data not available",
        "migration": migration or "Migration data not available for this species",
        "similar":   similar,
        "extras":    extras,
        "used_llm":  used_llm,
    }


def species_detail(folder):
    """The full structured record for one species — everything the UI can show.

    species_info() exists to feed the legacy text renderer and deliberately
    collapses `similar_species` to a single tuple. This does not: it keeps every
    look-alike, and surfaces `family`, `order` and `range_description`, which
    the knowledge base has always carried but nothing ever displayed.
    """
    kb = SPECIES_KB.get(folder, {})
    merged = species_info(folder)

    # Curated look-alikes are a (name, how) tuple; KB ones are a list of dicts.
    # Normalise to the list form, curated first since it is hand-verified.
    similar = []
    curated = SIMILAR_SPECIES.get(folder)
    if curated:
        similar.append({"name": curated[0], "how_to_distinguish": curated[1],
                        "source": "curated"})
    seen = {s["name"].lower() for s in similar}
    for entry in kb.get("similar_species") or []:
        if (entry.get("name") or "").lower() not in seen:
            similar.append({"name": entry.get("name", ""),
                            "how_to_distinguish": entry.get("how_to_distinguish", ""),
                            "source": "llm"})

    return {
        "folder":            folder,
        "display_name":      display_name(folder),
        "scientific_name":   kb.get("scientific_name"),
        "family":            kb.get("family"),
        "order":             kb.get("order"),
        "habitat":           merged["habitat"],
        "range_description": kb.get("range_description"),
        "migration":         merged["migration"],
        "field_marks":       kb.get("field_marks") or [],
        "size_cm":           kb.get("size_cm"),
        "diet":              kb.get("diet"),
        "similar_species":   similar,
        "conservation_status": kb.get("conservation_status"),
        "fun_fact":          kb.get("fun_fact"),
        "has_curated_notes": folder in HABITAT_MAP or folder in MIGRATION_MAP,
        "used_llm":          merged["used_llm"],
    }


def match_cub_species(external_name):
    """Map a BirdNET common name onto a CUB folder, or None.

    The two label spaces do not line up: BirdNET knows ~6,500 species and uses
    current standard names, while CUB uses 200 abbreviated and occasionally
    misspelled folder names ('Cardinal' for Northern Cardinal, 'Artic_Tern',
    bare genera like 'Geococcyx'). Exact matching fails on most of them, so this
    falls back to containment and prefers the longest match — otherwise 'Tern'
    would swallow 'Arctic Tern'.

    This is a heuristic, not a taxonomy. A hand-verified CUB -> scientific name
    mapping is the real fix, and the KB's scientific_name field is the natural
    place to build it from.
    """
    if not external_name:
        return None
    target = external_name.lower().strip()
    best, best_len = None, 0

    for folder in CLASS_NAMES:
        key = display_name(folder).lower()
        if key == target:
            return folder
        if (key in target or target in key) and len(key) > best_len:
            best, best_len = folder, len(key)

    # Fall back to the KB's scientific name, which is often more reliable than
    # the abbreviated common name.
    if not best:
        for folder, rec in SPECIES_KB.items():
            sci = (rec.get("scientific_name") or "").lower()
            if sci and sci == target:
                return folder
    return best


# ════════════════════════════════════════════════════════════
# OPEN-SET GATE — "is this even one of my 200 birds?"
# ════════════════════════════════════════════════════════════
# A 200-way softmax cannot say "none of the above": it renormalises over the
# 200 classes it knows, so a photo of a dog still comes back as some bird with
# a confident-looking score. Energy scoring (Liu et al. 2020) reads the raw
# logit magnitudes instead of the normalised probabilities and separates
# in-distribution from out-of-distribution inputs without retraining.
#
#   energy = -T · logsumexp(logits / T)      lower = more in-distribution
#
# The threshold is fitted offline by scripts/fit_openset.py against the 541
# labelled images in data/openset/manifest.json. Absent that file the gate is
# disabled and everything is treated as a bird — same behaviour as before.
OPENSET = {"enabled": False, "method": "energy", "threshold": None, "temperature": 1.0}

if os.path.exists(OPENSET_THRESHOLD_PATH):
    try:
        with open(OPENSET_THRESHOLD_PATH) as _f:
            _cfg = json.load(_f)
        OPENSET.update({
            "enabled":     True,
            "method":      _cfg.get("method", "energy"),
            "threshold":   float(_cfg["threshold"]),
            "temperature": float(_cfg.get("temperature", 1.0)),
            "tpr":         _cfg.get("tpr"),
            "fpr_near_bird":  _cfg.get("fpr_near_bird"),
            "fpr_animal":     _cfg.get("fpr_animal"),
            "fpr_nonanimal":  _cfg.get("fpr_nonanimal"),
            "fitted":      _cfg.get("created"),
        })
        print(f"🚧 Open-set gate ON — {OPENSET['method']} "
              f"threshold {OPENSET['threshold']:.3f} "
              f"(TPR {(_cfg.get('tpr') or 0)*100:.0f}% on real birds)")
    except (json.JSONDecodeError, OSError, KeyError, ValueError) as e:
        print(f"⚠️  Could not read {OPENSET_THRESHOLD_PATH}: {e} — gate stays off")
else:
    print("ℹ️  No open-set threshold yet — run scripts/fit_openset.py "
          "(every input will be treated as a known bird)")


def _score_energy(logits, t):
    """Free energy — reads un-normalised logit mass. Lower = more in-distribution."""
    return -t * torch.logsumexp(logits / t, dim=1)


def _score_msp(logits, t):
    """Negated max softmax probability (Hendrycks & Gimpel 2017)."""
    return -F.softmax(logits / t, dim=1).max(dim=1).values


def _score_entropy(logits, t):
    """Softmax entropy — high when the model is spread across many species."""
    logp = F.log_softmax(logits / t, dim=1)
    return -(logp.exp() * logp).sum(dim=1)


def _score_margin(logits, _t):
    """Negated gap between the top two logits — thin gap means 'between classes'."""
    top2 = logits.topk(2, dim=1).values
    return -(top2[:, 0] - top2[:, 1])


# Registry shared with scripts/fit_openset.py, which imports it rather than
# reimplementing the maths. A threshold is only meaningful against the exact
# score it was fitted with, so there must be exactly one definition of each.
OPENSET_SCORES = {
    "energy":  _score_energy,
    "msp":     _score_msp,
    "entropy": _score_entropy,
    "margin":  _score_margin,
}


def openset_score(logits, method="msp", temperature=1.0):
    """Open-set score for a batch of logits. Lower always means more in-distribution."""
    try:
        fn = OPENSET_SCORES[method]
    except KeyError:
        raise ValueError(f"unknown open-set method {method!r}; "
                         f"expected one of {sorted(OPENSET_SCORES)}") from None
    t = torch.as_tensor(temperature, dtype=logits.dtype, device=logits.device)
    return fn(logits, t).detach().cpu().numpy()


def openset_verdict(logits):
    """Decide whether `logits` came from one of the 200 known species."""
    score = float(openset_score(logits, OPENSET["method"], OPENSET["temperature"])[0])
    if not OPENSET["enabled"]:
        return {"enabled": False, "is_bird": True, "score": score,
                "threshold": None, "method": OPENSET["method"], "margin": None}
    threshold = OPENSET["threshold"]
    return {
        "enabled":   True,
        "is_bird":   score <= threshold,
        "score":     score,
        "threshold": threshold,
        "method":    OPENSET["method"],
        # How far the input sits from the decision boundary, normalised so the
        # UI can show "just barely" versus "not remotely" a bird.
        "margin":    float(threshold - score),
    }


def provenance():
    """Where the numbers and the prose each came from."""
    return {
        "checkpoint":       os.path.basename(MODEL_PATH),
        "val_acc":          CHECKPOINT_VAL_ACC,
        "model_audited":    MODEL_IS_AUDITED,
        "caveat":           None if MODEL_IS_AUDITED else CONTAMINATION_SUMMARY,
        "kb_source":        KB_SOURCE_MODEL,
        "kb_verified":      False if SPECIES_KB else None,
        "num_species":      NUM_SPECIES,
        "openset_enabled":  OPENSET["enabled"],
    }


def confidence_band(conf, low=0.60, high=0.80):
    """'high' | 'moderate' | 'low' for a 0-1 confidence."""
    if conf >= high:
        return "high"
    if conf >= low:
        return "moderate"
    return "low"


# ════════════════════════════════════════════════════════════
# IMAGE IDENTIFICATION
# ════════════════════════════════════════════════════════════
def _as_pil(image):
    """Accept a numpy array (Gradio), a PIL image, or a path."""
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def identify_image(image, topk=5):
    """Identify a bird from a photo. Returns the structured result dict."""
    t0 = time.perf_counter()
    img = _as_pil(image)
    x = val_transform(img).unsqueeze(0).to(device)
    t_pre = time.perf_counter()

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        top = probs.topk(min(topk, NUM_SPECIES))
    t_inf = time.perf_counter()

    ranked = []
    for prob, idx in zip(top.values[0], top.indices[0]):
        folder = CLASS_NAMES[idx.item()]
        ranked.append({
            "folder":       folder,
            "display_name": display_name(folder),
            "confidence":   float(prob.item()),
            "habitat":      species_info(folder)["habitat"],
        })

    top_folder = ranked[0]["folder"]
    detail = species_detail(top_folder)
    conf = ranked[0]["confidence"]

    return {
        "kind": "image",
        "species": {
            "folder":          top_folder,
            "display_name":    detail["display_name"],
            "scientific_name": detail["scientific_name"],
            "family":          detail["family"],
            "order":           detail["order"],
        },
        "confidence":      conf,
        "confidence_band": confidence_band(conf),
        "top5":            ranked,
        "info":            detail,
        "openset":         openset_verdict(logits),
        "provenance":      provenance(),
        "timing_ms": {
            "preprocess": round((t_pre - t0) * 1000, 1),
            "inference":  round((t_inf - t_pre) * 1000, 1),
            "total":      round((time.perf_counter() - t0) * 1000, 1),
        },
    }


def generate_gradcam(image, save_path=None):
    """Render a Grad-CAM overlay. Returns (path, info_dict) — path is None on failure."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        img        = _as_pil(image)
        raw_tensor = raw_transform(img)
        inp_tensor = norm_transform(raw_tensor).unsqueeze(0).to(device)

        with torch.no_grad():
            out       = model(inp_tensor)
            probs     = F.softmax(out, dim=1)
            pred_idx  = probs.argmax().item()
            pred_conf = float(probs[0][pred_idx].item())
            pred_name = display_name(CLASS_NAMES[pred_idx])

        cam = GradCAM(model=model, target_layers=[model.features[-1]])
        grayscale_cam = cam(input_tensor=inp_tensor,
                            targets=[ClassifierOutputTarget(pred_idx)])[0]

        rgb_img = raw_tensor.permute(1, 2, 0).numpy()
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(rgb_img)
        ax1.set_title("Original Image", fontsize=13)
        ax1.axis("off")
        ax2.imshow(cam_image)
        ax2.set_title(
            f"Grad-CAM — Model focuses on red/yellow areas\n"
            f"Predicted: {pred_name} ({pred_conf*100:.1f}%)", fontsize=11)
        ax2.axis("off")
        plt.suptitle("Grad-CAM Explainability — What the model looks at", fontsize=13)
        plt.tight_layout()

        cam_path = save_path or os.path.join(SAVE_DIR, "gradcam_single.png")
        plt.savefig(cam_path, dpi=150, bbox_inches="tight")
        plt.close()

        return cam_path, {
            "folder":       CLASS_NAMES[pred_idx],
            "display_name": pred_name,
            "confidence":   pred_conf,
            "path":         cam_path,
        }
    except ImportError:
        return None, {"error": "grad-cam not installed. Run: pip3 install grad-cam"}
    except Exception as e:
        return None, {"error": f"{type(e).__name__}: {e}"}


# ════════════════════════════════════════════════════════════
# BIRDNET AUDIO IDENTIFICATION
# ════════════════════════════════════════════════════════════
def decode_audio(audio_path):
    """Decode any librosa-readable file to a temp 48 kHz mono WAV.

    BirdNET expects 48 kHz mono. Handing it an MP3 directly routes through
    pydub, which shells out to ffmpeg — a system dependency this project does
    not need: libsndfile (via soundfile) decodes MP3 since 1.1, and librosa
    resamples. Returns (wav_path, samples, sample_rate).
    """
    y, sr = librosa.load(audio_path, sr=48000, mono=True)
    if y.size == 0:
        raise ValueError("audio file decoded to zero samples")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, y, sr)
    return tmp.name, y, sr


def render_spectrogram(y, sr, detections, save_path):
    """Mel-spectrogram plus a lane-per-species detection timeline."""
    duration = len(y) / sr
    by_species = {}
    for det in detections:
        by_species.setdefault(det["common_name"], []).append(det)
    lanes = sorted(by_species,
                   key=lambda n: max(d["confidence"] for d in by_species[n]),
                   reverse=True)[:6]

    height = 4.2 + 0.42 * max(len(lanes), 1)
    fig, (ax_spec, ax_time) = plt.subplots(
        2, 1, figsize=(12, height), sharex=True,
        gridspec_kw={"height_ratios": [3, max(1.1, 0.42 * max(len(lanes), 1))]},
    )

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=15000)
    img = librosa.display.specshow(
        librosa.power_to_db(mel, ref=np.max),
        sr=sr, x_axis="time", y_axis="mel", fmax=15000, ax=ax_spec, cmap="magma",
    )
    ax_spec.set_title("Mel spectrogram — what BirdNET listens to", fontsize=12)
    ax_spec.set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=ax_spec, format="%+2.0f dB", pad=0.01)

    cmap = plt.get_cmap("viridis")
    for row, name in enumerate(lanes):
        for det in by_species[name]:
            conf = det["confidence"]
            ax_time.barh(row, det["end_time"] - det["start_time"],
                         left=det["start_time"], height=0.62,
                         color=cmap(conf), edgecolor="white", linewidth=0.5)
            ax_time.text(det["start_time"] + 0.1, row, f"{conf*100:.0f}%",
                         va="center", fontsize=7, color="white")
    ax_time.set_yticks(range(len(lanes)))
    ax_time.set_yticklabels([n[:28] for n in lanes], fontsize=8)
    ax_time.set_xlim(0, duration)
    ax_time.invert_yaxis()
    ax_time.set_xlabel("Time (seconds)")
    ax_time.set_title("Detections over time — colour = confidence", fontsize=10)
    ax_time.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()
    return save_path


def identify_audio(audio_path, use_location=False, lat=39.83, lon=-98.58,
                   obs_date=None, min_conf=0.25):
    """Identify a bird from a recording. Returns the structured result dict.

    `use_location` is off by default and that is deliberate. BirdNET treats
    lat/lon/date as a species-occurrence prior, so a wrong location actively
    suppresses the correct species. This used to be hardcoded to 20.59 N,
    78.96 E — central India — while every CUB species is North American, so the
    prior was fighting the classifier on every single call.
    """
    if not BIRDNET_AVAILABLE:
        return {"kind": "audio", "ok": False,
                "error": "BirdNET not installed. Fix: pip3 install birdnetlib, then restart."}
    if audio_path is None:
        return {"kind": "audio", "ok": False,
                "error": "Please upload or record some audio first."}

    t0 = time.perf_counter()
    wav_path = None
    try:
        wav_path, y, sr = decode_audio(audio_path)
        duration = len(y) / sr

        kwargs = {"min_conf": float(min_conf)}
        if use_location:
            from datetime import date, datetime
            if isinstance(obs_date, str) and obs_date.strip():
                try:
                    parsed = datetime.strptime(obs_date.strip(), "%Y-%m-%d").date()
                except ValueError:
                    parsed = date.today()
            else:
                parsed = date.today()
            kwargs.update(lat=float(lat), lon=float(lon), date=parsed)

        recording = Recording(BIRDNET_ANALYZER, wav_path, **kwargs)
        recording.analyze()
        detections = sorted(recording.detections,
                            key=lambda d: d["confidence"], reverse=True)

        settings = {
            "min_conf":      float(min_conf),
            "use_location":  bool(use_location),
            "lat":           float(lat) if use_location else None,
            "lon":           float(lon) if use_location else None,
            "duration_sec":  round(duration, 1),
        }

        if not detections:
            return {"kind": "audio", "ok": True, "detected": False,
                    "settings": settings, "detections": [], "grouped": [],
                    "spectrogram": None, "provenance": provenance(),
                    "hints": [
                        "Lower the confidence threshold",
                        "Turn the location prior off if it is on",
                        "Use a clip of at least 3 seconds with little background noise",
                        "Grab test recordings from xeno-canto.org",
                    ]}

        spec_path = render_spectrogram(
            y, sr, detections, os.path.join(SAVE_DIR, "audio_analysis.png"))

        # Group by species: one bird singing eight times is one bird, not eight
        # detections. Report its best score and how often it was heard.
        grouped = {}
        for det in detections:
            g = grouped.setdefault(det["common_name"],
                                   {"best": 0.0, "count": 0, "first": det["start_time"]})
            g["best"] = max(g["best"], det["confidence"])
            g["count"] += 1
            g["first"] = min(g["first"], det["start_time"])
        ranked = [
            {"common_name": name, "confidence": g["best"], "count": g["count"],
             "first_heard": round(g["first"], 1),
             "cub_folder": match_cub_species(name)}
            for name, g in sorted(grouped.items(),
                                  key=lambda kv: kv[1]["best"], reverse=True)
        ]

        top = detections[0]
        matched = match_cub_species(top["common_name"])
        conf = float(top["confidence"])

        return {
            "kind": "audio",
            "ok": True,
            "detected": True,
            "species": {
                "common_name":     top["common_name"],
                "scientific_name": top["scientific_name"],
                "cub_folder":      matched,
                "in_image_model":  matched is not None,
            },
            "confidence":      conf,
            # Audio confidence runs lower than image softmax; BirdNET's own
            # guidance treats 0.7+ as strong, so the bands differ on purpose.
            "confidence_band": confidence_band(conf, low=0.40, high=0.70),
            "settings":        settings,
            "detections": [
                {"common_name": d["common_name"],
                 "scientific_name": d["scientific_name"],
                 "confidence": float(d["confidence"]),
                 "start_time": round(d["start_time"], 1),
                 "end_time":   round(d["end_time"], 1)}
                for d in detections
            ],
            "grouped":     ranked,
            "info":        species_detail(matched) if matched else None,
            "spectrogram": spec_path,
            "provenance":  {**provenance(), "audio_model": "BirdNET (Cornell Lab)"},
            "timing_ms":   {"total": round((time.perf_counter() - t0) * 1000, 1)},
        }

    except Exception as e:
        return {"kind": "audio", "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "hints": ["Supported: .wav, .mp3, .flac, .ogg, .m4a",
                          "At least 3 seconds of audio is recommended"]}
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except OSError:
                pass


# ════════════════════════════════════════════════════════════
# CATALOGUE + HEALTH
# ════════════════════════════════════════════════════════════
def species_catalogue():
    """One summary row per known species — powers the field-guide grid."""
    rows = []
    for folder in CLASS_NAMES:
        kb = SPECIES_KB.get(folder, {})
        rows.append({
            "folder":              folder,
            "display_name":        display_name(folder),
            "scientific_name":     kb.get("scientific_name"),
            "family":              kb.get("family"),
            "order":               kb.get("order"),
            "conservation_status": kb.get("conservation_status"),
            "size_cm":             kb.get("size_cm"),
        })
    return rows


def health():
    """Everything the dashboard's status bar needs, in one call."""
    return {
        "device":        device.type,
        "device_label":  DEVICE_LABEL,
        "checkpoint":    os.path.basename(MODEL_PATH),
        "num_species":   NUM_SPECIES,
        "input_size":    380,
        "model_audited": MODEL_IS_AUDITED,
        "val_acc":       CHECKPOINT_VAL_ACC,
        "caveat":        None if MODEL_IS_AUDITED else CONTAMINATION_SUMMARY,
        "kb_records":    len(SPECIES_KB),
        "kb_source":     KB_SOURCE_MODEL,
        "birdnet":       BIRDNET_AVAILABLE,
        "openset":       OPENSET,
        "coverage": {
            "habitat":   len(HABITAT_MAP),
            "migration": len(MIGRATION_MAP),
            "similar":   len(SIMILAR_SPECIES),
        },
    }
