"""HTTP API for the bird dashboard.

Everything the React app in bird-frontend/ needs, as JSON. The Gradio app stays
on :7860 as a debug surface; this is the interface the real UI talks to.

    uvicorn api:app --port 8000 --reload        # or ./dev.sh

Design notes:

- The model loads once, at import, because bird_core does. Requests never pay
  for it. That is also why this must not be run with multiple workers on a
  laptop: each worker would load its own copy of a 79 MB checkpoint.
- Nothing here formats prose for display. Endpoints return the same dicts
  bird_core produces, so the dashboard, the voice narrator and the tests all
  read one shape.
- The LLM endpoints are the only ones that can be slow (seconds) and the only
  ones that can degrade. They always return 200 with `generated: false` and a
  reason rather than an error status, because a missing Ollama daemon is a
  degraded feature, not a failed request.
"""

import io
import json
import os
import tempfile
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError

import bird_core
from services import llm, sightings, verifier

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(PROJECT_DIR, "bird_audio_samples")
SAMPLES_MANIFEST = os.path.join(SAMPLES_DIR, "manifest.json")

# Vite's dev server. The proxy in vite.config.js means the browser normally
# sees same-origin requests, but allowing these keeps direct calls working.
ALLOWED_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:4173", "http://127.0.0.1:4173",
]

MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# Below this confidence the classifier's own answer is not trusted on its own and
# the open-vocabulary verifier gets a say. Measured justification: across the 541
# labelled open-set photos, a <60% rule catches 98% of birds outside the 200 — at
# the cost of also firing on 59% of correct identifications, which is affordable
# only because the verifier is local. See services/verifier.py.
VERIFY_BELOW_CONFIDENCE = float(os.environ.get("BIRD_VERIFY_BELOW", "0.60"))


@asynccontextmanager
async def lifespan(app):
    print(f"\n🌐 API ready — {bird_core.NUM_SPECIES} species on "
          f"{bird_core.DEVICE_LABEL}, open-set "
          f"{'ON' if bird_core.OPENSET['enabled'] else 'OFF'}")
    status = llm.available()
    print(f"🤖 LLM {'ready via ' + status['provider'] if status['ok'] else 'unavailable — template text'}"
          f" — {status.get('reason', status['model'])}")
    verify_status = verifier.available()
    if verify_status["ok"]:
        print(f"🔍 Verifier ready — {verify_status['species']} species, loads on "
              f"first use (below {VERIFY_BELOW_CONFIDENCE:.0%} confidence)")
    else:
        print(f"🔍 Verifier unavailable — {verify_status['reason']}")
    yield


app = FastAPI(title="Bird Identifier API", version="2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════
async def _read_upload(upload, kinds=("image",)):
    """Read an upload with a size cap, and reject the obvious wrong type."""
    data = await upload.read()
    if not data:
        raise HTTPException(400, "uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD_BYTES // 2**20} MB")
    ctype = upload.content_type or ""
    if kinds and not any(ctype.startswith(k) for k in kinds) and ctype:
        raise HTTPException(415, f"expected {'/'.join(kinds)}, got {ctype}")
    return data


def _open_image(data):
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "could not decode that image") from None


def _species_or_404(folder):
    if folder not in bird_core.CLASS_NAMES:
        raise HTTPException(404, f"unknown species folder {folder!r}")
    return bird_core.species_detail(folder)


def _verification_reason(result):
    """Why the verifier should run on this result, or None to skip it.

    Two triggers with complementary error profiles. The open-set gate is the precise
    one (flags every non-bird, only 5% false alarms) and the confidence bar is the
    sensitive one (catches 98% of birds outside the 200). Either is enough.
    """
    gate = result.get("openset") or {}
    if gate.get("enabled") and not gate.get("is_bird", True):
        return "rejected by the open-set gate"
    if (result.get("confidence") or 0.0) < VERIFY_BELOW_CONFIDENCE:
        return f"confidence below {VERIFY_BELOW_CONFIDENCE:.0%}"
    return None


def _verify(pil_image, result):
    """Run the open-vocabulary verifier when warranted. Never raises.

    Returns None when nothing triggered it, or a dict that always carries `ran` so
    the UI can tell "not needed" from "could not run".
    """
    reason = _verification_reason(result)
    if reason is None:
        return None
    try:
        payload = verifier.identify(pil_image)
        payload["reason"] = reason
        classifier_folder = (result.get("species") or {}).get("folder")
        payload["agrees_with_classifier"] = (
            payload.get("cub_folder") == classifier_folder
            if payload.get("cub_folder") and classifier_folder else None
        )
        return payload
    except verifier.VerifierUnavailable as e:
        return {"ran": False, "reason": reason, "error": str(e)}
    except Exception as e:
        return {"ran": False, "reason": reason, "error": f"{type(e).__name__}: {e}"}


# ════════════════════════════════════════════════════════════
# STATUS
# ════════════════════════════════════════════════════════════
@app.get("/api/health")
def health():
    """Model, device, knowledge base, open-set gate, LLM and verifier in one call."""
    return {
        **bird_core.health(),
        "llm": llm.available(),
        "verifier": {**verifier.available(),
                     "triggers_below_confidence": VERIFY_BELOW_CONFIDENCE},
    }


# ════════════════════════════════════════════════════════════
# IDENTIFICATION
# ════════════════════════════════════════════════════════════
@app.post("/api/identify")
async def identify(
    image: UploadFile = File(...),
    verify: bool = Form(True),
    add_to_deck: bool = Form(False),
):
    """Identify a bird from a photo, optionally verifying and filing it in the deck.

    Both extras happen in this one request because the image is already in hand —
    a separate register call would mean uploading the photo twice. `add_to_deck`
    defaults to false so the Gradio app, bird_text and the existing tests see
    exactly the previous behaviour; the dashboard sends true.

    Named `add_to_deck` rather than `register` because pydantic warns that a body
    field called `register` shadows an attribute on BaseModel.
    """
    data = await _read_upload(image)
    pil = _open_image(data)
    result = bird_core.identify_image(pil)

    result["verification"] = _verify(pil, result) if verify else None

    if add_to_deck:
        thumb = sightings.save_thumbnail(pil, uuid.uuid4().hex[:12])
        outcome = sightings.deck_register(result, result["verification"], thumb=thumb)
        # Nothing was filed, so the thumbnail we just wrote is an orphan.
        if outcome["entry"] is None:
            stale = sightings.thumbnail_path(thumb)
            if stale:
                try:
                    os.remove(stale)
                except OSError:
                    pass
        result["deck"] = outcome
    else:
        result["deck"] = None

    result["answer"] = _answer(result)
    return result


def _answer(result):
    """The resolved answer the dashboard shows, with field notes for *that* bird.

    Added alongside the classifier's own fields, never instead of them: callers
    that read `species`/`confidence` see exactly what they always did.
    """
    answer = sightings.resolve_answer(result, result.get("verification"))
    if answer["identified_by"] == "cub":
        answer["info"] = result.get("info") or {}
    elif answer["folder"]:
        # The verifier corrected the pick into the 200: show the corrected
        # species' notes, not the ones for the classifier's mistaken guess.
        try:
            info = bird_core.species_detail(answer["folder"])
        except Exception:
            info = {}
        answer["info"] = info
        answer["family"] = answer["family"] or info.get("family")
        answer["order"] = answer["order"] or info.get("order")
    else:
        answer["info"] = {}
    return answer


@app.post("/api/identify/audio")
async def identify_audio_endpoint(
    audio: UploadFile = File(...),
    min_conf: float = Form(0.25),
    use_location: bool = Form(False),
    lat: float = Form(39.83),
    lon: float = Form(-98.58),
    obs_date: str = Form(""),
):
    """Identify a bird from a recording, via BirdNET."""
    data = await _read_upload(audio, kinds=("audio", "video", "application/octet-stream"))
    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        token = uuid.uuid4().hex[:12]
        result = bird_core.identify_audio(
            tmp.name, use_location=use_location, lat=lat, lon=lon,
            obs_date=obs_date or None, min_conf=min_conf,
            spectrogram_path=os.path.join(bird_core.SAVE_DIR, f"spectrogram_{token}.png"),
        )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    # Hand back a URL, not a filesystem path: the browser cannot read the latter.
    if result.get("spectrogram"):
        result["spectrogram_url"] = f"/api/media/{os.path.basename(result['spectrogram'])}"
    return result


@app.post("/api/gradcam")
async def gradcam(image: UploadFile = File(...)):
    """Grad-CAM overlay showing which pixels drove the prediction."""
    data = await _read_upload(image)
    token = uuid.uuid4().hex[:12]
    path, info = bird_core.generate_gradcam(
        _open_image(data),
        save_path=os.path.join(bird_core.SAVE_DIR, f"gradcam_{token}.png"))
    if path is None:
        raise HTTPException(500, info.get("error", "Grad-CAM failed"))
    return {**info, "url": f"/api/media/{os.path.basename(path)}"}


@app.get("/api/media/{name}")
def media(name: str):
    """Serve a generated PNG (spectrogram, Grad-CAM) out of the evaluation dir."""
    if os.path.basename(name) != name or not name.endswith(".png"):
        raise HTTPException(400, "bad media name")
    path = os.path.join(bird_core.SAVE_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(404, "no such artifact")
    return FileResponse(path, media_type="image/png")


# ════════════════════════════════════════════════════════════
# FIELD GUIDE
# ════════════════════════════════════════════════════════════
@app.get("/api/species")
def species_list():
    """Summary row for all known species — powers the field-guide grid."""
    return {"species": bird_core.species_catalogue(),
            "total": bird_core.NUM_SPECIES}


@app.get("/api/species/{folder}")
def species_one(folder: str):
    """Everything the knowledge base holds for one species."""
    return _species_or_404(folder)


@app.get("/api/compare")
def compare(a: str = Query(...), b: str = Query(...)):
    """Side-by-side comparison of two species, with a generated summary."""
    if a == b:
        raise HTTPException(400, "pick two different species")
    return llm.compare(_species_or_404(a), _species_or_404(b))


@app.get("/api/samples")
def samples():
    """The bundled demo recordings, so the audio view is never empty."""
    if not os.path.exists(SAMPLES_MANIFEST):
        return {"samples": []}
    with open(SAMPLES_MANIFEST) as f:
        entries = json.load(f)
    for entry in entries:
        entry["url"] = f"/api/samples/audio?file={entry['file']}"
        entry["cub_folder"] = bird_core.match_cub_species(entry.get("common_name"))
    return {"samples": entries}


@app.get("/api/samples/audio")
def sample_audio(file: str = Query(...)):
    """Stream one bundled sample. Only paths listed in the manifest are served."""
    with open(SAMPLES_MANIFEST) as f:
        allowed = {e["file"] for e in json.load(f)}
    if file not in allowed:
        raise HTTPException(404, "not a bundled sample")
    path = os.path.join(SAMPLES_DIR, file)
    if not os.path.exists(path):
        raise HTTPException(404, "sample file missing on disk")
    return FileResponse(path)


# ════════════════════════════════════════════════════════════
# LLM: NARRATION + CHAT
# ════════════════════════════════════════════════════════════
@app.post("/api/narrate")
async def narrate(payload: dict):
    """Turn a result dict into one paragraph of spoken field-guide prose.

    The browser speaks this with the Web Speech API. Reading the on-screen
    report aloud instead would produce "SPECIES colon Blue Jay, FIELD MARKS
    colon bullet…", which is why this endpoint exists.
    """
    result = payload.get("result") or payload
    if not isinstance(result, dict) or not result.get("species"):
        raise HTTPException(400, "expected an identification result under 'result'")
    return llm.narrate(result, model=payload.get("model"))


def _chat_context(question, folder=None, subject=None):
    """Retrieved context for a question, preferring the species on screen.

    `subject` names a bird on screen that has no knowledge-base record — one the
    open-vocabulary verifier identified outside the 200. Without it the model
    would be asked "what does it eat?" with no idea which bird "it" is.
    """
    folders = []
    on_screen = bool(folder and folder in bird_core.CLASS_NAMES)
    if on_screen:
        folders.append(folder)
    for extra in llm.retrieve(question, bird_core.species_catalogue(),
                              bird_core.SPECIES_KB, limit=3):
        if extra not in folders:
            folders.append(extra)
    folders = folders[:3]
    context = llm.build_context(folders, bird_core.species_detail)
    subject = (subject or "").strip()[:160]
    if subject and not on_screen:
        note = (f"Species on screen: {subject}. It was identified by an open-vocabulary "
                f"model and is not in this knowledge base, so any records below are "
                f"about other species.")
        context = note + ("\n\n---\n\n" + context if context else "")
    return folders, context


@app.post("/api/chat")
async def chat(payload: dict):
    """Answer a birding question, grounded in the species knowledge base."""
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "question is required")

    folders, context = _chat_context(question, payload.get("folder"), payload.get("subject"))
    answer = llm.chat(question, context, history=payload.get("history"),
                      model=payload.get("model"))
    return {**answer, "grounded_in": folders}


@app.post("/api/chat/stream")
async def chat_stream(payload: dict):
    """Same as /api/chat but server-sent events, so the UI can render tokens live."""
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "question is required")

    folders, context = _chat_context(question, payload.get("folder"), payload.get("subject"))

    def events():
        yield f"data: {json.dumps({'type': 'grounding', 'folders': folders})}\n\n"
        streamed = False
        try:
            for piece in llm.chat(question, context, history=payload.get("history"),
                                  model=payload.get("model"), stream=True):
                streamed = True
                yield f"data: {json.dumps({'type': 'token', 'text': piece})}\n\n"
        except Exception as e:
            if not streamed:
                # Every provider failed before saying anything: answer from the
                # field guide instead, as ordinary text. Never an error screen.
                yield f"data: {json.dumps({'type': 'token', 'text': llm.template_answer(context)})}\n\n"
            else:
                # Cut off mid-answer: keep what arrived and say so in-band.
                yield ("data: " + json.dumps({
                    "type": "error",
                    "text": " …the answer was cut off. Try asking again.",
                    "reason": f"{type(e).__name__}: {e}"}) + "\n\n")
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ════════════════════════════════════════════════════════════
# LIFE LIST
# ════════════════════════════════════════════════════════════
@app.get("/api/sightings")
def list_sightings(limit: int = Query(200, ge=1, le=1000), offset: int = Query(0, ge=0)):
    return sightings.list_all(limit=limit, offset=offset)


@app.post("/api/sightings")
async def add_sighting(
    result: str = Form(...),
    notes: str = Form(""),
    image: UploadFile = File(None),
):
    """Save an identification to the life list.

    `result` is the JSON result dict as a string, so the optional thumbnail can
    ride along in the same multipart request.
    """
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        raise HTTPException(400, "result must be valid JSON") from None
    if not isinstance(parsed, dict) or not parsed.get("species"):
        raise HTTPException(400, "result does not look like an identification")

    thumb = None
    if image is not None:
        data = await _read_upload(image)
        # The id is generated inside add(), so the thumbnail is written first
        # under its own name and the row simply points at it.
        thumb = sightings.save_thumbnail(_open_image(data), uuid.uuid4().hex[:12])

    return sightings.add(parsed, notes=notes or None, thumb=thumb)


@app.delete("/api/sightings/{sighting_id}")
def delete_sighting(sighting_id: str):
    if not sightings.delete(sighting_id):
        raise HTTPException(404, "no such sighting")
    return JSONResponse({"deleted": sighting_id})


@app.get("/api/sightings/stats")
def sighting_stats():
    return sightings.stats(all_species_folders=bird_core.CLASS_NAMES)


# ════════════════════════════════════════════════════════════
# THE DECK
# ════════════════════════════════════════════════════════════
@app.get("/api/deck")
def deck():
    """Both sides of the collection plus progress, in one call for the deck view."""
    sides = sightings.deck_list()
    return {**sides,
            "stats": sightings.deck_stats(all_species_folders=bird_core.CLASS_NAMES)}


@app.get("/api/deck/stats")
def deck_stats():
    return sightings.deck_stats(all_species_folders=bird_core.CLASS_NAMES)


@app.post("/api/deck/register")
async def deck_register(
    result: str = Form(...),
    force: bool = Form(False),
    image: UploadFile = File(None),
):
    """File an identification into the deck by hand.

    Backs the "add it anyway" action offered when neither the classifier nor the
    verifier was confident enough to file it automatically.
    """
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        raise HTTPException(400, "result must be valid JSON") from None
    if not isinstance(parsed, dict) or not parsed.get("species"):
        raise HTTPException(400, "result does not look like an identification")

    thumb = None
    if image is not None:
        data = await _read_upload(image)
        thumb = sightings.save_thumbnail(_open_image(data), uuid.uuid4().hex[:12])

    return sightings.deck_register(parsed, parsed.get("verification"),
                                   thumb=thumb, force=force)


@app.delete("/api/deck/{key}")
def deck_delete(key: str):
    """Remove a card. Plain {key}, not {key:path}: deck keys ("073.Blue_Jay",
    "ext:Ara macao") never contain a slash, and the greedy path convertor would
    happily swallow the /thumb suffix of the route below."""
    if not sightings.deck_delete(key):
        raise HTTPException(404, "no such card")
    return JSONResponse({"deleted": key})


@app.get("/api/deck/{key}/thumb")
def deck_thumb(key: str):
    entry = sightings.deck_get(key)
    if not entry:
        raise HTTPException(404, "no such card")
    path = sightings.thumbnail_path(entry.get("thumb"))
    if not path:
        raise HTTPException(404, "no photo for that card")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/sightings/{sighting_id}/thumb")
def sighting_thumb(sighting_id: str):
    row = sightings._get(sighting_id)
    if not row:
        raise HTTPException(404, "no such sighting")
    path = sightings.thumbnail_path(row.get("thumb"))
    if not path:
        raise HTTPException(404, "no thumbnail for that sighting")
    return FileResponse(path, media_type="image/jpeg")
