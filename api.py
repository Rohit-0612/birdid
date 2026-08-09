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
from services import llm, sightings

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


@asynccontextmanager
async def lifespan(app):
    print(f"\n🌐 API ready — {bird_core.NUM_SPECIES} species on "
          f"{bird_core.DEVICE_LABEL}, open-set "
          f"{'ON' if bird_core.OPENSET['enabled'] else 'OFF'}")
    status = llm.available()
    print(f"🤖 Ollama {'ready' if status['ok'] else 'unavailable'} — "
          f"{status.get('reason', status['model'])}")
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


# ════════════════════════════════════════════════════════════
# STATUS
# ════════════════════════════════════════════════════════════
@app.get("/api/health")
def health():
    """Model, device, knowledge base, open-set gate and LLM status in one call."""
    return {**bird_core.health(), "llm": llm.available()}


# ════════════════════════════════════════════════════════════
# IDENTIFICATION
# ════════════════════════════════════════════════════════════
@app.post("/api/identify")
async def identify(image: UploadFile = File(...)):
    """Identify a bird from a photo."""
    data = await _read_upload(image)
    return bird_core.identify_image(_open_image(data))


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


def _chat_context(question, folder=None):
    """Retrieved context for a question, preferring the species on screen."""
    folders = []
    if folder and folder in bird_core.CLASS_NAMES:
        folders.append(folder)
    for extra in llm.retrieve(question, bird_core.species_catalogue(),
                              bird_core.SPECIES_KB, limit=3):
        if extra not in folders:
            folders.append(extra)
    folders = folders[:3]
    return folders, llm.build_context(folders, bird_core.species_detail)


@app.post("/api/chat")
async def chat(payload: dict):
    """Answer a birding question, grounded in the species knowledge base."""
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "question is required")

    folders, context = _chat_context(question, payload.get("folder"))
    answer = llm.chat(question, context, history=payload.get("history"),
                      model=payload.get("model"))
    return {**answer, "grounded_in": folders}


@app.post("/api/chat/stream")
async def chat_stream(payload: dict):
    """Same as /api/chat but server-sent events, so the UI can render tokens live."""
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "question is required")

    folders, context = _chat_context(question, payload.get("folder"))

    def events():
        yield f"data: {json.dumps({'type': 'grounding', 'folders': folders})}\n\n"
        try:
            for piece in llm.chat(question, context, history=payload.get("history"),
                                  model=payload.get("model"), stream=True):
                yield f"data: {json.dumps({'type': 'token', 'text': piece})}\n\n"
        except Exception as e:
            # The stream has already started, so an error has to arrive in-band.
            yield ("data: " + json.dumps({
                "type": "error",
                "text": "The local language model is not reachable. "
                        "Start it with `ollama serve`.",
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


@app.get("/api/sightings/{sighting_id}/thumb")
def sighting_thumb(sighting_id: str):
    row = sightings._get(sighting_id)
    if not row:
        raise HTTPException(404, "no such sighting")
    path = sightings.thumbnail_path(row.get("thumb"))
    if not path:
        raise HTTPException(404, "no thumbnail for that sighting")
    return FileResponse(path, media_type="image/jpeg")
