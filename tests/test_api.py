"""API smoke tests over FastAPI's TestClient — no server needed.

The LLM is stubbed throughout. These tests assert the HTTP contract: status
codes, response shapes, path-traversal guards, and that a missing Ollama daemon
degrades instead of 500ing. Whether qwen3 writes good prose is not something a
test can pin, and mocking it keeps the suite fast and offline.
"""

import glob
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api
from services import sightings


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    # Point the life list at a scratch database so a test run never touches the
    # user's real sightings.
    db = tmp_path_factory.mktemp("db") / "sightings.db"
    thumbs = tmp_path_factory.mktemp("thumbs")
    monkeypatch_module.setattr(sightings, "DB_PATH", str(db))
    monkeypatch_module.setattr(sightings, "THUMB_DIR", str(thumbs))
    with TestClient(api.app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def bird_bytes():
    matches = sorted(glob.glob("CUB_200_2011/images/*/*.jpg"))
    if not matches:
        pytest.skip("CUB_200_2011 images not present (gitignored dataset)")
    return open(matches[0], "rb").read()


def _png_bytes(color=(120, 40, 160)):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, "PNG")
    return buf.getvalue()


# ── Status ───────────────────────────────────────────────────
def test_health(client):
    body = client.get("/api/health").json()
    assert body["num_species"] == 200
    assert "openset" in body and "llm" in body
    assert "caveat" in body


# ── Identification ───────────────────────────────────────────
def test_identify(client, bird_bytes):
    res = client.post("/api/identify", files={"image": ("b.jpg", bird_bytes, "image/jpeg")})
    assert res.status_code == 200
    body = res.json()
    assert body["species"]["display_name"]
    assert 0 <= body["confidence"] <= 1
    assert len(body["top5"]) == 5


def test_identify_rejects_a_non_image(client):
    res = client.post("/api/identify",
                      files={"image": ("notes.txt", b"this is not an image", "text/plain")})
    assert res.status_code == 415


def test_identify_rejects_undecodable_bytes(client):
    res = client.post("/api/identify",
                      files={"image": ("broken.jpg", b"\xff\xd8not-a-jpeg", "image/jpeg")})
    assert res.status_code == 400


def test_identify_rejects_an_empty_upload(client):
    res = client.post("/api/identify", files={"image": ("empty.jpg", b"", "image/jpeg")})
    assert res.status_code == 400


def test_gradcam_returns_a_fetchable_url(client, bird_bytes):
    body = client.post("/api/gradcam",
                       files={"image": ("b.jpg", bird_bytes, "image/jpeg")}).json()
    assert body["url"].startswith("/api/media/")
    assert client.get(body["url"]).status_code == 200


# ── Field guide ──────────────────────────────────────────────
def test_species_list(client):
    body = client.get("/api/species").json()
    assert body["total"] == 200
    assert len(body["species"]) == 200


def test_species_detail(client):
    body = client.get("/api/species/073.Blue_Jay").json()
    assert body["display_name"] == "Blue Jay"
    assert body["family"]


def test_unknown_species_is_404(client):
    assert client.get("/api/species/999.Not_A_Bird").status_code == 404


def test_compare_requires_two_different_species(client):
    assert client.get("/api/compare?a=073.Blue_Jay&b=073.Blue_Jay").status_code == 400
    assert client.get("/api/compare?a=073.Blue_Jay&b=999.Nope").status_code == 404


def test_compare_table_is_built_from_the_kb_not_the_llm(client, monkeypatch):
    """The prose may be generated; the rows must come from real KB fields."""
    from services import llm
    monkeypatch.setattr(llm, "_chat", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    body = client.get("/api/compare?a=017.Cardinal&b=073.Blue_Jay").json()
    assert body["generated"] is False       # LLM was unavailable
    assert len(body["rows"]) > 0            # table survived anyway
    assert body["summary"]                  # and so did a summary
    labels = {row["label"] for row in body["rows"]}
    assert "Scientific name" in labels


def test_samples_listing_and_streaming(client):
    body = client.get("/api/samples").json()
    if not body["samples"]:
        pytest.skip("bird_audio_samples not downloaded")
    first = body["samples"][0]
    assert first["url"].startswith("/api/samples/audio?file=")
    assert client.get(first["url"]).status_code == 200


def test_sample_audio_refuses_paths_not_in_the_manifest(client):
    res = client.get("/api/samples/audio", params={"file": "../../bird_core.py"})
    assert res.status_code == 404


def test_media_refuses_traversal_and_non_png(client):
    assert client.get("/api/media/..%2F..%2Fbird_core.py").status_code in (400, 404)
    assert client.get("/api/media/secrets.txt").status_code == 400


# ── LLM endpoints degrade rather than fail ───────────────────
def test_narrate_needs_a_result(client):
    assert client.post("/api/narrate", json={"nonsense": True}).status_code == 400


def test_narrate_falls_back_to_template_text(client, bird_bytes, monkeypatch):
    from services import llm
    monkeypatch.setattr(llm, "_chat", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ollama down")))
    result = client.post("/api/identify",
                         files={"image": ("b.jpg", bird_bytes, "image/jpeg")}).json()
    res = client.post("/api/narrate", json={"result": result})
    assert res.status_code == 200          # a dead LLM is degraded, not an error
    body = res.json()
    assert body["generated"] is False
    assert body["text"]
    assert result["species"]["display_name"] in body["text"]


def test_narrate_says_so_when_the_photo_was_rejected(client, monkeypatch):
    from services import llm
    fake = {
        "kind": "image",
        "species": {"display_name": "Some Bird"},
        "confidence": 0.9, "confidence_band": "high",
        "info": {"habitat": "x"},
        "openset": {"enabled": True, "is_bird": False},
    }
    body = llm.narrate(fake)
    assert body["generated"] is False
    assert "two hundred" in body["text"].lower()


def test_chat_requires_a_question(client):
    assert client.post("/api/chat", json={}).status_code == 400


def test_chat_grounds_on_the_named_species(client, monkeypatch):
    from services import llm
    monkeypatch.setattr(llm, "_chat", lambda *a, **k: "A stubbed answer.")
    body = client.post("/api/chat", json={"question": "what does a blue jay eat?"}).json()
    assert body["text"] == "A stubbed answer."
    assert "073.Blue_Jay" in body["grounded_in"]


def test_chat_prefers_the_species_on_screen(client, monkeypatch):
    from services import llm
    monkeypatch.setattr(llm, "_chat", lambda *a, **k: "ok")
    body = client.post("/api/chat",
                       json={"question": "how big is it?", "folder": "017.Cardinal"}).json()
    assert body["grounded_in"][0] == "017.Cardinal"


def test_chat_stream_emits_grounding_then_tokens_then_done(client, monkeypatch):
    from services import llm
    monkeypatch.setattr(llm, "chat",
                        lambda *a, **k: iter(["Blue ", "Jays ", "eat ", "acorns."]))
    with client.stream("POST", "/api/chat/stream",
                       json={"question": "what do blue jays eat?"}) as res:
        assert res.status_code == 200
        events = [json.loads(line[6:]) for line in res.iter_lines()
                  if line.startswith("data: ")]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "grounding"
    assert kinds[-1] == "done"
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Blue Jays eat acorns."


def test_chat_stream_reports_an_llm_failure_in_band(client, monkeypatch):
    """Once the stream has started, an error can only arrive as an event."""
    from services import llm

    def explode(*a, **k):
        raise RuntimeError("ollama died mid-stream")
        yield  # pragma: no cover — makes this a generator function

    monkeypatch.setattr(llm, "chat", explode)
    with client.stream("POST", "/api/chat/stream", json={"question": "hi"}) as res:
        assert res.status_code == 200
        events = [json.loads(line[6:]) for line in res.iter_lines()
                  if line.startswith("data: ")]
    assert any(e["type"] == "error" for e in events)
    assert events[-1]["type"] == "done"


# ── Life list ────────────────────────────────────────────────
def test_sighting_lifecycle(client, bird_bytes):
    result = client.post("/api/identify",
                         files={"image": ("b.jpg", bird_bytes, "image/jpeg")}).json()

    saved = client.post("/api/sightings",
                        data={"result": json.dumps(result), "notes": "test row"},
                        files={"image": ("b.jpg", bird_bytes, "image/jpeg")}).json()
    assert saved["display_name"] == result["species"]["display_name"]
    assert saved["is_bird"] is True
    assert saved["notes"] == "test row"

    assert client.get(f"/api/sightings/{saved['id']}/thumb").status_code == 200

    listing = client.get("/api/sightings").json()
    assert any(s["id"] == saved["id"] for s in listing["sightings"])

    stats = client.get("/api/sightings/stats").json()
    assert stats["species_seen"] >= 1
    assert stats["total_species"] == 200

    assert client.delete(f"/api/sightings/{saved['id']}").status_code == 200
    assert client.delete(f"/api/sightings/{saved['id']}").status_code == 404


def test_saving_rejects_malformed_json(client):
    assert client.post("/api/sightings", data={"result": "not json{"}).status_code == 400
    assert client.post("/api/sightings", data={"result": '{"no":"species"}'}).status_code == 400


def test_rejected_photos_are_logged_but_not_counted_as_species(client):
    """A photo the gate rejected is not a bird you saw."""
    before = client.get("/api/sightings/stats").json()["species_seen"]
    rejected = {
        "kind": "image",
        "species": {"folder": "017.Cardinal", "display_name": "Cardinal",
                    "scientific_name": "Cardinalis cardinalis",
                    "family": "Cardinalidae", "order": "Passeriformes"},
        "confidence": 0.5, "confidence_band": "low",
        "openset": {"enabled": True, "is_bird": False},
    }
    saved = client.post("/api/sightings", data={"result": json.dumps(rejected)}).json()
    assert saved["is_bird"] is False

    stats = client.get("/api/sightings/stats").json()
    assert stats["species_seen"] == before      # unchanged
    assert stats["rejected_count"] >= 1         # but recorded
    client.delete(f"/api/sightings/{saved['id']}")


def test_thumbnail_path_blocks_traversal():
    assert sightings.thumbnail_path("../../../etc/passwd") is None
    assert sightings.thumbnail_path("") is None
    assert sightings.thumbnail_path(None) is None
