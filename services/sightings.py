"""The life list — every identification you choose to keep.

Birders keep a life list: the running record of every species they have
personally seen. Until now every identification this app made was thrown away
the moment the next photo was uploaded, which is what made it a demo rather
than a tool. This is a SQLite table plus thumbnails, no ORM, no migrations
framework — one file, one schema, `CREATE TABLE IF NOT EXISTS` on open.

Deliberately *not* one row per species: the same bird seen twice is two
sightings on two dates, and the interesting stats (how many of the 200 have you
found, what did you see this month) are derived, not stored.
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get("BIRD_DB_PATH", os.path.join(PROJECT_DIR, "data", "sightings.db"))
THUMB_DIR = os.path.join(PROJECT_DIR, "data", "sightings", "thumbs")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sightings (
    id                 TEXT PRIMARY KEY,
    ts                 TEXT NOT NULL,
    kind               TEXT NOT NULL CHECK (kind IN ('image', 'audio')),
    folder             TEXT,
    display_name       TEXT NOT NULL,
    scientific_name    TEXT,
    family             TEXT,
    "order"            TEXT,
    confidence         REAL,
    confidence_band    TEXT,
    is_bird            INTEGER NOT NULL DEFAULT 1,
    thumb              TEXT,
    notes              TEXT
);
CREATE INDEX IF NOT EXISTS idx_sightings_ts     ON sightings(ts DESC);
CREATE INDEX IF NOT EXISTS idx_sightings_folder ON sightings(folder);
"""


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # WAL so a long-running read (the stats query) never blocks a save.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def _row_to_dict(row):
    d = dict(row)
    d["is_bird"] = bool(d["is_bird"])
    return d


def save_thumbnail(pil_image, sighting_id, size=320):
    """Persist a small square-ish thumbnail. Returns the stored filename."""
    os.makedirs(THUMB_DIR, exist_ok=True)
    img = pil_image.convert("RGB")
    img.thumbnail((size, size))
    name = f"{sighting_id}.jpg"
    img.save(os.path.join(THUMB_DIR, name), "JPEG", quality=82, optimize=True)
    return name


def thumbnail_path(name):
    """Resolve a stored thumbnail name to a path, or None if it is gone.

    Guards against path traversal: `name` reaches here from a URL, and the API
    hands the result straight to a file response.
    """
    if not name or os.path.basename(name) != name:
        return None
    path = os.path.join(THUMB_DIR, name)
    return path if os.path.exists(path) else None


def add(result, notes=None, thumb=None):
    """Record one identification. `result` is a bird_core result dict."""
    sighting_id = uuid.uuid4().hex[:12]

    if result.get("kind") == "audio":
        sp = result.get("species") or {}
        info = result.get("info") or {}
        row = {
            "folder":          sp.get("cub_folder"),
            "display_name":    sp.get("common_name") or "Unknown",
            "scientific_name": sp.get("scientific_name"),
            "family":          info.get("family"),
            "order":           info.get("order"),
            "is_bird":         1,
        }
    else:
        sp = result.get("species") or {}
        row = {
            "folder":          sp.get("folder"),
            "display_name":    sp.get("display_name") or "Unknown",
            "scientific_name": sp.get("scientific_name"),
            "family":          sp.get("family"),
            "order":           sp.get("order"),
            "is_bird":         int((result.get("openset") or {}).get("is_bird", True)),
        }

    record = {
        "id":              sighting_id,
        "ts":              datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind":            result.get("kind", "image"),
        "confidence":      result.get("confidence"),
        "confidence_band": result.get("confidence_band"),
        "thumb":           thumb,
        "notes":           notes,
        **row,
    }

    with _connect() as conn:
        conn.execute(
            'INSERT INTO sightings (id, ts, kind, folder, display_name, '
            'scientific_name, family, "order", confidence, confidence_band, '
            'is_bird, thumb, notes) VALUES (:id, :ts, :kind, :folder, '
            ':display_name, :scientific_name, :family, :order, :confidence, '
            ':confidence_band, :is_bird, :thumb, :notes)',
            record,
        )
    return _get(sighting_id)


def _get(sighting_id):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM sightings WHERE id = ?", (sighting_id,)).fetchone()
    return _row_to_dict(row) if row else None


def list_all(limit=200, offset=0):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM sightings ORDER BY ts DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) AS n FROM sightings").fetchone()["n"]
    return {"sightings": [_row_to_dict(r) for r in rows], "total": total,
            "limit": limit, "offset": offset}


def delete(sighting_id):
    """Remove a sighting and its thumbnail. Returns True if a row was deleted."""
    existing = _get(sighting_id)
    if not existing:
        return False
    with _connect() as conn:
        conn.execute("DELETE FROM sightings WHERE id = ?", (sighting_id,))
    path = thumbnail_path(existing.get("thumb"))
    if path:
        try:
            os.remove(path)
        except OSError:
            pass  # the row is gone either way; an orphan thumb is harmless
    return True


def stats(all_species_folders=None):
    """Progress and breakdowns for the life-list view.

    `all_species_folders` is the model's full class list, so "37 of 200" can be
    reported without this module importing torch.
    """
    with _connect() as conn:
        rows = [_row_to_dict(r) for r in
                conn.execute("SELECT * FROM sightings").fetchall()]

    # Only confirmed birds count toward the list. A rejected open-set photo is
    # kept for the timeline but is not a species you have seen.
    real = [r for r in rows if r["is_bird"] and r["folder"]]
    seen_folders = {r["folder"] for r in real}

    by_family, by_month = {}, {}
    for r in real:
        by_family[r["family"] or "Unknown"] = by_family.get(r["family"] or "Unknown", 0) + 1
        by_month[r["ts"][:7]] = by_month.get(r["ts"][:7], 0) + 1

    first_seen = {}
    for r in sorted(real, key=lambda x: x["ts"]):
        first_seen.setdefault(r["folder"], r["ts"])

    total_species = len(all_species_folders) if all_species_folders else None
    return {
        "total_sightings":  len(rows),
        "species_seen":     len(seen_folders),
        "total_species":    total_species,
        "progress":         (len(seen_folders) / total_species) if total_species else None,
        "by_family":        dict(sorted(by_family.items(), key=lambda kv: -kv[1])),
        "by_month":         dict(sorted(by_month.items())),
        "rejected_count":   sum(1 for r in rows if not r["is_bird"]),
        "audio_count":      sum(1 for r in rows if r["kind"] == "audio"),
        "first_seen":       first_seen,
        "latest":           rows and max(r["ts"] for r in rows) or None,
    }


def export_json():
    """The whole list, for backup or for moving to a real birding app."""
    return json.dumps(list_all(limit=10**6)["sightings"], indent=2)
