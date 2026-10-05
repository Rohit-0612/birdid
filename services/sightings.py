"""The life list and the deck — what you have seen, and what you have collected.

Two tables, because they answer two different questions:

`sightings` is the **event log**: one row per identification you keep, with its
date and confidence. "What did I see last Tuesday" and "how many sightings this
month" come from here.

`deck` is the **collection**: one row per *species*, Pokédex-style. The first photo
of a bird creates its card; later photos bump the encounter count and can improve
the card's photo. "How many of the 200 have I found" comes from here, and it is why
uploading the same pigeon ten times yields one card rather than ten.

The deck has two sides. Birds the trained classifier recognises land in the `cub`
side; birds identified by the open-vocabulary verifier (services/verifier.py) land
in `external`, keyed on scientific name. Routing lives in `deck_key_for()`.

No ORM and no migration framework — one file, `CREATE TABLE IF NOT EXISTS` on open,
so an existing database picks up the new table without ceremony.
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

CREATE TABLE IF NOT EXISTS deck (
    key              TEXT PRIMARY KEY,   -- '073.Blue_Jay' | 'ext:Ara macao'
    source           TEXT NOT NULL CHECK (source IN ('cub', 'external')),
    display_name     TEXT NOT NULL,
    scientific_name  TEXT,
    family           TEXT,
    "order"          TEXT,
    first_seen       TEXT NOT NULL,
    last_seen        TEXT NOT NULL,
    encounters       INTEGER NOT NULL DEFAULT 1,
    thumb            TEXT,
    best_confidence  REAL,
    identified_by    TEXT,               -- 'cub' | 'verifier'
    agreed           INTEGER             -- 1/0 when both models ran, else NULL
);
CREATE INDEX IF NOT EXISTS idx_deck_source ON deck(source);
CREATE INDEX IF NOT EXISTS idx_deck_first  ON deck(first_seen DESC);
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


# ════════════════════════════════════════════════════════════
# THE DECK — one card per species
# ════════════════════════════════════════════════════════════
# Confidence below which the classifier's own answer is not trusted on its own and
# the verifier's opinion decides. Matches the API's verification trigger.
TRUST_CONFIDENCE = 0.60


def deck_key_for(result, verification=None, trust_confidence=TRUST_CONFIDENCE):
    """Decide which card an identification belongs to, or None to register nothing.

    Four outcomes, in priority order:

    1. The classifier is trusted — the open-set gate accepted the photo and
       confidence clears the bar — so the card is its species.
    2. The classifier is not trusted but the verifier is confident, and the species
       it names *is* one of the 200. The verifier has corrected the pick; the card is
       still on the CUB side, under the corrected species.
    3. The verifier is confident about a species outside the 200 — a new bird.
    4. Neither is confident. Register nothing rather than guess: a wrong card is
       worse than a missing one, because the deck is meant to be a record.

    Returns (key, source, fields) or (None, None, reason).
    """
    species = result.get("species") or {}
    gate = result.get("openset") or {}
    confidence = result.get("confidence") or 0.0

    gate_ok = gate.get("is_bird", True)
    classifier_trusted = gate_ok and confidence >= trust_confidence

    if classifier_trusted and species.get("folder"):
        info = result.get("info") or {}
        return species["folder"], "cub", {
            "display_name":    species.get("display_name") or species["folder"],
            "scientific_name": species.get("scientific_name"),
            "family":          species.get("family") or info.get("family"),
            "order":           species.get("order") or info.get("order"),
            "best_confidence": confidence,
            "identified_by":   "cub",
        }

    verified = verification or {}
    if verified.get("ran") and verified.get("confident") and verified.get("best"):
        best = verified["best"]
        if verified.get("cub_folder"):
            return verified["cub_folder"], "cub", {
                "display_name":    best["common_name"],
                "scientific_name": best["scientific_name"],
                "family":          None,
                "order":           None,
                "best_confidence": best.get("similarity"),
                "identified_by":   "verifier",
            }
        return f"ext:{best['scientific_name']}", "external", {
            "display_name":    best["common_name"],
            "scientific_name": best["scientific_name"],
            "family":          None,
            "order":           None,
            "best_confidence": best.get("similarity"),
            "identified_by":   "verifier",
        }

    if not gate_ok and not verified.get("ran"):
        return None, None, "rejected by the open-set gate and no verifier available"
    if not gate_ok:
        return None, None, "not one of the 200, and the verifier could not name it"
    return None, None, (f"confidence {confidence:.0%} is below the "
                        f"{trust_confidence:.0%} bar and the verifier could not name it")


def resolve_answer(result, verification=None, trust_confidence=TRUST_CONFIDENCE):
    """The one answer to show the user, decided exactly as the deck decides.

    The classifier and the verifier both report on every photo that needs a
    second look, and the deck already picks between them. This reuses that pick,
    so the headline on screen and the card in the deck can never disagree.

    `band` is the classifier's confidence band when it answered, "confirmed" when
    the verifier did (its similarity is not a probability, so no percentage is
    implied), and None when neither model would commit — `status` is then
    "unsure" and the classifier's pick is offered only as a guess.
    """
    key, source, fields = deck_key_for(result, verification, trust_confidence)
    species = result.get("species") or {}

    if key is None:
        return {
            "status": "unsure",
            "identified_by": None,
            "display_name": None,
            "scientific_name": None,
            "family": None,
            "order": None,
            "folder": None,
            "source": None,
            "confidence": None,
            "band": None,
            "reason": fields,
            "best_guess": species.get("display_name"),
        }

    by_classifier = fields["identified_by"] == "cub"
    return {
        "status": "identified",
        "identified_by": fields["identified_by"],
        "display_name": fields["display_name"],
        "scientific_name": fields["scientific_name"],
        "family": fields["family"],
        "order": fields["order"],
        "folder": key if source == "cub" else None,
        "source": source,
        "confidence": result.get("confidence") if by_classifier else None,
        "band": result.get("confidence_band") if by_classifier else "confirmed",
        "reason": None,
        "best_guess": None,
    }


def deck_register(result, verification=None, thumb=None, force=False):
    """Add or update this species' card. Returns {entry, created, reason}.

    On a repeat encounter the counter goes up, `last_seen` moves, and the thumbnail
    is replaced **only if the new photo scored higher** — so a card drifts toward
    your best shot of a bird rather than your most recent one.
    """
    key, source, extra = deck_key_for(result, verification)
    if key is None:
        if not force:
            return {"entry": None, "created": False, "reason": extra}
        # Manual "add anyway": fall back to whatever the classifier said.
        species = result.get("species") or {}
        key = species.get("folder") or f"ext:{species.get('display_name', 'unknown')}"
        source = "cub" if species.get("folder") else "external"
        extra = {
            "display_name":    species.get("display_name") or "Unknown",
            "scientific_name": species.get("scientific_name"),
            "family":          species.get("family"),
            "order":           species.get("order"),
            "best_confidence": result.get("confidence"),
            "identified_by":   "manual",
        }

    verified = verification or {}
    agreed = None
    if verified.get("ran") and verified.get("cub_folder") and (result.get("species") or {}).get("folder"):
        agreed = int(verified["cub_folder"] == result["species"]["folder"])

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new_confidence = extra.get("best_confidence")

    with _connect() as conn:
        existing = conn.execute("SELECT * FROM deck WHERE key = ?", (key,)).fetchone()

        if existing is None:
            conn.execute(
                'INSERT INTO deck (key, source, display_name, scientific_name, '
                'family, "order", first_seen, last_seen, encounters, thumb, '
                'best_confidence, identified_by, agreed) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)',
                (key, source, extra["display_name"], extra["scientific_name"],
                 extra["family"], extra["order"], now, now, thumb,
                 new_confidence, extra["identified_by"], agreed),
            )
            created = True
        else:
            old_confidence = existing["best_confidence"]
            better = (new_confidence is not None
                      and (old_confidence is None or new_confidence > old_confidence))
            conn.execute(
                'UPDATE deck SET last_seen = ?, encounters = encounters + 1, '
                'thumb = CASE WHEN ? THEN ? ELSE thumb END, '
                'best_confidence = CASE WHEN ? THEN ? ELSE best_confidence END, '
                'family = COALESCE(family, ?), "order" = COALESCE("order", ?), '
                'scientific_name = COALESCE(scientific_name, ?), '
                'agreed = COALESCE(?, agreed) '
                'WHERE key = ?',
                (now,
                 1 if (better and thumb) else 0, thumb,
                 1 if better else 0, new_confidence,
                 extra["family"], extra["order"], extra["scientific_name"],
                 agreed, key),
            )
            created = False

            # A superseded thumbnail would otherwise leak a file per encounter.
            if better and thumb and existing["thumb"] and existing["thumb"] != thumb:
                stale = thumbnail_path(existing["thumb"])
                if stale:
                    try:
                        os.remove(stale)
                    except OSError:
                        pass

    return {"entry": deck_get(key), "created": created, "reason": None}


def deck_get(key):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM deck WHERE key = ?", (key,)).fetchone()
    return _deck_row(row) if row else None


def _deck_row(row):
    entry = dict(row)
    entry["agreed"] = None if entry["agreed"] is None else bool(entry["agreed"])
    return entry


def deck_list():
    """Both sides of the deck, newest discovery first."""
    with _connect() as conn:
        rows = [_deck_row(r) for r in
                conn.execute("SELECT * FROM deck ORDER BY first_seen DESC").fetchall()]
    return {
        "cub":      [r for r in rows if r["source"] == "cub"],
        "external": [r for r in rows if r["source"] == "external"],
    }


def deck_delete(key):
    entry = deck_get(key)
    if not entry:
        return False
    with _connect() as conn:
        conn.execute("DELETE FROM deck WHERE key = ?", (key,))
    path = thumbnail_path(entry.get("thumb"))
    if path:
        try:
            os.remove(path)
        except OSError:
            pass
    return True


def deck_stats(all_species_folders=None):
    """Collection progress. `all_species_folders` is the model's class list."""
    sides = deck_list()
    total = len(all_species_folders) if all_species_folders else None
    found = len(sides["cub"])

    by_family = {}
    for entry in sides["cub"] + sides["external"]:
        name = entry["family"] or "Unknown"
        by_family[name] = by_family.get(name, 0) + 1

    encounters = sum(e["encounters"] for e in sides["cub"] + sides["external"])
    return {
        "found":            found,
        "total_species":    total,
        "progress":         (found / total) if total else None,
        "new_birds":        len(sides["external"]),
        "total_encounters": encounters,
        "by_family":        dict(sorted(by_family.items(), key=lambda kv: -kv[1])),
        "most_seen":        max((sides["cub"] + sides["external"]),
                                key=lambda e: e["encounters"], default=None),
    }
