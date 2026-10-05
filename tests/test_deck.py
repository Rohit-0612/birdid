"""The deck: routing, species-level dedupe, and what must never be filed.

Routing is the part worth pinning. It has four branches and they decide what ends
up in a permanent record of the user's birding, so a regression here quietly
corrupts data rather than throwing. The verifier is never loaded in these tests —
its answers are plain dicts, so they are constructed directly.
"""

import os

import pytest

from services import sightings


@pytest.fixture(autouse=True)
def scratch_db(tmp_path, monkeypatch):
    """Every test gets its own database and thumbnail directory."""
    monkeypatch.setattr(sightings, "DB_PATH", str(tmp_path / "deck.db"))
    monkeypatch.setattr(sightings, "THUMB_DIR", str(tmp_path / "thumbs"))
    return tmp_path


def result(folder="073.Blue_Jay", name="Blue Jay", confidence=0.85,
           is_bird=True, family="Corvidae"):
    return {
        "kind": "image",
        "species": {"folder": folder, "display_name": name,
                    "scientific_name": "Cyanocitta cristata",
                    "family": family, "order": "Passeriformes"},
        "confidence": confidence, "confidence_band": "high",
        "openset": {"enabled": True, "is_bird": is_bird},
        "info": {},
    }


def verification(common="Scarlet Macaw", sci="Ara macao", confident=True,
                 cub_folder=None, similarity=0.36):
    return {"ran": True, "confident": confident, "cub_folder": cub_folder,
            "in_cub_200": cub_folder is not None,
            "best": {"common_name": common, "scientific_name": sci,
                     "similarity": similarity, "score": 0.9}}


# ── Routing: the four branches ───────────────────────────────
def test_trusted_classifier_files_to_the_cub_side():
    key, source, extra = sightings.deck_key_for(result(confidence=0.85))
    assert (key, source) == ("073.Blue_Jay", "cub")
    assert extra["identified_by"] == "cub"


def test_verifier_correcting_into_the_200_stays_on_the_cub_side():
    """A corrected pick is still one of the 200, so it belongs in that section."""
    key, source, extra = sightings.deck_key_for(
        result(folder="017.Cardinal", confidence=0.35),
        verification(common="American Crow", sci="Corvus brachyrhynchos",
                     cub_folder="029.American_Crow"),
    )
    assert (key, source) == ("029.American_Crow", "cub")
    assert extra["identified_by"] == "verifier"


def test_verifier_naming_a_species_outside_the_200_files_to_new_birds():
    key, source, extra = sightings.deck_key_for(
        result(confidence=0.15, is_bird=False), verification())
    assert (key, source) == ("ext:Ara macao", "external")
    assert extra["display_name"] == "Scarlet Macaw"


def test_nothing_is_filed_when_neither_model_is_confident():
    key, source, reason = sightings.deck_key_for(
        result(confidence=0.20, is_bird=False),
        verification(confident=False))
    assert key is None and source is None
    assert "verifier could not name it" in reason


def test_nothing_is_filed_on_low_confidence_with_no_verifier():
    key, _, reason = sightings.deck_key_for(result(confidence=0.45))
    assert key is None
    assert "below" in reason


def test_gate_rejection_without_a_verifier_explains_itself():
    key, _, reason = sightings.deck_key_for(result(confidence=0.9, is_bird=False))
    assert key is None
    assert "no verifier available" in reason


def test_a_rejected_photo_is_not_filed_even_at_high_confidence():
    """The open-set gate outranks confidence: 95% sure of the wrong universe."""
    outcome = sightings.deck_register(result(confidence=0.95, is_bird=False))
    assert outcome["entry"] is None


# ── Species-level behaviour ──────────────────────────────────
def test_the_same_species_twice_is_one_card():
    first = sightings.deck_register(result())
    second = sightings.deck_register(result())
    assert first["created"] is True
    assert second["created"] is False
    assert second["entry"]["encounters"] == 2
    assert len(sightings.deck_list()["cub"]) == 1


def test_thumbnail_is_replaced_only_by_a_better_photo(scratch_db):
    os.makedirs(sightings.THUMB_DIR, exist_ok=True)
    for name in ("first.jpg", "worse.jpg", "better.jpg"):
        open(os.path.join(sightings.THUMB_DIR, name), "wb").write(b"x")

    sightings.deck_register(result(confidence=0.80), thumb="first.jpg")

    kept = sightings.deck_register(result(confidence=0.60), thumb="worse.jpg")
    assert kept["entry"]["thumb"] == "first.jpg"
    assert kept["entry"]["best_confidence"] == pytest.approx(0.80)

    upgraded = sightings.deck_register(result(confidence=0.93), thumb="better.jpg")
    assert upgraded["entry"]["thumb"] == "better.jpg"
    assert upgraded["entry"]["best_confidence"] == pytest.approx(0.93)
    # The superseded file is deleted rather than leaking one per encounter.
    assert not os.path.exists(os.path.join(sightings.THUMB_DIR, "first.jpg"))


def test_agreement_is_recorded_when_both_models_ran():
    agreed = sightings.deck_register(
        result(confidence=0.85), verification(cub_folder="073.Blue_Jay"))
    assert agreed["entry"]["agreed"] is True

    sightings.deck_delete("073.Blue_Jay")
    disagreed = sightings.deck_register(
        result(confidence=0.85), verification(cub_folder="029.American_Crow"))
    assert disagreed["entry"]["agreed"] is False


def test_agreement_is_unknown_when_only_the_classifier_ran():
    outcome = sightings.deck_register(result(confidence=0.85))
    assert outcome["entry"]["agreed"] is None


def test_force_files_a_card_the_routing_declined():
    declined = sightings.deck_register(result(confidence=0.30))
    assert declined["entry"] is None

    forced = sightings.deck_register(result(confidence=0.30), force=True)
    assert forced["created"] is True
    assert forced["entry"]["identified_by"] == "manual"


# ── Stats ────────────────────────────────────────────────────
def test_stats_count_species_not_encounters():
    classes = [f"{i:03d}.species" for i in range(1, 201)]
    sightings.deck_register(result())
    sightings.deck_register(result())
    sightings.deck_register(result(folder="017.Cardinal", name="Cardinal",
                                  family="Cardinalidae"))
    sightings.deck_register(result(confidence=0.1, is_bird=False), verification())

    stats = sightings.deck_stats(all_species_folders=classes)
    assert stats["found"] == 2                 # two CUB species
    assert stats["new_birds"] == 1             # one beyond the 200
    assert stats["total_species"] == 200
    assert stats["progress"] == pytest.approx(2 / 200)
    assert stats["total_encounters"] == 4      # every encounter still counted
    assert stats["most_seen"]["encounters"] == 2


def test_new_birds_do_not_inflate_progress_over_the_200():
    """The external side has no ceiling, so it must not count toward found/200."""
    classes = [f"{i:03d}.species" for i in range(1, 201)]
    for sci in ("Ara macao", "Tyto alba", "Spheniscus demersus"):
        sightings.deck_register(result(confidence=0.1, is_bird=False),
                                verification(sci=sci, common=sci))
    stats = sightings.deck_stats(all_species_folders=classes)
    assert stats["found"] == 0
    assert stats["new_birds"] == 3
    assert stats["progress"] == 0


def test_delete_removes_the_card_and_its_thumbnail(scratch_db):
    os.makedirs(sightings.THUMB_DIR, exist_ok=True)
    open(os.path.join(sightings.THUMB_DIR, "t.jpg"), "wb").write(b"x")
    sightings.deck_register(result(), thumb="t.jpg")

    assert sightings.deck_delete("073.Blue_Jay") is True
    assert sightings.deck_delete("073.Blue_Jay") is False
    assert sightings.deck_get("073.Blue_Jay") is None
    assert not os.path.exists(os.path.join(sightings.THUMB_DIR, "t.jpg"))


def test_external_keys_survive_a_round_trip():
    """Scientific names contain a space and the key a colon; both must round-trip."""
    sightings.deck_register(result(confidence=0.1, is_bird=False), verification())
    entry = sightings.deck_get("ext:Ara macao")
    assert entry is not None
    assert entry["source"] == "external"
    assert entry["scientific_name"] == "Ara macao"


def test_the_deck_and_the_sighting_log_are_independent():
    """The deck dedupes by species; the log keeps every event."""
    for _ in range(3):
        sightings.add(result())
        sightings.deck_register(result())
    assert len(sightings.deck_list()["cub"]) == 1
    assert sightings.list_all()["total"] == 3


# ── The answer shown to the user ─────────────────────────────
# resolve_answer() reuses deck_key_for(), so the headline on screen and the card
# filed in the deck must name the same bird in every branch.

def test_answer_is_the_classifier_when_it_is_trusted():
    a = sightings.resolve_answer(result(confidence=0.91), None)
    assert a["status"] == "identified" and a["identified_by"] == "cub"
    assert a["display_name"] == "Blue Jay" and a["folder"] == "073.Blue_Jay"
    assert a["confidence"] == 0.91 and a["band"] == "high"


def test_answer_is_the_verifiers_correction_into_the_200():
    a = sightings.resolve_answer(result(confidence=0.2),
                                 verification("American Crow", "Corvus brachyrhynchos",
                                              cub_folder="029.American_Crow"))
    assert a["identified_by"] == "verifier" and a["band"] == "confirmed"
    assert a["display_name"] == "American Crow" and a["folder"] == "029.American_Crow"
    assert a["source"] == "cub" and a["confidence"] is None


def test_answer_names_a_bird_outside_the_200_not_the_classifiers_guess():
    """The Barn Owl case: the classifier says Northern Fulmar at 9%, the verifier
    says Barn Owl — the user must see Barn Owl, with no percentage."""
    a = sightings.resolve_answer(result("091.Northern_Fulmar", "Northern Fulmar",
                                        confidence=0.09, is_bird=False),
                                 verification("Barn Owl", "Tyto alba"))
    assert a["display_name"] == "Barn Owl" and a["scientific_name"] == "Tyto alba"
    assert a["source"] == "external" and a["folder"] is None
    assert a["band"] == "confirmed" and a["confidence"] is None
    assert a["best_guess"] is None


def test_answer_is_unsure_when_neither_model_commits():
    a = sightings.resolve_answer(result(confidence=0.3),
                                 verification(confident=False))
    assert a["status"] == "unsure" and a["display_name"] is None
    assert a["best_guess"] == "Blue Jay"
    assert "verifier could not name it" in a["reason"]


@pytest.mark.parametrize("res,ver", [
    (result(confidence=0.91), None),
    (result(confidence=0.2), verification(cub_folder="029.American_Crow")),
    (result(confidence=0.1, is_bird=False), verification()),
    (result(confidence=0.3), verification(confident=False)),
])
def test_the_answer_and_the_deck_card_always_agree(res, ver):
    key, source, fields = sightings.deck_key_for(res, ver)
    a = sightings.resolve_answer(res, ver)
    if key is None:
        assert a["status"] == "unsure"
    else:
        assert a["display_name"] == fields["display_name"]
        assert a["source"] == source
