"""The result-dict contract.

Everything downstream reads these keys: the React dashboard, the voice narrator,
the chat grounding, the life-list writer and the text renderer. A silent rename
here breaks four consumers at once, and the browser would show blank fields
rather than an error — so the shape is pinned.

Importing bird_core loads the checkpoint, which takes a few seconds. That is
paid once for the whole module thanks to the session-scoped fixtures.
"""

import glob
import os

import pytest

import bird_core
import bird_text

CUB_GLOB = "CUB_200_2011/images/*/*.jpg"
OOD_GLOB = "data/openset/ood/nonanimal/*/*"


@pytest.fixture(scope="session")
def bird_image():
    matches = sorted(glob.glob(CUB_GLOB))
    if not matches:
        pytest.skip("CUB_200_2011 images not present (gitignored dataset)")
    return matches[0]


@pytest.fixture(scope="session")
def result(bird_image):
    return bird_core.identify_image(bird_image)


# ── Shape ────────────────────────────────────────────────────
def test_top_level_keys(result):
    assert set(result) >= {
        "kind", "species", "confidence", "confidence_band", "top5",
        "info", "openset", "provenance", "timing_ms",
    }
    assert result["kind"] == "image"


def test_species_block(result):
    assert set(result["species"]) == {
        "folder", "display_name", "scientific_name", "family", "order",
    }
    assert result["species"]["folder"] in bird_core.CLASS_NAMES


def test_info_block_exposes_what_the_old_text_dropped(result):
    # family, order and range_description have been in the knowledge base since
    # it was generated and were displayed nowhere. Regressing that is easy.
    info = result["info"]
    for key in ("habitat", "range_description", "migration", "field_marks",
                "similar_species", "size_cm", "diet", "conservation_status",
                "fun_fact", "family", "order"):
        assert key in info, f"info is missing {key}"


def test_similar_species_are_dicts_not_a_single_tuple(result):
    # The old species_info() collapsed look-alikes to one (name, how) tuple and
    # threw the rest away.
    for entry in result["info"]["similar_species"]:
        assert set(entry) >= {"name", "how_to_distinguish", "source"}
        assert entry["source"] in {"curated", "llm"}


def test_confidence_is_a_fraction_not_a_percentage(result):
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["confidence_band"] in {"high", "moderate", "low"}
    assert result["confidence"] == pytest.approx(result["top5"][0]["confidence"])


def test_top5_is_ranked(result):
    confidences = [row["confidence"] for row in result["top5"]]
    assert confidences == sorted(confidences, reverse=True)
    assert len(result["top5"]) == 5


def test_provenance_reports_the_unaudited_checkpoint_honestly(result):
    prov = result["provenance"]
    assert set(prov) >= {"checkpoint", "model_audited", "caveat", "kb_source",
                         "num_species", "openset_enabled"}
    # An unaudited checkpoint must carry a caveat, and an audited one must not.
    assert (prov["caveat"] is None) == bool(prov["model_audited"])


def test_confidence_band_thresholds():
    assert bird_core.confidence_band(0.95) == "high"
    assert bird_core.confidence_band(0.80) == "high"
    assert bird_core.confidence_band(0.70) == "moderate"
    assert bird_core.confidence_band(0.60) == "moderate"
    assert bird_core.confidence_band(0.10) == "low"
    # Audio uses looser bands on purpose: BirdNET scores run lower.
    assert bird_core.confidence_band(0.75, low=0.40, high=0.70) == "high"


# ── Species detail and catalogue ─────────────────────────────
def test_species_detail_for_every_class():
    """No class may blow up or return a partial record."""
    required = {"folder", "display_name", "habitat", "migration",
                "similar_species", "field_marks"}
    for folder in bird_core.CLASS_NAMES:
        detail = bird_core.species_detail(folder)
        assert required <= set(detail), f"{folder} is missing keys"
        assert detail["folder"] == folder


def test_catalogue_covers_all_classes():
    catalogue = bird_core.species_catalogue()
    assert len(catalogue) == bird_core.NUM_SPECIES
    assert [row["folder"] for row in catalogue] == list(bird_core.CLASS_NAMES)


def test_curated_notes_win_over_generated_ones():
    """Hand-written text must never be overwritten by the LLM overlay."""
    checked = 0
    for folder, curated in bird_core.HABITAT_MAP.items():
        if folder in bird_core.SPECIES_KB:
            assert bird_core.species_detail(folder)["habitat"] == curated
            checked += 1
    assert checked > 0, "no overlap to verify — the fixture data changed"


def test_health_is_serialisable_and_complete():
    health = bird_core.health()
    assert set(health) >= {"device", "num_species", "checkpoint", "openset",
                           "kb_records", "birdnet", "coverage", "model_audited"}
    import json
    json.dumps(health)  # the API returns this directly


# ── Text renderer ────────────────────────────────────────────
def test_text_renderer_round_trips(result):
    text = bird_text.format_image_result(result)
    assert result["species"]["display_name"] in text
    assert "TOP 5 PREDICTIONS:" in text
    for row in result["top5"]:
        assert row["display_name"] in text


def test_text_renderer_handles_a_species_with_no_notes():
    """Species with neither curated nor generated notes must still render."""
    bare = {
        "kind": "image",
        "species": {"folder": "001.X", "display_name": "Mystery Bird",
                    "scientific_name": None, "family": None, "order": None},
        "confidence": 0.42, "confidence_band": "low",
        "top5": [{"folder": "001.X", "display_name": "Mystery Bird",
                  "confidence": 0.42, "habitat": "unknown"}],
        "info": {"habitat": "unknown", "migration": "unknown",
                 "field_marks": [], "similar_species": [], "used_llm": False},
        "openset": {"enabled": False, "is_bird": True},
        "provenance": {"num_species": 200},
        "timing_ms": {"total": 1.0},
    }
    text = bird_text.format_image_result(bare)
    assert "Mystery Bird" in text
    assert "LOW CONFIDENCE" in text


# ── Open-set gate ────────────────────────────────────────────
def test_openset_block_present(result):
    gate = result["openset"]
    assert set(gate) >= {"enabled", "is_bird", "score", "threshold", "method"}
    assert isinstance(gate["is_bird"], bool)


@pytest.mark.skipif(not bird_core.OPENSET["enabled"],
                    reason="threshold.json not fitted; run scripts/fit_openset.py")
def test_a_real_bird_is_accepted(result):
    assert result["openset"]["is_bird"] is True


@pytest.mark.skipif(not bird_core.OPENSET["enabled"],
                    reason="threshold.json not fitted; run scripts/fit_openset.py")
def test_a_photo_of_an_object_is_rejected():
    """The behaviour the gate exists for: a bicycle is not a Brown Thrasher."""
    matches = sorted(glob.glob(OOD_GLOB))
    if not matches:
        pytest.skip("open-set OOD images not downloaded")
    rejected = sum(
        not bird_core.identify_image(path)["openset"]["is_bird"]
        for path in matches[:6]
    )
    # The fitted rate on this tier is 0-3% acceptance, so demand a clear majority
    # rather than perfection — one borderline image should not fail the suite.
    assert rejected >= 5, f"only {rejected}/6 non-animal photos were rejected"


def test_the_rejection_banner_appears_in_the_text_report():
    text = bird_text.format_image_result({
        "kind": "image",
        "species": {"folder": "001.X", "display_name": "Some Bird",
                    "scientific_name": None, "family": None, "order": None},
        "confidence": 0.9, "confidence_band": "high",
        "top5": [{"folder": "001.X", "display_name": "Some Bird",
                  "confidence": 0.9, "habitat": "x"}],
        "info": {"habitat": "x", "migration": "y", "field_marks": [],
                 "similar_species": [], "used_llm": False},
        "openset": {"enabled": True, "is_bird": False, "score": 6.0,
                    "threshold": 5.0, "method": "entropy"},
        "provenance": {"num_species": 200},
        "timing_ms": {"total": 1.0},
    })
    assert "NOT ONE OF MY 200 BIRDS" in text
    # The warning has to come before the ranking, or it reads as a footnote.
    assert text.index("NOT ONE OF MY") < text.index("Some Bird")


def test_openset_scores_are_shared_with_the_fitting_script():
    """fit_openset.py imports these; a threshold is invalid against other maths."""
    import torch
    assert set(bird_core.OPENSET_SCORES) == {"energy", "msp", "entropy", "margin"}
    logits = torch.randn(2, bird_core.NUM_SPECIES)
    for method in bird_core.OPENSET_SCORES:
        scores = bird_core.openset_score(logits, method, 2.0)
        assert scores.shape == (2,)
    with pytest.raises(ValueError):
        bird_core.openset_score(logits, "nonexistent")


# ── Name matching ────────────────────────────────────────────
@pytest.mark.parametrize("birdnet_name,expected", [
    ("Northern Cardinal", "017.Cardinal"),
    ("Blue Jay", "073.Blue_Jay"),
    ("American Crow", "029.American_Crow"),
    ("Tyrannosaurus rex", None),
])
def test_match_cub_species(birdnet_name, expected):
    assert bird_core.match_cub_species(birdnet_name) == expected


def test_lookup_tables_are_keyed_on_folder_names_not_display_names():
    """The bug that made 173/200 lookups silently fail."""
    from bird_data import HABITAT_MAP, MIGRATION_MAP, SIMILAR_SPECIES
    valid = set(bird_core.CLASS_NAMES)
    for name, table in (("habitat", HABITAT_MAP), ("migration", MIGRATION_MAP),
                        ("similar", SIMILAR_SPECIES)):
        unreachable = set(table) - valid
        assert not unreachable, f"{name} has unreachable keys: {sorted(unreachable)}"
