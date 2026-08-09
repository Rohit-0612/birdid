"""Render bird_core result dicts as the plain-text report the Gradio app shows.

Kept deliberately separate from bird_core so there is exactly one place that
knows about emoji, column alignment and box-drawing characters. The JSON API
never touches this module; the Gradio tabs render through it, which is what
makes the refactor a no-op for the existing UI.
"""

from bird_core import LLM_DISCLAIMER

RULE = "─" * 55


def _confidence_line(result):
    conf = result["confidence"] * 100
    band = result["confidence_band"]
    if band == "low":
        return f"⚠️  LOW CONFIDENCE ({conf:.1f}%) — Try a clearer photo"
    if band == "moderate":
        return f"🟡  MODERATE CONFIDENCE ({conf:.1f}%) — Likely correct"
    return f"✅  HIGH CONFIDENCE ({conf:.1f}%) — Very sure"


def _openset_banner(result):
    """Warn up front when the photo probably is not one of the 200 species.

    The classifier renormalises over 200 classes, so without this the report
    below reads as a confident identification of a dog. Shown before anything
    else because it changes how every number underneath should be read.
    """
    gate = result.get("openset") or {}
    if not gate.get("enabled") or gate.get("is_bird", True):
        return ""
    return (
        "🚫  NOT ONE OF MY 200 BIRDS\n"
        f"    Open-set score {gate['score']:.2f} vs threshold "
        f"{gate['threshold']:.2f} ({gate['method']}).\n"
        "    The ranking below is what the model would say if forced to pick,\n"
        "    not an identification. Try a clearer photo of a single bird.\n"
        + RULE + "\n"
    )


def format_image_result(result):
    """The photo-identification report, byte-identical to the pre-refactor text."""
    info = result["info"]
    out = _openset_banner(result)

    out += f"\n🐦  SPECIES     : {result['species']['display_name']}"
    if result["species"].get("scientific_name"):
        out += f"\n🔬  SCIENTIFIC  : {result['species']['scientific_name']}"
    out += f"\n{_confidence_line(result)}\n"

    facts = [("📏  SIZE", info.get("size_cm")),
             ("🍽️   DIET", info.get("diet"))]
    for label, value in facts:
        if value:
            out += f"\n{label}        : {value}"
    if any(v for _, v in facts):
        out += "\n"

    out += f"\n📍  FOUND IN    : {info['habitat']}\n"
    out += f"\n✈️   MIGRATION   : {info['migration']}\n"

    marks = info.get("field_marks") or []
    if marks:
        out += "\n🔎  FIELD MARKS :\n"
        for mark in marks[:6]:
            out += f"    • {mark}\n"

    similar = info.get("similar_species") or []
    if similar:
        out += (f"\n⚠️  LOOKS SIMILAR TO : {similar[0]['name']}"
                f"\n    How to tell apart  : {similar[0]['how_to_distinguish']}\n")

    if info.get("fun_fact"):
        out += f"\n💡  DID YOU KNOW : {info['fun_fact']}\n"

    out += f"\n{RULE}\nTOP 5 PREDICTIONS:\n"
    for i, row in enumerate(result["top5"]):
        marker = " ◀ TOP PICK" if i == 0 else ""
        out += (f"\n#{i+1}  {row['display_name']}  "
                f"({row['confidence']*100:.1f}%){marker}\n    📍 {row['habitat']}\n")

    if info.get("used_llm") and LLM_DISCLAIMER:
        out += f"\n{RULE}\n{LLM_DISCLAIMER}\n"

    return out


def _prior_note(settings):
    if settings["use_location"]:
        return f"📍 Location prior ON — {settings['lat']:.2f}, {settings['lon']:.2f}"
    return "🌍 Location prior OFF — scoring against all species"


def format_audio_result(result):
    """The call-identification report, byte-identical to the pre-refactor text."""
    if not result.get("ok"):
        out = f"❌ {result['error']}"
        for hint in result.get("hints", []):
            out += f"\n{hint}"
        return out

    settings = result["settings"]
    prior = _prior_note(settings)

    if not result["detected"]:
        out = (f"🔇 No bird detected above {settings['min_conf']*100:.0f}% confidence.\n\n"
               f"{prior}\n"
               f"Clip length: {settings['duration_sec']:.1f}s\n\n"
               "Things to try:\n")
        for hint in result.get("hints", []):
            out += f"  • {hint}\n"
        return out

    sp = result["species"]
    conf = result["confidence"] * 100
    band = result["confidence_band"]
    conf_msg = {"low":      f"⚠️  LOW CONFIDENCE ({conf:.1f}%)",
                "moderate": f"🟡  MODERATE CONFIDENCE ({conf:.1f}%)",
                "high":     f"✅  HIGH CONFIDENCE ({conf:.1f}%)"}[band]

    info = result.get("info") or {}
    habitat   = info.get("habitat", "Location data not available")
    migration = info.get("migration", "Migration data not available")

    out = f"""
🎵  IDENTIFIED FROM CALL : {sp['common_name']}
🔬  Scientific name      : {sp['scientific_name']}
{conf_msg}

{prior}
🎧  Clip: {settings['duration_sec']:.1f}s · threshold {settings['min_conf']*100:.0f}% · {len(result['detections'])} detection(s)

📍  FOUND IN   : {habitat}

✈️   MIGRATION  : {migration}

🤖  Powered by : BirdNET (Cornell Lab of Ornithology)
"""

    if sp["in_image_model"]:
        out += (f"\n🔗  In this project's image model as: "
                f"{info.get('display_name', sp['cub_folder'])}\n")
    else:
        out += (
            f"\n⚠️  '{sp['common_name']}' is not one of the "
            f"{result['provenance']['num_species']} CUB species\n"
            "    this project's image model knows. BirdNET covers ~6,500\n"
            "    species, so audio can identify birds the photo tab cannot.\n"
        )

    similar = info.get("similar_species") or []
    if similar:
        out += (f"\n⚠️  SOUNDS SIMILAR TO : {similar[0]['name']}\n"
                f"    How to tell apart  : {similar[0]['how_to_distinguish']}\n")

    grouped = result["grouped"]
    if len(grouped) > 1 or result["detections"][0]["confidence"] < 1.0:
        out += f"\n{RULE}\nSPECIES HEARD IN THIS RECORDING:\n"
        for i, g in enumerate(grouped[:8], 1):
            bar = "█" * int(g["confidence"] * 20)
            times = f"{g['count']}× from {g['first_heard']:.0f}s"
            out += (f"\n#{i}  {g['common_name']}  ({g['confidence']*100:.1f}%)"
                    f"  · {times}\n    {bar}\n")

    return out


def format_gradcam_info(info):
    """Status line under the Grad-CAM image."""
    if "error" in info:
        return f"❌ {info['error']}"
    return (f"✅ Grad-CAM generated!\n"
            f"Predicted: {info['display_name']} ({info['confidence']*100:.1f}%)\n"
            f"Red/Yellow areas = what the model focused on\n"
            f"Saved to: {info['path']}")
