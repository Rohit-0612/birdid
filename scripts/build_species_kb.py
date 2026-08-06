#!/usr/bin/env python3
"""
Build a structured knowledge base for all 200 CUB-200-2011 species using a
local LLM served by Ollama. No API key, no network calls beyond localhost.

    python3 scripts/build_species_kb.py                 # build everything
    python3 scripts/build_species_kb.py --limit 3       # try 3 species first
    python3 scripts/build_species_kb.py --model gemma3:12b
    python3 scripts/build_species_kb.py --redo 044.Frigatebird

Why this exists
---------------
The hand-written lookup tables in the app cover 185/200 species for habitat
and migration and only 16/200 for look-alikes, and the numeric prefixes were
originally keyed to a different species list entirely. This regenerates the
whole thing from one consistent source, keyed on the model's real class names.

Three things worth understanding in here
----------------------------------------
1. Schema-constrained decoding. We hand Ollama a JSON Schema (generated from
   the Pydantic model) via `format=`. The model is then *constrained* during
   decoding to emit conforming JSON -- structurally different from asking for
   JSON in the prompt and hoping. We still validate, because a schema
   guarantees shape, never truth.
2. Resumability. Records append to a JSONL as they finish, and a rerun skips
   what is already there. 200 sequential local generations take a while and
   you do not want a laptop sleeping to cost you the whole run.
3. The output is UNVERIFIED. A local 8B model gets natural-history facts wrong
   sometimes. Every record carries provenance saying so, and the app labels it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List

try:
    import ollama
    from pydantic import BaseModel, Field, ValidationError
except ImportError:
    sys.exit("Missing deps. Run: pip3 install -r requirements-llm.txt")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES_TXT = os.path.join(PROJECT_DIR, "CUB_200_2011", "classes.txt")
JSONL_PATH = os.path.join(PROJECT_DIR, "data", "species_kb.jsonl")
JSON_PATH = os.path.join(PROJECT_DIR, "data", "species_kb.json")

DEFAULT_MODEL = "qwen3:8b"


# ── Schema ───────────────────────────────────────────────────────────────
# This doubles as the prompt: the field descriptions are carried into the
# JSON Schema, so the model sees them while decoding.

class SimilarSpecies(BaseModel):
    name: str = Field(description="Common name of a species that is easily confused with this one")
    how_to_distinguish: str = Field(description="One concrete visual difference a birder could check in the field")


class SpeciesRecord(BaseModel):
    display_name: str = Field(description="Corrected common English name, e.g. 'Arctic Tern'")
    scientific_name: str = Field(description="Binomial name, e.g. 'Sterna paradisaea'")
    family: str = Field(description="Taxonomic family, e.g. 'Laridae'")
    order: str = Field(description="Taxonomic order, e.g. 'Charadriiformes'")
    habitat: str = Field(description="Countries/regions and the habitat type it occupies, one sentence")
    range_description: str = Field(description="Where it breeds and where it winters, one sentence")
    migration: str = Field(description="Migration pattern with rough months, or state that it is a year-round resident")
    field_marks: List[str] = Field(description="3 to 6 short visual identification marks")
    size_cm: str = Field(description="Typical body length in cm, e.g. '33-39 cm'")
    diet: str = Field(description="What it primarily eats, one short phrase")
    similar_species: List[SimilarSpecies] = Field(description="1 to 3 confusable species; empty list if genuinely distinctive")
    conservation_status: str = Field(description="IUCN category: Least Concern, Near Threatened, Vulnerable, Endangered, Critically Endangered, or Unknown")
    fun_fact: str = Field(description="One memorable, verifiable fact")


# CUB folder names contain misspellings, bare genus names, and species
# ambiguous without context. Steering these explicitly is far cheaper than
# correcting 200 records by hand afterwards.
NAME_HINTS = {
    "017.Cardinal": "Northern Cardinal (Cardinalis cardinalis)",
    "022.Chuck_will_Widow": "Chuck-will's-widow",
    "044.Frigatebird": "Magnificent Frigatebird (the CUB folder is just 'Frigatebird')",
    "074.Florida_Jay": "Florida Scrub-Jay",
    "091.Mockingbird": "Northern Mockingbird",
    "092.Nighthawk": "Common Nighthawk",
    "103.Sayornis": "Eastern Phoebe (the CUB folder uses the genus name Sayornis)",
    "105.Whip_poor_Will": "Eastern Whip-poor-will",
    "110.Geococcyx": "Greater Roadrunner (the CUB folder uses the genus name Geococcyx)",
    "124.Le_Conte_Sparrow": "LeConte's Sparrow",
    "130.Tree_Sparrow": "American Tree Sparrow",
    "134.Cape_Glossy_Starling": "Cape Starling / Cape Glossy Starling",
    "141.Artic_Tern": "Arctic Tern -- note the CUB folder misspells it as 'Artic'",
    "146.Forsters_Tern": "Forster's Tern",
    "070.Green_Violetear": "Mexican Violetear (formerly Green Violetear)",
    "018.Spotted_Catbird": "Spotted Catbird, an Australian rainforest species (NOT the North American Gray Catbird)",
}

SYSTEM_PROMPT = """\
You are an expert ornithologist writing entries for a field guide.

You will be given one bird species from the CUB-200-2011 dataset. Return a \
single JSON object describing it, conforming exactly to the provided schema.

Rules:
- CUB folder names sometimes contain misspellings or bare genus names. Always \
return the CORRECT common and scientific name, not the folder's spelling.
- Be specific and concrete. "Eastern North America" beats "various regions".
- field_marks must be visual things a person can see: plumage colour, bill \
shape, wing bars, tail shape, eye rings. Not behaviour, not song.
- similar_species must name real bird species and give a difference that can \
actually be checked in a photograph.
- If you are unsure of a fact, give the most widely accepted answer rather \
than hedging. Do not write "unknown" except for conservation_status.
- Keep every string field to one or two sentences.
"""


def load_class_names() -> List[str]:
    """Prefer the model's own class list; fall back to the CUB classes file."""
    for name in ("best_bird_model_inference.pth", "best_bird_model.pth"):
        path = os.path.join(PROJECT_DIR, name)
        if os.path.exists(path):
            import torch
            return torch.load(path, map_location="cpu")["class_names"]
    if os.path.exists(CLASSES_TXT):
        return [l.split(" ", 1)[1].strip() for l in open(CLASSES_TXT) if l.strip()]
    sys.exit("Could not find a checkpoint or CUB_200_2011/classes.txt")


def display_name(folder: str) -> str:
    return folder.split(".", 1)[-1].replace("_", " ")


def build_prompt(folder: str) -> str:
    hint = NAME_HINTS.get(folder)
    lines = [
        f"CUB folder name: {folder}",
        f"Species as written in the dataset: {display_name(folder)}",
    ]
    if hint:
        lines.append(f"This species is: {hint}")
    lines.append("\nWrite the field-guide entry for this species.")
    return "\n".join(lines)


def generate(client, model: str, folder: str, retries: int = 3) -> SpeciesRecord:
    schema = SpeciesRecord.model_json_schema()
    last = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(folder)},
                ],
                format=schema,          # <- schema-constrained decoding
                think=False,            # qwen3 is a thinking model; skip it here
                options={"temperature": 0.3, "num_ctx": 4096},
            )
            return SpeciesRecord.model_validate_json(resp["message"]["content"])
        except ValidationError as e:
            last = f"schema validation failed: {e.error_count()} error(s)"
        except Exception as e:                                   # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        if attempt < retries:
            time.sleep(2 * attempt)
    raise RuntimeError(f"{folder}: gave up after {retries} attempts -- {last}")


def load_done() -> dict:
    if not os.path.exists(JSONL_PATH):
        return {}
    done = {}
    with open(JSONL_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                done[rec["folder"]] = rec
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("BIRD_LLM_MODEL", DEFAULT_MODEL))
    ap.add_argument("--limit", type=int, help="only process the first N missing species")
    ap.add_argument("--redo", nargs="*", default=[], help="folder names to regenerate")
    args = ap.parse_args()

    class_names = load_class_names()
    done = load_done()
    for folder in args.redo:
        done.pop(folder, None)

    todo = [f for f in class_names if f not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"model={args.model}  total={len(class_names)}  "
          f"already done={len(done)}  to generate={len(todo)}")

    client = ollama.Client()
    try:
        available = {m.model for m in client.list().models}
    except Exception as e:                                       # noqa: BLE001
        sys.exit(f"Cannot reach Ollama ({e}). Is the daemon running?")
    if args.model not in available and f"{args.model}:latest" not in available:
        sys.exit(f"Model '{args.model}' not pulled. Run: ollama pull {args.model}")

    os.makedirs(os.path.dirname(JSONL_PATH), exist_ok=True)
    failures = []
    t_start = time.time()

    for i, folder in enumerate(todo, 1):
        t0 = time.time()
        try:
            rec = generate(client, args.model, folder)
        except RuntimeError as e:
            print(f"[{i:3d}/{len(todo)}] FAILED {folder}: {e}")
            failures.append(folder)
            continue

        row = {
            "folder": folder,
            **rec.model_dump(),
            # Provenance travels with the data. This is generated text, not
            # a reference source, and everything downstream must be able to
            # say so without guessing.
            "_source": args.model,
            "_generated": time.strftime("%Y-%m-%d"),
            "_verified": False,
        }
        with open(JSONL_PATH, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        rate = time.time() - t0
        eta = (len(todo) - i) * (time.time() - t_start) / i
        print(f"[{i:3d}/{len(todo)}] {folder:<38} {rec.scientific_name:<28} "
              f"{rate:5.1f}s  eta {eta/60:4.1f}m")

    # Assemble the flat JSON the app reads, ordered by class index.
    done = load_done()
    ordered = {f: done[f] for f in class_names if f in done}
    with open(JSON_PATH, "w") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)

    print(f"\nwrote {len(ordered)}/{len(class_names)} records -> {JSON_PATH}")
    if failures:
        print(f"{len(failures)} failed: {failures}")
        print("Rerun the script to retry only those.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
