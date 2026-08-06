#!/usr/bin/env python3
"""
Validate data/species_kb.json before the app is allowed to depend on it.

    python3 scripts/validate_kb.py
    python3 scripts/validate_kb.py --strict   # warnings become failures

A JSON Schema guarantees the *shape* of generated data. It says nothing about
whether the content is right, and nothing about whether it lines up with the
rest of this project. That is what this script is for.

Checks, in rough order of severity:

  ERROR   a record exists for a folder the model cannot predict (dead data)
  ERROR   a required field is empty or a placeholder like "unknown"
  ERROR   a duplicated scientific name (usually two species collapsed into one)
  WARN    a species the model predicts has no record yet
  WARN    a similar_species entry naming a bird outside the 200 classes
  WARN    a scientific name that is not a plausible binomial
  WARN    a suspiciously short field

Exit code is non-zero if anything failed, so this can gate a build.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(PROJECT_DIR, "data", "species_kb.json")
CLASSES_TXT = os.path.join(PROJECT_DIR, "CUB_200_2011", "classes.txt")

REQUIRED_TEXT = [
    "display_name", "scientific_name", "family", "order", "habitat",
    "range_description", "migration", "size_cm", "diet",
    "conservation_status", "fun_fact",
]
PLACEHOLDERS = {"", "unknown", "n/a", "na", "none", "not available", "-", "tbd"}
IUCN = {
    "Least Concern", "Near Threatened", "Vulnerable", "Endangered",
    "Critically Endangered", "Extinct in the Wild", "Extinct",
    "Data Deficient", "Unknown",
}
BINOMIAL = re.compile(r"^[A-Z][a-z]+ [a-z-]+$")


def load_class_names():
    for name in ("best_bird_model_inference.pth", "best_bird_model.pth"):
        path = os.path.join(PROJECT_DIR, name)
        if os.path.exists(path):
            import torch
            return torch.load(path, map_location="cpu")["class_names"]
    if os.path.exists(CLASSES_TXT):
        return [l.split(" ", 1)[1].strip() for l in open(CLASSES_TXT) if l.strip()]
    sys.exit("Could not find a checkpoint or CUB_200_2011/classes.txt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="treat warnings as errors")
    args = ap.parse_args()

    if not os.path.exists(JSON_PATH):
        sys.exit(f"No knowledge base at {JSON_PATH}. Run scripts/build_species_kb.py first.")

    kb = json.load(open(JSON_PATH))
    class_names = load_class_names()
    valid_display = {f.split(".", 1)[-1].replace("_", " ").lower() for f in class_names}

    errors, warnings = [], []

    # --- Alignment with the model's label space -------------------------
    unknown = sorted(set(kb) - set(class_names))
    for folder in unknown:
        errors.append(f"{folder}: record exists but the model has no such class (dead data)")

    missing = [f for f in class_names if f not in kb]
    for folder in missing:
        warnings.append(f"{folder}: no record yet")

    # --- Per-record content ---------------------------------------------
    sci_names = defaultdict(list)
    for folder, rec in kb.items():
        for field in REQUIRED_TEXT:
            val = str(rec.get(field, "")).strip()
            if val.lower() in PLACEHOLDERS:
                errors.append(f"{folder}: '{field}' is empty or a placeholder ({val!r})")
            elif len(val) < 3:
                warnings.append(f"{folder}: '{field}' looks too short ({val!r})")

        sci = str(rec.get("scientific_name", "")).strip()
        if sci and not BINOMIAL.match(sci):
            warnings.append(f"{folder}: scientific_name {sci!r} is not a plain binomial")
        sci_names[sci.lower()].append(folder)

        marks = rec.get("field_marks") or []
        if len(marks) < 3:
            warnings.append(f"{folder}: only {len(marks)} field_marks (want 3+)")

        status = str(rec.get("conservation_status", "")).strip()
        if status and status not in IUCN:
            warnings.append(f"{folder}: conservation_status {status!r} is not an IUCN category")

        for sim in rec.get("similar_species") or []:
            name = str(sim.get("name", "")).strip()
            if name.lower() not in valid_display:
                warnings.append(f"{folder}: similar_species {name!r} is outside the 200 CUB classes")
            if len(str(sim.get("how_to_distinguish", "")).strip()) < 10:
                warnings.append(f"{folder}: similar_species {name!r} has a thin distinguishing note")

        if rec.get("_verified") is not False and "_verified" not in rec:
            warnings.append(f"{folder}: missing _verified provenance flag")

    for sci, folders in sci_names.items():
        if sci and len(folders) > 1:
            errors.append(f"duplicate scientific_name {sci!r} across: {folders}")

    # --- Report ----------------------------------------------------------
    print(f"knowledge base : {JSON_PATH}")
    print(f"records        : {len(kb)}/{len(class_names)}")
    if kb:
        sample = next(iter(kb.values()))
        print(f"generated by   : {sample.get('_source', '?')} on {sample.get('_generated', '?')}")
        print(f"verified       : {sample.get('_verified')}  <- LLM output, treat as unverified")

    def show(label, items, cap=25):
        if not items:
            return
        print(f"\n{label} ({len(items)}):")
        for line in items[:cap]:
            print(f"  - {line}")
        if len(items) > cap:
            print(f"  ... and {len(items) - cap} more")

    show("ERRORS", errors)
    show("WARNINGS", warnings)

    failed = bool(errors) or (args.strict and bool(warnings))
    print("\n" + ("FAILED" if failed else "PASSED") +
          f"  ({len(errors)} errors, {len(warnings)} warnings)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
