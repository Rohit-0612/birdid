#!/usr/bin/env python3
"""Measure the open-vocabulary verifier, then fit its acceptance threshold.

The verifier exists to name birds the trained classifier cannot, so the number
that matters is its accuracy on the `ood/near_bird` tier of
data/openset/manifest.json — 130 photos of 26 bird species that are deliberately
NOT among the CUB 200. Those entries carry ground-truth `label` values, which makes
this a real evaluation rather than a demo.

Three outcomes are reported separately, because collapsing them would hide the
interesting part:

  correct          top-1 common name matches the label
  wrong species    named a different species, right or wrong genus
  not a candidate  the true species is absent from the label file entirely,
                   so no ranking could ever have been right

That last category is a property of the vocabulary, not the model: BirdNET lists
only species it can hear, so Common Ostrich is not in the file at all.

The threshold is then fitted on cosine similarity, not on the softmax score. The
softmax uses BioCLIP's learned logit scale (~100), which saturates: a *wrong*
macaw species came back at p=0.986. Similarity separates cleanly (~0.38 for a
confident bird, ~0.23 for a squirrel), so that is what gets thresholded.

Usage:
    python3 scripts/eval_verifier.py                 # evaluate + write threshold.json
    python3 scripts/eval_verifier.py --dry-run       # report only
    python3 scripts/eval_verifier.py --min-precision 0.75
"""

import argparse
import json
import os
import re
import sys
from datetime import date

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import verifier  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENSET_DIR = os.path.join(PROJECT_DIR, "data", "openset")
MANIFEST = os.path.join(OPENSET_DIR, "manifest.json")


def truth_name(entry):
    """Ground-truth common name for a manifest entry.

    The two tiers label differently: `ood/*` entries carry a common name
    ("Scarlet Macaw") while `id` entries carry the CUB folder
    ("001.Black_footed_Albatross"). Comparing the folder form against BirdNET's
    common names matches nothing, which is what made the first run of this script
    report 94% of in-distribution species as absent from the vocabulary.
    """
    label = entry.get("label") or ""
    if re.match(r"^\d{3}\.", label):
        import bird_core
        return bird_core.display_name(label)
    return label


def norm(name):
    return "".join(c for c in (name or "").lower() if c.isalnum() or c == " ").strip()


def names_match(truth, predicted):
    """Common-name match, tolerant of the manifest's and BirdNET's spellings."""
    t, p = norm(truth), norm(predicted)
    if not t or not p:
        return False
    return t == p or t in p or p in t


def evaluate(entries, kind_label, candidates, embeddings_cache=None):
    """Rank every image in `entries` and bucket the outcome."""
    candidate_commons = [c for _, c in candidates]
    in_vocabulary = {norm(c) for c in candidate_commons}

    images, kept = [], []
    for entry in entries:
        path = os.path.join(OPENSET_DIR, entry["file"])
        try:
            images.append(Image.open(path).convert("RGB"))
            kept.append(entry)
        except (OSError, ValueError):
            continue

    print(f"  encoding {len(kept)} images…")
    feats = verifier.encode_images(images)
    ranked = verifier.rank(feats, topk=5)

    rows = []
    for entry, candidates_for_image in zip(kept, ranked):
        truth = truth_name(entry)
        best = candidates_for_image[0]
        # Is the true species even in the vocabulary?
        coverable = any(names_match(truth, c) for c in candidate_commons) \
            if norm(truth) not in in_vocabulary else True

        top1 = names_match(truth, best["common_name"])
        top5 = any(names_match(truth, c["common_name"]) for c in candidates_for_image)
        same_genus = (best["scientific_name"].split()[0].lower()
                      in {c["scientific_name"].split()[0].lower()
                          for c in candidates_for_image if names_match(truth, c["common_name"])}) \
            if top5 and not top1 else False

        rows.append({
            "file": entry["file"], "truth": truth,
            "predicted": best["common_name"],
            "predicted_sci": best["scientific_name"],
            "similarity": best["similarity"], "score": best["score"],
            "margin": best["margin"], "top1": top1, "top5": top5,
            "coverable": coverable, "same_genus": same_genus,
        })

    n = len(rows)
    coverable = [r for r in rows if r["coverable"]]
    uncoverable = [r for r in rows if not r["coverable"]]

    print(f"\n  {kind_label}: {n} images, {len({r['truth'] for r in rows})} species")
    print(f"    top-1 overall            {sum(r['top1'] for r in rows) / n:>6.1%}")
    print(f"    top-5 overall            {sum(r['top5'] for r in rows) / n:>6.1%}")
    if uncoverable:
        print(f"    not in the vocabulary    {len(uncoverable) / n:>6.1%}  "
              f"({len({r['truth'] for r in uncoverable})} species, "
              f"e.g. {sorted({r['truth'] for r in uncoverable})[:3]})")
    if coverable:
        print(f"    top-1 where a correct answer was possible "
              f"{sum(r['top1'] for r in coverable) / len(coverable):>6.1%}")
        print(f"    top-5 where a correct answer was possible "
              f"{sum(r['top5'] for r in coverable) / len(coverable):>6.1%}")
    return rows


def fit_threshold(rows, min_precision, grid=None):
    """Pick the lowest similarity cut that reaches `min_precision` on accepted rows.

    Only rows where a correct answer was reachable count toward precision — being
    penalised for a species missing from the vocabulary would push the threshold up
    for no benefit.
    """
    usable = [r for r in rows if r["coverable"]]
    if not usable:
        return None

    sims = np.array([r["similarity"] for r in usable])
    correct = np.array([r["top1"] for r in usable], dtype=bool)
    grid = grid if grid is not None else np.quantile(sims, np.linspace(0, 0.95, 40))

    print(f"\n  {'min similarity':>15}{'accepted':>11}{'precision':>11}")
    print("  " + "-" * 37)
    chosen = None
    for cut in grid:
        accepted = sims >= cut
        if accepted.sum() < 10:
            continue
        precision = correct[accepted].mean()
        coverage = accepted.mean()
        marker = ""
        if chosen is None and precision >= min_precision:
            chosen, marker = float(cut), "  <- chosen"
        print(f"  {cut:>15.3f}{coverage:>10.0%}{precision:>11.0%}{marker}")
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # 0.90 by default: when the app puts a name on a new-bird card it should be
    # right about nine times in ten. Measured, not guessed — see the table this
    # script prints.
    ap.add_argument("--min-precision", type=float, default=0.90,
                    help="required accuracy among accepted answers (default 0.90)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    status = verifier.available()
    if not status["ok"]:
        sys.exit(f"Verifier unavailable: {status['reason']}")

    print(f"Verifier: {status['model']}")
    print(f"Candidates: {status['species']} bird species "
          f"(filtered from {len(open(verifier.labels_path()).read().splitlines())} "
          f"BirdNET labels)")
    verifier.warm(progress=True)
    candidates = verifier.load_labels()

    with open(MANIFEST) as f:
        manifest = json.load(f)

    print("\n" + "═" * 60)
    print("BIRDS OUTSIDE THE 200 — the population this feature serves")
    print("═" * 60)
    near_bird = [e for e in manifest if e["kind"] == "ood/near_bird"]
    nb_rows = evaluate(near_bird, "ood/near_bird", candidates)

    print("\n" + "═" * 60)
    print("BIRDS INSIDE THE 200 — agreement check against the classifier")
    print("═" * 60)
    id_rows = evaluate([e for e in manifest if e["kind"] == "id"], "id", candidates)

    print("\n" + "═" * 60)
    print("THRESHOLD FIT (on birds outside the 200)")
    print("═" * 60)
    min_score = fit_threshold(nb_rows, args.min_precision)
    if min_score is None:
        print(f"\n  No cut reaches {args.min_precision:.0%} precision — "
              "the verifier is not reliable enough to name new species.")
        print("  Writing no threshold; identify() will report confident=None.")
        if not args.dry_run:
            print("  (nothing written)")
        return

    accepted = [r for r in nb_rows if r["similarity"] >= min_score and r["coverable"]]
    margins = [r["margin"] for r in nb_rows
               if r["coverable"] and r["top1"] and r["margin"] is not None]

    payload = {
        "model": verifier.MODEL_ID,
        "prompt": verifier.PROMPT,
        "min_score": float(min_score),
        # A thin top-two gap means the model is torn; the 10th percentile of gaps on
        # correct answers is a light floor rather than an aggressive filter.
        "min_margin": float(np.quantile(margins, 0.10)) if margins else 0.0,
        "target_precision": args.min_precision,
        "measured": {
            "near_bird": {
                "n": len(nb_rows),
                "species": len({r["truth"] for r in nb_rows}),
                "top1": sum(r["top1"] for r in nb_rows) / len(nb_rows),
                "top5": sum(r["top5"] for r in nb_rows) / len(nb_rows),
                "not_in_vocabulary": sum(not r["coverable"] for r in nb_rows) / len(nb_rows),
                "top1_when_possible": (
                    sum(r["top1"] for r in nb_rows if r["coverable"])
                    / max(1, sum(r["coverable"] for r in nb_rows))),
                "precision_at_threshold": (
                    sum(r["top1"] for r in accepted) / len(accepted)) if accepted else None,
                "coverage_at_threshold": len(accepted) / max(1, sum(r["coverable"] for r in nb_rows)),
            },
            "id": {
                "n": len(id_rows),
                "top1": sum(r["top1"] for r in id_rows) / len(id_rows),
                "top5": sum(r["top5"] for r in id_rows) / len(id_rows),
            },
        },
        "candidates": len(candidates),
        "labels_sha256": verifier.labels_sha256(),
        "created": date.today().isoformat(),
        "note": ("Fitted by scripts/eval_verifier.py on the labelled ood/near_bird "
                 "tier of data/openset/manifest.json. Accept a name when "
                 "similarity >= min_score and margin >= min_margin. Rerun if the "
                 "BirdNET label file or the prompt template changes."),
    }

    print(f"\n  min_score  {payload['min_score']:.3f}")
    print(f"  min_margin {payload['min_margin']:.3f}")
    print(f"  precision among accepted: "
          f"{(payload['measured']['near_bird']['precision_at_threshold'] or 0):.0%}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    os.makedirs(verifier.CACHE_DIR, exist_ok=True)
    with open(verifier.THRESHOLD_PATH, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\n✅ Wrote {os.path.relpath(verifier.THRESHOLD_PATH, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
