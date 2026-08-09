#!/usr/bin/env python3
"""Fit the open-set rejection threshold — "is this even one of my 200 birds?"

A 200-way softmax cannot answer that question. It renormalises over the classes
it knows, so a photo of a golden retriever comes back as some bird with a
confident-looking score. That is the single most misleading behaviour in the
app, and it needs no retraining to fix: the raw logit magnitudes already carry
the signal, they are just thrown away by softmax.

Four score families are swept over four temperatures and the winner is chosen
by AUROC, because guessing wrong here is easy: energy scoring (Liu et al. 2020)
is the usual recommendation and it performs *worse than random* on this
checkpoint — AUROC 0.37 versus 0.90 for plain max-softmax. A checkpoint that
has memorised CUB produces large logit mass for almost any bird-shaped input,
which is exactly what energy measures. So the script measures instead of
asserting:

  msp      -max(softmax(logits / T))         Hendrycks & Gimpel 2017
  energy   -T · logsumexp(logits / T)        Liu et al. 2020, NeurIPS
  entropy  softmax entropy                   spread across many classes
  margin   -(top1 logit - top2 logit)        "between classes" detector

Known limitation, visible in the printed table: the gate is good at "not a
bird" and only partial at "a bird outside my 200". The near_bird tier is real
birds the model has no class for, and bird features fire for them by design.
Treat the near_bird number as the honest ceiling on this approach.

Fitting data is data/openset/manifest.json — 341 in-distribution photos of CUB
species (sourced from Wikimedia, NOT from CUB itself, so a contaminated
checkpoint cannot cheat) and 200 out-of-distribution images in three tiers:
near_bird (macaws, eagles, owls, penguins — the hard case), animal, nonanimal.

The reported numbers come from a held-out half the threshold never saw. The
shipped threshold is then refitted on everything, which is standard practice
and is why the two sets of numbers are printed separately.

Usage:
    python3 scripts/fit_openset.py                  # fit, report, write threshold.json
    python3 scripts/fit_openset.py --tpr 0.90       # reject more aggressively
    python3 scripts/fit_openset.py --score msp      # force the baseline score
    python3 scripts/fit_openset.py --dry-run        # report only, write nothing
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import date

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bird_core import (  # noqa: E402
    CLASS_NAMES,
    MODEL_PATH,
    device,
    model,
    openset_score,
    val_transform,
)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENSET_DIR = os.path.join(PROJECT_DIR, "data", "openset")
MANIFEST = os.path.join(OPENSET_DIR, "manifest.json")
OUT_PATH = os.path.join(OPENSET_DIR, "threshold.json")

# Temperatures swept for the energy score. T=1 is the paper's default; higher T
# flattens the logits and sometimes separates better on fine-grained models.
TEMPERATURES = [1.0, 2.0, 5.0, 10.0]


# ── scoring ──────────────────────────────────────────────────
def collect_logits(entries, batch_size=16):
    """Run the model over every manifest entry. Returns (logits, kept_entries)."""
    logits, kept, skipped = [], [], []
    batch, batch_entries = [], []

    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        with torch.no_grad():
            out = model(x)
        logits.append(out.float().cpu())
        kept.extend(batch_entries)
        batch.clear()
        batch_entries.clear()

    for i, entry in enumerate(entries, 1):
        path = os.path.join(OPENSET_DIR, entry["file"])
        try:
            img = Image.open(path).convert("RGB")
        except (OSError, ValueError) as e:
            skipped.append((entry["file"], str(e)))
            continue
        batch.append(val_transform(img))
        batch_entries.append(entry)
        if len(batch) == batch_size:
            flush()
            print(f"\r  scored {len(kept)}/{len(entries)}…", end="", flush=True)
    flush()
    print(f"\r  scored {len(kept)}/{len(entries)} images"
          + (f" ({len(skipped)} unreadable)" if skipped else ""))
    for f, err in skipped[:5]:
        print(f"    skipped {f}: {err}")

    return torch.cat(logits) if logits else torch.empty(0, len(CLASS_NAMES)), kept


def score_as(logits, method, temperature=1.0):
    """Score via bird_core's registry, so fit-time and run-time maths cannot drift."""
    return openset_score(logits, method, temperature)


def auroc(scores, is_id):
    """AUROC for separating ID from OOD. Scores are 'lower = ID', so negate.

    Hand-rolled via the rank identity rather than pulling in sklearn: this
    script already costs a model load, and the formula is three lines.
    """
    x = -np.asarray(scores, dtype=float)
    pos, neg = x[is_id], x[~is_id]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def threshold_at_tpr(id_scores, target_tpr):
    """Largest score still accepted when we insist on keeping `target_tpr` of birds.

    Accept when score <= threshold, so the cut is the target_tpr quantile of the
    in-distribution scores.
    """
    return float(np.quantile(id_scores, target_tpr))


def rates(scores, threshold):
    """Fraction of `scores` that would be accepted as a known bird."""
    if len(scores) == 0:
        return float("nan")
    return float((np.asarray(scores) <= threshold).mean())


# ── reporting ────────────────────────────────────────────────
def report(name, id_scores, ood_by_tier, threshold, tpr_target):
    tpr = rates(id_scores, threshold)
    all_ood = np.concatenate([v for v in ood_by_tier.values() if len(v)])
    print(f"\n  {name}")
    print(f"    threshold {threshold:>9.4f}   (target TPR {tpr_target:.0%})")
    print(f"    {'accepted as a known bird':<34}{'rate':>8}   {'n':>5}")
    print(f"    {'─' * 49}")
    print(f"    {'in-distribution birds (want high)':<34}{tpr:>7.1%}   {len(id_scores):>5}")
    for tier, scores in ood_by_tier.items():
        if len(scores) == 0:
            continue
        label = f"OOD {tier} (want low)"
        print(f"    {label:<34}{rates(scores, threshold):>7.1%}   {len(scores):>5}")
    print(f"    {'─' * 49}")
    print(f"    {'all OOD':<34}{rates(all_ood, threshold):>7.1%}   {len(all_ood):>5}")
    return tpr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tpr", type=float, default=0.95,
                    help="fraction of real birds that must still be accepted (default 0.95)")
    ap.add_argument("--score", default="auto",
                    choices=["auto", "msp", "energy", "entropy", "margin"],
                    help="scoring family; 'auto' picks the best AUROC (default)")
    ap.add_argument("--seed", type=int, default=42, help="split seed (default 42)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true", help="report without writing threshold.json")
    args = ap.parse_args()

    if not 0.5 <= args.tpr < 1.0:
        ap.error("--tpr should be between 0.5 and 1.0")

    with open(MANIFEST) as f:
        entries = json.load(f)
    manifest_sha = hashlib.sha256(
        open(MANIFEST, "rb").read()).hexdigest()

    print(f"Open-set fit — {len(entries)} manifest entries")
    print(f"Checkpoint: {os.path.basename(MODEL_PATH)} on {device}")

    # Cache the forward pass. Comparing score functions is the iterative part of
    # this work and it does not need the model re-run. Keyed on checkpoint +
    # manifest so a changed checkpoint or a re-download can never reuse stale
    # logits — the threshold is only valid for the logit scale it was fitted on.
    cache_key = hashlib.sha256(
        f"{os.path.basename(MODEL_PATH)}:{manifest_sha}".encode()).hexdigest()[:16]
    cache_path = os.path.join(OPENSET_DIR, f"logits_cache_{cache_key}.npz")

    if os.path.exists(cache_path):
        blob = np.load(cache_path, allow_pickle=True)
        logits = torch.from_numpy(blob["logits"])
        kept = [e for e in entries if e["file"] in set(blob["files"].tolist())]
        # Preserve manifest order so `kept` lines up with the cached rows.
        order = {f: i for i, f in enumerate(blob["files"].tolist())}
        kept.sort(key=lambda e: order[e["file"]])
        print(f"  reusing cached logits for {len(kept)} images "
              f"({os.path.basename(cache_path)})")
    else:
        logits, kept = collect_logits(entries, args.batch_size)
        np.savez_compressed(cache_path, logits=logits.numpy(),
                            files=np.array([e["file"] for e in kept]))
    if len(kept) < 50:
        sys.exit("Not enough readable images to fit a threshold. "
                 "Run scripts/download_openset_photos.py first.")

    kinds = np.array([e["kind"] for e in kept])
    is_id = kinds == "id"
    tiers = ["near_bird", "animal", "nonanimal"]

    # ── pick the score function ──────────────────────────────
    candidates = {"margin": score_as(logits, "margin")}
    for t in TEMPERATURES:
        for m in ("msp", "energy", "entropy"):
            candidates[f"{m}@T={t:g}"] = score_as(logits, m, t)

    print("\nScore comparison (AUROC, ID vs all OOD — higher is better):")
    ranked = sorted(candidates.items(), key=lambda kv: auroc(kv[1], is_id), reverse=True)
    for label, scores in ranked:
        print(f"  {label:<16}{auroc(scores, is_id):.4f}")

    if args.score == "auto":
        chosen = ranked[0][0]
    else:
        # Best temperature within the requested family.
        chosen = next((l for l, _ in ranked if l.split("@")[0] == args.score), None)
        if chosen is None:
            sys.exit(f"no candidate for --score {args.score}")
    scores = candidates[chosen]
    method = chosen.split("@")[0]
    temperature = float(chosen.split("=")[1]) if "@T=" in chosen else 1.0
    print(f"\nUsing: {chosen}  (AUROC {auroc(scores, is_id):.4f})")

    # ── honest estimate on a held-out half ───────────────────
    # Fitting the threshold and reporting its FPR on the same images would
    # flatter it. Split stratified by kind so every tier appears in both halves.
    rng = np.random.default_rng(args.seed)
    fit_mask = np.zeros(len(kept), dtype=bool)
    for kind in np.unique(kinds):
        idx = np.flatnonzero(kinds == kind)
        rng.shuffle(idx)
        fit_mask[idx[: len(idx) // 2]] = True

    def tiers_of(mask):
        return {t: scores[mask & (kinds == f"ood/{t}")] for t in tiers}

    held_threshold = threshold_at_tpr(scores[fit_mask & is_id], args.tpr)
    print("\n" + "═" * 55)
    print("HELD-OUT ESTIMATE — threshold fitted on one half, measured on the other")
    print("═" * 55)
    report("evaluation half", scores[~fit_mask & is_id],
           tiers_of(~fit_mask), held_threshold, args.tpr)

    # ── shipped threshold, refitted on everything ────────────
    final_threshold = threshold_at_tpr(scores[is_id], args.tpr)
    print("\n" + "═" * 55)
    print("SHIPPED THRESHOLD — refitted on all data (in-sample, optimistic)")
    print("═" * 55)
    final_tpr = report("all data", scores[is_id], tiers_of(np.ones(len(kept), bool)),
                       final_threshold, args.tpr)

    payload = {
        "method":        method,
        "score":         chosen,
        "temperature":   temperature,
        "threshold":     final_threshold,
        "target_tpr":    args.tpr,
        "tpr":           final_tpr,
        "auroc":         auroc(scores, is_id),
        "fpr_near_bird": rates(scores[kinds == "ood/near_bird"], final_threshold),
        "fpr_animal":    rates(scores[kinds == "ood/animal"], final_threshold),
        "fpr_nonanimal": rates(scores[kinds == "ood/nonanimal"], final_threshold),
        "held_out": {
            "threshold":     held_threshold,
            "tpr":           rates(scores[~fit_mask & is_id], held_threshold),
            "fpr_near_bird": rates(scores[~fit_mask & (kinds == "ood/near_bird")], held_threshold),
            "fpr_animal":    rates(scores[~fit_mask & (kinds == "ood/animal")], held_threshold),
            "fpr_nonanimal": rates(scores[~fit_mask & (kinds == "ood/nonanimal")], held_threshold),
            "seed":          args.seed,
        },
        "n_id":            int(is_id.sum()),
        "n_ood":           int((~is_id).sum()),
        "checkpoint":      os.path.basename(MODEL_PATH),
        "manifest_sha256": manifest_sha,
        "created":         date.today().isoformat(),
        "note": ("Fitted by scripts/fit_openset.py. Accept an input as one of the "
                 "200 known species when score <= threshold. Rerun after changing "
                 "the checkpoint — logit scale is checkpoint-specific and a stale "
                 "threshold silently rejects real birds."),
    }

    if args.dry_run:
        print("\n--dry-run: not writing threshold.json")
        print(json.dumps(payload, indent=2))
        return

    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\n✅ Wrote {os.path.relpath(OUT_PATH, PROJECT_DIR)}")
    print("   Restart the app to pick it up.")


if __name__ == "__main__":
    main()
