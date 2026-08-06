#!/usr/bin/env python3
"""
Measure how well the Audio ID tab actually performs.

    python3 scripts/eval_audio.py
    python3 scripts/eval_audio.py --min-conf 0.1 --with-prior

Reads bird_audio_samples/manifest.json (written by download_bird_audio.py),
runs every clip through BirdNET exactly the way the app does, and reports:

  top-1 accuracy   how often the highest-confidence detection is the
                   species the recording is actually of
  recall@any       how often the correct species appears anywhere in the
                   clip's detections, even if not ranked first
  false species    other species BirdNET reported, which matters because a
                   long field recording legitimately contains several birds

What this number is and is not
------------------------------
This is a small, favourable test set: curated xeno-canto recordings, mostly
one bird, mostly clean. Real field audio is harder. Treat the result as a
sanity check on the pipeline, not as a published benchmark figure -- for
that, see BirdSet (Rauch et al., ICLR 2025).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import date

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(PROJECT_DIR, "bird_audio_samples", "manifest.json")
OUT_DIR = os.path.join(PROJECT_DIR, "evaluation")
RESULTS = os.path.join(OUT_DIR, "audio_results.json")


def norm(s):
    return " ".join((s or "").lower().replace("-", " ").split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-conf", type=float, default=0.25)
    ap.add_argument("--with-prior", action="store_true",
                    help="enable the lat/lon/date occurrence prior")
    ap.add_argument("--lat", type=float, default=39.83)
    ap.add_argument("--lon", type=float, default=-98.58)
    args = ap.parse_args()

    if not os.path.exists(MANIFEST):
        sys.exit(f"No {MANIFEST}. Run scripts/download_bird_audio.py first.")
    manifest = json.load(open(MANIFEST))
    if not manifest:
        sys.exit("Manifest is empty.")

    import librosa
    import soundfile as sf
    from birdnetlib import Recording
    from birdnetlib.analyzer import Analyzer

    analyzer = Analyzer()
    os.makedirs(OUT_DIR, exist_ok=True)

    rows, correct_top1, correct_any = [], 0, 0
    for i, entry in enumerate(manifest, 1):
        path = os.path.join(PROJECT_DIR, "bird_audio_samples", entry["file"])
        if not os.path.exists(path):
            print(f"[{i}/{len(manifest)}] missing {entry['file']}")
            continue

        expected = entry["common_name"]
        t0 = time.time()
        y, sr = librosa.load(path, sr=48000, mono=True)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, y, sr)

        kwargs = {"min_conf": args.min_conf}
        if args.with_prior:
            kwargs.update(lat=args.lat, lon=args.lon, date=date(2026, 5, 14))
        rec = Recording(analyzer, tmp.name, **kwargs)
        rec.analyze()
        os.unlink(tmp.name)

        dets = sorted(rec.detections, key=lambda d: -d["confidence"])
        names = [d["common_name"] for d in dets]
        top1 = names[0] if names else None

        # BirdNET uses current standard names; the manifest uses the same,
        # but allow containment so "Cardinal" vs "Northern Cardinal" passes.
        hit1 = bool(top1) and (norm(expected) in norm(top1) or norm(top1) in norm(expected))
        hitany = any(norm(expected) in norm(n) or norm(n) in norm(expected) for n in names)
        correct_top1 += hit1
        correct_any += hitany

        others = sorted({n for n in names
                         if not (norm(expected) in norm(n) or norm(n) in norm(expected))})
        rows.append({
            "file": entry["file"], "expected": expected, "top1": top1,
            "top1_conf": round(dets[0]["confidence"], 3) if dets else None,
            "correct_top1": hit1, "correct_any": hitany,
            "n_detections": len(dets), "other_species": others,
            "duration_s": round(len(y) / sr, 1),
        })
        mark = "OK  " if hit1 else ("~   " if hitany else "MISS")
        print(f"[{i}/{len(manifest)}] {mark} {expected:<22} -> "
              f"{str(top1):<26} {rows[-1]['top1_conf']}  "
              f"({rows[-1]['duration_s']}s, {time.time()-t0:.0f}s)")

    n = len(rows)
    if not n:
        sys.exit("No clips evaluated.")

    summary = {
        "n_clips": n,
        "n_species": len({r["expected"] for r in rows}),
        "min_conf": args.min_conf,
        "location_prior": args.with_prior,
        "top1_accuracy": round(100 * correct_top1 / n, 1),
        "recall_any": round(100 * correct_any / n, 1),
        "evaluated": date.today().isoformat(),
        "caveat": ("small curated xeno-canto set, mostly single-bird clean "
                   "recordings; not a field-conditions benchmark"),
    }
    with open(RESULTS, "w") as f:
        json.dump({"summary": summary, "clips": rows}, f, indent=2)

    print("\n" + "=" * 58)
    print(f"clips              : {n}  ({summary['n_species']} species)")
    print(f"threshold          : {args.min_conf}   location prior: "
          f"{'ON' if args.with_prior else 'OFF'}")
    print(f"top-1 accuracy     : {summary['top1_accuracy']}%  "
          f"({correct_top1}/{n})")
    print(f"correct anywhere   : {summary['recall_any']}%  ({correct_any}/{n})")
    print("=" * 58)
    print(f"written to {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
