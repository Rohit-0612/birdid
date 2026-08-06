#!/usr/bin/env python3
"""
Collect photos from Wikimedia Commons for open-set evaluation.

    python3 scripts/download_openset_photos.py --what all
    python3 scripts/download_openset_photos.py --what ood --per-label 6
    python3 scripts/download_openset_photos.py --what id --species-limit 70

Two sets, both needed because nothing in this repo can currently measure
anything honestly:

  ID  (in-distribution)  photos of CUB species that are NOT CUB images.
                         Used for out-of-dataset accuracy and, before a clean
                         retrain exists, for fitting the rejection threshold.

  OOD (out-of-distribution) photos of things the model must refuse, in three
                         tiers. near_bird is the tier that matters -- a parrot
                         is the case that actually broke.

Guards
------
1. Every OOD label is checked against CUB_200_2011/classes.txt before any
   download. CUB contains traps: 5 kingfishers, 8 gulls, 7 terns, 2 pelicans,
   and 007.Parakeet_Auklet -- an auk, not a parrot. Downloading a "gull" as
   OOD would silently poison the measurement.
2. Every downloaded file is MD5-checked against all 11,788 CUB images and
   rejected on an exact match. Commons and CUB both draw on Flickr, so real
   overlap is possible. This catches byte-identical duplicates only, NOT
   re-encodings or crops -- say so in the README rather than implying the set
   is provably disjoint.

Licensing: Commons content is CC-licensed; attribution is a condition of use.
Author, licence and source URL for every file go into manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUB_IMAGES = os.path.join(PROJECT_DIR, "CUB_200_2011", "images")
CLASSES_TXT = os.path.join(PROJECT_DIR, "CUB_200_2011", "classes.txt")
KB_PATH = os.path.join(PROJECT_DIR, "data", "species_kb.json")
OUT_DIR = os.path.join(PROJECT_DIR, "data", "openset")
MANIFEST = os.path.join(OUT_DIR, "manifest.json")
MD5_CACHE = os.path.join(OUT_DIR, "cub_md5.json")

API = "https://commons.wikimedia.org/w/api.php"
UA = "birdsproject/1.0 (educational bird-ID project; open-set evaluation)"
IMG_EXT = (".jpg", ".jpeg", ".png")

# CUB folder names that are abbreviated, misspelled, or bare genera. Searching
# Commons for "Cardinal" or "Geococcyx" returns the wrong thing.
NAME_FIXUPS = {
    "017.Cardinal": "Northern Cardinal",
    "022.Chuck_will_Widow": "Chuck-will's-widow",
    "044.Frigatebird": "Magnificent Frigatebird",
    "074.Florida_Jay": "Florida Scrub-Jay",
    "091.Mockingbird": "Northern Mockingbird",
    "092.Nighthawk": "Common Nighthawk",
    "103.Sayornis": "Eastern Phoebe",
    "105.Whip_poor_Will": "Eastern Whip-poor-will",
    "110.Geococcyx": "Greater Roadrunner",
    "124.Le_Conte_Sparrow": "LeConte's Sparrow",
    "130.Tree_Sparrow": "American Tree Sparrow",
    "134.Cape_Glossy_Starling": "Cape Starling",
    "141.Artic_Tern": "Arctic Tern",
    "146.Forsters_Tern": "Forster's Tern",
    "070.Green_Violetear": "Mexican Violetear",
}

# Nothing here may be a CUB species; enforced at runtime against classes.txt.
OOD_LABELS = {
    "near_bird": [
        "Scarlet Macaw", "Blue-and-yellow Macaw", "Eclectus Parrot",
        "Grey Parrot", "Budgerigar", "Cockatiel", "Rainbow Lorikeet",
        "Sulphur-crested Cockatoo",
        "Red-tailed Hawk", "Bald Eagle", "Peregrine Falcon", "Golden Eagle",
        "Barn Owl", "Great Horned Owl", "Snowy Owl",
        "Emperor Penguin", "African Penguin",
        # NB: no "Mallard" -- CUB has 087.Mallard. The trap check caught it.
        "Wood Duck", "Canada Goose", "Mute Swan",
        "Greater Flamingo", "Indian Peafowl", "Common Ostrich",
        "Toco Toucan", "Southern Ground Hornbill", "Grey Crowned Crane",
    ],
    "animal": [
        "Golden Retriever", "Domestic cat", "Eastern gray squirrel",
        "Red fox", "Monarch butterfly", "Green iguana", "Common frog",
        "Honey bee",
    ],
    "nonanimal": [
        "Sunflower", "Oak tree", "Sports car", "Mountain landscape",
        "Coffee cup", "Bicycle",
    ],
}


# ── HTTP, cloned from scripts/download_bird_audio.py ─────────────────────

def _get(url, timeout=30, tries=5):
    delay = 2.0
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            return urllib.request.urlopen(req, timeout=timeout).read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == tries:
                raise
            wait = float(e.headers.get("Retry-After") or delay)
            print(f"      rate limited, waiting {wait:.0f}s ({attempt}/{tries})")
            time.sleep(wait)
            delay = min(delay * 2, 60)
    raise RuntimeError("unreachable")


def api(**params):
    params.update(action="query", format="json")
    return json.loads(_get(API + "?" + urllib.parse.urlencode(params)))


def strip_html(s):
    return re.sub(r"<[^>]+>", "", s or "").strip()


def search_images(query, limit):
    res = api(list="search", srsearch=f'filetype:bitmap "{query}"',
              srnamespace=6, srlimit=min(limit * 3, 40))
    return [r["title"] for r in res.get("query", {}).get("search", [])
            if r["title"].lower().endswith(IMG_EXT)][:limit]


def file_info(title, width=800):
    res = api(titles=title, prop="imageinfo",
              iiprop="url|extmetadata|mime|size|user", iiurlwidth=width)
    page = next(iter(res["query"]["pages"].values()))
    info = (page.get("imageinfo") or [{}])[0]
    if not info:
        return None
    meta = info.get("extmetadata", {})
    return {
        "url": info.get("thumburl") or info["url"],
        "author": strip_html(meta.get("Artist", {}).get("value", "")) or info.get("user"),
        "license": meta.get("LicenseShortName", {}).get("value", "see Commons page"),
        "source_page": "https://commons.wikimedia.org/wiki/" +
                       urllib.parse.quote(title.replace(" ", "_")),
    }


# ── CUB overlap guard ────────────────────────────────────────────────────

def cub_md5_set():
    if os.path.exists(MD5_CACHE):
        return set(json.load(open(MD5_CACHE)))
    print("hashing CUB images once (~11,788 files)...")
    md5s = set()
    for root, _, files in os.walk(CUB_IMAGES):
        for fn in files:
            if fn.lower().endswith(IMG_EXT):
                with open(os.path.join(root, fn), "rb") as f:
                    md5s.add(hashlib.md5(f.read()).hexdigest())
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(sorted(md5s), open(MD5_CACHE, "w"))
    print(f"  cached {len(md5s)} hashes -> {MD5_CACHE}")
    return md5s


def cub_species_tokens():
    """Lowercase word tokens of every CUB species, for the OOD trap check."""
    names = [l.split(" ", 1)[1].strip() for l in open(CLASSES_TXT)]
    return {n.split(".", 1)[1].replace("_", " ").lower() for n in names}


def cub_collision(label, cub_names):
    """Return a reason string if this OOD label overlaps CUB, else None.

    Two checks. Exact/containment catches 'Mallard' (CUB has 087.Mallard --
    this fired on the first run and would otherwise have put a real CUB
    species into the out-of-distribution set). The head-noun check catches
    'Common Tern', which shares 'tern' with CUB's 7 terns even though that
    exact species is absent.
    """
    low = label.lower()
    for name in cub_names:
        if name == low or name in low:
            return f"names CUB species {name!r}"
    head = low.split()[-1]
    clash = sorted(n for n in cub_names if n.split()[-1] == head)
    if clash:
        return f"shares head noun {head!r} with CUB: {clash[:4]}"
    return None


# ── main ─────────────────────────────────────────────────────────────────

def load_manifest():
    if os.path.exists(MANIFEST):
        return json.load(open(MANIFEST))
    return []


def fetch_group(targets, subdir, per_label, manifest, have, cub_md5, kind):
    added, skipped = 0, 0
    for label, query in targets:
        folder = os.path.join(OUT_DIR, subdir, re.sub(r"[^\w]+", "_", label))
        print(f"  {label}  <- \"{query}\"")
        try:
            titles = search_images(query, per_label)
        except Exception as e:                                   # noqa: BLE001
            print(f"      search failed: {e}")
            continue
        if not titles:
            print("      no results")
            continue
        os.makedirs(folder, exist_ok=True)
        for title in titles:
            try:
                info = file_info(title)
                if not info:
                    continue
                fname = os.path.basename(urllib.parse.unquote(info["url"].split("?")[0]))
                rel = os.path.relpath(os.path.join(folder, fname), OUT_DIR)
                if rel in have:
                    continue
                blob = _get(info["url"], timeout=90)
                if hashlib.md5(blob).hexdigest() in cub_md5:
                    print(f"      SKIP {fname} -- byte-identical to a CUB image")
                    skipped += 1
                    continue
                with open(os.path.join(folder, fname), "wb") as f:
                    f.write(blob)
                manifest.append({
                    "file": rel, "kind": kind, "label": label, "query": query,
                    "author": info["author"], "license": info["license"],
                    "source": info["source_page"], "bytes": len(blob),
                })
                have.add(rel)
                added += 1
                time.sleep(3)
            except Exception as e:                               # noqa: BLE001
                print(f"      failed {title}: {e}")
        json.dump(manifest, open(MANIFEST, "w"), indent=2, ensure_ascii=False)
    return added, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", choices=["id", "ood", "all"], default="all")
    ap.add_argument("--per-label", type=int, default=5)
    ap.add_argument("--species-limit", type=int, default=70)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    cub_md5 = cub_md5_set()
    manifest = load_manifest()
    have = {m["file"] for m in manifest}
    total_added = total_skipped = 0

    if args.what in ("ood", "all"):
        cub_names = cub_species_tokens()
        print("\n=== OOD (must NOT be CUB species) ===")
        problems = [(t, l, r) for t, labels in OOD_LABELS.items()
                    for l in labels if (r := cub_collision(l, cub_names))]
        if problems:
            print(f"  {len(problems)} label(s) collide with CUB and must be removed:")
            for tier, label, reason in problems:
                print(f"    [{tier}] {label}: {reason}")
            sys.exit("Fix OOD_LABELS before downloading.")
        n_labels = sum(len(v) for v in OOD_LABELS.values())
        print(f"  trap check passed for {n_labels} labels against 200 CUB species")
        for tier, labels in OOD_LABELS.items():
            print(f"\n-- tier: {tier} --")
            a, s = fetch_group([(l, l) for l in labels], os.path.join("ood", tier),
                               args.per_label, manifest, have, cub_md5, f"ood/{tier}")
            total_added += a
            total_skipped += s

    if args.what in ("id", "all"):
        classes = [l.split(" ", 1)[1].strip() for l in open(CLASSES_TXT)]
        kb = json.load(open(KB_PATH)) if os.path.exists(KB_PATH) else {}
        targets = []
        for folder in classes[:args.species_limit]:
            # Prefer the KB's scientific name; Commons files are titled by
            # binomial far more consistently than by common name.
            sci = (kb.get(folder) or {}).get("scientific_name")
            query = sci or NAME_FIXUPS.get(folder) or folder.split(".", 1)[1].replace("_", " ")
            targets.append((folder, query))
        print(f"\n=== ID (CUB species, non-CUB photos) — {len(targets)} species ===")
        a, s = fetch_group(targets, "id_commons", args.per_label,
                           manifest, have, cub_md5, "id")
        total_added += a
        total_skipped += s

    json.dump(manifest, open(MANIFEST, "w"), indent=2, ensure_ascii=False)
    by_kind = {}
    for m in manifest:
        by_kind[m["kind"]] = by_kind.get(m["kind"], 0) + 1
    print(f"\nadded {total_added}, skipped {total_skipped} as CUB duplicates")
    print(f"manifest: {len(manifest)} files -> {MANIFEST}")
    for k in sorted(by_kind):
        print(f"   {k:<18} {by_kind[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
