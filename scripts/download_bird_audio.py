#!/usr/bin/env python3
"""
Download CC-licensed bird call recordings for testing the Audio ID tab.

    python3 scripts/download_bird_audio.py                  # default species
    python3 scripts/download_bird_audio.py --per-species 3
    python3 scripts/download_bird_audio.py --species "Blue Jay" "American Crow"

Why not xeno-canto directly
---------------------------
The original download_bird_audio.py called xeno-canto API **v2**, which has
been retired — it now returns HTTP 404 with a pointer to v3, and v3 requires
a free account key. Rather than gate the project's test data behind a
credential, this pulls the same xeno-canto recordings from their Wikimedia
Commons mirror, which needs no key.

Licensing
---------
These recordings are CC-licensed and attribution is a condition of use, not a
courtesy. Every download is recorded in manifest.json with its author, licence
and source URL, and the app surfaces that alongside the audio.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_DIR, "bird_audio_samples")
MANIFEST = os.path.join(OUT_DIR, "manifest.json")
API = "https://commons.wikimedia.org/w/api.php"
UA = "birdsproject/1.0 (educational bird-ID project; contact: local user)"

# Common name -> scientific name. Commons files are titled by binomial, and
# these all correspond to species in CUB-200-2011.
DEFAULT_SPECIES = {
    "Northern Cardinal":  "Cardinalis cardinalis",
    "Blue Jay":           "Cyanocitta cristata",
    "American Crow":      "Corvus brachyrhynchos",
    "Barn Swallow":       "Hirundo rustica",
    "House Wren":         "Troglodytes aedon",
    "Song Sparrow":       "Melospiza melodia",
    "Red-winged Blackbird": "Agelaius phoeniceus",
    "Common Yellowthroat": "Geothlypis trichas",
}

AUDIO_EXT = (".mp3", ".ogg", ".oga", ".flac", ".wav")


def api(**params):
    params.update(action="query", format="json")
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def strip_html(s):
    return re.sub(r"<[^>]+>", "", s or "").strip()


def find_recordings(scientific_name, limit):
    q = f'filetype:audio "{scientific_name}"'
    res = api(list="search", srsearch=q, srnamespace=6, srlimit=limit * 3)
    return [r["title"] for r in res.get("query", {}).get("search", [])
            if r["title"].lower().endswith(AUDIO_EXT)][:limit]


def file_info(title):
    res = api(titles=title, prop="imageinfo",
              iiprop="url|extmetadata|mime|size|user")
    page = next(iter(res["query"]["pages"].values()))
    info = page.get("imageinfo", [{}])[0]
    if not info:
        return None
    meta = info.get("extmetadata", {})
    return {
        "title": title,
        "url": info["url"].split("?")[0],
        "mime": info.get("mime"),
        "bytes": info.get("size"),
        "author": strip_html(meta.get("Artist", {}).get("value", "")) or info.get("user"),
        "license": meta.get("LicenseShortName", {}).get("value", "see Commons page"),
        "source_page": "https://commons.wikimedia.org/wiki/" +
                       urllib.parse.quote(title.replace(" ", "_")),
    }


def download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r, open(dest, "wb") as f:
        f.write(r.read())
    return os.path.getsize(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-species", type=int, default=2)
    ap.add_argument("--species", nargs="*", help="common names from the default list")
    args = ap.parse_args()

    wanted = DEFAULT_SPECIES
    if args.species:
        missing = [s for s in args.species if s not in DEFAULT_SPECIES]
        if missing:
            sys.exit(f"Unknown species {missing}. Known: {sorted(DEFAULT_SPECIES)}")
        wanted = {s: DEFAULT_SPECIES[s] for s in args.species}

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else []
    have = {m["file"] for m in manifest}
    added = 0

    for common, sci in wanted.items():
        print(f"\n{common} ({sci})")
        try:
            titles = find_recordings(sci, args.per_species)
        except Exception as e:                                   # noqa: BLE001
            print(f"  search failed: {e}")
            continue
        if not titles:
            print("  no recordings found")
            continue

        folder = os.path.join(OUT_DIR, common.replace(" ", "_"))
        os.makedirs(folder, exist_ok=True)

        for title in titles:
            try:
                info = file_info(title)
                if not info:
                    continue
                fname = os.path.basename(urllib.parse.unquote(info["url"]))
                rel = os.path.join(os.path.basename(folder), fname)
                if rel in have:
                    print(f"  skip (have)  {fname}")
                    continue
                size = download(info["url"], os.path.join(folder, fname))
                manifest.append({
                    "file": rel,
                    "common_name": common,
                    "scientific_name": sci,
                    "author": info["author"],
                    "license": info["license"],
                    "source": info["source_page"],
                    "bytes": size,
                })
                have.add(rel)
                added += 1
                print(f"  saved  {fname}  ({size/1024:.0f} KB, {info['license']}, by {info['author']})")
                time.sleep(1)          # be polite to Commons
            except Exception as e:                               # noqa: BLE001
                print(f"  failed {title}: {e}")

    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{added} new file(s); {len(manifest)} total in {OUT_DIR}")
    print(f"Attribution recorded in {MANIFEST} — keep it with the audio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
