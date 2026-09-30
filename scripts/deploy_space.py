#!/usr/bin/env python3
"""Publish the BirdID API to a Hugging Face Space.

    python3 scripts/deploy_space.py --repo USER/birdid-api --dry-run
    python3 scripts/deploy_space.py --repo USER/birdid-api --site https://birdid.vercel.app

Uploads only what the API needs at runtime — the code, the knowledge base, the
open-set and verifier thresholds, the verifier's text-embedding cache, the
bundled audio samples and the inference checkpoint — plus deploy/space/* at
the Space root. Datasets, the 235 MB training checkpoint, logs and the local
life list never leave this machine.

Creates the Space on first run (Gradio SDK, ZeroGPU hardware — the free tier)
and sets its non-secret variables. GROQ_API_KEY is a secret: add it under the
Space's Settings → Secrets, or pass --groq-secret to copy it from this shell's
environment (it is never printed).

Needs `hf auth login` with a token that has write access.
"""

import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CODE = [
    "api.py", "bird_core.py", "bird_data.py", "bird_text.py", "LICENSE",
    "services/__init__.py", "services/llm.py", "services/sightings.py", "services/verifier.py",
]
DATA = [
    "data/species_kb.json",
    "data/openset/threshold.json",
    "data/verifier/threshold.json",
]
DATA_GLOBS = ["data/verifier/text_embeddings_*.npz", "bird_audio_samples/**"]
CHECKPOINT = "best_bird_model_inference.pth"
SPACE_FILES = "deploy/space"

VARIABLES = {"BIRD_DEVICE": "cpu", "BIRD_LLM_ORDER": "ollama,groq"}


def collect():
    """(local path, path in the Space repo) for everything to upload."""
    files = [(p, p) for p in CODE + DATA]
    for pattern in DATA_GLOBS:
        files += [(p, p) for p in sorted(glob.glob(pattern, root_dir=ROOT, recursive=True))
                  if os.path.isfile(os.path.join(ROOT, p))]
    files.append((CHECKPOINT, CHECKPOINT))
    for name in sorted(os.listdir(os.path.join(ROOT, SPACE_FILES))):
        files.append((f"{SPACE_FILES}/{name}", name))

    missing = [local for local, _ in files if not os.path.isfile(os.path.join(ROOT, local))]
    if missing:
        sys.exit("missing files:\n  " + "\n  ".join(missing))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", required=True, help="Space id, e.g. USER/birdid-api")
    parser.add_argument("--site", default="", help="the Vercel URL, shown on the Space page")
    parser.add_argument("--groq-secret", action="store_true",
                        help="copy GROQ_API_KEY from this environment into the Space secrets")
    parser.add_argument("--dry-run", action="store_true", help="list what would be uploaded")
    args = parser.parse_args()

    files = collect()
    total = sum(os.path.getsize(os.path.join(ROOT, local)) for local, _ in files)
    print(f"{len(files)} files, {total / 1e6:.1f} MB → {args.repo}")
    if args.dry_run:
        for local, remote in files:
            size = os.path.getsize(os.path.join(ROOT, local)) / 1e6
            print(f"  {size:8.2f} MB  {remote}" + (f"   (from {local})" if local != remote else ""))
        return

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="space", space_sdk="gradio",
                    space_hardware="zero-a10g", exist_ok=True)

    variables = dict(VARIABLES)
    if args.site:
        variables["BIRDID_SITE_URL"] = args.site
    for key, value in variables.items():
        api.add_space_variable(args.repo, key, value)

    if args.groq_secret:
        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            sys.exit("--groq-secret given but GROQ_API_KEY is not set in this environment")
        api.add_space_secret(args.repo, "GROQ_API_KEY", key)
        print("GROQ_API_KEY secret set")

    operations = [CommitOperationAdd(path_in_repo=remote,
                                     path_or_fileobj=os.path.join(ROOT, local))
                  for local, remote in files]
    commit = api.create_commit(args.repo, repo_type="space", operations=operations,
                               commit_message="Deploy BirdID API")
    host = args.repo.lower().replace("/", "-").replace("_", "-").replace(".", "-")
    print(f"pushed {commit.oid[:8]}\n  Space: https://huggingface.co/spaces/{args.repo}"
          f"\n  API:   https://{host}.hf.space/api/health")


if __name__ == "__main__":
    main()
