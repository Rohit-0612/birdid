# BirdID

Identify a bird from a photo or a recording, hear it described aloud in a
field-guide voice, ask follow-up questions, and keep a life list. Everything runs
locally — no API keys, no cloud calls, no telemetry.

<!-- Screenshots: run ./dev.sh and grab the Identify and Life list views. -->

## What it does

**Identify by photo.** EfficientNetV2-S over the 200 species of CUB-200-2011.
You get the species, a confidence gauge, the taxonomy trail, the field marks that
confirm it, every look-alike with how to tell them apart, and a Grad-CAM overlay
showing which pixels drove the answer.

**Say when it does not know.** A 200-way softmax cannot answer "none of the
above" — it renormalises over the classes it has, so a photo of a frog used to
come back as a confident *Chuck-will's-widow*. An open-set gate now reads the raw
logits and rejects inputs that are not one of the known species. It reliably
rejects things that are not birds; it only partially rejects birds outside the
200, and the numbers for both are printed in the app.

**Identify by call.** BirdNET (Cornell Lab) covers ~6,500 species, so the Listen
view can name birds the photo model cannot — and it tells you which. Sixteen
Creative-Commons clips ship with the project so you can try it immediately.

**Voice, both directions.** The browser's Web Speech API reads a written summary
aloud, highlighting each sentence as it is spoken. Hold the mic button or the
space bar to speak a command: *"tell me about it"*, *"read the field marks"*,
*"save this"*, *"stop"*. Anything that is not a command becomes a question for the
grounded chat, and the answer is spoken back.

**Ask the field guide.** A local LLM answers questions grounded in a 200-species
knowledge base. Facts are supplied from the knowledge base and echoed, never
recalled from the model, so sizes and conservation statuses cannot be invented.

**Life list.** Save sightings to a local SQLite file: progress over the 200
species, family breakdown, timeline with thumbnails.

## Accuracy: read this first

**The bundled checkpoint's accuracy figures are not trustworthy, and the app says
so on every screen that shows one.**

It carries no record of what it was trained on, and measured evidence says it saw
essentially all of CUB-200-2011: official-test images that were never held out
score ~95%, *above* the ~86–88% realistic ceiling for this architecture, and
higher than images it definitely trained on. That is the signature of a data
leak. The species it names are usually right; the confidence number is not a
calibrated probability of being right.

The fix is in the repository. [`train.py`](train.py) trains against the committed
split manifests in [`data/splits/`](data/splits/) and records their SHA-256 in the
checkpoint. Any checkpoint carrying that hash is picked up automatically and the
warnings disappear:

```bash
python3 scripts/make_split.py          # official CUB split, hashed manifests
python3 train.py                       # ~85-88% top-1 is the honest range
python3 scripts/fit_openset.py         # refit the gate for the new logit scale
```

Species notes are machine-written too: [`data/species_kb.json`](data/species_kb.json)
was generated locally by `qwen3:8b` and is not expert-verified. Hand-written
entries are labelled *curated* / *verified* in the UI; everything else is
labelled as generated. The **How it works** view lays all of this out.

## Setup

Requires Python 3.11+ and Node 20+.

```bash
pip3 install -r requirements.txt -r requirements-api.txt
cd bird-frontend && npm install && cd ..
```

A checkpoint named `best_bird_model_inference.pth`, `best_bird_model.pth` or
`best_bird_model_official.pth` must be in the project root (or set
`BIRD_MODEL_PATH`). Checkpoints and datasets are gitignored — they are large and
re-downloadable.

Optional, for the spoken summaries, chat and comparison prose:

```bash
pip3 install -r requirements-llm.txt
ollama pull qwen3:8b && ollama serve
```

Without Ollama everything still works; those three features fall back to template
text built from the knowledge base and are labelled as such.

## Running

```bash
./dev.sh                 # API :8000 + dashboard :5173, Ctrl-C stops both
./dev.sh start|stop|status
```

Then open **http://localhost:5173**.

| | | |
|---|---|---|
| **:5173** | dashboard | the real UI — React + Vite |
| **:8000** | JSON API | `/docs` for the interactive schema |
| **:7860** | Gradio | debug surface, `./run.sh start` |

The Gradio app is kept deliberately: one process, no npm, direct access to the
model, and it renders the same results as plain text.

## Architecture

```
bird_data.py      curated lookup tables + display_name(), no torch
bird_core.py      model, knowledge base, open-set gate — returns dicts
bird_text.py      dict -> the plain-text report Gradio shows
bird_eval.py      slow offline harnesses (metrics, calibration)
api.py            FastAPI — the interface the dashboard talks to
services/
  llm.py          Ollama: narration, grounded chat, comparison
  sightings.py    SQLite life list
bird_complete_local.py   Gradio UI
bird-frontend/    React dashboard
```

The important rule: **`bird_core` returns dictionaries, never formatted text.**
Before this, the two predictors returned ASCII reports, so the frontend had to
regex-scrape them and most of the knowledge base never reached the screen —
`family`, `order` and `range_description` were loaded and displayed nowhere, and
every look-alike after the first was discarded. One structured payload now feeds
the dashboard, the narrator, the chat grounding and the life list.

`api.py` loads the model once at import. Do not run it with multiple workers on a
laptop: each worker would load its own copy of the checkpoint.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/make_split.py` | Official CUB split with hashed manifests |
| `train.py` | Train an auditable checkpoint (Colab notebook in `notebooks/`) |
| `scripts/fit_openset.py` | Fit the open-set threshold; sweeps four score families and picks by AUROC |
| `scripts/build_species_kb.py` | Generate the species knowledge base via Ollama |
| `scripts/validate_kb.py` | Gate the knowledge base (placeholders, IUCN values, binomials) |
| `scripts/download_bird_audio.py` | Fetch CC-licensed calls from Wikimedia |
| `scripts/download_openset_photos.py` | Fetch the in-distribution and OOD photo sets |

Rerun `fit_openset.py` after changing the checkpoint — the logit scale is
checkpoint-specific, and a stale threshold silently rejects real birds.

## Tests

```bash
python3 -m pytest              # 50 tests, ~15s from cold
```

`tests/test_core_schema.py` pins the result-dict contract that four consumers
read, and covers the open-set gate end to end. `tests/test_api.py` covers the
HTTP surface with `TestClient` and the LLM stubbed, including that a missing
Ollama daemon degrades rather than 500s. Tests that need the gitignored datasets
skip rather than fail, and the life list is redirected to a scratch database.

Frontend: `cd bird-frontend && npx eslint src && npm run build`.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `BIRD_MODEL_PATH` | auto-detected | Checkpoint to load |
| `BIRD_SAVE_DIR` | `evaluation/` | Where generated PNGs go |
| `BIRD_TEST_DIR` | `birds_split/test` | Evaluation image folder |
| `BIRD_DB_PATH` | `data/sightings.db` | Life-list database |
| `BIRD_LLM_MODEL` | `qwen3:8b` | Ollama model |
| `BIRD_LLM_KEEP_ALIVE` | `15m` | How long Ollama holds the weights |

## Credits and licensing

- **CUB-200-2011** — Caltech-UCSD Birds, Wah et al. 2011.
- **BirdNET** — Cornell Lab of Ornithology, via `birdnetlib`.
- **Audio and open-set photos** — Wikimedia Commons, all CC or public domain.
  Attribution for every file is in `bird_audio_samples/manifest.json` and
  `data/openset/manifest.json`, which are committed for exactly that reason.
- **Open-set scoring** — max-softmax (Hendrycks & Gimpel 2017), energy
  (Liu et al. 2020). Energy is the usual recommendation and measures *worse than
  random* on this checkpoint; softmax entropy wins here. The script measures
  rather than assumes.
