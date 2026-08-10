# BirdID

Identify a bird from a photo or a recording, hear it described aloud in a
field-guide voice, ask follow-up questions, and collect every bird you find into a
deck. Everything runs locally — no API keys, no telemetry. The only network access
is a one-time model download, and only if you enable the second identifier.

<!-- Screenshots: run ./dev.sh and grab the Identify and Deck views. -->

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

**Name the birds it was never trained on.** When the gate rejects a photo or
confidence drops below 60%, a second model gets a say. BioCLIP scores the image
against species *names* rather than a fixed class list, so its vocabulary is a data
file — 6,423 birds, 32× the trained model's reach. A macaw that the classifier
calls "White-breasted Kingfisher, 15%" comes back correctly named. Measured on 130
photos of 26 species deliberately outside the 200: **79% top-1** where the species
was in vocabulary, **91.5% precision** among the answers it commits to. It runs
locally and is optional.

**Collect them.** Every photo files itself into a deck, one card per species,
Pokédex-style — no save button. Two sides: **The 200** the classifier covers, shown
as a completable grid with locked cards for what you have not found, and **New
birds** for everything the second model named from outside those 200. Photograph
the same bird twice and you get one card reading "×2", with the better photo kept.
When neither model is confident, nothing is filed and the app tells you why; a
wrong card is worse than a missing one.

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

**Life list.** Every encounter is logged to a local SQLite file alongside the deck —
the deck dedupes by species, the log keeps the history.

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

Optional, to name birds outside the trained 200:

```bash
pip3 install -r requirements-verify.txt
python3 scripts/eval_verifier.py       # measures accuracy, fits the threshold
```

First use downloads ~400 MB of BioCLIP weights from HuggingFace and caches them.
That download is the project's only network dependency and it happens once —
inference afterwards is fully offline.

Without either extra everything still works. The LLM features fall back to template
text from the knowledge base, verification reports `ran: false`, and both are
labelled as such in the UI.

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
  verifier.py     BioCLIP: open-vocabulary ID for birds outside the 200
  sightings.py    SQLite life list + the deck
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
| `scripts/eval_verifier.py` | Measure the open-vocabulary verifier on 26 non-CUB species, fit its threshold |
| `scripts/build_species_kb.py` | Generate the species knowledge base via Ollama |
| `scripts/validate_kb.py` | Gate the knowledge base (placeholders, IUCN values, binomials) |
| `scripts/download_bird_audio.py` | Fetch CC-licensed calls from Wikimedia |
| `scripts/download_openset_photos.py` | Fetch the in-distribution and OOD photo sets |

Rerun `fit_openset.py` after changing the checkpoint — the logit scale is
checkpoint-specific, and a stale threshold silently rejects real birds. Rerun
`eval_verifier.py` if the BirdNET label file or the prompt template changes.

## Tests

```bash
python3 -m pytest              # 77 tests, ~15s from cold
```

`tests/test_core_schema.py` pins the result-dict contract that four consumers
read, and covers the open-set gate end to end. `tests/test_deck.py` covers all four
routing branches, species-level dedupe, and that new birds never inflate progress
over the 200. `tests/test_api.py` covers the HTTP surface with `TestClient` and both
the LLM and the verifier stubbed, including that a missing Ollama daemon or a
missing BioCLIP degrades rather than 500s. Tests that need the gitignored datasets
skip rather than fail, and the deck is redirected to a scratch database.

Frontend: `cd bird-frontend && npm run lint && npm run build`.

`npm run contrast` is the gate for the design tokens. `src/index.css` documents a
measured contrast ratio next to almost every colour and tells you to check them
before editing; this script is what does the checking. It verifies every
documented foreground/background pair in both themes, asserts the two
hand-duplicated dark blocks have not drifted apart, asserts the values the
comments record as *rejected* still fail, and holds the ambient forest and the
paper grain to a budget — the composited page must depart from the base colour
less than the canopy that shipped before it did. Run it after touching any token.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `BIRD_MODEL_PATH` | auto-detected | Checkpoint to load |
| `BIRD_SAVE_DIR` | `evaluation/` | Where generated PNGs go |
| `BIRD_TEST_DIR` | `birds_split/test` | Evaluation image folder |
| `BIRD_DB_PATH` | `data/sightings.db` | Life-list database |
| `BIRD_LLM_MODEL` | `qwen3:8b` | Ollama model |
| `BIRD_LLM_KEEP_ALIVE` | `15m` | How long Ollama holds the weights |
| `BIRD_VERIFIER_MODEL` | `hf-hub:imageomics/bioclip` | Open-vocabulary model |
| `BIRD_VERIFY_BELOW` | `0.60` | Confidence under which the verifier is consulted |

## Credits and licensing

This project is released under the [MIT License](LICENSE). The datasets, models
and media it depends on carry their own terms, listed below.

- **CUB-200-2011** — Caltech-UCSD Birds, Wah et al. 2011.
- **Hero bird model** — `bird-frontend/public/models/stork.glb`, from the three.js
  example assets and credited there as "Stork by mirada from rome". MIT, as part
  of the three.js repository. Full note in `public/models/ATTRIBUTION.txt`. The
  file is unmodified; the app rewrites its vertex colours and material at load.
- **Animated nav indicator** — the shared-`layoutId` technique is adapted from
  ibelick's Animated Tabs on 21st.dev (MIT), hand-ported to plain JSX.
- **BirdNET** — Cornell Lab of Ornithology, via `birdnetlib`.
- **Audio and open-set photos** — Wikimedia Commons, all CC or public domain.
  Attribution for every file is in `bird_audio_samples/manifest.json` and
  `data/openset/manifest.json`, which are committed for exactly that reason.
- **BioCLIP** — Stevens et al. 2024, `imageomics/bioclip`. Candidate species names
  come from BirdNET's label file, filtered to birds: it is a sound model, so it also
  lists 92 frogs, crickets and katydids, and unfiltered a Bald Eagle ranked as
  "Honey Bee".
- **Open-set scoring** — max-softmax (Hendrycks & Gimpel 2017), energy
  (Liu et al. 2020). Energy is the usual recommendation and measures *worse than
  random* on this checkpoint; softmax entropy wins here. The script measures
  rather than assumes.
