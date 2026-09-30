---
title: BirdID API
emoji: 🐦
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.12.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Backend for BirdID — birds by photo and by call
models:
  - imageomics/bioclip
---

# BirdID API

The backend of [BirdID](https://github.com/Rohit-0612/birdid): EfficientNetV2-S
over the 200 species of CUB-200-2011, BioCLIP for birds outside them, BirdNET
for calls, and a grounded field-guide chat (local Ollama → Groq → knowledge-base
text). The website is deployed separately and talks to this Space over `/api`.

Deployed from the main repository with `scripts/deploy_space.py`; edit there,
not here.
