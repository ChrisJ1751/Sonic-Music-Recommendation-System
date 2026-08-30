---
title: Sonic Recommender API
emoji: 🎧
colorFrom: green
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
short_description: FastAPI serving layer, EASE on Last.fm-360K.
---

# Sonic, recommender API

The **FastAPI serving layer** for the Last.fm-360K artist recommender. Served
model: **EASE** (Steck 2019), the linear item-item autoencoder that won the model
comparison on this data.

- **Interactive demo (Streamlit):** https://jone1751-sonic.hf.space
- **Write-up (Quarto report):** https://chrisj1751.github.io/Sonic-Music-Recommendation-System/
- **Source:** https://github.com/ChrisJ1751/Sonic-Music-Recommendation-System

## Try it

| Endpoint | What it does |
|---|---|
| `/docs` | Swagger explorer |
| `/health` | Liveness + catalogue size |
| `/recommendations/{user_id}?k=10&diversity=0.3` | Top-k for a user; `diversity` re-ranks with MMR |
| `/similar-artists/{artist_id}?k=10` | "Fans also like", from EASE's item-item weights |
| `/about` | Full results + methodology payload |

IDs are matrix indices: `user_id` is a user row (0 to 39,498), `artist_id` an artist
column (0 to 11,606). A `user_id` past the end falls back to popularity.

## How it boots

The EASE weight matrix is a dense float32 11,607² array, **514 MiB**. Refitting
it means an ~1.04e12 FLOP matrix inverse: measured **~85 to 100 s at ~2.5 to 3 GB peak**.
So the container never refits. It fetches the pre-fitted matrix from
[jone1751/sonic-ease-360k](https://huggingface.co/jone1751/sonic-ease-360k),
verifies its size and SHA-256, and fails the boot loudly if either is wrong.

Measured: **688 MiB** resident once loaded, **~0.2 ms** warm requests, and a cold
start dominated by the artifact download plus a ~5 s ALS fit (ALS supplies the
item embeddings used by the MMR diversity control).

Every request emits one structured JSON log line, request id, path, status,
duration, and echoes `X-Request-ID`.

> This Space is built from the same repository as the Streamlit demo; only this
> README differs, because a Space's SDK is declared in its README metadata.
