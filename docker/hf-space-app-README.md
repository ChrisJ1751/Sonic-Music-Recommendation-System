---
title: Sonic Music Recommender
emoji: 🎧
colorFrom: green
colorTo: purple
sdk: streamlit
app_file: app/streamlit_app.py
python_version: "3.11"
pinned: false
short_description: A disciplined music recommender, EASE on Last.fm-360K.
---

# Sonic

Music recommendations from listening patterns, on the Last.fm-360K dataset.
Served model: **EASE** (Steck 2019), a linear item-item autoencoder that beat
tuned ALS and a deep Mult-VAE on the same frozen split.

- **API:** https://jone1751-sonic-api.hf.space/docs
- **Write-up:** https://chrisj1751.github.io/Sonic-Music-Recommendation-System/
- **Source:** https://github.com/ChrisJ1751/Sonic-Music-Recommendation-System

Pick a listener to see real recommendations, or explore artist-to-artist radio.
The **diversity** slider is MMR re-ranking, the same lever the API exposes as its
`diversity` parameter.

> This Space is built from the repository above. Only this README differs,
> because a Space declares its SDK in README front matter and the repository's
> own README should not carry deployment config.
