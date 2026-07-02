# Deploying Sonic — permanent hosting

Two always-on, free surfaces:

- **Report** (static Quarto site) -> **GitHub Pages**
- **App** (live Streamlit + EASE model) -> **Hugging Face Spaces** (16 GB RAM free)

The repo is already prepared: `git` is initialised, the `.gitignore` commits only
the ~7.5 MB processed matrix (not the 514 MB EASE cache or 2.1 GB raw), the app is
`torch`-free, and cross-links are configurable. EASE refits from the matrix in ~15 s
at app startup, so no large model blob is needed in the cloud.

---

## 0. Push to GitHub (one time)

```bash
git add -A
git commit -m "Sonic music recommender: report + app + API"
gh repo create sonic-music-recsys --public --source=. --push   # or create on github.com and:
# git remote add origin https://github.com/<you>/sonic-music-recsys.git && git push -u origin main
```

Confirm the commit is small (a few MB): `git count-objects -vH`.

---

## 1. Report -> GitHub Pages

The workflow `.github/workflows/deploy-report.yml` renders the report and publishes
it on every push. Just enable Pages once:

1. GitHub repo -> **Settings -> Pages -> Build and deployment -> Source: GitHub Actions**.
2. Push (or re-run the workflow from the Actions tab). It builds `report/_site` with
   Quarto — no Python or data needed (`execute: enabled: false`).
3. Your report is live at `https://<you>.github.io/sonic-music-recsys/`.

## 2. App -> Hugging Face Spaces

1. Create a Space at <https://huggingface.co/new-space>: **SDK = Streamlit**, hardware
   = free CPU basic (16 GB RAM).
2. In the Space's **README.md**, set the config header so it runs our entry point:

   ```yaml
   ---
   title: Sonic Music Recommender
   sdk: streamlit
   app_file: app/streamlit_app.py
   pinned: false
   ---
   ```

   (You can add an `emoji:` line in the HF UI if you want a Space icon.)

3. Push this repo to the Space (it's a git remote):

   ```bash
   git remote add space https://huggingface.co/spaces/<you>/sonic
   git push space main
   ```

   The Space installs `requirements.txt` (Streamlit + the lean serving stack — no
   torch) and boots. First load fits EASE (~15 s); it's cached for later loads.
4. **Link back to the report:** Space -> **Settings -> Variables** -> add
   `REPORT_URL = https://<you>.github.io/sonic-music-recsys/`. The app's Overview
   page will link to it.

## 3. Point the report at the live app, then re-render

Once you know the Space URL, set it everywhere in the report and rebuild:

```bash
sed -i 's#http://localhost:8501#https://<you>-sonic.hf.space#g' report/_variables.yml report/_quarto.yml
quarto render report
git add report && git commit -m "Point report at the live app" && git push
```

(The in-page links use the `app_url` Quarto variable in `report/_variables.yml`; the
navbar "Live demo" button is in `report/_quarto.yml` — the one `sed` covers both.)

---

## Notes & gotchas

- **RAM:** EASE's weight matrix is ~0.5 GB in memory and peaks higher while fitting.
  HF's 16 GB free tier is comfortable; **Streamlit Community Cloud's ~1 GB tier is
  not** — it will likely OOM on the EASE fit. Prefer HF Spaces for the app.
- **Data:** only `data/processed/lastfm360k/{matrix.npz,item_ids,user_ids}` are
  committed. To rebuild them from scratch you need the raw dump (`python -m
  src.data_360k`), which is gitignored and not required for hosting.
- **Cold start:** the Space sleeps when idle (free tier) and takes ~30 s to wake and
  fit EASE. Fine for a portfolio link; mention it if you embed the demo somewhere.
- **API (optional):** the FastAPI service can also run on an HF **Docker** Space or
  Render using the same repo + `uvicorn api.main:app`; it's redundant with the app
  for a portfolio, so it's left un-deployed by default.
