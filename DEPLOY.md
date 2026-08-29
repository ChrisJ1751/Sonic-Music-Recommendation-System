# Deploying Sonic — permanent hosting

Two always-on, free surfaces:

- **Report** (static Quarto site) -> **GitHub Pages**
- **App** (live Streamlit + EASE model) -> **Hugging Face Spaces** (16 GB RAM free)

The repo is already prepared: `git` is initialised, the `.gitignore` commits only
the ~7.5 MB processed matrix (not the 514 MB EASE cache or 2.1 GB raw), the app is
`torch`-free, and cross-links are configurable. The Streamlit app refits EASE from
the matrix at startup, so no large model blob is needed for *that* surface — see the
RAM note below for why the containerised API does the opposite.

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
   torch) and boots. First load fits EASE (~1-2 min on the free CPU tier — see the
   RAM note below); it's cached for later loads in the same process.
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

## 4. API -> Fly.io (alternative, not the deployed path)

> **Not the deployed target — see section 5.** Kept because `fly.toml` is a
> working, fully-configured deployment and the record of the memory sizing.
> Fly requires a payment method: with `min_machines_running = 0` compute bills
> per-second while awake and the 1 GB volume bills continuously. See
> <https://fly.io/pricing>.

The FastAPI service is containerised and deployed separately from the app. Unlike
the Space, **it never refits EASE** — a refit needs ~2.5-3 GB and would OOM the
machine. It fetches the pre-fitted 514 MiB `ease_B.npy` at boot instead, so the
image stays artifact-free (~500 MB) and the model can be revised without
rebuilding the service.

### 4.1 Publish the EASE matrix (one time)

Create a **model** repo (not a Space) on Hugging Face and upload the artifact:

```bash
pip install huggingface_hub    # one-off tool, NOT a runtime dependency
hf auth login                  # needs a token with WRITE permission
hf upload jone1751/sonic-ease-360k data/processed/lastfm360k/ease_B.npy ease_B.npy
```

`hf upload` creates the repo if it does not exist, **public by default** — which
is required here, because the container fetches the artifact over plain HTTPS
with no token. If you make it private the Fly boot will fail on a 401.

(`hf` is the CLI in `huggingface_hub` >= 1.0; `huggingface-cli` still works but is
deprecated. In PowerShell, keep each command on one line — `\` is not a line
continuation there.)

The current artifact's identity, already recorded in `fly.toml`:

```
bytes  538889924
sha256 2ab7e37dab346cd88b7c36a3b218756f9382de1bc28bbf138ea0eeb431709b37
```

`scripts/fetch_ease_b.py` verifies both before putting the file in place, so a
truncated or corrupted download fails the boot instead of quietly serving a wrong
model. **Re-run the upload and update `EASE_B_SHA256` whenever EASE is refit.**

### 4.2 Deploy

```bash
git lfs pull                    # the Dockerfile refuses to build from LFS stubs
flyctl auth login
flyctl launch --no-deploy       # pick a globally-unique app name; keep fly.toml
flyctl volumes create sonic_artifacts --size 1 --region lhr
flyctl deploy
```

`flyctl launch` will offer to overwrite `fly.toml` — decline; the committed one
carries the measured memory sizing and the artifact configuration.

Then check it:

```bash
curl -s https://<your-app>.fly.dev/health
curl -s "https://<your-app>.fly.dev/recommendations/0?k=5"
flyctl logs        # one JSON object per request
```

### 4.3 What the settings mean

| Setting | Value | Why |
|---|---|---|
| `[[vm]] memory` | `2gb` | Measured floor is 688 MiB. 1 GB fits but leaves ~300 MiB; 2 GB is headroom for the refit failure mode. |
| `[[mounts]]` | `/artifacts` | Caches the 514 MiB download across restarts. Mounted here, **not** on `data/processed/lastfm360k`, which would shadow the artifacts baked into the image. |
| `auto_stop_machines` | `stop` | Scales to zero between visits; ~10 s cold start with the volume warm. Set `min_machines_running = 1` to eliminate cold starts. |
| `grace_period` | `180s` | The *first* boot on an empty volume downloads 514 MiB before binding. |

### 4.4 Cold-start budget (measured)

| Scenario | Time |
|---|---|
| Warm request | ~0.2 ms |
| Cold start, volume populated | ~10 s (0.7 s import + 5.5 s `load_state` + machine start) |
| First boot, empty volume | + 514 MiB download |
| If the artifact were missing | ~85-100 s refit at ~2.5-3 GB peak → OOM |

Of the 5.5 s `load_state`, **5.06 s is the ALS retrain** (`np.load` of the 514 MiB
matrix is only 0.11 s). ALS is trained purely to supply item embeddings for the MMR
diversity control. Caching its `item_factors` (2.8 MiB) would cut startup to well
under a second, but that is a change to `src/serving.py` and was left out of scope.

---

## 5. API -> Hugging Face Docker Space (DEPLOYED)

**Live: <https://jone1751-sonic-api.hf.space>** — `/docs`, `/health`, `/about`.

> **Requires PRO.** Hugging Face restricts Gradio and Docker Spaces on the
> `cpu-basic` tier to **PRO** subscribers (~$9/mo); only *Static* Spaces are free
> for everyone. `hf repos create --type space --sdk docker` returns
> **402 Payment Required** without it. The Streamlit Space in section 2 is
> unaffected.

cpu-basic is 2 vCPU / **16 GB RAM** — roomy for a service whose measured floor is
688 MiB. The artifact already lives on HF Hub, so the boot fetch stays in-network.
Chosen over Fly to consolidate the app, the model repo and the API on one
platform, which is the surface a reviewer actually browses.

**Known cost:** `cpu-basic` has **ephemeral disk** (persistent storage is a paid
add-on), so a Space that has slept re-downloads the 514 MiB artifact on wake.
Fly's volume avoided that. Measured build + boot from a cold push: ~100 s.

**Same image as Fly.** The Dockerfile bakes `EASE_B_URL` / `EASE_B_SHA256` /
`EASE_B_BYTES` as defaults, so the Space needs **no** variables configured. It
leaves `EASE_B_CACHE_DIR` unset, so the artifact downloads straight to the path
`serving.py` reads — correct here, because the free tier has no persistent volume.

### 5.1 Why a separate branch

A Space declares its SDK in its **README metadata**, and this repo's `README.md`
already carries the *Streamlit* Space header (section 2). One repo cannot carry
two different Space headers on the same branch, so the API Space is pushed from a
branch whose only difference is `README.md`, generated from
`docker/hf-space-README.md`.

### 5.2 Create and push

```bash
hf repos create sonic-api --type space --sdk docker
git remote add space-api https://huggingface.co/spaces/jone1751/sonic-api
git push space-api hf-space-api:main
```

The push carries the Git-LFS objects (the 7.7 MB processed core), which the
Dockerfile needs — its LFS-pointer guard fails the build otherwise.

Watch the build in the Space's **Logs** tab. First boot downloads the 514 MiB
artifact; you should see `fetch_ease_b | verified` then `model loaded`.

### 5.3 Check it

```bash
curl -s https://jone1751-sonic-api.hf.space/health
curl -s "https://jone1751-sonic-api.hf.space/recommendations/17084?k=5"
```

### 5.4 Updating it later

```bash
git checkout hf-space-api
git merge main                     # keep this branch's README.md on conflict
git push space-api hf-space-api:main
```

### 5.5 Tradeoffs vs Fly

| | HF Docker Space | Fly.io |
|---|---|---|
| Cost | PRO subscription (~$9/mo) — **deployed here** | Card required (~$2/mo at this config) |
| RAM | 16 GB (not configurable) | Pinned in `fly.toml` (2 GB) |
| Artifact caching | None — re-downloads on cold start | Persistent volume |
| Sleep | After ~48 h idle | Scale-to-zero per request |

The honest summary: HF is roomy but you cannot demonstrate right-sizing, because
you are simply given 16 GB — and it now costs more than Fly. The memory analysis
in `decisions.md` and `fly.toml` is what makes the sizing defensible, not the
platform. Fly is the deployed target.

---

## Notes & gotchas

- **RAM (measured 2026-08-28):** the EASE weight matrix is a dense float32
  11,607 x 11,607 array = **514 MiB** resident; the loaded service settles at
  **688 MiB RSS**. *Fitting* it is the expensive part: ~1.04e12 FLOPs for the
  inverse, i.e. **~85-100 s and ~2.5-3 GB peak** at 10-12 GFLOP/s with BLAS pinned
  to one thread. HF's 16 GB free tier absorbs that; **Streamlit Community Cloud's
  ~1 GB tier will OOM**. The containerised API never refits — it fetches the
  pre-fitted artifact (section 4).
- **Data:** only `data/processed/lastfm360k/{matrix.npz,item_ids,user_ids}` are
  committed. To rebuild them from scratch you need the raw dump (`python -m
  src.data_360k`), which is gitignored and not required for hosting.
- **Cold start:** the Space sleeps when idle (free tier) and must wake *and refit*
  EASE, which is ~1-2 min on the free CPU tier — not the ~30 s previously claimed
  here. The Fly API (section 4) ships the pre-fitted matrix instead and cold-starts
  in ~10 s. Mention this if you embed the demo anywhere.
- **API:** now deployed on Fly.io — see section 4. (This supersedes the previous
  note that the API was "redundant with the app ... left un-deployed by default";
  see decisions.md 2026-08-28.)
