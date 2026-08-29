# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Container for the Sonic recommendation API (EASE on Last.fm-360K).
#
# Base image: python:3.12-slim-bookworm
#   * 3.12 matches .github/workflows/ci.yml, so the interpreter that runs the
#     tests is the interpreter that ships.
#   * `slim` is Debian/glibc: numpy, scipy and implicit all publish manylinux
#     wheels, so nothing compiles here. Alpine's musl has no manylinux wheels and
#     would force slow source builds against a different BLAS. Note that "no
#     compiler needed" is not "no system libraries needed" -- see libgomp1 below.
#   * Not full `python:3.12` (~1 GB) -- the build toolchain is not needed at run
#     time, which is also why this is a two-stage build.
#
# The 514 MiB EASE weight matrix is deliberately NOT in this image. It is fetched
# and verified at boot by scripts/fetch_ease_b.py; see that file for why
# refitting in-container is not an option (~1.04e12 FLOPs, ~2.5-3 GB peak).
# The three small processed artifacts (~7.7 MB) ARE baked in, so the container
# needs the network for exactly one file.
# ---------------------------------------------------------------------------

FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
# Dependency layer first so it is cached until the manifest actually changes.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .


# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1
# NOTE: EASE_B_CACHE_DIR is deliberately NOT set here. It is platform config:
# fly.toml sets it to /artifacts because Fly attaches a persistent volume there.
# Hugging Face Spaces have no persistent volume on the free tier, so leaving it
# unset makes the artifact download straight to the path serving.py reads.

# Which model artifact to serve. Baked in as a DEFAULT (not platform config) so
# `docker run sonic-api` just works, and so platforms that cannot declare env
# vars in config -- a Hugging Face Space, for instance -- need no manual setup.
# Override at runtime to serve a different EASE fit; fly.toml does exactly that.
# The digest is verified at boot, so the image states precisely which model it
# serves and a mismatch fails the boot rather than serving the wrong weights.
ENV EASE_B_URL=https://huggingface.co/jone1751/sonic-ease-360k/resolve/main/ease_B.npy \
    EASE_B_SHA256=2ab7e37dab346cd88b7c36a3b218756f9382de1bc28bbf138ea0eeb431709b37 \
    EASE_B_BYTES=538889924

# implicit's compiled extension dlopens the OpenMP runtime at IMPORT time. The
# wheel is manylinux so nothing is built, but libgomp.so.1 is not present in
# python-slim, and without it `import implicit` dies with
#   ImportError: libgomp.so.1: cannot open shared object file
# which kills the container before uvicorn ever binds. Runtime stage only: the
# builder does not need it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

COPY src ./src
COPY api ./api
COPY configs ./configs
COPY scripts ./scripts
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

# The committed 360K core (~7.7 MB). If the build context was checked out
# without Git-LFS these are ~130-byte pointer stubs -- fail the build loudly
# here rather than at runtime, since a silent pointer stub is exactly what has
# been quietly skipping the API tests in CI.
COPY data/processed/lastfm360k/matrix.npz ./data/processed/lastfm360k/
COPY data/processed/lastfm360k/item_ids.parquet ./data/processed/lastfm360k/
COPY data/processed/lastfm360k/user_ids.parquet ./data/processed/lastfm360k/
RUN if head -c 64 data/processed/lastfm360k/matrix.npz | grep -q 'git-lfs'; then \
        echo "ERROR: data/processed/lastfm360k/* are Git-LFS pointer stubs." >&2; \
        echo "       Run 'git lfs pull' before 'docker build'." >&2; \
        exit 1; \
    fi

# src.utils.get_logger() writes into outputs/logs/, and /artifacts is the volume
# mount point. Both must exist and be writable by the unprivileged user.
# Two directories are written at runtime: outputs/logs (src.utils.get_logger)
# and data/processed/lastfm360k (where scripts/fetch_ease_b.py lands the artifact
# when no cache dir is configured). Fly runs the container as root and the
# entrypoint drops to `app`; Hugging Face Spaces may run it as an arbitrary UID,
# so make those writable by any user rather than betting on one.
RUN useradd --create-home --uid 10001 app \
 && mkdir -p outputs/logs /artifacts \
 && chown -R app:app /app /artifacts \
 && chmod -R a+rwX /app/outputs /app/data /artifacts \
 && chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
