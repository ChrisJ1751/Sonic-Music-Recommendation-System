#!/usr/bin/env python
"""Fetch the pre-fitted EASE weight matrix before the API starts.

Why this exists
---------------
`src/serving.py` will happily fit EASE itself if the cache is missing -- but on
this catalogue that means inverting an 11,607 x 11,607 Gram matrix: ~1.04e12
FLOPs (measured at ~85-100 s single-threaded, with BLAS pinned to 1 thread) and
a ~2.5-3 GB peak. That does not fit a 2 GB container. So the 514 MiB artifact is
shipped, not recomputed.

It is fetched at boot rather than baked into the image so the image stays ~350 MB
and the model can be revised without rebuilding the service.

Design notes
------------
* Stdlib only (`urllib`) -- no new runtime dependency, per project constraint.
* The destination is imported from `src.serving`, never re-hardcoded, so it can
  never drift from the path the app actually reads (AGENTS.md: paths go through
  `src/utils.py`). That costs ~1 s of import at boot; correctness is worth it.
* No-op when EASE_B_URL is unset, so local development keeps today's
  fit-on-demand behaviour untouched.
* Size and SHA-256 are verified before the file is moved into place, so a
  truncated or CDN-corrupted download fails loudly at boot instead of silently
  serving a subtly wrong model.

Environment
-----------
EASE_B_URL        required to do anything; unset => no-op
EASE_B_SHA256     optional, hex digest; verified when present
EASE_B_BYTES      optional, expected size in bytes; verified when present
EASE_B_CACHE_DIR  optional; download here and symlink into place. Used on Fly so
                  the artifact lives on the persistent volume and survives
                  restarts -- mounting the volume over the data directory itself
                  would shadow the small artifacts baked into the image.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Same bootstrap as app/_shared.py: make the repo root importable so this runs
# standalone (`python scripts/fetch_ease_b.py`) without PYTHONPATH gymnastics.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.serving import EASE_B_PATH  # noqa: E402

CHUNK = 1 << 20
ATTEMPTS = 3


def log(msg: str, **fields: object) -> None:
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    print(f"fetch_ease_b | {msg}{' | ' + extra if extra else ''}", flush=True)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def is_valid(path: Path, expect_sha: str | None, expect_bytes: int | None) -> bool:
    """Is an existing file already the artifact we want?"""
    if not path.exists():
        return False
    if expect_bytes is not None and path.stat().st_size != expect_bytes:
        log("cached file has wrong size, refetching",
            have=path.stat().st_size, want=expect_bytes)
        return False
    if expect_sha is not None:
        actual = digest(path)
        if actual.lower() != expect_sha.lower():
            log("cached file has wrong digest, refetching", have=actual[:16])
            return False
    return True


def download(url: str, target: Path) -> None:
    """Stream to a .part file, then rename. Retries transient network errors."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")

    last: Exception | None = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            started = time.perf_counter()
            request = urllib.request.Request(url, headers={"User-Agent": "sonic-api"})
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                total = int(response.headers.get("Content-Length") or 0)
                written = 0
                with part.open("wb") as fh:
                    while chunk := response.read(CHUNK):
                        fh.write(chunk)
                        written += len(chunk)
            elapsed = time.perf_counter() - started
            log("downloaded", bytes=written, expected=total or "unknown",
                seconds=round(elapsed, 1), mib_s=round(written / 2**20 / max(elapsed, 1e-9), 1))
            if total and written != total:
                raise OSError(f"truncated download: got {written} of {total} bytes")
            part.replace(target)
            return
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = exc
            part.unlink(missing_ok=True)
            log("attempt failed", attempt=attempt, error=repr(exc))
            if attempt < ATTEMPTS:
                time.sleep(2 ** attempt)
    raise SystemExit(f"fetch_ease_b: giving up after {ATTEMPTS} attempts: {last}")


def main() -> int:
    url = os.environ.get("EASE_B_URL", "").strip()
    if not url:
        log("EASE_B_URL unset -- skipping; serving.py will fit EASE on demand")
        return 0

    expect_sha = os.environ.get("EASE_B_SHA256", "").strip() or None
    raw_bytes = os.environ.get("EASE_B_BYTES", "").strip()
    expect_bytes = int(raw_bytes) if raw_bytes else None

    dest = Path(EASE_B_PATH)
    cache_dir = os.environ.get("EASE_B_CACHE_DIR", "").strip()
    target = Path(cache_dir) / dest.name if cache_dir else dest

    if is_valid(target, expect_sha, expect_bytes):
        log("artifact already present", path=str(target), size=target.stat().st_size)
    else:
        log("fetching artifact", url=url, target=str(target))
        download(url, target)
        if not is_valid(target, expect_sha, expect_bytes):
            target.unlink(missing_ok=True)
            raise SystemExit("fetch_ease_b: downloaded artifact failed verification")
        log("verified", path=str(target), size=target.stat().st_size)

    # Link the cached copy into the path serving.py reads.
    if target != dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        try:
            dest.symlink_to(target)
            log("linked", link=str(dest), to=str(target))
        except OSError:
            # Some filesystems disallow symlinks; a hard link or copy still works.
            os.link(target, dest)
            log("hard-linked", link=str(dest), to=str(target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
