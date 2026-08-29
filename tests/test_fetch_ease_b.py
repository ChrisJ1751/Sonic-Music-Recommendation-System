"""Tests for the boot-time artifact fetch (scripts/fetch_ease_b.py).

This runs before uvicorn in the container, so a silent failure here means the
service either serves a corrupt model or falls back to a ~90 s, ~3 GB refit that
OOMs a 2 GB machine. Worth testing properly.

Uses file:// URLs so nothing touches the network.
"""
from __future__ import annotations

import hashlib

import pytest

from scripts import fetch_ease_b

PAYLOAD = b"not-really-a-numpy-array" * 100
SHA = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture()
def source(tmp_path):
    src = tmp_path / "source" / "ease_B.npy"
    src.parent.mkdir(parents=True)
    src.write_bytes(PAYLOAD)
    return src


@pytest.fixture()
def dest(tmp_path, monkeypatch):
    target = tmp_path / "data" / "processed" / "lastfm360k" / "ease_B.npy"
    monkeypatch.setattr(fetch_ease_b, "EASE_B_PATH", target)
    return target


def _env(monkeypatch, **kwargs):
    for key in ("EASE_B_URL", "EASE_B_SHA256", "EASE_B_BYTES", "EASE_B_CACHE_DIR"):
        monkeypatch.delenv(key, raising=False)
    for key, value in kwargs.items():
        monkeypatch.setenv(key, str(value))


def test_noop_when_url_unset(dest, monkeypatch):
    _env(monkeypatch)
    assert fetch_ease_b.main() == 0
    assert not dest.exists(), "must not create anything when unconfigured"


def test_downloads_and_verifies(source, dest, monkeypatch):
    _env(monkeypatch, EASE_B_URL=source.as_uri(), EASE_B_SHA256=SHA,
         EASE_B_BYTES=len(PAYLOAD))
    assert fetch_ease_b.main() == 0
    assert dest.read_bytes() == PAYLOAD


def test_rejects_wrong_digest_and_leaves_nothing_behind(source, dest, monkeypatch):
    _env(monkeypatch, EASE_B_URL=source.as_uri(), EASE_B_SHA256="00" * 32)
    with pytest.raises(SystemExit, match="failed verification"):
        fetch_ease_b.main()
    assert not dest.exists(), "a corrupt artifact must never be left in place"


def test_rejects_wrong_size(source, dest, monkeypatch):
    _env(monkeypatch, EASE_B_URL=source.as_uri(), EASE_B_BYTES=len(PAYLOAD) + 1)
    with pytest.raises(SystemExit, match="failed verification"):
        fetch_ease_b.main()
    assert not dest.exists()


def test_skips_redownload_when_already_valid(source, dest, monkeypatch):
    dest.parent.mkdir(parents=True)
    dest.write_bytes(PAYLOAD)
    # Point at a URL that would fail if it were actually fetched.
    _env(monkeypatch, EASE_B_URL="file:///nonexistent/ease_B.npy", EASE_B_SHA256=SHA)
    assert fetch_ease_b.main() == 0
    assert dest.read_bytes() == PAYLOAD


def test_refetches_when_cached_file_is_wrong(source, dest, monkeypatch):
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"stale-truncated-junk")
    _env(monkeypatch, EASE_B_URL=source.as_uri(), EASE_B_SHA256=SHA)
    assert fetch_ease_b.main() == 0
    assert dest.read_bytes() == PAYLOAD


def test_cache_dir_is_linked_into_place(source, dest, tmp_path, monkeypatch):
    """Fly mounts a volume at /artifacts; the real file lives there and is linked
    into the data dir, so the volume never shadows the image's baked-in files."""
    cache = tmp_path / "artifacts"
    cache.mkdir()
    _env(monkeypatch, EASE_B_URL=source.as_uri(), EASE_B_SHA256=SHA,
         EASE_B_CACHE_DIR=str(cache))
    assert fetch_ease_b.main() == 0
    assert (cache / "ease_B.npy").read_bytes() == PAYLOAD
    assert dest.read_bytes() == PAYLOAD
    # second run is a no-op that still repairs the link
    assert fetch_ease_b.main() == 0
    assert dest.read_bytes() == PAYLOAD
