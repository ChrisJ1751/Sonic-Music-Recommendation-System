"""Tests for the Streamlit app package.

app/ had no test coverage at all despite being the primary deployed surface.
These do not need a Streamlit runtime: they cover the pure helpers plus a set of
source-level guards that pin bugs which actually reached production.
"""
from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from app import _shared

APP_DIR = pathlib.Path(_shared.__file__).parent
VIEW_FILES = sorted((APP_DIR / "views").glob("*.py"))


# --- pure helpers --------------------------------------------------------

def test_gini_is_zero_for_a_perfectly_even_distribution():
    assert _shared._gini(np.array([5.0, 5.0, 5.0, 5.0])) == pytest.approx(0.0)


def test_gini_by_hand():
    # sorted [0,0,0,4]: sum((2i - n - 1) * x) = (2*4-4-1)*4 = 12; n*sum = 16
    assert _shared._gini(np.array([0.0, 0.0, 0.0, 4.0])) == pytest.approx(0.75)


def test_gini_handles_degenerate_input():
    assert _shared._gini(np.array([])) == 0.0
    assert _shared._gini(np.array([0.0, 0.0])) == 0.0


def test_gini_increases_with_concentration():
    even = _shared._gini(np.array([10.0, 10.0, 10.0, 10.0]))
    skewed = _shared._gini(np.array([1.0, 1.0, 1.0, 97.0]))
    assert even < skewed < 1.0


def test_gini_agrees_with_the_src_implementation():
    """app/_shared._gini duplicates the formula in src.metrics.gini (different
    call shapes, same maths). Pin that they agree, so the copy cannot drift."""
    from src import metrics

    counts = np.array([0, 3, 1, 0, 7, 2], dtype=float)
    recs = np.concatenate([np.full(int(c), i) for i, c in enumerate(counts)]).reshape(1, -1)
    assert _shared._gini(counts) == pytest.approx(metrics.gini(recs, n_items=len(counts)))


# --- regression guards for bugs that shipped -----------------------------

def test_no_localhost_defaults_in_deployed_code():
    """REPORT_URL defaulted to http://localhost:8080 and was never set on the
    Space, so every visitor to the live app got a dead 'open the report' link.
    A localhost default is a loaded gun in anything that gets deployed."""
    offenders = []
    for path in [*VIEW_FILES, APP_DIR / "_shared.py", APP_DIR / "streamlit_app.py"]:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "os.environ.get" in line and "localhost" in line:
                offenders.append(f"{path.name}:{i}")
    assert not offenders, f"localhost used as a deployed default: {offenders}"


def test_no_deprecated_use_container_width():
    """Streamlit deprecated use_container_width with removal 'after 2025-12-31'
    -- already past. requirements.txt pins streamlit with no upper bound, so the
    release that removes it would break the live app. Use width= instead."""
    offenders = [p.name for p in [*VIEW_FILES, APP_DIR / "_shared.py"]
                 if "use_container_width" in p.read_text(encoding="utf-8")]
    assert not offenders, f"deprecated use_container_width in: {offenders}"


def test_blas_is_pinned_before_numpy_is_imported():
    """src/serving.py pins BLAS threads, but Streamlit has already imported numpy
    by then, so the setdefault is a no-op and OpenBLAS starts a 12-thread pool.
    The entry point has to do it first for the determinism claim to hold.

    Parsed with ast rather than matched with a regex: `from __future__` and the
    `import os` the pin itself needs are legitimately allowed to come first, and
    distinguishing those from a library import is exactly what a parser is for.
    """
    source = (APP_DIR / "streamlit_app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    pin_line = min(
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value == "OPENBLAS_NUM_THREADS"
    )

    exempt = {"os", "__future__"}
    library_imports = [
        (node.lineno, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in ([ast.alias(name=node.module or "")] if isinstance(node, ast.ImportFrom)
                      else node.names)
        if (alias.name or "").split(".")[0] not in exempt
    ]
    assert library_imports, "expected the entry point to import a library at all"
    too_early = [(line, name) for line, name in library_imports if line < pin_line]
    assert not too_early, f"imported before BLAS was pinned: {too_early}"


def test_views_do_not_reimplement_recommendation_logic():
    """The whole point of src/serving.py is one shared inference core. A view
    reaching for the model internals means the logic has started to fork."""
    banned = ("fit_ease", "np.linalg.inv", "argpartition")
    offenders = [(p.name, tok) for p in VIEW_FILES
                 for tok in banned if tok in p.read_text(encoding="utf-8")]
    assert not offenders, f"views reaching past serving.py: {offenders}"


# --- artifact fetch ------------------------------------------------------

def test_ensure_artifact_is_a_noop_when_already_present(tmp_path, monkeypatch):
    """Must not re-download 514 MiB on every process start."""
    calls = []
    artifact = tmp_path / "ease_B.npy"
    artifact.write_bytes(b"x")

    import scripts.fetch_ease_b as fetch

    monkeypatch.setattr("src.serving.EASE_B_PATH", artifact)
    monkeypatch.setattr(fetch, "main", lambda: calls.append(1))
    _shared._ensure_ease_artifact()
    assert calls == []


def test_ensure_artifact_never_raises(tmp_path, monkeypatch):
    """A demo that boots slowly beats a demo that does not boot: a Hub outage
    must fall through to serving.py's refit, not crash the app."""
    import scripts.fetch_ease_b as fetch

    monkeypatch.setattr("src.serving.EASE_B_PATH", tmp_path / "missing.npy")
    monkeypatch.setattr(fetch, "main", lambda: (_ for _ in ()).throw(OSError("hub down")))
    _shared._ensure_ease_artifact()      # must not raise


def test_get_state_surfaces_a_human_error_instead_of_a_traceback(monkeypatch):
    """Every page calls get_state() first, so an unhandled exception there shows
    a raw Python traceback to whoever opened the live demo."""
    shown: dict[str, str] = {}

    def boom():
        raise RuntimeError("matrix.npz is missing")

    monkeypatch.setattr(_shared, "_load_state", boom)
    monkeypatch.setattr(_shared.st, "error", lambda msg, **_: shown.setdefault("error", msg))
    monkeypatch.setattr(_shared.st, "caption", lambda msg, **_: shown.setdefault("caption", msg))
    monkeypatch.setattr(_shared.st, "stop", lambda: (_ for _ in ()).throw(SystemExit))

    with pytest.raises(SystemExit):
        _shared.get_state()

    assert "could not be loaded" in shown["error"]
    assert "RuntimeError" in shown["caption"], "the underlying cause must still be visible"


# --- colour contrast -----------------------------------------------------
#
# The palette is a handful of string constants, so contrast regresses silently
# the moment someone nudges one. WCAG 2.1: text needs 4.5:1 (1.4.3), non-text
# graphics that carry meaning need 3:1 (1.4.11), both against the background.

BACKGROUND = "#0c0d11"      # matches .streamlit/config.toml backgroundColor


def _relative_luminance(hex_colour: str) -> float:
    channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(foreground: str, background: str = BACKGROUND) -> float:
    a, b = _relative_luminance(foreground), _relative_luminance(background)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def test_background_token_matches_the_streamlit_theme():
    """These ratios are meaningless if the assumed background is not the real one."""
    config = (pathlib.Path(_shared.__file__).parents[1] / ".streamlit/config.toml").read_text(encoding="utf-8")
    assert BACKGROUND in config


@pytest.mark.parametrize("token", ["TEXT", "LABEL"])
def test_text_tokens_meet_wcag_aa(token):
    ratio = _contrast(getattr(_shared, token))
    assert ratio >= 4.5, f"{token} is {ratio:.2f}:1 against the background; AA text needs 4.5:1"


@pytest.mark.parametrize("token", ["GREEN", "PURPLE", "BLUE", "AMBER", "FAINT"])
def test_graphic_tokens_meet_wcag_non_text_contrast(token):
    """These colour bars, markers and lines that carry meaning. FAINT was
    #3a4150 at 1.90:1, the non-served models' bars were nearly invisible."""
    ratio = _contrast(getattr(_shared, token))
    assert ratio >= 3.0, f"{token} is {ratio:.2f}:1; meaningful non-text graphics need 3:1"


def test_muted_is_not_used_for_text_in_the_app():
    """MUTED is 4.23:1, below the 4.5:1 text bar. It is fine for a reference
    line and was briefly used for a footer, which is the mistake this pins."""
    assert _contrast(_shared.MUTED) < 4.5, "MUTED now passes AA; promote it and delete this guard"
    offenders = []
    for path in [*VIEW_FILES, APP_DIR / "_shared.py"]:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "MUTED" in line and ("textfont" in line or "st.markdown" in line or "color:{MUTED}" in line):
                offenders.append(f"{path.name}:{i}")
    assert not offenders, f"MUTED used as text: {offenders}"
