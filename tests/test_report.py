"""Guards for the Quarto report and the docs that quote its numbers.

report/_quarto.yml sets `execute: enabled: false`, so every figure in the report
is authored prose rather than computed output. That is a deliberate choice — the
render is fast and deterministic — but it means the numbers can silently drift
away from the model they describe. These tests are the tripwire for that.
"""
from __future__ import annotations

import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
QMD = sorted(REPORT.glob("*.qmd"))


@pytest.fixture(scope="module")
def report_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in QMD)


def _mentioned(value: float, text: str) -> bool:
    """Numbers appear variously as 0.219 / 0.22 / .219 in prose."""
    return any(fmt in text for fmt in (f"{value}", f"{value:.3f}", f"{value:.2f}"))


def test_report_files_exist():
    assert QMD, "no .qmd files found — did the report move?"


def test_headline_metrics_match_the_served_model(report_text):
    """The report quotes the same numbers the API serves at /about."""
    from src import serving

    missing = [f"{m['label']}={m['value']}" for m in serving.about_payload()["headline"]
               if not _mentioned(m["value"], report_text)]
    assert not missing, f"headline metrics absent from the report: {missing}"


def test_leaderboard_matches_the_served_model(report_text):
    from src import serving

    missing = [f"{r['model']}={r['ndcg10']}" for r in serving.about_payload()["leaderboard"]
               if not _mentioned(r["ndcg10"], report_text)]
    assert not missing, f"leaderboard values absent from the report: {missing}"


def test_beyond_accuracy_matches_the_served_model(report_text):
    from src import serving

    missing = []
    for row in serving.about_payload()["beyond_accuracy"]:
        for key in ("coverage", "novelty"):
            if not _mentioned(row[key], report_text):
                missing.append(f"{row['model']}.{key}={row[key]}")
    assert not missing, f"beyond-accuracy values absent from the report: {missing}"


def test_cutoff_curve_matches_the_served_model(report_text):
    from src import serving

    curve = serving.about_payload()["cutoff_curve"]
    missing = [f"{name}@{k}={v}"
               for name in ("ndcg", "recall")
               for k, v in zip(curve["k"], curve[name], strict=True)
               if not _mentioned(v, report_text)]
    assert not missing, f"cutoff-curve values absent from the report: {missing}"


def test_no_relative_file_links_escape_the_rendered_site():
    """Only report/_site is published to Pages, so a `../docs/...` link 404s for
    every reader. This shipped: limitations.qmd linked ../docs/specs/model_card.md."""
    offenders = []
    for path in QMD:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for href in re.findall(r"\]\(([^)]+)\)", line):
                if href.startswith("../") or (href.endswith(".md") and "://" not in href):
                    offenders.append(f"{path.name}:{i} -> {href}")
    assert not offenders, f"links that escape the published site: {offenders}"


def test_docs_agree_on_the_test_count():
    """The suite size is quoted in several files by hand and has drifted before
    (README claimed 45 and 53 simultaneously). Pin that they all agree, and that
    the number is not below the count of test functions actually defined."""
    # docs/phase_plan.md is excluded on purpose: it is an append-only milestone
    # log, so "12 tests pass. Done 2026-06-30." is accurate history, not drift.
    quoting = {}
    for path in [ROOT / "README.md", REPORT / "index.qmd", REPORT / "limitations.qmd"]:
        found = {int(n) for n in re.findall(r"(\d+) tests\b", path.read_text(encoding="utf-8"))}
        if found:
            quoting[path.name] = found

    claimed = set().union(*quoting.values())
    assert len(claimed) == 1, f"docs disagree on the test count: {quoting}"

    defined = sum(
        1
        for f in (ROOT / "tests").glob("test_*.py")
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    )
    stated = claimed.pop()
    assert stated >= defined, (
        f"docs claim {stated} tests but {defined} test functions are defined "
        "(parametrised cases only push the real count higher)"
    )
