"""
Shared utilities for the music recommender.

Kept deliberately minimal. Add functions here only when they are used in more
than one notebook or script. Premature abstraction is the enemy of a young
project — start with notebook-local code and promote to src/ once the shape is
clear.

The point of the path constants below: anchor everything to PROJECT_ROOT so code
works the same whether it's run from the repo root, from notebooks/, or via
`python -m`. Nothing should hardcode a relative data path.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

# -------------------------------------------------------------------------
# Project paths — single source of truth
# -------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
RUNS_DIR = OUTPUTS_DIR / "runs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"

# Search-visible split artifacts (written by make_split, read by the loop). These
# are NOT secret. The LOCKED holdout path is deliberately NOT here — it lives only
# in src/harness/make_split.py so loop code has no symbol that could load it.
HARNESS_DIR = PROCESSED_DIR / "harness"
TRAIN_PATH = HARNESS_DIR / "train.npz"
TEST_PATH = HARNESS_DIR / "test.npz"
USER_INDEX_PATH = PROCESSED_DIR / "user_index.parquet"
ITEM_INDEX_PATH = PROCESSED_DIR / "item_index.parquet"


# -------------------------------------------------------------------------
# Config loading
# -------------------------------------------------------------------------

def load_config(name: str = "data_config") -> dict:
    """
    Load a YAML config from the configs/ directory.

    Parameters
    ----------
    name : str
        The config filename without extension, e.g. "data_config".
    """
    path = CONFIG_DIR / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------------------------------------------------------
# Data integrity
# -------------------------------------------------------------------------

def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 of a file (chunked read). Used to version raw data drops."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# -------------------------------------------------------------------------
# Run directory — timestamped output folder per execution
# -------------------------------------------------------------------------

def make_run_dir(label: str | None = None) -> Path:
    """
    Create a timestamped run directory under outputs/runs/.

    Use this for any run that produces artifacts (metrics, figures, model files)
    so results are not overwritten between runs.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{label}" if label else stamp
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------

def get_logger(name: str = "musicrec", log_to_file: bool = True) -> logging.Logger:
    """Configure a logger writing to stdout and (optionally) outputs/logs/."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_to_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d")
        file_handler = logging.FileHandler(LOGS_DIR / f"{name}_{stamp}.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def raw_data_dir() -> Path:
    """
    Resolve the raw-data directory: LASTFM_RAW_DIR env override, else the
    `data.raw_dir` in data_config.yaml, resolved against PROJECT_ROOT.
    """
    override = os.environ.get("LASTFM_RAW_DIR")
    if override:
        p = Path(override)
        return p if p.is_absolute() else PROJECT_ROOT / p
    rel = load_config("data_config")["data"]["raw_dir"]
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p
