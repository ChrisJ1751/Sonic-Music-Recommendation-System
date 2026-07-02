"""search_config.py — THE AGENT-EDITABLE FILE (one of three).

This is the ONLY file the search loop is permitted to modify. It carries ALS
hyperparameters and nothing else: no data loading, no metric definitions, no
split logic. Those live in the FROZEN layer (`eval_core.py`) and must not be
touched.

Contract (enforced by `run_search.py`):
  - Every value must stay inside the documented bound below. The runner
    validates and refuses to run an out-of-bounds config — that keeps the
    search honest and the log comparable.
  - One CONFIG dict per attempt. `program.md` says what to try and in what
    order; this file holds the *current* attempt's values.

Why these four knobs (implicit ALS on listen-count feedback):
  - factors          latent dimensionality of user/item vectors. More factors =
                     more capacity (and more overfitting risk on sparse data).
  - regularization   L2 on the factors. The main guard against overfitting the
                     long tail of barely-listened artists.
  - iterations       ALS alternating-least-squares sweeps. More = better fit up
                     to a plateau; cheap to raise.
  - alpha            implicit-feedback confidence scaling. Confidence on a
                     (user, artist) cell is c = 1 + alpha * log(1 + count). Counts
                     are log-scaled first because Last.fm play counts span ~6
                     orders of magnitude (decisions.md 2026-06-30; ablation in
                     notebooks/02). Higher alpha trusts heavy listening more.

See `program.md` for the protocol and `docs/specs/eval_design.md` for how these
are scored.
"""

# --- the current attempt -----------------------------------------------------
CONFIG = {
    "factors": 32,
    "regularization": 0.01,
    "iterations": 15,
    "alpha": 1.0,
    # Bookkeeping (the runner copies these into the log row):
    "tag": "chosen",
    "note": "promoted winner of the Milestone-4 session (decisions.md 2026-06-30): "
            "top of the low-capacity plateau with the tightest seed band",
}

# --- allowed ranges (the runner validates CONFIG against these) --------------
# Edit CONFIG within these bounds. To change a BOUND is a protocol change, not a
# tuning move — that requires a decisions.md entry, not just an edit here.
BOUNDS = {
    "factors": (16, 256),           # int
    "regularization": (1e-4, 1.0),  # float, log-scale in practice
    "iterations": (5, 50),          # int
    "alpha": (1.0, 100.0),          # float
}
