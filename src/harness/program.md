# Search Protocol (`program.md`)

> **You (Chris) own this file.** It's drafted here as a starting template
> modeled on the FPD `program.md`; edit it to say exactly what you want the
> search to do. The runner and the agent follow it; they do not invent moves it
> doesn't sanction.
>
> Like the FPD harness — and unlike upstream autoresearch — this protocol
> **proposes and reports; it never auto-promotes.** A human reads the log and
> decides.

## Frozen — never edited by the loop

The loop imports these and must not modify them (doing so is an integrity bug,
not a tuning choice):

- `src/harness/eval_core.py` — the per-user split and the precision@k /
  recall@k / NDCG@k definitions.
- The **locked holdout** (`data/processed/harness/_LOCKED_holdout/`). The loop
  has no symbol that references it. It is never read during search.
- `docs/specs/*` and `decisions.md` — human-owned.

## The editable surface

Exactly one file: `src/harness/search_config.py`. Four knobs, each bounded:
`factors`, `regularization`, `iterations`, `alpha`. Out-of-bounds configs are
rejected by the runner.

## Primary metric

**NDCG@10 on the per-user test holdout**, averaged over scored users. Report
precision@10 and recall@10 alongside it every time — a move that lifts NDCG but
craters recall is not obviously a win.

> Decide the `k` and the primary metric here and then keep them fixed for the
> whole session. Changing the metric mid-search is metric-shopping.

## What to try, in what order

Pre-register the moves before looking at results. Suggested opening sequence
(edit freely):

1. **Baseline** — the shipped `CONFIG` (`factors=64, reg=0.01, iters=15,
   alpha=40`). Establishes the number every later move is compared against.
2. **`alpha` sweep** — {1, 15, 40, 80}. Confidence scaling is usually the
   highest-leverage implicit-ALS knob; find its neighborhood first.
3. **`factors` sweep** at the best alpha — {32, 64, 128}. Watch for the point
   where more capacity stops helping the holdout (overfitting the sparse tail).
4. **`regularization`** at the best (alpha, factors) — {0.001, 0.01, 0.1}.
5. **`iterations`** — confirm the chosen config has converged (e.g. 15 vs 30).

Write each planned config down first, with a one-line reason. Adding configs
*after* seeing results is p-hacking — if you must, log it as a new
pre-registration round and treat the extra comparisons honestly.

> The concrete pre-registered set for the Milestone-4 session lives in
> `src/harness/run_session.py` (`PRE_REGISTERED`), each config tagged with its
> reason. `python -m src.harness.run_session` runs them all and prints a ranked
> leaderboard. The readout and the chosen config are in
> `notebooks/03_search_readout.ipynb` and `decisions.md`.

## How to log each attempt

`python -m src.harness.run_search` reads `search_config.CONFIG`, trains,
evaluates on the per-user test split, and **appends one row** to
`outputs/experiments/log.jsonl` (never overwrites). Each row carries the config,
the metrics, a `config_hash`, and a timestamp.

## What counts as an improvement

- A later config beats the incumbent only if it improves the **primary metric**
  without a meaningful regression in the others.
- **If the sparsity-fragility investigation concludes a stability layer is
  warranted** (see `docs/specs/sparsity_fragility_investigation.md`), "improves"
  means improves across the agreed seeds/folds, not on a single split. **If it
  concludes the single per-user split is stable enough**, a single-split
  comparison is sufficient. Resolve that question before the search session, not
  during it.

## Stopping criteria

- A fixed budget: stop after the pre-registered moves above (≈ 5–8 configs), or
- a wall-clock box, or
- two consecutive moves that don't beat the incumbent.

Then **stop and report.** Do not keep roaming the space — at this sparsity a
wide enough search will manufacture a lucky winner.

## Acceptance

`propose → evaluate → log → rank`. Nothing is automatic. The runner promotes
nothing. A human reads `log.jsonl`, picks a config with reasoning, records it in
`decisions.md`, and only then is the locked holdout read (once) to confirm.
