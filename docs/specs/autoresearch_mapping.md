# Autoresearch, mapped onto this project (and deliberately inverted)

This project's search harness is an adaptation of Andrej Karpathy's
[**autoresearch**](https://github.com/karpathy/autoresearch) — a minimal
three-file scaffold for letting an agent run its own ML experiments. We keep its
*structure* (which is excellent for auditability) and invert its *objective* and
*safety posture* (which suit an overnight GPT-search, not a portfolio recommender
on tiny, sparse data). Reading the upstream repo and being explicit about what we
changed — and why — is the point of this page.

## Upstream autoresearch in one paragraph

Three files. `prepare.py` is frozen: data prep, the dataloader, and the
evaluation metric (`val_bpb`, validation bits-per-byte — lower is better and
vocab-size-independent, so architectural changes compare fairly). `train.py` is
the agent's entire sandbox — "everything is fair game: architecture,
hyperparameters, optimizer, batch size." `program.md` is human-written guidance.
The loop: the agent edits `train.py`, trains for a fixed **5-minute** budget,
evaluates, **keeps the change if `val_bpb` improved and auto-commits**, and
**never stops** — running unattended to produce "a log of experiments and
(hopefully) a better model." Roughly 12 experiments/hour. The whole design rests
on "single file to modify, diffs reviewable, scope manageable."

## The mapping

| autoresearch | role | this project |
|---|---|---|
| `prepare.py` (frozen) | data + the ground-truth metric | **`src/harness/eval_core.py`** — per-user split + precision@k/recall@k/NDCG@k, toy-validated, frozen |
| `train.py` (agent rewrites freely) | the mutable surface | **`src/harness/search_config.py`** — *only* four bounded hyperparameters, not arbitrary code |
| `program.md` | loop protocol | **`src/harness/program.md`** — what to try, what counts as a win, stopping rule |
| `results.tsv` | experiment log | **`outputs/experiments/log.jsonl`** — one JSON row per config, written by the runner |
| the training loop | runs + keeps/commits | **`run_search.py`** (one config) + **`run_session.py`** (the pre-registered set) — evaluates, logs, **promotes nothing** |
| `val_bpb`, lower-better, vocab-independent | the fair metric | **NDCG@10**, bounded [0,1], on a **fixed** split so every config is compared on identical held-out users |

## The four deliberate inversions

1. **A validated config, not free-form code.** Upstream lets the agent rewrite an
   entire Python file. Here the editable surface is a dict of four numbers, each
   validated against documented `BOUNDS` before a run. All executable logic lives
   in the frozen layer. *Why:* arbitrary code is the right call when the goal is
   to discover novel architectures with a human reviewing diffs; for a bounded
   hyperparameter search it only adds risk and unreviewable surface area.

2. **A human gate, not auto-commit.** Upstream keeps any change that improves the
   metric and commits it forever. Here the runner ranks and logs; a human reads
   the leaderboard, records the choice in `decisions.md`, and only then reads the
   locked holdout once. *Why:* at 99.7% sparsity with ~1,883 scored users, a wide
   enough automated search **will** manufacture a lucky winner. Pre-registration +
   a held-out arbiter + a human gate is the honest counter.

3. **A bounded budget, not run-forever.** Upstream never stops. Our `program.md`
   defines a small pre-registered set and then stops and reports. *Why:* the
   multiplicity of an unbounded search is exactly the failure mode the sparsity
   makes dangerous (`docs/specs/sparsity_fragility_investigation.md`).

4. **A sealed holdout the loop cannot see.** Upstream has no notion of a final
   arbiter — `val_bpb` on the eval set is the whole story. We seal a per-user
   locked holdout in `make_split.py` (the only module that knows its path) and
   read it exactly once, by hand, after the choice is made. *Why:* it is the only
   defense against tuning to the validation split.

## What we kept verbatim, because it is genuinely good

- **The frozen/mutable separation.** The thing that defines a win cannot be
  edited by the thing trying to win. This is the core idea and we keep it strictly.
- **A fair, comparable metric on a fixed evaluation set.** `val_bpb`'s
  vocab-independence becomes our fixed per-user split: every config sees the same
  held-out users, so NDCG differences are about the model, not the split.
- **"Diffs reviewable, scope manageable."** A four-number config diff is about as
  reviewable as it gets.

## Lineage note

The same three-file discipline was used on a prior professional project (First
Payment Delinquency), which *also* inverted autoresearch — there to characterise
*stability* of a near-baseline model on tiny, imbalanced credit data. This
project shares the discipline but not that stability apparatus, because its
problem is sparsity, not imbalance (see the sparsity investigation). Citing both
adaptations is deliberate: the scaffold is reusable; what you optimise for, and
how much you defend against overfitting, must be re-derived from each dataset.
