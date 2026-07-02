"""The three-file disciplined-search harness.

  eval_core.py     FROZEN  — per-user split + precision@k/recall@k/NDCG@k
  search_config.py EDITABLE — ALS hyperparameters (the only file the loop edits)
  program.md       PROTOCOL — what to try, how to log, what counts as a win
  run_search.py    RUNNER   — imports the above, evaluates, logs; promotes nothing
"""
