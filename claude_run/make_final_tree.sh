#!/bin/bash
# Run ON dsinlp01. Builds /tmp/ronke21_rlfr/rl-for-real-final: a copy of the experiment code with the two
# Track-1 final-evaluation changes required by the course advisor (2026-10-14):
#   1. eval seed 62, changed in code: grpo/dataset.py  eval_data = reasoning_gym.create_dataset("composite", seed=62, ...)
#   2. eval_num_samples: 200 in config/multi_tasks.yaml itself
# Since 13:10 (Ron) the main code itself uses eval seed 62 + 200 samples in the YAML, so this copy is identical; kept because run_lane_final.sh runs from it.
set -euo pipefail
R=/tmp/ronke21_rlfr
rsync -a --delete --exclude claude_run "$R/rl-for-real/" "$R/rl-for-real-final/"
F=$R/rl-for-real-final
true  # seed already 62 in the main code
true  # already 200 in the main YAML
grep -q 'seed=62, size=config.eval_num_samples' $F/grpo/dataset.py
grep -q '^eval_num_samples: 200$' $F/config/multi_tasks.yaml
diff -r --exclude claude_run "$R/rl-for-real" "$F" || true
echo FINAL_TREE_OK
