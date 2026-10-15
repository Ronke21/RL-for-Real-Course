#!/bin/bash
# Phase 1 (Ex 2, 3, 4): sequential, ONE process, GPU 1 only (GPU 0 is Ron's).
cd "/home/nlp/ronke21/RL for Real Course/rl-sentiment-exercise/claude_run/work"
PY=$HOME/miniconda3/envs/sentiment/bin/python
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/sentiment/lib CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTHONUNBUFFERED=1
run() {  # name, args...
  name=$1; shift
  [ -f ../runs/$name/final/config.json ] && { echo "skip $name"; return; }
  echo "=== $name: $* ($(date))"
  $PY train.py "$@" --no_wandb --output_dir ../runs/$name > ../logs/$name.log 2>&1 || echo "FAILED $name"
  rm -rf ../runs/$name/checkpoint-*
}
run e2_five_stars  --reward_shaping five_stars
run e3_expectation --reward_shaping expectation
run e4_fwd_c5      --kl_type forward  --kl_coef 5
run e4_bwd_c5      --kl_type backward --kl_coef 5
for c in 1 0.3 0.1; do
  run e4_fwd_c$c --kl_type forward  --kl_coef $c
  run e4_bwd_c$c --kl_type backward --kl_coef $c
done
echo "phase1 finished $(date)"
