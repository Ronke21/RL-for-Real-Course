#!/bin/bash
# usage: lane.sh GPU PHASE "name|train.py args" ...   (one process at a time on the given GPU, never GPU 0)
GPU=$1; PHASE=$2; shift 2
[ "$GPU" = 0 ] && { echo "GPU 0 is reserved"; exit 1; }
D="/home/nlp/ronke21/RL for Real Course/rl-sentiment-exercise/claude_run"
cd "$D/work"
PY=$HOME/miniconda3/envs/sentiment/bin/python
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/sentiment/lib CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTHONUNBUFFERED=1
for item in "$@"; do
  name=${item%%|*}; args=${item#*|}
  [ -f ../runs/$name/final/config.json ] && { echo "skip $name"; continue; }
  s=$(date +%s)
  echo "=== gpu$GPU $name: $args ($(date))"
  $PY train.py $args --no_wandb --output_dir ../runs/$name > ../logs/$name.log 2>&1 || echo "FAILED $name"
  rm -rf ../runs/$name/checkpoint-*
  e=$(date +%s); printf "%s\t%s\t%s\t%s\t%s\n" "$PHASE" "$name" "gpu$GPU" "$s" "$e" >> ../results/timing.tsv
done
echo "lane gpu$GPU $PHASE finished $(date)"
