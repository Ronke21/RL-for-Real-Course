#!/bin/bash
# Exercise 6 on GPU 1 (never GPU 0): prompt engineering vs RL, 2 independent repeats per metric
cd "$(dirname "$0")/work"
PY=$HOME/miniconda3/envs/sentiment/bin/python
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/sentiment/lib CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
s=$(date +%s)
for set in norep whiplash; do
  for rep in 1 2; do
    SHAPE_MODE=$set PE_SET=$set $PY prompt_engineering.py --use_shaped_reward --trained_model ../runs/e5_$set/final --num_prompts 64 > ../logs/e6_${set}_rep$rep.log 2>&1
  done
done
printf "phase6\te6_prompt_engineering\tgpu1\t%s\t%s\n" $s $(date +%s) >> ../results/timing.tsv
echo "pe finished $(date)"
