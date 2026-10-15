#!/bin/bash
cd "$(dirname "$0")"
PY=$HOME/miniconda3/envs/sentiment/bin/python
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/sentiment/lib CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
s=$(date +%s)
$PY eval_models.py results/eval_b.json e5_short=runs/e5_short/final
printf "phase5eval\teval_b\tgpu2\t%s\t%s\n" $s $(date +%s) >> results/timing.tsv
echo "eval_b finished $(date)"
