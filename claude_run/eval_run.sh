#!/bin/bash
# post-hoc scoring of finished models on GPU 2 (never GPU 0)
cd "$(dirname "$0")"
PY=$HOME/miniconda3/envs/sentiment/bin/python
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/sentiment/lib CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
s=$(date +%s)
$PY eval_models.py results/eval_a.json base_gpt2=gpt2 e3_no_kl=runs/e3_expectation/final e4_fwd_c0.1_s400=runs/e4_fwd_c0.1_s400/final e5_power=runs/e5_power/final e5_norep=runs/e5_norep/final e5_whiplash=runs/e5_whiplash/final
printf "phase5eval\teval_a\tgpu2\t%s\t%s\n" $s $(date +%s) >> results/timing.tsv
echo "eval_a finished $(date)"
