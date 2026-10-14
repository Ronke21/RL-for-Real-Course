#!/bin/bash
# Run ON dsinlp01:  setsid nohup timeout -k 60 4h bash run_lane.sh GPU > logs/lane_gpuGPU.log 2>&1 &
# Sequential runner for one A100: takes the next unstarted line "name|train_grpo.py args" from
# queue/gpuGPU.txt, runs it, repeats. Waits for new lines until its 4 h budget ends.
# Authorised for RL-for-Real: all 8 A100 GPUs 0-7 (Ron, 2026-10-14 ~14:10). Do not raise unasked.
set -u
GPU=$1
R=/tmp/ronke21_rlfr
PY=$R/conda/envs/rlfr/bin/python
Q=$R/queue/gpu$GPU.txt
case "$GPU" in 0|1|2|3|4|5|6|7) ;; *) echo "GPU $GPU not authorised (0-7 only)"; exit 1;; esac
export CUDA_VISIBLE_DEVICES=$GPU HF_HOME=$R/hf WANDB_PROJECT=rl-for-real PYTHONUNBUFFERED=1 \
       OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
mkdir -p $R/runs $R/logs $R/state
export WANDB_DIR=$R/wandb_files; mkdir -p $WANDB_DIR
cd $R/rl-for-real
while true; do
  line=""
  if [ -f "$Q" ]; then
    while IFS= read -r l; do
      [ -z "$l" ] && continue; case "$l" in \#*) continue;; esac
      n=${l%%|*}
      [ -e $R/state/$n.started ] && continue
      line=$l; break
    done < "$Q"
  fi
  if [ -z "$line" ]; then sleep 60; continue; fi
  name=${line%%|*}; args=${line#*|}
  used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ -z "$used" ] || [ "$used" -ge 2000 ]; then echo "$(date +%T) GPU $GPU not empty (${used:-?} MiB), waiting"; sleep 60; continue; fi
  # Online only if this host has a wandb login (checked per run, so a later `wandb login` takes effect)
  if grep -qs api.wandb.ai ~/.netrc; then unset WANDB_MODE; else export WANDB_MODE=offline; fi
  touch $R/state/$name.started
  echo "=== $(date +%T) gpu$GPU START $name: $args"
  s=$(date +%s)
  timeout -k 60 90m $PY -u train_grpo.py --config config/multi_tasks.yaml $args \
      --output_dir $R/runs --run_name $name > $R/logs/$name.log 2>&1
  rc=$?
  e=$(date +%s)
  echo "=== $(date +%T) gpu$GPU END $name rc=$rc $(( (e-s)/60 )) min"
  printf "%s\tgpu%s\t%s\t%s\t%s\n" "$name" "$GPU" "$s" "$e" "$rc" >> $R/state/timing.tsv
  rm -f $R/runs/$name/model*.safetensors   # keep final_model/ only; drop the duplicate root copy
done
