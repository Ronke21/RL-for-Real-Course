#!/bin/bash
# Run ON dsinlp01:  setsid nohup timeout -k 60 4h bash run_lane_final.sh GPU LAST_EXP_RUN > logs/final_lane_gpuGPU.log 2>&1 &
# FINAL-EVALUATION lane: waits until the experiment lane on this GPU has finished LAST_EXP_RUN, stops that lane,
# then runs queue/final_gpuGPU.txt from rl-for-real-final (eval seed 62 in code, eval_num_samples 200 in the YAML).
# Sequential runner for one A100: takes the next unstarted line "name|train_grpo.py args" from
# queue/gpuGPU.txt, runs it, repeats. Waits for new lines until its 4 h budget ends.
# Authorised for RL-for-Real: A100 GPUs 0-5 only (Ron, 2026-10-14; 6-7 were lent for one baseline re-check, now returned). Do not raise unasked.
set -u
GPU=$1
R=/tmp/ronke21_rlfr
PY=$R/conda/envs/rlfr/bin/python
Q=$R/queue/final_gpu$GPU.txt
LAST=$2
case "$GPU" in 0|1|2|3|4|5) ;; *) echo "GPU $GPU not authorised (0-5 only)"; exit 1;; esac
export CUDA_VISIBLE_DEVICES=$GPU HF_HOME=$R/hf WANDB_PROJECT=rl-for-real PYTHONUNBUFFERED=1 \
       OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
mkdir -p $R/runs $R/logs $R/state
export WANDB_DIR=$R/wandb_files; mkdir -p $WANDB_DIR
cd $R/rl-for-real-final
until grep -q "END $LAST " $R/logs/lane_gpu$GPU.log; do sleep 60; done
pkill -u ronke21 -f "^bash $R/rl-for-real/claude_run/run_lane.sh $GPU\$" || true
echo "$(date +%T) experiment lane $GPU stopped after $LAST; final lane takes over"
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
  timeout -k 60 100m $PY -u train_grpo.py --config config/multi_tasks.yaml $args \
      --output_dir $R/runs --run_name $name > $R/logs/$name.log 2>&1
  rc=$?
  e=$(date +%s)
  echo "=== $(date +%T) gpu$GPU END $name rc=$rc $(( (e-s)/60 )) min"
  printf "%s\tgpu%s\t%s\t%s\t%s\n" "$name" "$GPU" "$s" "$e" "$rc" >> $R/state/timing.tsv
  rm -f $R/runs/$name/model*.safetensors   # keep final_model/ only; drop the duplicate root copy
done
