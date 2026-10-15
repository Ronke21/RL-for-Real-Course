#!/bin/bash
# Short recorded rollouts (16 eps, body trajectories) for the report GIFs. Two lanes: GPUs 1 and 2.
# Launch ON nlp10 from claude_run/: setsid nohup timeout -k 60 1h bash gifs.sh 1 > logs/gifs_1.log 2>&1 &   (and 2)
set -uo pipefail
trap 'kill 0' EXIT INT TERM
cd "/home/nlp/ronke21/RL for Real Course/robotics_rl_course_students/claude_run"
export STEP_GPU=$1 STEP_COURSE=work
source ./lib.sh
rec() { local name=$1 out=$2; shift 2; [ -s "$W/$out" ] || step "$name" 15 python -m course.physical_eval --num-envs 16 --episodes 16 --record-trajectory --output "$out" "$@"; }
if [ "$1" = 1 ]; then
  rec rec_m1_weak    results/rec/m1_weak.json    --checkpoint course/pretrained/candidates/early_candidate_epoch_70/last.ckpt --seed 0
  rec rec_m1_control results/rec/m1_control.json --checkpoint results/m1_control/last.ckpt --seed 0
  rec rec_m2_expert  results/rec/m2_expert.json  --checkpoint course/pretrained/m1/last.ckpt --seed 0
else
  rec rec_m2_bc      results/rec/m2_bc.json      --checkpoint course/pretrained/m1/last.ckpt --bc-model bc_policy.pt --seed 0
  rec rec_m3_before  results/rec/m3_before_perturbed.json --checkpoint results/m1_control/last.ckpt --seed 1 --perturbed
  rec rec_m3_after   results/rec/m3_after_perturbed.json  --checkpoint results/m3_g1/last.ckpt --seed 1 --perturbed
fi
say "gifs lane $1 ALL DONE"
