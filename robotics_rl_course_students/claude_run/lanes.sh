#!/bin/bash
# Three parallel lanes of Claude's M0-M3 run on nlp10 (GPUs 0-2; GPU 3 is Ron's). One GPU job per lane at a time.
# Launch ON nlp10 from claude_run/:  setsid nohup timeout -k 120 6h bash lanes.sh A > logs/lane_A.log 2>&1 < /dev/null &   (same for B, C)
set -uo pipefail
trap 'kill 0' EXIT INT TERM
LANE=$1
cd "/home/nlp/ronke21/RL for Real Course/robotics_rl_course_students/claude_run"
mkdir -p logs
case $LANE in A) export STEP_GPU=2;; B) export STEP_GPU=1;; C) export STEP_GPU=0;; *) exit 2;; esac
export STEP_COURSE=work
source ./lib.sh

lane_A() {  # M1 core, then the clean equal-budget M3 control and the M3 "before" evaluations
  evalp m1_baseline results/m1_baseline.json --checkpoint course/pretrained/candidates/early_candidate_epoch_70/last.ckpt
  train m1_control 30 1
  evalp m1_control_eval results/m1_control.json --checkpoint results/m1_control/last.ckpt
  local START; START=$(m3_start); echo "$START" > "$W/results/m3_start_checkpoint.txt"; say "M3 start: $START"
  train m3_clean_control 15 3 --training-max-steps 10000000 --checkpoint "$START" --time-limit-minutes 15
  evalp m3_cleanctl_clean results/m3_cleanctl_clean.json --checkpoint results/m3_clean_control/last.ckpt --seed 1
  evalp m3_cleanctl_perturbed results/m3_cleanctl_perturbed.json --checkpoint results/m3_clean_control/last.ckpt --seed 1 --perturbed
  evalp m3_before_clean results/m3_before_clean.json --checkpoint "$START" --seed 1
  evalp m3_before_perturbed results/m3_before_perturbed.json --checkpoint "$START" --seed 1 --perturbed
}

lane_B() {  # M2, then the optional M1 action-rate trade-off (x1 control vs x10 ablation)
  [ -s "$W/bc_data.pt" ] || step m2_collect 20 python -m course.module_2_behavior_cloning.collect_data --experiment-name m1_frozen \
    --checkpoint course/pretrained/m1/last.ckpt --simulator isaaclab --num-envs 16 --num-steps 1000 --output bc_data.pt
  [ -s "$W/bc_policy.pt" ] || step m2_train_bc 30 python -m course.module_2_behavior_cloning.train_bc --data bc_data.pt --epochs 30 --output bc_policy.pt
  grep -q "=== END m2_test_bc rc=0" logs/m2_test_bc.log 2>/dev/null || step m2_test_bc 20 python -m course.module_2_behavior_cloning.test_bc \
    --simulator isaaclab --bc-model bc_policy.pt --checkpoint course/pretrained/m1/last.ckpt --num-envs 16 --num-steps 3000 --headless
  evalp m2_expert_clean results/m2_expert_clean.json --checkpoint course/pretrained/m1/last.ckpt --seed 0
  evalp m2_bc_clean results/m2_bc_clean.json --checkpoint course/pretrained/m1/last.ckpt --bc-model bc_policy.pt --seed 0
  evalp m2_expert_perturbed results/m2_expert_perturbed.json --checkpoint course/pretrained/m1/last.ckpt --seed 0 --perturbed
  evalp m2_bc_perturbed results/m2_bc_perturbed.json --checkpoint course/pretrained/m1/last.ckpt --bc-model bc_policy.pt --seed 0 --perturbed
  train m1_reward_control 10 1 --checkpoint course/pretrained/m1/last.ckpt --training-max-steps 2000000 --time-limit-minutes 10
  evalp m1_reward_control_eval results/m1_reward_control.json --checkpoint results/m1_reward_control/last.ckpt
  export CLAUDE_ACTION_RATE_SCALE=10
  train m1_ablation 10 1 --checkpoint course/pretrained/m1/last.ckpt --training-max-steps 2000000 --time-limit-minutes 10
  unset CLAUDE_ACTION_RATE_SCALE
  evalp m1_ablation_eval results/m1_ablation.json --checkpoint results/m1_ablation/last.ckpt
}

lane_C() {  # M0, then the M3 randomized fine-tune in work_dr (flag True there only), then Metatron
  # random_headings hard-codes --viewer and 1 env; --headless (wins in run_frames) and 16 envs for a display-less host
  [ -s "$W/results/m0_before/broken_random.json" ] || step m0_random_before 20 python -m course.module_0_hello_world.run_frames \
    --random-headings --variant broken --headless --num-envs 16 --output-dir results/m0_before
  [ -s "$W/results/m0_after/student_random.json" ] || step m0_random_after 20 python -m course.module_0_hello_world.random_headings \
    --headless --num-envs 16 --output-dir results/m0_after
  [ -s "$W/results/m0_frames/student_heading180.json" ] || step m0_compare 60 python -m course.module_0_hello_world.run_frames --compare --num-envs 128
  say "waiting for lane A's m1_control.json"
  until [ -s "$W/results/m1_control.json" ] || grep -q "end m1_control_eval rc=[1-9]" logs/lane_A.log 2>/dev/null; do sleep 60; done
  local START; START=$(m3_start); say "M3 start: $START"
  mkdir -p "$WD/results"
  rsync -a --exclude results --exclude bc_data.pt --exclude bc_policy.pt --exclude '__pycache__' "$W/" "$WD/"
  [ "$START" = "results/m1_control/last.ckpt" ] && rsync -a "$W/results/m1_control" "$WD/results/"
  E="$WD/course/module_1_train_no_dr/experiment.py"
  sed "s/^DOMAIN_RANDOMIZATION = .*/DOMAIN_RANDOMIZATION = True/" "$E" > "$E.new" && mv "$E.new" "$E"
  export STEP_COURSE=work_dr
  step m3_check_wiring 5 python -m course.module_3_train_with_dr.check_wiring
  train m3_g1 15 3 --training-max-steps 10000000 --checkpoint "$START" --time-limit-minutes 15
  evalp m3_after_clean results/m3_after_clean.json --checkpoint results/m3_g1/last.ckpt --seed 1
  evalp m3_after_perturbed results/m3_after_perturbed.json --checkpoint results/m3_g1/last.ckpt --seed 1 --perturbed
  rsync -a "$WD/results/m3_g1" "$WD/results/m3_after_clean.json" "$WD/results/m3_after_perturbed.json" "$W/results/" 2>/dev/null
  export STEP_COURSE=work
  grep -q "=== END metatron rc=0" logs/metatron.log 2>/dev/null || step metatron 20 python -m course.metatron_hurdles.run --headless --episodes 5
}

say "lane $LANE start"
lane_$LANE
say "lane $LANE ALL DONE"
