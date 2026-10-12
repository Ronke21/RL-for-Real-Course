#!/bin/bash
# Runs every README step. Three independent lanes in parallel (2 threads each), then figures.
set -euo pipefail
cd "/home/nlp/ronke21/RL for Real Course/toy-ppo"
PY=$HOME/miniconda3/envs/simple_ppo/bin/python
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/simple_ppo/lib:${LD_LIBRARY_PATH:-}
export MPLBACKEND=Agg SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
L=claude_run/logs
laneA() {  # README steps 3,4,5,9
  $PY tester.py
  $PY main.py
  $PY -m extensions.actor_critic
  for env in CartPole-v1 LunarLander-v3; do
    for ag in random REINFORCE PPO Actor-Critic; do
      $PY visualize.py --agent $ag --env $env --out "plots/${ag}_${env}.gif"
    done
  done
}
laneB() { $PY claude_run/ext_mountaincar.py; }                       # ext 6
laneC() { $PY claude_run/ext_gae.py; $PY claude_run/ext_tuning.py; } # ext 8, 7
mkdir -p plots
laneA > $L/laneA_main.log 2>&1 & A=$!
laneB > $L/laneB_mountaincar.log 2>&1 & B=$!
laneC > $L/laneC_tuning_gae.log 2>&1 & C=$!
rc=0; wait $A || rc=1; wait $B || rc=1; wait $C || rc=1
$PY claude_run/make_figures.py > $L/figures.log 2>&1 || rc=1
echo "run_all finished rc=$rc at $(date)"; exit $rc
