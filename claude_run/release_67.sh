#!/bin/bash
# Run ON dsinlp01, detached: when both GPU 6/7 baseline re-check runs have ended, stop lanes 6 and 7.
R=/tmp/ronke21_rlfr
until grep -q "END p6_A_repro_evalbs16" $R/logs/lane_gpu6.log && grep -q "END p6_A_repro_evalbs64" $R/logs/lane_gpu7.log; do sleep 60; done
for g in 6 7; do pkill -u ronke21 -f "^bash $R/rl-for-real/claude_run/run_lane.sh $g\$"; done
echo "$(date +%T) lanes 6-7 released" >> $R/logs/release_67.log
