#!/bin/bash
# Run one course command in a Claude copy on nlp10, logging to logs/<name>.log.
# Usage (on nlp10): STEP_GPU=N STEP_COURSE=work|work_dr bash step.sh NAME MINUTES python -m course....
# The container is named and always stopped on exit: killing the podman client alone leaves it running on the GPU.
set -uo pipefail
C="/home/nlp/ronke21/RL for Real Course/robotics_rl_course_students/claude_run"
NAME=$1; MIN=$2; shift 2
GPU=${STEP_GPU:-2}
# Ron authorised nlp10 GPUs 0-2 for Claude's runs (2026-10-15, "run a few in parallel"); GPU 3 is Ron's. Do not change unasked.
case "$GPU" in 0|1|2) ;; *) echo "GPU $GPU not authorised for claude_run" >&2; exit 2;; esac
export RRL_COURSE="$C/${STEP_COURSE:-work}" RRL_GPU=$GPU RRL_NAME="crl_${NAME}_g${GPU}"
R=/tmp/ronke21_rrl
stop_ctr() { timeout 90 podman --root $R/podman/root --runroot $R/podman/run --storage-driver overlay \
  --storage-opt overlay.ignore_chown_errors=true --storage-opt overlay.mount_program=/usr/bin/fuse-overlayfs \
  stop -i -t 10 "$RRL_NAME" > /dev/null 2>&1 || true; }
stop_ctr   # a leftover container with this name would block the run
# Staggered start: Isaac Sim segfaulted (rc 139) when 3 instances initialised at once (2026-10-15).
# Hold a lock while this simulator starts; release after 90 s. fd 9 is closed for the child so no
# long-running process ever holds the lock.
exec 9>>"$R/sim_start.lock"; flock 9
echo "=== START $NAME $(date +%F_%T) gpu=$GPU course=${STEP_COURSE:-work}: $*" >> "$C/logs/$NAME.log"
timeout -k 60 "${MIN}m" bash "/home/nlp/ronke21/RL for Real Course/robotics_setup/rrl.sh" "$@" >> "$C/logs/$NAME.log" 2>&1 < /dev/null 9>&- &
CHILD=$!
trap 'kill $CHILD 2>/dev/null; stop_ctr; echo "=== END $NAME rc=killed $(date +%F_%T)" >> "$C/logs/$NAME.log"; exit 143' INT TERM
for i in $(seq 90); do kill -0 $CHILD 2>/dev/null || break; sleep 1; done
exec 9>&-
wait $CHILD; rc=$?
trap - INT TERM
stop_ctr
echo "=== END $NAME rc=$rc $(date +%F_%T)" >> "$C/logs/$NAME.log"
exit $rc
