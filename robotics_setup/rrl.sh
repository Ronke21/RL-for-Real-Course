#!/bin/bash
# Run a command inside the course container on nlp10 (Ubuntu 24.04; Rocky 9's glibc 2.34 is too old for Isaac Sim 6).
# Usage (on nlp10):  bash rrl.sh <command...>      e.g.  bash rrl.sh python -m course.preflight
# Inside: course at /work/robotics-course/course (= the folder in your NFS home), IsaacLab + .venv on nlp10 local disk.
# GPU: RRL_GPU (default 3). One GPU only.
set -eu
R=/tmp/ronke21_rrl
COURSE="${RRL_COURSE:-/home/nlp/ronke21/RL for Real Course/robotics_rl_course_students}"   # RRL_COURSE: Claude's copy = .../claude_run/work
GPU=${RRL_GPU:-3}
mkdir -p $R/IsaacLab $R/home $R/podman/root $R/podman/run
TTY=""; [ -t 0 ] && TTY="-it"
exec podman --root $R/podman/root --runroot $R/podman/run --storage-driver overlay \
  --storage-opt overlay.ignore_chown_errors=true --storage-opt overlay.mount_program=/usr/bin/fuse-overlayfs \
  run --rm $TTY ${RRL_NAME:+--name $RRL_NAME} --device nvidia.com/gpu=$GPU --ipc=host \
  -v "$COURSE":/work/robotics-course/course -v $R/IsaacLab:/work/robotics-course/IsaacLab -v $R/home:/root -v "/home/nlp/ronke21/RL for Real Course/robotics_setup":/work/setup:ro \
  -e HOME=/root -e PYTHONPATH=/work/robotics-course/course -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e UV_CACHE_DIR=/root/.cache/uv -e CLAUDE_ACTION_RATE_SCALE -w /work/robotics-course/course \
  localhost/rrl:24.04 bash -lc 'source /work/robotics-course/IsaacLab/.venv/bin/activate 2>/dev/null || true; exec "$@"' _ "$@"
