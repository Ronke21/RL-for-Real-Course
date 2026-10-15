#!/bin/bash
# Launched detached on nlp10: build the image, then run the install inside it. Self-limits to 3 h.
set -euo pipefail
trap 'kill 0' EXIT INT TERM
D="/home/nlp/ronke21/RL for Real Course/robotics_setup"
R=/tmp/ronke21_rrl
mkdir -p $R/podman/root $R/podman/run
podman --root $R/podman/root --runroot $R/podman/run --storage-driver overlay \
  --storage-opt overlay.ignore_chown_errors=true --storage-opt overlay.mount_program=/usr/bin/fuse-overlayfs \
  build -t rrl:24.04 -f "$D/Containerfile" "$D"
bash "$D/rrl.sh" bash /work/setup/install_in_container.sh
