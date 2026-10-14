#!/bin/bash
# Run ON dsinlp01: local Miniconda + env for rl-for-real under /tmp/ronke21_rlfr (local disk).
# dsinlp01's /home/nlp/ronke21 is a separate, empty NFS home, so nothing from nlp07 is visible there.
set -euo pipefail
R=/tmp/ronke21_rlfr
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PIP_NO_INPUT=1
[ -x $R/conda/bin/conda ] || bash $R/Miniconda3-latest-Linux-x86_64.sh -b -p $R/conda
C=$R/conda/bin/conda
$C tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
$C tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
[ -x $R/conda/envs/rlfr/bin/python ] || $C create -y -p $R/conda/envs/rlfr python=3.10
PY=$R/conda/envs/rlfr/bin/python
$PY -m pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
$PY -m pip install -q "transformers==4.57.6" "trl==0.27.2" "accelerate>=0.30.0" "peft>=0.10.0" \
    "reasoning-gym>=0.1.19" "wandb>=0.16.0" "tqdm>=4.65.0" "pyyaml>=6.0"
$PY -c "import torch, transformers, trl, reasoning_gym; print('ENV_OK', torch.__version__, transformers.__version__, trl.__version__)"
# Model into a local HF cache (one download, not per run)
HF_HOME=$R/hf $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-3B-Instruct'))"
echo "SETUP_DONE $(date)"
