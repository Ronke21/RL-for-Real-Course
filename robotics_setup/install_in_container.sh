#!/bin/bash
# Steps 2-3 of course/install.html, run inside the container. Stops on the first failure.
set -euo pipefail
cd /work/robotics-course
test -f course/pyproject.toml
test -f course/course/constraints_isaaclab.txt
if [ ! -d IsaacLab/.git ]; then git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab; fi
cd IsaacLab
git checkout 4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8
[ -d .venv ] || uv venv --python 3.12 .venv
source .venv/bin/activate
python --version
uv pip install --python .venv/bin/python -r pyproject.toml --extra isaacsim \
  -e '../course[isaaclab,dev]' \
  --index https://download.pytorch.org/whl/cu128 \
  --prerelease if-necessary \
  'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' 'torchaudio==2.11.0+cu128'
cd ../course
uv --no-config pip install --python ../IsaacLab/.venv/bin/python -r course/constraints_isaaclab.txt
uv pip check --python ../IsaacLab/.venv/bin/python
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
python -m course.preflight
echo "__INSTALL_DONE__"
