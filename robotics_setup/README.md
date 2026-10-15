# Robotics RL course: how to run it here

Why a container: the course needs Ubuntu + an RTX GPU. The nlp hosts are Rocky 9 (glibc 2.34, too old for
Isaac Sim 6 wheels) and SLURM's B200 / dsinlp01's A100 have no RT cores. nlp10 (4x RTX 2080 Ti, 11 GB) runs an
Ubuntu 24.04 rootless podman container with GPU access.

- Course code (you edit this): `/home/nlp/ronke21/RL for Real Course/robotics_rl_course_students`, seen inside the container as `/work/robotics-course/course`.
- IsaacLab + `.venv` (Python 3.12, Isaac Sim 6.0.0.1, torch 2.11+cu128), caches: nlp10 local disk `/tmp/ronke21_rrl` (54 GB). Gone if nlp10's /tmp is wiped; rerun `setup_nlp10.sh` (about 30 min).
- `rrl.sh` starts the container, activates the venv, sets `PYTHONPATH` and `OMP_NUM_THREADS=1`, and runs one command from the course root. GPU 3 by default (`RRL_GPU=N` to change; check `nvidia-smi` first).

```bash
ssh nlp10
RRL="/home/nlp/ronke21/RL for Real Course/robotics_setup/rrl.sh"

bash "$RRL" python -c "import isaacsim"          # once: review and accept the Isaac Sim licence
bash "$RRL" python -m course.module_0_hello_world.walk_forward --headless --episodes 16      # install step 4
bash "$RRL" python -m course.launch 1 train --setup-check --simulator isaaclab --headless \
  --num-envs 512 --batch-size 512 --training-max-steps 32768 --experiment-name setup_smoke --time-limit-minutes 5   # step 5
bash "$RRL" python -m course.physical_eval --checkpoint results/setup_smoke/last.ckpt \
  --num-envs 16 --episodes 16 --steps 200 --output results/setup_smoke/reload_check.json                         # step 6
bash "$RRL" bash                                   # an interactive shell inside the container
```

There is no display: always pass `--headless`. The 2080 Ti has 11 GB vs the RTX 5090 the course was timed on,
so reduce `--num-envs` and `--batch-size` together when training runs out of memory.
