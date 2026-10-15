# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This folder holds the user's exercises for the RL-for-Real course (NVIDIA Research and partners; exercises by Ido Greenberg). It is itself a git repo whose `main` branch is pushed to the user's private `Ronke21/RL-for-Real-Course` (remote `origin`). The exercises are separate clones of upstream `github.com/ido90/*` repos with their own `CLAUDE.md`, env and history; this repo's `.gitignore` excludes them, and each is pushed to a **branch** of the same private repo:

- `toy-ppo/` (branch `toy-ppo`): REINFORCE, Actor-Critic and PPO on Gymnasium CartPole/LunarLander. CPU only, env `simple_ppo`. Remote `rlforreal`.
- `rl-sentiment-exercise/` (branch `rl-sentiment`, local branch of the same name): GRPO fine-tuning of GPT-2 for positive sentiment with TRL. GPU, env `sentiment`. Remote `rlforreal`.
- `rl-for-real/` (branch `rl-for-real-grpo`): the GRPO reasoning exercise (Qwen2.5-3B on reasoning-gym, TRL). Runs on the A100 host `dsinlp01` (env under `/tmp/ronke21_rlfr`); see its `CLAUDE.md` and `SUMMARY.md`. Remote `rlforreal`.
- `robotics_rl_course_students/` (tracked on `main`, whitelist in `.gitignore`): the instructor's ProtoMotions/IsaacLab bundle for the G1 robotics day (M0-M3). Only the four student files, the user's `results/` and Claude's `claude_run/` (report, figures, logs, scripts, result JSONs, solved copy's four files) are tracked; the framework, assets and checkpoints (one is 259 MB) stay local.
- `robotics_setup/` (tracked): runs the robotics course in an Ubuntu 24.04 rootless podman container on **nlp10** (RTX 2080 Ti). Rocky 9's glibc 2.34 is too old for Isaac Sim 6 wheels; B200/A100/V100 have no RT cores. Install and caches live on nlp10's local `/tmp/ronke21_rrl`. The user runs commands with `~/.local/bin/rrl` (GPU 3, their course folder); Claude's runs use `claude_run/step.sh` (GPUs 0-2, authorised 2026-10-15).
- `Course Materials/` (tracked): lecture slides and course documents.

Shared conventions:

- The user solves the exercises themselves. Student stubs (`raise NotImplementedError`) and in-code questions are theirs; reference code lives in `solutions.py` / `rewards_solution.py`. Don't fill stubs or overwrite their uncommitted edits unless asked.
- Claude's own end-to-end runs live in each exercise's `claude_run/` (scripts, logs, results, plus a report: `toy-ppo/claude_run/REPORT.md`, `rl-sentiment-exercise/claude_run/RESULTS.md`, `robotics_rl_course_students/claude_run/REPORT.md`), never in the student files.
- toy-ppo and sentiment run locally on the current compute host, not via SLURM and never on the 8-core login node. Their envs live in `~/miniconda3/envs/`; set `LD_LIBRARY_PATH` to the env's `lib/` first (in `simple_ppo`, imports fail with `CXXABI` errors without it).
- Robotics gotchas: nlp10 has no display (always `--headless`; "see the robot" = `course/render_rollouts.py` GIFs from `--record-trajectory` reports); 4096 envs OOM on 11 GB, use 2048; several Isaac Sim instances starting at once segfault (`step.sh` staggers starts with a lock); killing the podman client leaves the container on the GPU (`step.sh` names containers and stops them on exit).
- `Miniconda3-latest-Linux-x86_64.sh` here is just the installer from the sentiment README setup (gitignored).
