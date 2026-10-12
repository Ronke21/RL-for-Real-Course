# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A teaching exercise (RL-for-Real course, upstream `github.com/ido90/toy-ppo`): REINFORCE → Actor-Critic → PPO on CartPole-v1 and LunarLander-v3 (discrete actions only; both nets use `Categorical`). Students fill in five `raise NotImplementedError` stubs in `main.py`; `solutions.py` holds the reference versions. The user is implementing the stubs themselves (uncommitted edits in `main.py`: TODOs 1-4 filled in as of 2026-10-11, check `git diff` for the current state). Do not fill in or rewrite remaining stubs unless asked.

## Commands

Environment is `simple_ppo` (`conda env create -f environment.yml`; CPU-only torch, gymnasium with box2d). It exists on this machine (`~/miniconda3/envs/simple_ppo`). On Linux, `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` first, or imports fail with `CXXABI` errors.

```bash
python tester.py                       # unit-test the 5 TODO functions (no training)
python main.py                         # train REINFORCE + PPO on ENV_IDS; ~8-15 min CPU; writes checkpoints/ and plots/
python -m extensions.actor_critic      # REINFORCE vs Actor-Critic vs PPO
python visualize.py --agent PPO --env CartPole-v1   # needs checkpoints/; writes plots/{agent}_{env}.gif
```

There is no pytest, linter, or build step. To run a single test, call its function: `python -c "import tester; tester.test_compute_returns()"`. Tests return `True/False` and print; `tester.py` exits 1 on failure, 2 on unimplemented stubs.

Training is CPU-bound (~10-15 min). For this repo Ron wants runs done locally on the current host (e.g. nlp07), not via SLURM; launch detached with `setsid nohup timeout 3h`, cap `OMP_NUM_THREADS`/`MKL_NUM_THREADS` (2 per parallel lane), and never run it on the 8-core login node. Run `python tester.py` first.

`claude_run/` holds the full README run done on 2026-10-12: `run_all.sh` (all steps + extensions), `ext_*.py` (MountainCar, PPO tuning, GAE check; they import `main` and patch its globals instead of editing it), and `REPORT.md` with results. Git remote `rlforreal` is the user's private repo; `origin` is upstream.

## Architecture

- `main.py` is the whole pipeline: config constants, rollout collection, returns/advantages, losses, `run_reinforce`, `run_ppo`. `compute_gae` is already implemented here (not a TODO; README extension 8 asks students to rewrite it).
- Rollout flow: `collect_rollouts` loops `num_steps` times over the TODO-1 function `collect_rollout_step` and returns a dict of stacked tensors (`obs, actions, log_probs, rewards, dones, values, last_value, ep_returns`). It spans several episodes, resetting inside the step function on `done`. `dones` is `terminated or truncated`, so GAE does not bootstrap across a time-limit truncation.
- REINFORCE recomputes `log_probs` with gradients from `rollout["obs"]` (rollout log-probs are detached) and takes one gradient step per rollout. PPO uses GAE, normalizes advantages, then runs `PPO_UPDATE_EPOCHS` × minibatch steps with grad-clip 0.5 and `policy_loss + 0.5 * value_loss` on one shared Adam.
- `rl_utils.py`: `PolicyNetwork`/`ValueNetwork` (64-64 tanh MLPs), `make_env` (wraps in `RecordEpisodeStatistics`, which is where `info["episode"]["r"]` comes from), `plot_comparison`. It seeds `random`/`numpy`/`torch` with 42 at import and pins `DEVICE = cpu`.
- `extensions/actor_critic.py` imports `collect_rollouts`, `compute_returns` and constants from `main`, so it inherits `USE_SOLUTIONS`.

## Gotchas

- `USE_SOLUTIONS` in `main.py` (currently `False`; `True` replaces the five TODO functions with imports from `solutions.py` at import time). Because `tester.py`, `extensions/` and everything else import from `main`, this flag also decides whether `tester.py` tests the student code or the reference code. Check it before interpreting test results.
- `solutions.compute_returns` defaults `gamma=1.0`; `main.compute_returns` defaults to `GAMMA`. Tests pass `gamma` explicitly.
- Checkpoints store only the policy state dict plus `obs_dim`/`act_dim`, named `checkpoints/{REINFORCE|PPO|Actor-Critic}_{env_id}.pt`; `visualize.py` builds the path from `--agent`/`--env`, so those names must match. Both `checkpoints/` and `plots/` are gitignored.
- `plot_comparison` calls `plt.show()` after saving; on a headless host use `MPLBACKEND=Agg`.
- Adding an env to `ENV_IDS` only works for discrete-action Gymnasium envs (`env.action_space.n`).
