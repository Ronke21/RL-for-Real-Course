# RL for Real: my course workspace

My exercises, results and course materials for **RL for Real**, a hands-on course organised by NVIDIA Research with Google Research, Mentee Robotics, Tel-Aviv University, Bar-Ilan University and the Technion. Exercises are by Ido Greenberg (upstream repos at `github.com/ido90/*`).

Each exercise has two layers:
- **My own solutions:** the student files with my answers.
- **A Claude end-to-end run:** in a `claude_run/` folder, with a full report, kept separate from my student files.

## Where everything is

This repo uses one branch per exercise, because each exercise is a clone of its own upstream repo with its own history.

| Exercise | Topic | Branch | Report |
|---|---|---|---|
| **Toy PPO** | REINFORCE → Actor-Critic → PPO on CartPole / LunarLander (CPU) | [`toy-ppo`](../../tree/toy-ppo) | `claude_run/REPORT.md` |
| **RL sentiment** | GRPO fine-tuning of GPT-2 toward positive reviews (TRL) | [`rl-sentiment`](../../tree/rl-sentiment) | `claude_run/RESULTS.md` |
| **RL for Real (GRPO)** | GRPO on Qwen2.5-3B-Instruct, reasoning-gym tasks | [`rl-for-real-grpo`](../../tree/rl-for-real-grpo) | `SUMMARY.md` |
| **Robotics** | Unitree G1 humanoid tracking in Isaac Sim: frames, PPO, behaviour cloning, domain randomisation | `main` (this branch) | [`robotics_rl_course_students/claude_run/REPORT.md`](robotics_rl_course_students/claude_run/REPORT.md) |

This branch (`main`) holds:

| Path | What |
|---|---|
| `Course Materials/` | Lecture slides and course documents (see below) |
| `robotics_rl_course_students/` | My robotics work. Only our own files are tracked; the instructor bundle (ProtoMotions framework, assets, checkpoints, ~2.5 GB) stays local. |
| `robotics_setup/` | How the robotics course runs on the lab machines: an Ubuntu 24.04 container with Isaac Sim on nlp10 |
| `CLAUDE.md` | Guide for Claude Code across all exercises |

## Course materials

| Topic | Files |
|---|---|
| Course overview | `RL for Real.docx`, `RL for Real_ Resources.docx`, `RL for Real_ project proposals.docx`, `RL exercises.pdf` |
| Foundations | `Intro to RL.pdf`, `rl-foundationsNVDA.pdf`, `Deep RL.pdf`, `deep_rl_2.pdf`, `main.pdf` |
| Methods | `Model-Based_RL_Revised_v4.pptx`, `Practical RL Design.pdf`, `exploration_experiments.ipynb` |
| Applications | `RL_in_LLMs.pdf`, `RL for Diffusion Models.pdf`, `RL for Optimization.pdf`, `[RL Course] Robotics.pptx` |

## Headline results

**Toy PPO** (single seed, 100 iterations):
- On CartPole both methods learn, and PPO is about twice as fast (473 vs 440 at the end).
- On LunarLander only PPO makes real progress (−13.5 vs −103), but it has not learned to land with the defaults.

**RL sentiment** (GPT-2, GRPO):
- The sparse five-star reward gives zero gradient: every group of 8 completions scores 0, so nothing is learned.
- The dense expected-stars reward is learned fast and the text degrades into nonsense, which motivates the KL penalty and reward shaping.
- RL beats the best prompt for the learned metric (0.70 vs 0.60).

**RL for Real, GRPO track 1** (Qwen2.5-3B-Instruct):
- Best eval accuracy is 54.5% (eval seed 62, 200 samples), against 10.5–12.5% untrained, with lr 5e-6 for 300 steps.
- Results vary a lot by training seed: 40.0 / 43.0 / 54.5% for the same config.

**Robotics** (500 physical episodes per number):

| Module | Result |
|---|---|
| M0, observation frame | The buggy world-frame velocity walks only at 0°: 0% survival at 90° and 180°. The heading-frame fix gives 100% at all headings. |
| M1, PPO | The weak tracker goes from 0% to 80.8% full-clip survival in 30 min, though error (0.14 m) stays above the 0.10 m reference: only 38% of the training budget fits on a 2080 Ti in that time. |
| M2, behaviour cloning | Validation MSE is 1,400× below the constant baseline, yet survival is 0/500 (expert 500/500): compounding error. |
| M3, domain randomisation | Perturbed survival goes 3% → 81% (clean fine-tune) → 91.6% (DR fine-tune), with no clean-tracking cost. |

## Running the robotics course here

The course needs Ubuntu and an RTX GPU. Lab hosts run Rocky 9 (glibc too old for Isaac Sim 6), and the SLURM B200 and A100 cards have no RT cores. So it runs in a podman container on **nlp10** (RTX 2080 Ti):

```bash
ssh nlp10
rrl python -m course.module_0_hello_world.walk_forward --headless --episodes 16   # rrl = robotics_setup/rrl.sh, GPU 3
```

- **No display:** nlp10 has no screen, so always pass `--headless`. To see the robot, render GIFs from recorded rollouts (`robotics_setup/README.md`).
- **11 GB card:** use 2048 environments for training, not 4096.
- **Install:** the simulator install lives on nlp10's local disk under `/tmp/ronke21_rrl`. If it disappears, rerun `robotics_setup/setup_nlp10.sh`.
