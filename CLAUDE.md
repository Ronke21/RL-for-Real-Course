# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A teaching exercise (RL-for-Real course, upstream `github.com/ido90/rl-sentiment-exercise`): GRPO fine-tuning of GPT-2 (TRL) toward positive movie-review sentiment, scored by `nlptown/bert-base-multilingual-uncased-sentiment` (5-star classifier). README Exercises 1-6 cover the sparse reward, a dense reward, hand-written KL regularization, reward shaping, and RL vs prompt engineering. Students fill in `raise NotImplementedError` stubs in `rewards.py` (`expected_stars`, `kl_penalty_forward`, `kl_penalty_backward`, `shaped_reward`) and answer `QUESTION Q1`-`Q10` comments spread across `data.py`, `rewards.py`, `reward_utils.py`, `train.py`. `rewards_solution.py` holds the reference versions.

The user is doing the exercise themselves: uncommitted edits in the repo root hold their Q answers and their `rewards.py` implementations (`shaped_reward` still a stub as of 2026-10-14; check `git diff`). Do not fill in stubs, answer questions, or "correct" their implementations unless asked. Their `kl_penalty_backward` differs from the solution (it normalizes by the sum of importance weights and does not clip them), so results from `rewards.py` and `--use_solution` are not interchangeable.

## Commands

Conda env `sentiment` (Python 3.10, CUDA torch, pinned `transformers==4.57.6`, `trl==0.27.2`; see README for install). It exists at `~/miniconda3/envs/sentiment`; set `LD_LIBRARY_PATH=$HOME/miniconda3/envs/sentiment/lib` before running.

```bash
python rewards.py                  # smoke-test the student reward functions (prints NotImplemented for open stubs)
python rewards_solution.py         # same tests on the reference code
python train.py --reward_shaping five_stars                  # Ex 2 (sparse)
python train.py                                              # Ex 3 (default: expectation, no KL)
python train.py --kl_type forward --kl_coef 0.1              # Ex 4
python train.py --reward_shaping custom --kl_type forward    # Ex 5
python prompt_engineering.py --use_shaped_reward --trained_model ./outputs/final   # Ex 6
python evaluate.py {evaluate,compare,plot} ...               # post-hoc scoring / plots
```

There is no test suite, linter, or build. Useful `train.py` flags: `--max_steps N` (default 200; a 20-step run is the dry run), `--use_solution`, `--no_wandb` (wandb is ON by default), `--no_perplexity`, `--output_dir`, `--use_peft` (LoRA if OOM), `--negate_reward`, `--beta` (TRL's own KL, default 0; distinct from `--kl_coef`). A 200-step run is about 8 min on a GTX 1080 Ti; 400 steps about 15 min.

Runs here are local GPU runs on the current host (2026-10-13 on nlp07: GPU 0 is left free for Ron, one training process per GPU on 1-3, `OMP_NUM_THREADS=2`, `TOKENIZERS_PARALLELISM=false`, `WANDB_MODE=disabled`). Check the cards are empty before launching.

## Architecture

- `train.py`: picks device/dtype (bf16 if `torch.cuda.is_bf16_supported()`, which is true via emulation on 1080 Ti cards, else fp16; fp32 on CPU), imports `rewards` or `rewards_solution` as `reward_module`, builds the reward with `reward_utils.make_reward_function`, configures `trl.GRPOConfig` from a `TrainingPreset` dataclass (8 generations/prompt, batch 2 x grad-accum 4, lr 1e-4, temperature 1.5, 64 new tokens, save every 100 steps), and attaches `ValidationCallback`, which every `logging_steps` (20) generates on the 64 `VALIDATION_PROMPTS` and reports sentiment, positive ratio, reference perplexity and first-tokens perplexity. Final model goes to `{output_dir}/final`.
- `reward_utils.make_reward_function` returns the closure TRL calls: base reward (`five_stars_reward` / `expected_stars` / `expected_stars` then `shaped_reward`), optional negation, then minus the KL penalty. For KL, `train.py` loads a separate copy of GPT-2 as `policy_model` plus a frozen `ref_model`, and `compute_log_probs` recomputes per-token log-probs under both outside TRL (prompt+completion re-tokenized). The `policy_model` object is the one handed to `GRPOTrainer`, which is why its log-probs track training.
- `ref_model` is also loaded when KL is off, solely for validation perplexity (skip with `--no_perplexity`).
- `sentiment.py` caches the classifier in module globals; `get_star_probs` returns a (batch, 5) tensor, `get_sentiment_scores` the [0,1] expected-stars score used for validation.
- `prompt_engineering.py` scores base GPT-2 vs a trained model under each prefix in `PROMPT_STRATEGIES`; with `--use_shaped_reward` it imports `shaped_reward` from `rewards` (or `rewards_solution` with `--use_solution`), so it fails until that stub is filled.

## `claude_run/`

A full exercise run done by Claude on 2026-10-13, kept separate from the user's work: `work/` is a copy of the clean repo with Claude's own TODO implementations (`work/rewards.py` selects the Ex 5 shaping via `SHAPE_MODE`; `work/prompt_engineering.py` selects prefix sets via `PE_SET`). Driver scripts (`phase1.sh`, `lane.sh GPU PHASE "name|args" ...`, `pe_run.sh`, `eval_*.sh`) write to `runs/<name>/final` (gitignored), `logs/`, `results/`, and delete intermediate `checkpoint-*` dirs. `RESULTS.md` has the answers and numbers (single seed). Key findings: the five_stars reward gives zero gradient (all-zero groups); `--kl_coef 5` default is far too strong; sentiment above 0.8 came from 400 steps, not smaller coefficients.

## Gotchas

- `kl_coef` scales a mean per-token KL against a reward in [0, 1]; the README calls the default 5.0 "very conservative".
- `make_reward_function` defaults `reward_module` to `rewards_solution` if none is passed; `train.py` always passes one.
- `outputs/`, `checkpoints/`, `wandb/`, `logs/`, `claude_run/runs/` are gitignored; `outputs/` currently holds an earlier run.
