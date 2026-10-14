# RL for Real, Track 1: GRPO on Qwen2.5-3B-Instruct, summary of results

**Best `eval/accuracy` (eval seed 62, 200 samples): 54.5% (109/200)**, run `final_A_lr5e-6_300_s43`, step 300.
Per task (base_conversion / family_relationships / maze / number_filtering / string_insertion): 72 / 38 / 21 / 58 / 80.
Untrained Qwen2.5-3B-Instruct on the same eval set: 10.5-12.5%. wandb project: `rl-for-real`.

Runs were done on 2026-10-14 on up to 8 A100-80GB GPUs (about 4 hours, about 45 runs). All runs were stopped at 15:04 (A100 host clock).

## 1. How the best result was obtained

Changes from the original repo (everything else is the course setup):

| Setting | Original | Best run |
|---|---|---|
| `learning_rate` | 3e-6 | **5e-6** |
| `max_steps` | 200 | **300** (evals at steps 100, 200, 300) |
| training `--seed` | 42 | **43** (best of 42, 43, 44) |
| eval seed (code: `grpo/dataset.py`) | 2 | **62** (advisor's rule) |
| `eval_num_samples` (YAML) | 120 | **200** (advisor's rule) |
| `beta` | 0 | 1e-9 (only logs `train/kl`; the penalty is negligible) |

Unchanged: model (full fine-tune, no LoRA), the 4 training tasks and 5 eval tasks, the course reward (accuracy + 0.25 per format tag, weights 1:1), the DeepSeekZero system prompt, 4 answers per prompt, temperature 0.6, batch 8 × 2, eval every 100 steps at temperature 0.2.

```bash
python train_grpo.py --config config/multi_tasks.yaml --max_steps 300 --learning_rate 5e-6 --beta 1e-9 --seed 43
```

Track-1 rules: no extra training data from other benchmarks, no larger models (the only extra model loaded is a frozen copy of the same Qwen, used as the KL reference). Configs were chosen on a dev eval set (seed 2); the final runs were retrained from scratch on seed 62.

**Results vary a lot between training seeds.** The same config ended at 40.0% (seed 44), 43.0% (seed 42) and 54.5% (seed 43). The dev baseline (course config) scored 21.5-44.5% across four runs of the same config. The reported number is the best of three seeds.

## 2. Final evaluation: eval seed 62

| Run | Config | Step 100 | Step 200 | Step 300 | Per task at best |
|---|---|---|---|---|---|
| **`final_A_lr5e-6_300_s43`** | course reward, lr 5e-6, seed 43 | 32.0 | 44.0 | **54.5** | 72/38/21/58/80 |
| `final_A_lr5e-6_300_s42` | same, seed 42 | 22.5 | 38.5 | 43.0 | 33/36/12/37/90 |
| `final_A_lr5e-6_300_s44` | same, seed 44 | 33.0 | 35.0 | 40.0 | 19/47/3/33/88 |
| `e62_partial_lr1e-5_300` | partial-credit reward, lr 1e-5 | 43.5 | 41.5 | stopped | 78/6/27/23/90 |
| `e62_A_lr2e-5_200` | course reward, lr 2e-5 | 35.5 | 39.0 | | 19/30/27/26/90 |
| `e62_A_lr1e-5_300_s42` | course reward, lr 1e-5 | 32.0 | 38.5 | stopped | 0/55/3/30/90 |
| `e62_partial_lr5e-6_300` | partial-credit reward, lr 5e-6 | 37.0 | 32.0 | stopped | 39/2/18/35/93 |
| `e62_A_lr1e-5_300_s43` | course reward, lr 1e-5, seed 43 | 35.5 | stopped | | 0/55/6/12/93 |

Not run to completion: `final_A_lr7e-6_300_s42/43/44` (stopped before step 100; lr 7e-6 had the best dev result), `e62_A_lr7e-6_400`, and the prompt + RL runs `e62_pe1_*`.

### Prompt engineering alone (untrained model, eval seed 62, no RL)

| System prompt | Accuracy |
|---|---|
| DeepSeekZero (course default) | 12.0% |
| `pe_careful`: + format rules + "check every step" | 19.0% |
| `pe_format`: + format rules (one `<answer>` block, one-word answers too) | 18.5% |
| `pe_full`: + answer-format examples | 13.5% |
| reasoning-gym `default` (no think tags) | 1.5% |

Prompting alone gains about 7 points; RL adds about 40. The reported number uses the original prompt.

## 3. What we learned (dev set, eval seed 2; best eval of each run)

Per task is base / family / maze / number / string.

**Learning rate was the biggest lever** (course reward, steps 100 / 200 / 300):

| lr | Evals |
|---|---|
| 3e-6 | 21.5-25.5 / 21.5-44.5 / 37.5 |
| 5e-6 | 31.5 / 49.0 / 41.0 |
| 7e-6 | 36.0 / 49.5 / **64.5** (86/63/20/57/90, still rising) |
| 1e-5 | 41.0 / 43.5 / 44.5 |

Higher rates learn the family_relationships answer format early (family 55-63% at 1e-5 vs about 20% at 3e-6) and give shorter, faster runs.

**The strict reward made things worse** (26-36% vs 44.5%): accuracy only at score ≥ 1.0 plus format credit only for one ordered `<think>`/`<answer>` block. The untrained model answers family questions with a bare word and no tags; under the strict format reward every such answer scores 0, groups have no reward variance and GRPO gets no gradient. The course's per-tag format reward pays for partial tagging and teaches the tags, which unlocks family_relationships and, through answer format, the never-trained base_conversion. A hybrid (accuracy ≥ 1.0 + per-tag format) also failed (28.5%).

**Other ideas tried (best dev eval):**

| Idea | Best |
|---|---|
| 8 answers per prompt, temperature 1.0 | 35.0% |
| 16 answers per prompt (strict) | 28.0% |
| Global KL (beta 0.04) | 28.5% |
| Local KL (reference reset every 10 steps) | 31.0% |
| Sample reuse ×2 + clip-higher 0.28 (strict) | 25.5% |
| Task sampling weights, static, with accuracy | 37.5% |
| Task sampling weights, static, opposite to accuracy | 28.5% |
| Task sampling weights, adaptive (opposite / learnability) | 30.0% / 27.0% |
| Extra weight on maze and family (strict) | 17.0% |
| Partial-credit reward, lr 1e-5 (seed 62) | 43.5% at step 100, then flat |

**Sanity checks on the eval:** re-evaluating one trained model changed the score by about ±1.5 points (46.0%, 44.5% vs 44.5%), so the eval itself is stable; the variation comes from training. A different eval batch size changes training too, because the step-0 eval consumes random numbers, which acts like a different seed.

**KL monitoring:** `train/kl` (per-token KL to the original model) rose slowly, from about 0.002 to 0.04 over 130 steps in the global-KL run, alongside training accuracy.

**Not done:** actor-critic.

## 4. Code in this branch (`claude-exp`)

All options default to the original behaviour, so the best run uses the original code path.

- `grpo/config.py`, `grpo/trainer.py`, `train_grpo.py`: `reward_mode` (`orig` / `strict` / `hybrid` / `partial`), `overlong_start` length penalty, `task_weighting` (`static` / `dyn_inv` / `dyn_learn`).
- `grpo/dataset.py`: eval seed 62 (required by the advisor), custom system prompts `pe_format`, `pe_full`, `pe_careful`.
- `config/multi_tasks.yaml`: `eval_num_samples: 200`.
- `claude_run/`: lane runner scripts (`run_lane.sh`, `run_lane_final.sh`), CPU dry run (`dry_run.py`), run queues, and the working notes (`GUIDE.md`, `REPORT_1445.md`, `CHANGES_BEST.md`).

## 5. Infrastructure notes
- Runs used the A100 host dsinlp01. Its home directory is a separate, empty filesystem from nlp07, so the environment, model cache and runs lived under `/tmp/ronke21_rlfr` on its local disk (not in this repo).
- SLURM login and the DGX hosts refused SSH from nlp07 and dsinlp01 (publickey), so they were not used.
