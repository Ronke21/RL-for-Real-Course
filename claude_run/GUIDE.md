# RL for Real: the GRPO reasoning exercise

Guide to the exercise, the tools, the starting settings, what we change and why, and the run plan.
Status as of 2026-10-14, 11:10 (dsinlp01 clock).

---

## 1. The task

Fine-tune **Qwen2.5-3B-Instruct** with RL (GRPO) so it solves reasoning puzzles, and get the best test score you can.

| | Challenge I: best overall test score | Challenge II: coolest training results |
|---|---|---|
| Metric | highest `eval/accuracy` reached at any eval point during training | vote after a 2-minute presentation |
| Rules | in `config/multi_tasks.yaml`, **do not change** `model_name_or_path`, `eval_datasets`, or the **keys** of `datasets` (task parameters can change, other training tasks cannot be added); **set** `eval_num_samples = 200`; do not hack the eligibility conditions | none (other tasks, larger models, creative constraints) |

Our choices for Challenge I, made to stay clearly eligible:
- Eval stays as given: every 100 steps, eval temperature 0.2, the original system prompt.
- All changes are to training: rewards, hyperparameters, training-task weights.
- `multi_tasks.yaml` itself is unchanged; every change is a command-line override.

---

## 2. The benchmark: Reasoning Gym

[Reasoning Gym](https://github.com/open-thought/reasoning-gym) (Stojanovski et al. 2025, arXiv 2505.24760) is a library of **procedurally generated** reasoning tasks. Each task has:
- **A generator:** produces unlimited questions with a known answer, with difficulty knobs (list length, grid size, and so on).
- **A scorer:** `score_answer(answer, entry)`, returning a value in [0, 1].

There's no fixed dataset to overfit: train and eval questions come from the same generators with different seeds (train seed 1, 20,000 items; eval seed 2, 200 items).

### The five tasks

Train on the first four; eval on all five. `base_conversion` is never trained on, so it measures transfer.

The eval set has 200 items (default task configs, equal weights). The step-0 (untrained) scores below are from our run A.

| Task | Eval items | Step-0 accuracy | Step-0 format score | Answer must be exactly |
|---|---|---|---|---|
| family_relationships | 51 | **0.0%** | **0%** | one lowercase word, e.g. `father` |
| number_filtering | 44 | 13.6% | 91% | Python list repr, e.g. `['85.3', '-1']` |
| string_insertion | 39 | 10.3% | 98% | the modified string, e.g. `CEADBABACDBDE` |
| base_conversion | 36 | 13.9% | 100% | lowercase digits, no prefix, e.g. `56` |
| maze | 30 | 6.7% | 94% | the number of steps, e.g. `7` |
| **Overall** | **200** | **8.5% (17/200)** | 71% | |

### How an answer is scored

1. The system prompt (reasoning-gym's `DeepSeekZero`) asks for `<think> reasoning </think><answer> answer </answer>`.
2. `extract_answer` takes the content of the **last** `<answer>…</answer>` block and strips whitespace. No block means no answer.
3. The task scorer compares it with the gold answer. Most tasks use exact match. If the gold is a *substring* of the answer, the base scorer gives partial credit `len(gold)/len(answer)`; string_insertion gives 0.1 for a list of characters; family_relationships lowercases.
4. **Eval counts a sample as correct only if the score is ≥ 1.0.** Partial credit is worth nothing at eval.

### Real examples (untrained model, step-0 eval)

**family_relationships: right answer, scored 0 for missing tags**
```
Q:    Benjamin is married to Rain. They have a child called Charles. Charles is married to Grace.
      They have a child called Noah. Aiden is married to Abigail. They have a child called Grace.
      How is Aiden related to Grace? Provide the relationship in one word.
GOLD: father
OUT:  father                      <- no <answer> tags, so nothing is extracted: score 0
```
This happens on all 51 family items: the task prompt says "one word", the model obeys that over the system prompt, and the eval sees nothing. It's the single biggest source of free points.

**base_conversion: answer first, reasoning after, several answer blocks**
```
Q:    convert the base-3 number 2211 to base-14            GOLD: 56
OUT:  <think>...convert to decimal first...</think> <answer>71</answer>
      <think>(2*27)+(2*9)+(1*3)+(1*1) = 76</think> <answer>76</answer>
      <think>Next, convert 76 to base-14 ...</think> ... <answer>56</answer>   <- last block counts: correct
```
The untrained model often guesses an answer before reasoning, then "re-answers" in more blocks. Only the last `<answer>` counts, so this sometimes works by luck and often wastes the token budget.

**number_filtering: format right, logic wrong**
```
Q:    Keep all numbers smaller than -50.45 in this list: ['-98', '-68.4', '-96.6', '71.47', '-52.06', '-6.73']
GOLD: ['-98', '-68.4', '-96.6', '-52.06']
OUT:  <answer>['-98', '-68.4', '-96.6', '-52.06', '-6.73']</answer>   <- answered before checking -6.73
```

**maze** (the grid uses random symbols for wall, passage, start and goal)
```
Navigate from 'L' (start) to 'y' (goal):
XXXXXXX
X;XX;;X
X;;L;XX          Legend: 'X' = Wall, ';' = Passage
X;;X;;X          GOLD: 5     OUT: <answer>7</answer>  (assumed only right/down moves)
X;XX;;X
X;;;XyX
XXXXXXX
```

**string_insertion**: 5 insertion rules over the letters A-E (e.g. "after ABCD insert A"), applied left to right, skipping over each insertion. The model usually checks the rules correctly but often answers before checking.

### Known traps

- **family_relationships label bug:** in-law questions are labelled with the wrong side, e.g. "father-in-law" where the right answer is "daughter-in-law". About 10% of items can't be answered correctly by correct reasoning, so 100% is impossible.
- **Partial credit can be gamed in training:** on maze, answering "16" when the gold is "6" earns 0.5 reward (the gold is a substring), though eval counts it as wrong.
- **Length:** eval generates at most 512 new tokens. An answer cut off before `</answer>` scores 0. So far the answers are short (75-110 tokens on average, about 1% hit the 384-token training limit), so this hasn't been a problem.

### Published reference point

The Reasoning Gym paper trained the same Qwen2.5-3B-Instruct with GRPO (lr 1e-6, 32 prompts × 8 answers per step, temperature 1.0, format reward weighted 0.2, 500 steps):
- **In-category gains:** a few points, e.g. their algorithmic category went from 52.3 to 59.7.
- **base_conversion transfer:** in their raw logs it went from 0.21 to 0.70 after RL on *other* categories. That's mostly the model learning to answer in the required form.

No one has published results for this specific exercise; the one GitHub fork was made today and is unchanged.

---

## 3. GRPO and TRL

### GRPO in one paragraph
For each prompt, the model samples a **group** of G answers. Each answer gets a reward. Its **advantage** is its reward minus the group mean, divided by the group standard deviation: answers better than their siblings are pushed up, worse ones pushed down. There's no value network (critic), and the group is the baseline. Consequence: **if all G answers get the same reward, the group teaches nothing.** TRL logs the share of such groups as `frac_reward_zero_std`. Bigger groups, a higher sampling temperature and rewards that separate good from bad answers all reduce it.

### TRL
[TRL](https://github.com/huggingface/trl) is Hugging Face's RL library for language models. `GRPOTrainer` handles generation, rewards, advantages and the policy-gradient loss. The repo pins **TRL 0.27.2**, whose defaults already follow the DAPO recipe:

| Option | Default in 0.27.2 | What it does |
|---|---|---|
| `loss_type` | `dapo` | token-level loss normalisation |
| `beta` | 0.0 | KL penalty to the original model (0 = none, no reference model loaded) |
| `scale_rewards` | `group` | divide advantages by the group std (`batch`, or `none` as in Dr. GRPO) |
| `num_generations` | 8 | group size G |
| `temperature` | 1.0 | sampling temperature for training answers |
| `num_iterations` | 1 | how many updates reuse each batch of answers |
| `epsilon` / `epsilon_high` | 0.2 / None | PPO clip range; `epsilon_high` = DAPO "clip-higher" |
| `mask_truncated_completions` | False | drop answers that hit the length limit from the loss |
| `off_policy_mask_threshold` | None | TRL's off-policy mask (the README calls it `off_policy_threshold`) |

**Important detail:** with `num_iterations=1` (the default), the PPO ratio is always 1, so `epsilon_high`, `off_policy_mask_threshold`, `importance_sampling_level` and `delta` **do nothing**. They only matter when samples are reused (`num_iterations > 1`).

---

## 4. The repo (`github.com/ido90/rl-for-real`)

About 1,000 lines of Python:

- `train_grpo.py`: entry point. `TrlParser` reads the YAML into three dataclasses: the repo's own `DatasetConfig` (`grpo/config.py`), TRL's `GRPOConfig`, and TRL's `ModelConfig`. Any field can be overridden on the command line, e.g. `--learning_rate 2e-6`. It then runs a step-0 eval, trains, and saves `final_model/`.
- `grpo/dataset.py`: builds a reasoning-gym **composite** dataset (train seed 1, eval seed 2) and wraps each question in the chat template with the `DeepSeekZero` system prompt.
- `grpo/trainer.py`: `CustomGRPOTrainer` (subclass of TRL's `GRPOTrainer`) with two rewards: **accuracy** (reasoning-gym's score of the last `<answer>`) and **format** (0.25 per tag found anywhere). It also logs `reward_variance_pct`. **This is where the README says to edit rewards.**
- `grpo/eval_callback.py`: `FixedEvalCallback` evaluates the fixed 200 samples at step 0 and every `fixed_eval_steps`. It reports overall and per-task accuracy (score ≥ 1.0), prints examples, writes `eval_results/eval_step_N.json`, and logs `eval/accuracy` to wandb. Generation: at most 512 new tokens, prompts truncated at 512 tokens, eval temperature from the config.

### Starting settings (`config/multi_tasks.yaml`)

| Group | Setting | Value |
|---|---|---|
| Data | `dataset_size` / `datasets` / `eval_datasets` | 20,000; 4 tasks, equal weight, default difficulty; the 4 + base_conversion |
| Steps | `max_steps` / `fixed_eval_steps` | 200 / 100 (evals at steps 0, 100 and 200) |
| Batch | `per_device_train_batch_size` × `gradient_accumulation_steps` | 8 × 2 = 16 answers per step = **4 prompts × 4 answers** |
| LR | `learning_rate` / schedule | 3e-6, constant with 10 warmup steps |
| GRPO | `num_generations` / `max_completion_length` / `reward_weights` | 4 / 384 / [1.0, 1.0] |
| Sampling | `temperature` / `eval_temperature` | 0.6 / 0.2 |
| Other | LoRA / bf16 / gradient checkpointing | off (full fine-tune) / on / on |

---

## 5. What we change and why

### Reward shaping (new `--reward_mode strict`, code in `grpo/trainer.py`)

| Change | Why | Source |
|---|---|---|
| **Accuracy binarised:** 1 if score ≥ 1.0, else 0 | matches exactly what eval counts; removes partial credit for "6 steps" or "16" | eval definition; reasoning-gym `dataset.py` |
| **Strict format:** credit only for one `<think>` block, then one `<answer>` block, nothing after; weight **0.2** (was 1.0, at 0.25 per tag anywhere) | the old reward pays for scattered tags (the multi-block answers above earn full format credit); a weight of 1.0 lets format alone drive learning when every answer in a group is wrong | Logic-RL (2502.14768), open-r1, Reasoning Gym paper (format 0.2) |
| **Soft length penalty:** 0 up to 256 tokens, falling linearly to -1 at 384 | keeps answers inside the eval's 512-token budget | DAPO (2503.14476) |

Upstream behaviour stays available as `--reward_mode orig` (the default), so run A reproduces the course baseline exactly. A CPU dry run checked every reward against hand-written answers.

### Hyperparameters

| Change | Why | Source |
|---|---|---|
| `num_generations` 4 → **8** (and 16 in one run) | more groups with some right and some wrong answers, so more learning signal; early logs: 25-47% "no signal" groups with 8 answers vs about 75% with 4 | Reasoning Gym config (n=8), DAPO/open-r1 (16) |
| `temperature` 0.6 → **1.0** | more exploration; 0.6 starts at low entropy and learns more slowly | Skywork-OR1 ablation, TRL/DAPO defaults |
| `learning_rate` 3e-6 → sweep **1e-6, 2e-6, 3e-6, 5e-6** | 3e-6 is aggressive for a 4-prompt batch, but the metric is the *best* eval point, so an early peak also counts | Reasoning Gym, TinyZero, open-r1 all use 1e-6 |
| batch **16 × 2** (4 prompts × 8), and **32 × 2** (8 × 8) in one run | the A100-80GB has room; larger generation batches use the GPU better | README direction 4 |
| training weights: **maze 2.0, family 1.5** (one run) | most of the headroom is there; changing weights keeps the task keys, so it's eligible | (our idea) |
| `num_iterations 2` + `epsilon_high 0.28` (one run) | reuses each batch of answers for 2 updates (cheaper per update); clip-higher only works with reuse | DAPO |

### Ideas considered but not run
- **vLLM generation:** faster, but needs a different torch version; too risky for a 3-hour budget.
- **`scale_rewards` batch/none:** mixed evidence.
- **Entropy-based token selection and GSPO:** gains reported only on 8B+ or MoE models.
- **Kimi-style length reward:** risks collapsing answers to very short ones.
- **Per-task answer-shape checks** (e.g. maze answer must be digits): unverified.
- **KL regularisation:** the README's "extended" item, more relevant to Challenge II.

---

## 6. The running plan

### Infrastructure
- **Host:** dsinlp01, **A100-80GB GPUs 0-5** (authorised 2026-10-14). One run per GPU at a time, each process killed after 4 h at most, a card taken only when empty, thread caps on every process.
- **Files:** dsinlp01's home is now a separate empty disk, so everything lives on its local disk under `/tmp/ronke21_rlfr/`:
  - `conda/`: Miniconda and the `rlfr` env (torch 2.6.0+cu124, transformers 4.57.6, trl 0.27.2, reasoning-gym)
  - `hf/`: the Qwen model cache
  - `rl-for-real/`: the code, rsynced from nlp07, branch `claude-exp`
  - `queue/gpuN.txt`: per-GPU run queue, one `name|args` line per run
  - `runs/<name>/`: outputs, including `eval_results/eval_step_*.json`
  - `logs/<name>.log`, `logs/lane_gpuN.log`, `state/timing.tsv`
- **Lane runner:** `claude_run/run_lane.sh N` runs the queue for GPU N in order. New lines can be appended or swapped while it runs.
- **wandb:** project `rl-for-real` (entity `ronke21-bar-ilan-university`). Runs that started after the login log live. The four phase-1 runs logged offline and are uploaded with `wandb sync` when they finish.

### Runs (each 200 steps; evals at steps 0, 100 and 200)

| Phase | GPU | Run | What it tests | Approx. time (dsinlp01 clock) |
|---|---|---|---|---|
| 1 | 1 | `p1_A_baseline` | course config as given (reference) | 10:53-~11:38 |
| 1 | 2 | `p1_B_strict` | strict reward, original hyperparameters | 10:53-~11:40 |
| 1 | 3 | `p1_C_strict_hp` | strict + G=8, T=1.0, lr 2e-6, batch 16×2 | 10:53-~11:47 |
| 1 | 4 | `p1_D_strict_hp_lr5e-6` | C at lr 5e-6 | 10:53-~11:32 |
| 2 | 0 | `p2_C_g16` | C with G=16 (16×4 batch) | 11:06-~12:22 (about 21 s/step) |
| 2 | 5 | `p2_C_taskw` | C with task weights maze 2.0, family 1.5 | 11:06-~12:03 |
| 2 (queued) | 1 | `p2_C_lr1e-6` | C at lr 1e-6 | after A |
| 2 (queued) | 2 | `p2_C_reuse_cliphigh` | C + `num_iterations 2`, `epsilon_high 0.28` | after B |
| 2 (queued) | 3 | `p2_C_bs32` | C with 8 prompts × 8 per step | after C |
| 2 (queued) | 4 | `p2_C_lr3e-6` | C at lr 3e-6 | after D |
| 3 | | best config, 300 steps, plus a second seed | confirm the winner, and give it more eval points | about 12:30-13:50 |

The queued phase-2 runs are fallbacks: I revise them from the step-100 evals of runs A-D (about 11:20-11:25). Phase 3 is chosen from the phase-2 results.

### Monitoring
```bash
ssh dsinlp01 'grep -E "START|END" /tmp/ronke21_rlfr/logs/lane_gpu*.log'                 # what ran / is running
ssh dsinlp01 'grep -h "Overall Accuracy" /tmp/ronke21_rlfr/logs/p*_*.log'               # eval results
ssh dsinlp01 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv'    # GPU use
```
Or on wandb: `eval/accuracy` and `eval/<task>/accuracy` per run, plus `train/rewards/_accuracy_reward/mean`, `train/frac_reward_zero_std` and `train/completions/clipped_ratio`.

### What to watch
- **family_relationships accuracy:** should jump first, as the model learns to put the word inside `<answer>` tags.
- **base_conversion:** transfer from format learning, never trained on.
- **`frac_reward_zero_std`:** lower means more learning signal.
- **`clipped_ratio`:** if it rises above about 10%, answers are getting long and the length penalty or max length matters.

### Results so far (11:50, dsinlp01 clock)

Overall eval accuracy (200 items). Per task at step 200: base_conversion / family / maze / number_filtering / string_insertion.

| Run | Step 0 | Step 100 | Step 200 | Per task at step 200 |
|---|---|---|---|---|
| **A: course config as given** | 8.5% | 21.5% | **44.5%** | 81 / 20 / 13 / 32 / 82 |
| B: strict reward | 8.5% | 16.5% | 26.0% | 19 / 0 / 10 / 20 / 85 |
| C: strict + G8, T1.0, lr 2e-6 | 8.5% | 11.5% | 29.0% | 11 / 0 / 17 / 34 / 87 |
| D: strict + G8, T1.0, lr 5e-6 | 8.5% | 26.5% | 36.0% | 53 / 12 / 3 / 25 / 90 |
| g16: strict, G16 | 8.5% | 21.0% | running | |
| taskw: strict, task weights | 8.5% | 11.5% | running | |

**Finding: the strict reward lost to the course's reward.**
- **family_relationships stays near 0% under strict.** The untrained model answers the bare word with no tags. Under strict format every answer in such a group scores 0, so GRPO gets no gradient.
- **The course reward gives credit for *some* tags.** That graded reward gives a gradient toward tags, which unlocks family (20%) and, through answer format, base_conversion (81%, never trained on).
- **Lesson:** a dense, graded shaping reward beats a strict one when the model starts far from the target format.
- **Higher lr helped** under strict: D (5e-6) beat C (2e-6).

**Changes at 11:47:**
- **Stopped** three early strict runs: lr 1e-6, batch 32, lr 3e-6.
- **Started 300-step runs** on the course reward, which give evals at steps 100, 200 and 300 (A was still climbing at 200).
- **New reward mode `hybrid`:** accuracy counted only at score ≥ 1.0, plus the course's per-tag format reward.

| GPU | Run | What it tests |
|---|---|---|
| 1 | `p3_A_300` | A, 300 steps |
| 3 | `p3_A_lr5e-6_300` | A at lr 5e-6 |
| 4 | `p3_hyb_300` | hybrid reward, A hyperparameters |
| 5 (after taskw) | `p3_A_G8T1_300` | A + G8, T1.0 |
| 2 (after reuse) | `p3_hyb_lr5e-6_300` | hybrid at lr 5e-6 |
| 0 (after g16) | `p3_A_seed43_300` | A, second seed |

### Round 4: the README's remaining ideas, on the winning course reward (queued 11:58)

| GPU | Starts (approx.) | Run | README item | Setting |
|---|---|---|---|---|
| 2 | 12:03 | `p4_A_klglobal_300` | global KL | `--beta 0.04` (reference = the original model) |
| 0 | 12:20 | `p4_A_kllocal_300` | local KL | `--beta 0.04 --sync_ref_model true --ref_model_sync_steps 10 --ref_model_mixup_alpha 1.0`: the reference is reset to the current policy every 10 steps (TR-DPO style), so the penalty limits recent drift, not total drift. Syncing every step would make the KL term vanish. |
| 1 | 13:00 | `p4_A_noGC_200` | GPU utilization / faster cycles | `--gradient_checkpointing false` (25-45 GB of the 80 GB was free) |
| 3 | 13:00 | `p4_A_reuse_opm_200` | off-policy mask | `--num_iterations 2 --off_policy_mask_threshold 0.01` (the mask only acts when samples are reused; without vLLM it compares against the pre-update log-probs) |
| 4 | 13:05 | `p4_A_lora_200` | LoRA | `--use_peft true` (r 256 from the config) `--learning_rate 3e-5` (LoRA needs about 10× the full fine-tune rate) |
| 5 | 13:10 | `p4_A_fmt0.5_200` | reward weights | `--reward_weights 1.0 0.5` |

Not attempted: actor-critic (needs a value head and a new loss; not feasible in the time left).
Each queue ends with a backup run (run A with seed 50-55), so a lane is never idle if a new-feature run crashes at start.

### Round 5: task sampling weights tied to per-task accuracy (queued 12:15, start ~13:00)

The 4 training tasks and the eval are unchanged; only how often each training task is sampled changes. Base = course config, 200 steps.
Static weights come from `p1_A_baseline`'s step-200 accuracy (family 20%, maze 13%, number 32%, string 82%).

| GPU | Run | Weighting | number / string / family / maze |
|---|---|---|---|
| 1 | `p7_w_inv_200` | static, opposite to accuracy: 1 − acc | 0.68 / 0.18 / 0.80 / 0.87 |
| 3 | `p7_w_corr_200` | static, with accuracy: acc | 0.32 / 0.82 / 0.20 / 0.13 |
| 4 | `p7_w_dyninv_200` | adaptive, 1 − recent training accuracy | starts uniform |
| 5 | `p7_w_dynlearn_200` | adaptive, learnability: 4·acc·(1 − acc) | starts uniform |

How the adaptive runs work (`--task_weighting dyn_inv|dyn_learn`, code in `grpo/trainer.py`):
- **What the trainer tracks:** each task's training accuracy (score ≥ 1.0) over the last 10 reward calls, which is about 10 steps.
- **How it sets the weights:** each update averages the new weight 50/50 with the old one, with a floor of 0.1, and writes the result into the composite dataset's `weights`. The dataset reads `weights` on every item, so the mix changes from the next batch on.
- **Groups stay intact:** all answers to one prompt are fetched together, so they always see the same task.
- **Logged:** the weights and per-task training accuracy go to wandb as `train/task_weight/<task>` and `train/task_acc/<task>`.

Why it might help: GRPO learns nothing from a group where every answer gets the same reward. `dyn_learn` sends more prompts to tasks near 50%, where groups disagree most. `dyn_inv` and the static inverse weights push training toward the weak tasks (maze, family) that hold most of the remaining eval headroom.

### Round 6: larger learning rates, plus KL monitoring on every new run (queued 12:30)

`p3_A_lr5e-6_300` (course config, lr 5e-6) led at step 100 with 31.5%, so we push the rate further.

| GPU | Starts ~ | Run | lr | Steps |
|---|---|---|---|---|
| 2 | 13:15 | `p8_A_lr1e-5_300` | 1e-5 | 300 |
| 0 | 13:45 | `p8_A_lr7e-6_300` | 7e-6 | 300 |
| 1 | 13:50 | `p8_A_lr2e-5_200` | 2e-5 | 200 (fits the lane's 4 h limit; replaces `p4_A_noGC_200`) |

**KL monitoring.** TRL computes the KL to the original model only when `beta > 0`. Every run that hadn't started by 12:30 gets `--beta 1e-9`:
- This loads Qwen2.5-3B-Instruct as a frozen reference and logs `train/kl`, the mean per-token KL (k3 estimator) between the current policy and the original model.
- A penalty of 1e-9 has no measurable effect on training. The cost is one extra forward pass per batch, about 10-15% slower.

Reference values, from the global-KL run (`beta 0.04`):
- KL rises from 0.002 at step 10 to 0.037 at step 130, while training accuracy rises.
- The local-KL run, whose reference resets every 10 steps, stays around 0.004.
- A run whose KL climbs far faster (e.g. above 0.2) with falling eval accuracy is drifting too far. This is most likely at lr 2e-5.

## 7. Final evaluation (course advisor's Track-1 rules, received 12:40)

Rules:
- **No extra training data from other benchmarks.** Variations of the existing tasks are allowed. We comply: always the same 4 tasks; only sampling weights vary.
- **No knowledge from larger models.** Copies of the same model are allowed. We comply: the only other model loaded is a frozen copy of Qwen2.5-3B-Instruct, used as the KL reference.
- **More compute is allowed.**
- **The final evaluation uses eval seed 62, set in code** (`eval_data = reasoning_gym.create_dataset("composite", seed=62, ...)` in `grpo/dataset.py`), and **`eval_num_samples: 200` set in `multi_tasks.yaml`**.

Protocol:
- **Dev set:** experiments use eval seed 2 (`/tmp/ronke21_rlfr/rl-for-real`), so all experiment results stay comparable.
- **Test set:** the chosen config is re-trained from scratch on the final tree, `/tmp/ronke21_rlfr/rl-for-real-final`, built by `claude_run/make_final_tree.sh`. It differs from the experiment code in exactly those two lines.
- **Why this order:** we choose on seed 2 and report on seed 62, so the final number isn't biased by our choice.

Final runs (`claude_run/run_lane_final.sh`; queues `queue/final_gpu{3,4,5}.txt`; each waits for its GPU's last experiment run, then takes over the GPU):

| GPU | Run | Config | Starts ~ |
|---|---|---|---|
| 3 | `final_A_lr5e-6_300_s42` | course config, lr 5e-6, 300 steps, seed 42, eval batch 16 (repo default), KL monitor | 13:55 |
| 4 | `final_A_lr5e-6_300_s43` | same, seed 43 | 13:55 |
| 5 | `final_A_lr5e-6_300_s44` | same, seed 44 | 13:55 |

The config will be revised at ~13:00 if lr 5e-6's step-300 result or the larger-lr runs point elsewhere. The final score is the highest `eval/accuracy` across these runs' evals on seed 62.

## 8. Results update (13:00, dev set = eval seed 2)

### The course baseline varies a lot between runs
Same config (course YAML + `eval_num_samples 200`), same seed 42:

| Run | Step 100 | Step 200 |
|---|---|---|
| `p1_A_baseline` (eval batch 64) | 21.5% | 44.5% |
| `p6_A_repro_evalbs64` (exact rerun) | 21.5% | 41.5% |
| `p3_A_300` (same, `max_steps 300`) | 25.5% | 34.0% |
| `p6_A_repro_evalbs16` (eval batch 16) | 16.0% | 21.5% |
| re-eval of `p1_A_baseline`'s step-200 model at eval batch 16, seeds 42 / 43 | | 46.0% / 44.5% |

- **The eval is stable.** Re-evaluating one model moves the score by about 1.5 points.
- **Training is not stable.** The eval at step 0 draws random numbers. A different eval batch size changes how many it draws, so training samples different answers from then on: in effect a different seed. Changing `max_steps` also changes the run.
- **Range:** at step 200 the same config scores **21.5-44.5%**, a mean of about 35%. The repo owner's "~30%" is a typical outcome; our first 44.5% was a lucky draw.
- **Implications:** compare configs across several runs, not one; and for the final score, running several seeds of the best config and taking the best is worth the compute.

### All finished evals so far
Per task is base / family / maze / number / string.

| Run | Step 100 | Step 200 | Step 300 | Per task at the best eval |
|---|---|---|---|---|
| **`p3_A_lr5e-6_300`** (course, lr 5e-6) | 31.5% | **49.0%** | 41.0% | 64 / 45 / 10 / 32 / 90 |
| `p1_A_baseline` (course) | 21.5% | 44.5% | | 81 / 20 / 13 / 32 / 82 |
| `p6_A_repro_evalbs64` (course rerun) | 21.5% | 41.5% | | 72 / 18 / 3 / 27 / 90 |
| `p1_D_strict_hp_lr5e-6` | 26.5% | 36.0% | | 53 / 12 / 3 / 25 / 90 |
| `p3_A_G8T1_300` (course + G8, T1.0) | 15.0% | 35.0% | running | 47 / 14 / 7 / 25 / 85 |
| `p3_A_300` (course, 300 steps) | 25.5% | 34.0% | running | 33 / 14 / 13 / 25 / 87 |
| `p1_C_strict_hp` | 11.5% | 29.0% | | |
| `p3_hyb_300` (hybrid reward) | 18.5% | 24.0% | 28.5% | 11 / 16 / 3 / 23 / 87 |
| `p4_A_klglobal_300` (beta 0.04) | 18.0% | 28.5% | running | 36 / 0 / 3 / 18 / 90 |
| `p2_C_g16` | 21.0% | 28.0% | | |
| `p1_B_strict` | 16.5% | 26.0% | | |
| `p2_C_reuse_cliphigh` | 11.0% | 25.5% | | |
| `p6_A_repro_evalbs16` (course rerun) | 16.0% | 21.5% | | |
| `p2_C_taskw` | 11.5% | 17.0% | | |
| `p4_A_kllocal_300` (local KL) | 16.5% | running | | |

- **lr 5e-6 holds the best dev score: 49.0% at step 200.** It's the only run where family_relationships learned well (45%). It fell to 41.0% at step 300, mostly because base_conversion dropped from 64% to 11%. Because the metric is the best eval at any point, 300 steps still helps: it gives three chances.
- **Final runs stay on the course config at lr 5e-6** for 300 steps, seeds 42/43/44 on eval seed 62. GPUs 6-7 were released at 13:00:53.
