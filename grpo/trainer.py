"""Custom GRPO Trainer with reasoning-gym reward functions."""

import ast
import difflib
import re

import torch
from trl import GRPOConfig, GRPOTrainer

from reasoning_gym.utils import extract_answer

from .dataset import ReasoningGymDataset

# Try to import wandb for logging (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def _closeness(task: str, answer, gold) -> float:
    """Similarity in [0, 1] between a wrong answer and the gold answer, used by reward_mode="partial"."""
    if answer is None:
        return 0.0
    a, g = answer.strip(), str(gold).strip()
    if task == "maze":
        try:
            return 1.0 / (1.0 + abs(int(a) - int(g)))
        except ValueError:
            return 0.0
    if task == "number_filtering":
        try:
            pa = {str(x) for x in ast.literal_eval(a)}
            pg = {str(x) for x in ast.literal_eval(g)}
        except (ValueError, SyntaxError, TypeError):
            return 0.0
        if not pa and not pg:
            return 1.0
        return 2 * len(pa & pg) / (len(pa) + len(pg))
    if task in ("string_insertion", "base_conversion"):
        return difflib.SequenceMatcher(None, a, g).ratio()
    return 0.0  # family_relationships: exact only


class CustomGRPOTrainer(GRPOTrainer):
    """
    Extended GRPOTrainer with reasoning-gym reward functions.
    
    Includes:
        - Accuracy reward based on reasoning-gym's scoring
        - Format reward for proper <think>/<answer> tags
        - Optional reward variance logging
    """
    
    def __init__(
        self,
        model,
        args: GRPOConfig,
        tokenizer,
        train_dataset: ReasoningGymDataset,
        eval_dataset: ReasoningGymDataset,
        log_reward_variance: bool = True,
        verbose_variance_logging: bool = False,
        batch_counts_as_step: bool = False,
        reward_mode: str = "orig",
        overlong_start: int = 256,
        task_weighting: str = "static",
        task_weight_every: int = 10,
        task_weight_floor: float = 0.1,
    ):
        if reward_mode not in ("orig", "strict", "hybrid", "partial"):
            raise ValueError(f"Unknown reward_mode: {reward_mode}")
        if task_weighting not in ("static", "dyn_inv", "dyn_learn"):
            raise ValueError(f"Unknown task_weighting: {task_weighting}")
        self.task_weighting = task_weighting
        self.task_weight_every = task_weight_every
        self.task_weight_floor = task_weight_floor
        self._task_counts = {}  # task -> [correct, total] since the last reweighting
        self._reward_calls = 0
        self.reward_mode = reward_mode
        self.overlong_start = overlong_start
        reward_funcs = [self._accuracy_reward, self._format_reward]
        if reward_mode == "strict":
            reward_funcs.append(self._length_reward)
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        self.log_reward_variance = log_reward_variance
        self.verbose_variance_logging = verbose_variance_logging
        self.batch_counts_as_step = batch_counts_as_step
        self._step_count = 0
    
    def log(self, logs: dict, start_time: float = None) -> None:
        """Override log to manually log to wandb with correct x-axis."""
        # First, let parent class merge _metrics into logs
        # (GRPOTrainer.log merges self._metrics[mode] into logs before calling super().log)
        mode = "train" if self.model.training else "eval"
        if hasattr(self, '_metrics') and mode in self._metrics:
            metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items() if val}
            if mode == "eval":
                metrics = {f"eval_{key}": val for key, val in metrics.items()}
            logs = {**logs, **metrics}
        
        # Now log to wandb with our custom step
        if self.state is not None and WANDB_AVAILABLE and wandb.run is not None:
            # Determine which step metric to use
            if self.batch_counts_as_step:
                step_value = self.state.global_step * self.args.gradient_accumulation_steps
            else:
                step_value = self.state.global_step
            
            try:
                wandb_logs = {}
                
                # Add all train metrics
                for key, value in logs.items():
                    # Convert string numbers to float (HuggingFace logs some metrics as strings)
                    if isinstance(value, str):
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            continue  # Skip non-numeric strings
                    
                    if isinstance(value, (int, float)):
                        # Prefix with train/ if not already prefixed
                        if not key.startswith("train/") and not key.startswith("eval/"):
                            wandb_logs[f"train/{key}"] = value
                        else:
                            wandb_logs[key] = value
                
                # Use step parameter to force wandb to use our step value as the x-axis
                if wandb_logs:
                    wandb.log(wandb_logs, step=step_value)
            except Exception as e:
                # Log error but don't crash training
                print(f"Warning: Failed to log to wandb at step {step_value}: {e}")
        
        # Clear metrics before calling parent (to avoid double logging in parent's log method)
        if hasattr(self, '_metrics') and mode in self._metrics:
            self._metrics[mode].clear()
        
        # Call grandparent's log (skip GRPOTrainer.log since we already handled metrics)
        from transformers import Trainer
        Trainer.log(self, logs, start_time)

    def _accuracy_reward(self, completions: list[str], **kwargs) -> list[float]:
        """Compute accuracy reward using reasoning-gym's scoring function."""
        assert "item" in kwargs, "The 'item' argument must be provided."
        assert len(kwargs["item"]) == len(completions)
        
        answers = [extract_answer(c) for c in completions]
        rewards = [
            self.train_dataset.data.score_answer(answer, item)
            for answer, item in zip(answers, kwargs["item"])
        ]
        if self.task_weighting != "static":
            self._update_task_weights(rewards, kwargs["item"])
        if self.reward_mode in ("strict", "hybrid"):
            # Eval counts a sample only at score >= 1.0, so partial credit rewards answers eval marks wrong
            rewards = [1.0 if r >= 1.0 else 0.0 for r in rewards]
        elif self.reward_mode == "partial":
            # Exact answers get 1.0; wrong ones get up to 0.5 for being close (task-specific similarity)
            rewards = [1.0 if r >= 1.0 else 0.5 * _closeness(item["metadata"]["source_dataset"], a, item["answer"])
                       for r, a, item in zip(rewards, answers, kwargs["item"])]

        if self.log_reward_variance:
            self._log_variance(rewards, answers, completions)
        
        return rewards
    
    def _update_task_weights(self, rewards: list[float], items: list[dict]):
        """Adaptive task sampling: every task_weight_every reward calls, reset the composite dataset's
        sampling weights from each task's recent training accuracy (score >= 1.0).
        dyn_inv:   w = 1 - acc            (more weight on tasks the model still fails)
        dyn_learn: w = acc * (1 - acc)    (more weight where GRPO groups are most likely to disagree)
        New weights are averaged 50/50 with the old ones and floored at task_weight_floor."""
        for r, item in zip(rewards, items):
            c = self._task_counts.setdefault(item["metadata"]["source_dataset"], [0, 0])
            c[0] += r >= 1.0
            c[1] += 1
        self._reward_calls += 1
        if self._reward_calls % self.task_weight_every:
            return
        composite = self.train_dataset.data
        new = []
        for name, old in zip(composite.dataset_names, composite.weights):
            correct, total = self._task_counts.get(name, [0, 0])
            if total == 0:
                new.append(old)
                continue
            acc = correct / total
            w = 1.0 - acc if self.task_weighting == "dyn_inv" else acc * (1.0 - acc) * 4.0
            new.append(max(self.task_weight_floor, 0.5 * old + 0.5 * w))
            self._metrics["train"][f"task_acc/{name}"].append(acc)
        composite.weights = new
        for name, w in zip(composite.dataset_names, new):
            self._metrics["train"][f"task_weight/{name}"].append(w)
        self._task_counts = {}

    def _log_variance(self, rewards: list[float], answers: list[str], completions: list[str]):
        """Log the percentage of prompts with reward variance across rollouts."""
        num_gens = self.args.num_generations
        num_prompts = len(rewards) // num_gens
        
        if num_prompts == 0:
            return
        
        prompts_with_reward_variance = 0
        prompts_with_answer_variance = 0
        
        for i in range(num_prompts):
            start_idx = i * num_gens
            end_idx = (i + 1) * num_gens
            group_rewards = rewards[start_idx:end_idx]
            group_answers = answers[start_idx:end_idx]
            
            has_reward_variance = len(set(group_rewards)) > 1
            has_answer_variance = len(set(group_answers)) > 1
            
            if has_reward_variance:
                prompts_with_reward_variance += 1
            if has_answer_variance:
                prompts_with_answer_variance += 1
            
            # Verbose: print full completions when there's variance
            if self.verbose_variance_logging and has_answer_variance:
                group_completions = completions[start_idx:end_idx]
                print(f"\n  {'─'*60}")
                print(f"  📊 Prompt {i}: rewards={group_rewards}")
                for gen_idx, (comp, rew, ans) in enumerate(zip(group_completions, group_rewards, group_answers)):
                    comp_preview = comp[:300].replace('\n', ' ↵ ')
                    if len(comp) > 300:
                        comp_preview += "..."
                    print(f"    Gen {gen_idx} [reward={rew}]: {comp_preview}")
                    print(f"           extracted: {ans}")
                print(f"  {'─'*60}")
        
        reward_variance_pct = (prompts_with_reward_variance / num_prompts * 100)
        answer_variance_pct = (prompts_with_answer_variance / num_prompts * 100)
        
        self._step_count += 1
        
        if self.verbose_variance_logging:
            print(f"🎲 Step {self._step_count}: {reward_variance_pct:.0f}% prompts have reward variance, "
                  f"{answer_variance_pct:.0f}% have answer variance ({num_prompts} prompts × {num_gens} gens)")
        
        # Log to wandb
        if WANDB_AVAILABLE and wandb.run is not None:
            try:
                wandb.log({
                    "train/reward_variance_pct": reward_variance_pct,
                    "train/answer_variance_pct": answer_variance_pct,
                }, commit=False)
            except:
                pass
        
        # Periodic GPU cleanup
        if self._step_count % 10 == 0:
            torch.cuda.empty_cache()

    # One <think> block, then one <answer> block, nothing after; no nested or repeated tags
    _STRICT_FORMAT = re.compile(
        r"\s*<think>(?:(?!</?(?:think|answer)>).)*</think>\s*<answer>(?:(?!</?(?:think|answer)>).)+</answer>\s*",
        re.DOTALL,
    )

    def _length_reward(self, completions: list[str], completion_ids=None, **kwargs) -> list[float]:
        """Soft overlong penalty (DAPO): 0 up to overlong_start tokens, linear down to -1 at max_completion_length."""
        max_len = self.args.max_completion_length
        span = max(max_len - self.overlong_start, 1)
        lengths = [len(ids) for ids in completion_ids]
        return [max(-1.0, min(0.0, (self.overlong_start - n) / span)) for n in lengths]

    def _format_reward(self, completions: list[str], **kwargs) -> list[float]:
        """Reward proper formatting with <think> and <answer> tags."""
        if self.reward_mode == "strict":
            return [1.0 if self._STRICT_FORMAT.fullmatch(c) else 0.0 for c in completions]

        def count_tags(text: str) -> float:
            count = 0.0
            if re.search(r"\s*<think>\s*", text):
                count += 0.25
            if re.search(r"\s*</think>\s*", text):
                count += 0.25
            if re.search(r"\s*<answer>\s*", text):
                count += 0.25
            if re.search(r"\s*</answer>\s*", text):
                count += 0.25
            return count

        return [count_tags(c) for c in completions]
