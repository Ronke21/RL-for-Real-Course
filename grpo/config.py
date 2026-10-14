"""Configuration dataclasses for GRPO training."""

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetConfigItem:
    """Configuration for a single dataset within the composite."""
    weight: Optional[float] = field(default=1.0)
    config: Optional[dict] = field(default_factory=dict)


def _parse_datasets(value):
    """Parse datasets from either dict or JSON string."""
    if value is None:
        return None
    if isinstance(value, str):
        # Handle JSON string from CLI (e.g., W&B sweep)
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # Try replacing single quotes with double quotes (common CLI issue)
            return json.loads(value.replace("'", '"'))
    return value


@dataclass
class DatasetConfig:
    """Configuration for datasets and evaluation."""
    dataset_size: int = field(default=10000)
    developer_prompt: str = field(default="DeepSeekZero")
    developer_role: str = field(default="system")
    datasets: Optional[str] = field(default=None)  # JSON string, e.g. '{"task": {}}'
    
    # Evaluation datasets (optional - if None, uses same config as training)
    eval_datasets: Optional[str] = field(default=None)  # JSON string, e.g. '{"task": {}}'
    
    # Fixed evaluation settings
    eval_num_samples: int = field(default=30)
    eval_num_examples_to_print: int = field(default=3)
    fixed_eval_steps: int = field(default=50)
    
    # Variance logging
    log_reward_variance: bool = field(default=True)
    verbose_variance_logging: bool = field(default=False)
    
    # Resume control
    auto_resume: bool = field(default=False)  # If True, auto-resume from last checkpoint
    
    # Eval-only mode
    eval_only: bool = field(default=False)  # If True, only run evaluation (no training)
    checkpoint_path: str = field(default=None)  # Path to checkpoint for eval-only mode
    
    # Eval generation settings
    eval_temperature: float = field(default=0.6)  # Temperature for eval generation (0 = greedy)
    
    # Step counting mode
    # If True: interpret max_steps/fixed_eval_steps as batch counts (divided by gradient_accumulation_steps)
    #          and use batch as wandb x-axis for fair comparison across different accumulation settings
    # If False: normal step-based counting
    batch_counts_as_step: bool = field(default=False)

    # Reward mode
    # "orig":   upstream rewards (raw reasoning-gym score + 0.25 per tag found anywhere)
    # "strict": accuracy binarised at score >= 1.0 (what eval counts), format credit only for one
    #           <think> block followed by one <answer> block, plus a soft overlong penalty
    #           (needs 3 reward_weights: [accuracy, format, length])
    # "hybrid": accuracy binarised as in "strict", upstream per-tag format reward, no length term
    #           (the graded tag reward is what teaches the bare-word family answers to use tags)
    reward_mode: str = field(default="orig")
    overlong_start: int = field(default=256)  # length penalty ramps from 0 here to -1 at max_completion_length

    # Task sampling weights during training ("static" = the weights in `datasets`, fixed)
    # "dyn_inv":   reweight every task_weight_every reward calls by 1 - recent training accuracy
    # "dyn_learn": reweight by acc * (1 - acc) (tasks where GRPO groups disagree most)
    task_weighting: str = field(default="static")
    task_weight_every: int = field(default=10)
    task_weight_floor: float = field(default=0.1)

    def __post_init__(self):
        # Parse datasets from JSON string if needed (for CLI/sweep support)
        self.datasets = _parse_datasets(self.datasets)
        self.eval_datasets = _parse_datasets(self.eval_datasets)
        
        # Convert training datasets
        if self.datasets:
            converted_datasets = {}
            for name, config_item in self.datasets.items():
                if isinstance(config_item, dict):
                    converted_datasets[name] = DatasetConfigItem(**config_item)
                else:
                    converted_datasets[name] = config_item
            self.datasets = converted_datasets
        
        # Convert eval datasets (if provided)
        if self.eval_datasets:
            converted_eval_datasets = {}
            for name, config_item in self.eval_datasets.items():
                if isinstance(config_item, dict):
                    converted_eval_datasets[name] = DatasetConfigItem(**config_item)
                else:
                    converted_eval_datasets[name] = config_item
            self.eval_datasets = converted_eval_datasets
