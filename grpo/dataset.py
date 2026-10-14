"""Dataset wrapper and preparation for reasoning-gym."""

from typing import Optional

from torch.utils.data import Dataset

import reasoning_gym
from reasoning_gym.composite import DatasetSpec
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import SYSTEM_PROMPTS

from .config import DatasetConfig

# Prompt-engineering variants of reasoning-gym's DeepSeekZero system prompt (select with developer_prompt=<name>).
# They target the untrained model's two main failures: bare-word answers without <answer> tags (family
# relationships) and answering before reasoning, then re-answering in several blocks (only the last counts).
_FORMAT_RULES = """
Rules: reason first inside <think> </think> and do not guess the answer early. Then write exactly one <answer> </answer> block containing only the final answer, in the exact format the question asks for (also for one-word answers), and nothing after it.
"""
_FORMAT_EXAMPLES = """
Answer examples: <answer>['-1.5', '23']</answer> (list in the question's format), <answer>7</answer>, <answer>1a3f</answer> (lowercase digits above 9), <answer>grandmother</answer>, <answer>ABCDAEB</answer>
"""
_CAREFUL = """
Check every element, rule or step one at a time, and verify the result before answering.
"""
CUSTOM_PROMPTS = {
    "pe_format": SYSTEM_PROMPTS["DeepSeekZero"] + _FORMAT_RULES,
    "pe_full": SYSTEM_PROMPTS["DeepSeekZero"] + _FORMAT_RULES + _FORMAT_EXAMPLES,
    "pe_careful": SYSTEM_PROMPTS["DeepSeekZero"] + _FORMAT_RULES + _CAREFUL,
}


class ReasoningGymDataset(Dataset):
    """
    Dataset wrapper for reasoning-gym procedural datasets.
    
    Formats prompts with chat templates for use with GRPOTrainer.
    """
    
    def __init__(
        self,
        tokenizer,
        procedural_dataset: ProceduralDataset,
        developer_prompt: Optional[str] = None,
        developer_role: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.data = procedural_dataset
        self.developer_prompt = developer_prompt
        self.developer_role = developer_role

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        question = item["question"]

        chat = []
        if self.developer_role is not None:
            chat.append({"role": self.developer_role, "content": self.developer_prompt})
        chat.append({"role": "user", "content": question})

        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "item": item}


def prepare_datasets(
    config: DatasetConfig,
    tokenizer,
) -> tuple[ReasoningGymDataset, ReasoningGymDataset]:
    """Prepare training and evaluation datasets from config."""
    developer_prompt = CUSTOM_PROMPTS.get(config.developer_prompt) or SYSTEM_PROMPTS[config.developer_prompt]

    # Training dataset specs
    train_dataset_specs = [
        DatasetSpec(
            name=name,
            weight=ds_config.weight,
            config=ds_config.config,
        )
        for name, ds_config in config.datasets.items()
    ]
    
    # Eval dataset specs: use eval_datasets if provided, otherwise use same as training
    if config.eval_datasets:
        eval_dataset_specs = [
            DatasetSpec(
                name=name,
                weight=ds_config.weight,
                config=ds_config.config,
            )
            for name, ds_config in config.eval_datasets.items()
        ]
    else:
        eval_dataset_specs = train_dataset_specs
    
    train_data = reasoning_gym.create_dataset(
        "composite", seed=1, size=config.dataset_size, datasets=train_dataset_specs
    )
    eval_data = reasoning_gym.create_dataset(
        "composite", seed=62, size=config.eval_num_samples, datasets=eval_dataset_specs
    )
    
    train_dataset = ReasoningGymDataset(
        tokenizer=tokenizer,
        procedural_dataset=train_data,
        developer_prompt=developer_prompt,
        developer_role=config.developer_role,
    )
    eval_dataset = ReasoningGymDataset(
        tokenizer=tokenizer,
        procedural_dataset=eval_data,
        developer_prompt=developer_prompt,
        developer_role=config.developer_role,
    )
    
    return train_dataset, eval_dataset
