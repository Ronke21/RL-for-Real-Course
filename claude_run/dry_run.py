"""CPU dry run (no model weights): datasets, prompt lengths, strict rewards, CLI overrides for every queued run."""
import shlex
import sys
from types import SimpleNamespace

sys.argv = [sys.argv[0]]
from transformers import AutoTokenizer
from trl import GRPOConfig, ModelConfig, TrlParser

import grpo
from grpo.trainer import CustomGRPOTrainer

import os
R = "/tmp/ronke21_rlfr"
QFILES = os.environ.get("QFILES", " ".join(f"gpu{g}" for g in range(8))).split()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# 1. Every queued arg string parses, reward_weights length matches the reward mode
for qf in QFILES:
    if not os.path.exists(f"{R}/queue/{qf}.txt"):
        continue
    for line in open(f"{R}/queue/{qf}.txt"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, args = line.split("|", 1)
        argv = ["--config", "config/multi_tasks.yaml", *args.split(), "--output_dir", "/tmp/x", "--run_name", name, "--bf16", "false"]
        rg, tr, md = TrlParser((grpo.DatasetConfig, GRPOConfig, ModelConfig)).parse_args_and_config(argv)
        assert md.model_name_or_path == "Qwen/Qwen2.5-3B-Instruct", (name, md.model_name_or_path)
        n_funcs = 3 if rg.reward_mode == "strict" else 2
        assert len(tr.reward_weights) == n_funcs, (name, tr.reward_weights)
        assert rg.eval_num_samples == 200 and set(rg.eval_datasets) == {
            "number_filtering", "string_insertion", "family_relationships", "maze", "base_conversion"}
        assert set(rg.datasets) == {"number_filtering", "string_insertion", "family_relationships", "maze"}
        gen_batch = tr.per_device_train_batch_size * tr.gradient_accumulation_steps
        assert gen_batch % tr.num_generations == 0, (name, gen_batch, tr.num_generations)
        print(f"OK {name}: mode={rg.reward_mode} w={tr.reward_weights} lr={tr.learning_rate} G={tr.num_generations} "
              f"T={tr.temperature} bs={tr.per_device_train_batch_size}x{tr.gradient_accumulation_steps} "
              f"maxlen={tr.max_completion_length} weights={ {k: v.weight for k, v in rg.datasets.items()} }")

# 2. Datasets and prompt lengths (eval truncates prompts at 512 tokens)
rg.eval_datasets = rg.eval_datasets
train, ev = grpo.prepare_datasets(rg, tok)
lens = {}
for i in range(len(ev)):
    it = ev[i]
    t = it["item"]["metadata"]["source_dataset"]
    lens.setdefault(t, []).append(len(tok(it["prompt"])["input_ids"]))
for t, L in sorted(lens.items()):
    print(f"eval {t}: n={len(L)} prompt tokens max={max(L)}")
assert max(max(L) for L in lens.values()) < 512

# 3. Strict rewards on hand-written completions against real items
fake = SimpleNamespace(reward_mode="strict", overlong_start=256, log_reward_variance=False, task_weighting="static",
                       args=SimpleNamespace(max_completion_length=384), train_dataset=train,
                       _STRICT_FORMAT=CustomGRPOTrainer._STRICT_FORMAT)
item = next(train[i]["item"] for i in range(50) if train[i]["item"]["metadata"]["source_dataset"] == "maze")
gold = item["answer"]
cases = {
    "correct": f"<think>count steps</think>\n<answer>{gold}</answer>",
    "verbose (partial credit upstream)": f"<think>x</think><answer>{gold} steps</answer>",
    "no tags": gold,
    "text after answer": f"<think>x</think><answer>{gold}</answer> extra",
    "two answers": f"<think>x</think><answer>1</answer><answer>{gold}</answer>",
}
acc = CustomGRPOTrainer._accuracy_reward(fake, list(cases.values()), item=[item] * len(cases))
fmt = CustomGRPOTrainer._format_reward(fake, list(cases.values()))
fake.reward_mode = "orig"
acc_orig = CustomGRPOTrainer._accuracy_reward(fake, list(cases.values()), item=[item] * len(cases))
for (k, _), a, f, ao in zip(cases.items(), acc, fmt, acc_orig):
    print(f"{k:35s} strict acc={a} fmt={f} | orig acc={ao:.2f}")
assert acc == [1.0, 0.0, 0.0, 1.0, 1.0] and fmt == [1.0, 1.0, 0.0, 0.0, 0.0]
fake.reward_mode = "strict"
ln = CustomGRPOTrainer._length_reward(fake, ["a"] * 4, completion_ids=[[0] * n for n in (100, 256, 320, 384)])
print("length reward at 100/256/320/384 tokens:", ln)
assert ln == [0.0, 0.0, -0.5, -1.0]

# 4. Adaptive task weights: string solved, maze failed -> dyn_inv favours maze, dyn_learn favours the 50% task
from collections import defaultdict
def mk(name):
    return {"metadata": {"source_dataset": name}}
for mode in ("dyn_inv", "dyn_learn"):
    comp = SimpleNamespace(dataset_names=["number_filtering", "string_insertion", "family_relationships", "maze"],
                           weights=[1.0, 1.0, 1.0, 1.0])
    fk = SimpleNamespace(task_weighting=mode, task_weight_every=2, task_weight_floor=0.1, _task_counts={},
                         _reward_calls=0, train_dataset=SimpleNamespace(data=comp),
                         _metrics={"train": defaultdict(list)})
    items = [mk("string_insertion")] * 4 + [mk("maze")] * 4 + [mk("number_filtering")] * 4 + [mk("family_relationships")] * 4
    rew = [1.0] * 4 + [0.0] * 4 + [1.0, 1.0, 0.0, 0.0] + [0.0, 0.0, 0.0, 0.5]
    for _ in range(2):
        CustomGRPOTrainer._update_task_weights(fk, rew, items)
    w = dict(zip(comp.dataset_names, comp.weights))
    print(mode, {k: round(v, 2) for k, v in w.items()})
    if mode == "dyn_inv":
        assert w["maze"] > w["number_filtering"] > w["string_insertion"] and w["string_insertion"] >= 0.1
    else:
        assert w["number_filtering"] > w["maze"] and w["number_filtering"] > w["string_insertion"]

# 5. Partial reward: exact 1.0, close answers up to 0.5, far/unparseable 0
from grpo.trainer import _closeness
cl = [_closeness("maze", "6", "7"), _closeness("maze", "x", "7"),
      _closeness("number_filtering", "['1', '-2.5']", "['1', '-2.5', '3']"),
      _closeness("string_insertion", "ABCDA", "ABCDE"), _closeness("family_relationships", "mother", "father")]
print("closeness:", [round(c, 2) for c in cl])
assert cl[0] == 0.5 and cl[1] == 0.0 and abs(cl[2] - 0.8) < 1e-9 and 0.7 < cl[3] < 0.9 and cl[4] == 0.0
fake.reward_mode = "partial"
pr = CustomGRPOTrainer._accuracy_reward(fake, list(cases.values()), item=[item] * len(cases))
print("partial acc rewards:", [round(x, 2) for x in pr])
assert pr == [1.0, 0.0, 0.0, 1.0, 1.0]
print("DRY_RUN_OK")
