"""Shared helpers for the extension experiments (no changes to main.py needed)."""
import json, os, random, sys, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import main            # noqa: E402
import rl_utils        # noqa: E402

RESULTS = os.path.join(ROOT, "claude_run", "results")
os.makedirs(RESULTS, exist_ok=True)
main.CHECKPOINT_DIR = os.path.join(RESULTS, "ckpt")   # keep main.py's own checkpoints/ untouched

if os.environ.get("DRYRUN"):                          # tiny smoke-test settings
    main.NUM_ITERATIONS, main.NUM_STEPS = 2, 256


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    main.make_env = lambda env_id, _s=s: rl_utils.make_env(env_id, seed=_s)


def summarize(curve, k=10):
    c = np.asarray(curve, dtype=float)
    return {"final_mean_last10": float(c[-k:].mean()), "best_iter_return": float(c.max()),
            "first_iter_mean_ge_200": next((i for i, v in enumerate(c) if v >= 200), None)}


def dump(name, obj):
    path = os.path.join(RESULTS, name + ("_dryrun" if os.environ.get("DRYRUN") else "") + ".json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print("wrote", path, flush=True)
