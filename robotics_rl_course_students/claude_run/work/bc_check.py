"""claude_run helper: verify the BC split is trajectory-disjoint and print the fit summary."""
import torch
b = torch.load("bc_policy.pt", weights_only=False)
d = torch.load("bc_data.pt", weights_only=False)
print("bc_policy keys:", list(b.keys()))
ids = d["episode_ids"]
print("samples", ids.numel(), "trajectories", torch.unique(ids).numel(), "obs keys", {k: tuple(v.shape) for k, v in d["obs"].items()}, "actions", tuple(d["actions"].shape))
for k, v in b.items():
    if "traj" in k.lower() or "episode" in k.lower() or "_ids" in k.lower():
        print(k, (len(v) if hasattr(v, "__len__") else v))
t = {int(x) for x in b["training_trajectory_ids"]}; v = {int(x) for x in b["validation_trajectory_ids"]}
print("train trajectories", len(t), "validation trajectories", len(v), "overlap", len(t & v))
h = b.get("history")
if h:
    print("history first", h[0]); print("history last", h[-1])
for k in ("constant_action_validation_mse", "baseline_mse", "validation_mse"):
    if k in b: print(k, b[k])
