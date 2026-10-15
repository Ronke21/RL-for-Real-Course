"""Collect claude_run/work/results/*.json into markdown tables (results/tables.md). Pure json, runs anywhere."""
import json
from pathlib import Path

W = Path(__file__).resolve().parent / "work"
R = W / "results"


def load(rel):
    p = R / rel
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get("summary", d), d


def row(label, rel):
    got = load(rel)
    if got is None:
        return f"| {label} | (missing: {rel}) | | | | | |"
    s, d = got
    lo, hi = s["survival_wilson_95_interval"]
    eps = d.get("episodes") or []
    delta = s.get("mean_squared_action_delta")
    if delta is None and eps and "mean_squared_action_delta" in eps[0]:
        delta = sum(e["mean_squared_action_delta"] for e in eps) / len(eps)
    delta = f"{delta:.4f}" if delta is not None else "n/a"
    return (f"| {label} | {s['survival_success_rate']*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}] | "
            f"{s['episode_count']} | {s['failures']} / {s['censored']} | {s['mean_error_before_failure']:.4f} | "
            f"{s['mean_survival_steps']:.1f} | {delta} |")


HEAD = ("| Policy / condition | Full-clip survival [95% Wilson] | Episodes | Failures / censored | "
        "Mean body-pos error (m) | Mean survival steps | Mean sq. action delta |\n|---|---|---|---|---|---|---|")

TABLES = {
    "Install check": [("setup_smoke reload (2 epochs, 16 eps)", "setup_smoke/reload_check.json")],
    "M0: velocity frame (frozen walker, 500 eps per cell)": [
        (f"{v} @ {h} deg", f"m0_frames/{v}_heading{h}.json") for h in (0, 90, 180) for v in ("broken", "student")],
    "M1: PPO fine-tuning of the weak epoch-70 tracker (clean, seed 0)": [
        ("weak starter (epoch 70)", "m1_baseline.json"), ("m1_control (my fine-tune)", "m1_control.json")],
    "M1 optional: action-rate x10 ablation (10 min from supplied expert)": [
        ("m1_reward_control (weight x1)", "m1_reward_control.json"), ("m1_ablation (weight x10)", "m1_ablation.json")],
    "M2: behavior cloning vs the supplied expert (seed 0)": [
        ("expert, clean", "m2_expert_clean.json"), ("BC student, clean", "m2_bc_clean.json"),
        ("expert, perturbed", "m2_expert_perturbed.json"), ("BC student, perturbed", "m2_bc_perturbed.json")],
    "M3: domain randomization (seed 1)": [
        ("before (M3 start ckpt), clean", "m3_before_clean.json"),
        ("equal-budget CLEAN fine-tune, clean", "m3_cleanctl_clean.json"),
        ("after DR fine-tune (m3_g1), clean", "m3_after_clean.json"),
        ("before (M3 start ckpt), perturbed", "m3_before_perturbed.json"),
        ("equal-budget CLEAN fine-tune, perturbed", "m3_cleanctl_perturbed.json"),
        ("after DR fine-tune (m3_g1), perturbed", "m3_after_perturbed.json")],
}

out = []
for title, rows in TABLES.items():
    out += [f"### {title}", "", HEAD] + [row(*r) for r in rows] + [""]
start = R / "m3_start_checkpoint.txt"
if start.exists():
    out.append(f"M3 start checkpoint: `{start.read_text().strip()}`\n")
text = "\n".join(out)
(R / "tables.md").write_text(text)
print(text)
