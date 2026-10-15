"""claude_run helper: report figures -> claude_run/figures/ (GIF pairs via render_rollouts, survival bar chart)."""
import json, shutil
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from course.render_rollouts import render

R = Path("results"); F = Path("figures"); F.mkdir(exist_ok=True)
ASSET = "protomotions/data/assets/mjcf/g1_bm.xml"
pairs = [("rec/m1_weak.json", "rec/m1_control.json", "weak starter (epoch 70)", "after PPO fine-tune", "m1_weak_vs_finetuned.gif"),
         ("rec/m2_expert.json", "rec/m2_bc.json", "RL expert", "BC student", "m2_expert_vs_bc.gif"),
         ("rec/m3_before_perturbed.json", "rec/m3_after_perturbed.json", "before DR (perturbed test)", "after DR fine-tune (perturbed test)", "m3_before_vs_after_perturbed.gif")]
for a, b, la, lb, out in pairs:
    try:
        render([str(R / a), str(R / b)], [la, lb], ASSET, str(F / out)); print("wrote", out)
    except ValueError as e:
        print(out, "->", e)
if (R / "m0_heading90_broken_vs_fixed.gif").exists():
    shutil.copy(R / "m0_heading90_broken_vs_fixed.gif", F / "m0_heading90_broken_vs_fixed.gif")
for p in R.glob("curves_*.png"):
    shutil.copy(p, F / p.name)

def s(f):
    d = json.loads((R / f).read_text())["summary"]; lo, hi = d["survival_wilson_95_interval"]
    p = d["survival_success_rate"]; return p * 100, max(0, p - lo) * 100, max(0, hi - p) * 100, d["mean_error_before_failure"]
groups = [("M0 frozen walker", [("broken 0°", "m0_frames/broken_heading0.json"), ("broken 90°", "m0_frames/broken_heading90.json"), ("broken 180°", "m0_frames/broken_heading180.json"),
                                ("fixed 0°", "m0_frames/student_heading0.json"), ("fixed 90°", "m0_frames/student_heading90.json"), ("fixed 180°", "m0_frames/student_heading180.json")]),
          ("M1 PPO (clean)", [("weak starter", "m1_baseline.json"), ("fine-tuned 30 min", "m1_control.json"), ("expert +2M, w×1", "m1_reward_control.json"), ("expert +2M, w×10", "m1_ablation.json")]),
          ("M2 BC", [("expert clean", "m2_expert_clean.json"), ("BC clean", "m2_bc_clean.json"), ("expert pert.", "m2_expert_perturbed.json"), ("BC pert.", "m2_bc_perturbed.json")]),
          ("M3 DR (seed 1)", [("start, clean", "m3_before_clean.json"), ("clean FT, clean", "m3_cleanctl_clean.json"), ("DR FT, clean", "m3_after_clean.json"),
                              ("start, pert.", "m3_before_perturbed.json"), ("clean FT, pert.", "m3_cleanctl_perturbed.json"), ("DR FT, pert.", "m3_after_perturbed.json")])]
fig, axes = plt.subplots(2, 4, figsize=(18, 7), gridspec_kw={"height_ratios": [2, 1]})
for col, (title, items) in enumerate(groups):
    vals = [s(f) for _, f in items]; x = range(len(items)); labels = [l for l, _ in items]
    ax = axes[0, col]
    ax.bar(x, [v[0] for v in vals], yerr=[[v[1] for v in vals], [v[2] for v in vals]], color="#4C78A8", capsize=3)
    ax.set_ylim(0, 105); ax.set_title(title); ax.set_ylabel("full-clip survival % (95% Wilson)") if col == 0 else None
    ax.set_xticks(list(x)); ax.set_xticklabels([]); [ax.text(i, v[0] + 2, f"{v[0]:.0f}", ha="center", fontsize=8) for i, v in enumerate(vals)]
    ax2 = axes[1, col]
    ax2.bar(x, [v[3] for v in vals], color="#F58518"); ax2.axhline(0.10, ls="--", c="gray", lw=1)
    ax2.set_xticks(list(x)); ax2.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax2.set_ylabel("mean body-pos error (m)") if col == 0 else None
fig.suptitle("Physical evaluation, 500 episodes per bar (dashed: guide's 0.10 m reference)"); fig.tight_layout()
fig.savefig(F / "summary_survival_error.png", dpi=100); print("wrote summary_survival_error.png")
