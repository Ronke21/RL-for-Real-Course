"""claude_run helper: list TensorBoard scalars of training runs and plot key curves to results/curves_<run>.png."""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = ["info/episode_length", "info/episode_reward", "info/gframes", "env/termination/tracking_error_mean",
        "env/raw_r/global_anchor_pos_mean", "env/raw_r/relative_body_pos_mean", "env/raw_r/body_lin_vel_mean", "env/raw_r/action_rate_mean",
        "actor/approx_kl", "actor/clip_frac", "losses/actor_loss", "losses/critic_loss",
        "critic/explained_variance", "actor/std_mean", "eval/success_rate", "times/training_minutes"]
for run in sys.argv[1:]:
    ev = sorted(Path(f"results/{run}/lightning_logs").rglob("events.out.tfevents*"))
    if not ev:
        print(run, "no events"); continue
    acc = EventAccumulator(str(ev[-1].parent), size_guidance={"scalars": 0}); acc.Reload()
    tags = acc.Tags()["scalars"]
    print(f"== {run}: {len(tags)} scalar tags")
    pick = [t for t in TAGS if t in tags]
    for t in pick:
        s = acc.Scalars(t)
        print(f"  {t}: first={s[0].value:.4g} last={s[-1].value:.4g} n={len(s)} last_step={s[-1].step}")
    if pick:
        n = len(pick); cols = 4; rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows), squeeze=False)
        for ax, t in zip(axes.flat, pick):
            s = acc.Scalars(t); ax.plot([x.step for x in s], [x.value for x in s]); ax.set_title(t, fontsize=8)
        for ax in list(axes.flat)[n:]: ax.axis("off")
        fig.suptitle(run); fig.tight_layout(); fig.savefig(f"results/curves_{run}.png", dpi=90); plt.close(fig)
