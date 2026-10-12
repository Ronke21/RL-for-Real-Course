"""Build all report figures from the JSON results + copy main.py plots/GIFs."""
import glob, json, os, shutil
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ROOT = os.path.dirname(os.path.dirname(R))
os.makedirs(f"{R}/figures", exist_ok=True)
for p in glob.glob(f"{ROOT}/plots/*.png") + glob.glob(f"{ROOT}/plots/*.gif"):
    shutil.copy(p, f"{R}/figures/")

def band(ax, runs, label, **kw):
    c = np.array([r["curve"] for r in runs]); m = c.mean(0); x = np.arange(1, len(m) + 1)
    ax.plot(x, m, label=label, **kw)
    if len(c) > 1: ax.fill_between(x, c.min(0), c.max(0), alpha=.15, color=ax.lines[-1].get_color())

f = f"{R}/mountaincar.json"
if os.path.exists(f):
    runs = json.load(open(f))["runs"]; fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, mode in zip(ax, ("plain", "shaped")):
        for algo in ("REINFORCE", "PPO"):
            band(a, [v for k, v in runs.items() if k.startswith(f"{algo}|{mode}")], algo)
        a.set_title(f"MountainCar-v0, {mode} reward"); a.set_xlabel("iteration"); a.legend(); a.grid(alpha=.3)
    ax[0].set_ylabel("true episode return (-200 = never reached goal)")
    plt.tight_layout(); plt.savefig(f"{R}/figures/mountaincar.png", dpi=130); plt.close()

f = f"{R}/tuning.json"
if os.path.exists(f):
    runs = json.load(open(f)); names = list(dict.fromkeys(k.rsplit("|", 1)[0] for k in runs))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for n in names:
        band(ax, [v for k, v in runs.items() if k.rsplit("|", 1)[0] == n], n)
    ax.set_title("PPO on LunarLander-v3: hyper-parameter variations (mean, min-max over 2 seeds)")
    ax.set_xlabel("iteration"); ax.set_ylabel("mean episode return"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{R}/figures/tuning_lunarlander.png", dpi=130); plt.close()
