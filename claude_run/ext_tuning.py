"""Extension 7 (+8 via GAE lambda): PPO hyper-parameter variations on LunarLander-v3, 2 seeds each."""
import functools, time
from common import main, seed_all, summarize, dump

ENV = "LunarLander-v3"
orig_gae = main.compute_gae
CONFIGS = {   # name: (ppo_lr, gamma, gae_lambda, update_epochs)
    "default (lr1e-4, g1.0, lam1.0, 10ep)": (main.PPO_LR, 1.0, 1.0, 10),
    "lr 3e-4":                               (3e-4, 1.0, 1.0, 10),
    "gamma 0.99":                            (main.PPO_LR, 0.99, 1.0, 10),
    "lambda 0.95":                           (main.PPO_LR, 1.0, 0.95, 10),
    "lr 3e-4 + g0.99 + lam0.95":             (3e-4, 0.99, 0.95, 10),
}
res = {}
for seed in (0, 1):
    for name, (lr, g, lam, ep) in CONFIGS.items():
        seed_all(seed)
        main.PPO_LR, main.PPO_UPDATE_EPOCHS = lr, ep
        main.compute_gae = functools.partial(orig_gae, gamma=g, gae_lambda=lam)
        t0 = time.time()
        curve = main.run_ppo(ENV)
        out = {"curve": [float(x) for x in curve], "seconds": time.time() - t0, **summarize(curve)}
        res[f"{name}|{seed}"] = out
        print(f"[{name} seed={seed}] last10={out['final_mean_last10']:.1f} ({out['seconds']:.0f}s)", flush=True)
        dump("tuning", res)
