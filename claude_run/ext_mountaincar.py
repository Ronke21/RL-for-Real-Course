"""Extension 6: MountainCar-v0 with the default REINFORCE / PPO, then with an energy-shaping fix."""
import gymnasium as gym
import numpy as np
import time
from common import main, rl_utils, seed_all, summarize, dump

ENV = "MountainCar-v0"
SHAPE_SCALE = 1e4


class EnergyShaping(gym.Wrapper):
    """Potential-based shaping r' = r + phi(s') - phi(s), phi = SCALE * mechanical energy.
    Sits OUTSIDE RecordEpisodeStatistics, so logged episode returns stay the true (-1/step) returns."""
    @staticmethod
    def phi(obs):
        x, v = float(obs[0]), float(obs[1])
        return SHAPE_SCALE * (0.5 * v * v + 0.0025 * np.sin(3 * x) / 3)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._phi = self.phi(obs)
        return obs, info

    def step(self, a):
        obs, r, term, trunc, info = self.env.step(a)
        p = self.phi(obs)
        r = r + p - self._phi
        self._phi = p
        return obs, r, term, trunc, info


def run(name, runner, shaped, seed):
    seed_all(seed)
    if shaped:
        base = main.make_env
        main.make_env = lambda env_id: EnergyShaping(base(env_id))
    t0 = time.time()
    curve = runner(ENV)
    out = {"curve": [float(x) for x in curve], "seconds": time.time() - t0, **summarize(curve)}
    print(f"[{name} seed={seed}] last10={out['final_mean_last10']:.1f} ({out['seconds']:.0f}s)", flush=True)
    return out


res = {}
for seed in (0, 1):
    res[f"REINFORCE|plain|{seed}"] = run("REINFORCE plain", main.run_reinforce, False, seed)
    res[f"PPO|plain|{seed}"] = run("PPO plain", main.run_ppo, False, seed)
    res[f"REINFORCE|shaped|{seed}"] = run("REINFORCE shaped", main.run_reinforce, True, seed)
    res[f"PPO|shaped|{seed}"] = run("PPO shaped", main.run_ppo, True, seed)
    dump("mountaincar", {"shape_scale": SHAPE_SCALE, "runs": res})
