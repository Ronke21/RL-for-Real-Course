"""Extension 8 check: independent O(T^2) GAE implementation vs main.compute_gae."""
import torch
from common import main, dump

def gae_reference(rewards, values, dones, last_value, gamma, lam):
    """A_t = sum_l (gamma*lam)^l * delta_{t+l}, stopping at the first episode end."""
    T = len(rewards)
    adv = torch.zeros(T)
    for t in range(T):
        total, coef = 0.0, 1.0
        for k in range(t, T):
            nv = last_value if k == T - 1 else values[k + 1]
            delta = rewards[k] + gamma * nv * (1 - dones[k]) - values[k]
            total += coef * delta
            coef *= gamma * lam
            if dones[k]:
                break
        adv[t] = total
    return adv

torch.manual_seed(0)
T = 300
rewards, values = torch.randn(T), torch.randn(T)
dones = (torch.rand(T) < 0.05).float()
last_value = torch.tensor(0.7)
out = {}
for gamma in (1.0, 0.99):
    for lam in (1.0, 0.95, 0.0):
        a, _ = main.compute_gae(rewards, values, dones, last_value, gamma=gamma, gae_lambda=lam)
        ref = gae_reference(rewards, values, dones, last_value, gamma, lam)
        err = float((a - ref).abs().max())
        out[f"gamma={gamma},lambda={lam}"] = err
        print(f"gamma={gamma} lambda={lam}: max|main - reference| = {err:.2e}")
# lambda=1 => Monte-Carlo advantage (only when the rollout ends on a done, so no bootstrap)
dones[-1] = 1.0
a, _ = main.compute_gae(rewards, values, dones, last_value, gamma=0.99, gae_lambda=1.0)
mc = main.compute_returns(rewards, dones, gamma=0.99) - values
out["lambda=1 equals G_t - V(s_t)"] = float((a - mc).abs().max())
print("lambda=1 vs G_t - V:", out["lambda=1 equals G_t - V(s_t)"])
dump("gae_check", out)
