### Install check

| Policy / condition | Full-clip survival [95% Wilson] | Episodes | Failures / censored | Mean body-pos error (m) | Mean survival steps | Mean sq. action delta |
|---|---|---|---|---|---|---|
| setup_smoke reload (2 epochs, 16 eps) | 0.0% [0.0, 19.4] | 16 | 16 / 0 | 0.1287 | 31.0 | 0.0001 |

### M0: velocity frame (frozen walker, 500 eps per cell)

| Policy / condition | Full-clip survival [95% Wilson] | Episodes | Failures / censored | Mean body-pos error (m) | Mean survival steps | Mean sq. action delta |
|---|---|---|---|---|---|---|
| broken @ 0 deg | 100.0% [99.2, 100.0] | 500 | 0 / 0 | 0.0739 | 161.0 | 0.0003 |
| student @ 0 deg | 100.0% [99.2, 100.0] | 500 | 0 / 0 | 0.0689 | 161.0 | 0.0003 |
| broken @ 90 deg | 0.0% [0.0, 0.8] | 500 | 500 / 0 | 0.1365 | 43.1 | 0.0002 |
| student @ 90 deg | 100.0% [99.2, 100.0] | 500 | 0 / 0 | 0.0689 | 161.0 | 0.0003 |
| broken @ 180 deg | 0.0% [0.0, 0.8] | 500 | 500 / 0 | 0.1560 | 34.0 | 0.0003 |
| student @ 180 deg | 100.0% [99.2, 100.0] | 500 | 0 / 0 | 0.0689 | 161.0 | 0.0003 |

### M1: PPO fine-tuning of the weak epoch-70 tracker (clean, seed 0)

| Policy / condition | Full-clip survival [95% Wilson] | Episodes | Failures / censored | Mean body-pos error (m) | Mean survival steps | Mean sq. action delta |
|---|---|---|---|---|---|---|
| weak starter (epoch 70) | 0.0% [0.0, 0.8] | 500 | 500 / 0 | 0.1289 | 31.0 | 0.0001 |
| m1_control (my fine-tune) | 80.8% [77.1, 84.0] | 500 | 96 / 0 | 0.1431 | 160.2 | 0.0004 |

### M1 optional: action-rate x10 ablation (10 min from supplied expert)

| Policy / condition | Full-clip survival [95% Wilson] | Episodes | Failures / censored | Mean body-pos error (m) | Mean survival steps | Mean sq. action delta |
|---|---|---|---|---|---|---|
| m1_reward_control (weight x1) | 100.0% [99.2, 100.0] | 500 | 0 / 0 | 0.0851 | 161.0 | 0.0005 |
| m1_ablation (weight x10) | 100.0% [99.2, 100.0] | 500 | 0 / 0 | 0.0852 | 161.0 | 0.0005 |

### M2: behavior cloning vs the supplied expert (seed 0)

| Policy / condition | Full-clip survival [95% Wilson] | Episodes | Failures / censored | Mean body-pos error (m) | Mean survival steps | Mean sq. action delta |
|---|---|---|---|---|---|---|
| expert, clean | 100.0% [99.2, 100.0] | 500 | 0 / 0 | 0.0856 | 161.0 | 0.0006 |
| BC student, clean | 0.0% [0.0, 0.8] | 500 | 500 / 0 | 0.1125 | 121.8 | 0.0005 |
| expert, perturbed | 92.4% [89.7, 94.4] | 500 | 38 / 0 | 0.1416 | 158.1 | 0.0009 |
| BC student, perturbed | 0.0% [0.0, 0.8] | 500 | 500 / 0 | 0.1218 | 59.5 | 0.0005 |

### M3: domain randomization (seed 1)

| Policy / condition | Full-clip survival [95% Wilson] | Episodes | Failures / censored | Mean body-pos error (m) | Mean survival steps | Mean sq. action delta |
|---|---|---|---|---|---|---|
| before (M3 start ckpt), clean | 78.8% [75.0, 82.2] | 500 | 106 / 0 | 0.1440 | 160.2 | 0.0004 |
| equal-budget CLEAN fine-tune, clean | 100.0% [99.2, 100.0] | 500 | 0 / 0 | 0.0937 | 161.0 | 0.0005 |
| after DR fine-tune (m3_g1), clean | 100.0% [99.2, 100.0] | 500 | 0 / 0 | 0.0895 | 161.0 | 0.0005 |
| before (M3 start ckpt), perturbed | 3.0% [1.8, 4.9] | 500 | 485 / 0 | 0.1653 | 95.0 | 0.0006 |
| equal-budget CLEAN fine-tune, perturbed | 81.0% [77.3, 84.2] | 500 | 95 / 0 | 0.1496 | 149.4 | 0.0008 |
| after DR fine-tune (m3_g1), perturbed | 91.6% [88.8, 93.7] | 500 | 42 / 0 | 0.1339 | 158.3 | 0.0007 |

M3 start checkpoint: `results/m1_control/last.ckpt`
