# DOF Lag OOD Evaluation Summary

## In-Distribution (lag=0)
| Model | Success | lin_vel_mae | fall_rate |
|-------|---------|-------------|-----------|
| baseline | 97.4% | 0.124 | 2.6% |
| dr_lag | 99.7% | 0.116 | 0.3% |
| weakrl | 99.0% | 0.122 | 1.0% |
| weakrl_adv_abs | 97.5% | 0.135 | 2.6% |
| weakrl_adv_pos | 95.1% | 0.136 | 5.2% |
| weakrl_adv_signed | 89.5% | 0.162 | 11.6% |
| weakrl_pos_dppo_ft | 94.9% | 0.142 | 5.3% |
| weakrl_pos_dppo_ft_align | 99.0% | 0.125 | 1.0% |

## OOD: Success rate (%) by DOF lag
| lag | baseline | dr_lag | weakrl | weakrl_adv_abs | weakrl_adv_pos | weakrl_adv_signed | weakrl_pos_dppo_ft | weakrl_pos_dppo_ft_align |
|-----|--------|--------|--------|--------|--------|--------|--------|--------|
| 5 | 98.0 | 99.2 | 99.5 | 98.4 | 96.5 | 93.0 | 96.2 | 98.8 |
| 15 | 96.8 | 98.8 | 99.2 | 99.0 | 97.4 | 94.3 | 96.4 | 99.0 |
| 40 | 94.3 | 99.2 | 98.2 | 96.4 | 95.1 | 92.3 | 98.0 | 99.7 |
| 50 | 90.8 | 98.9 | - | - | - | - | - | - |
| 60 | 82.1 | 98.3 | 98.3 | 89.7 | 86.3 | 91.0 | 97.3 | 99.7 |
| 80 | 0.0 | 96.4 | 94.4 | 84.2 | 81.6 | 89.6 | 97.8 | 97.5 |
| 100 | 0.0 | 95.5 | 87.8 | 66.8 | 68.5 | 84.9 | 90.1 | 84.2 |