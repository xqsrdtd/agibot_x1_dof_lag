# DOF Lag OOD Evaluation Summary

## In-Distribution (lag=0)
| Model | Success | lin_vel_mae | fall_rate |
|-------|---------|-------------|-----------|
| baseline | 93.3% | 0.317 | 6.7% |
| dr_lag | 99.3% | 0.297 | 0.7% |
| weakrl | 98.7% | 0.156 | 1.3% |

## OOD: Success rate (%) by DOF lag
| lag | baseline | dr_lag | weakrl |
|-----|--------|--------|--------|
| 5 | 94.7 | 97.8 | 99.3 |
| 15 | 91.2 | 96.8 | 99.0 |
| 40 | 84.5 | 97.8 | 97.7 |
| 50 | 74.7 | 97.2 | - |
| 60 | 46.8 | 95.7 | 97.8 |
| 80 | 0.0 | 90.3 | 92.7 |