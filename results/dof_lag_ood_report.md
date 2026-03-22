# DOF Lag OOD Evaluation Summary

## In-Distribution (lag=0)
| Model | Success | lin_vel_mae | fall_rate |
|-------|---------|-------------|-----------|
| baseline | 93.3% | 0.317 | 6.7% |
| robust | 99.3% | 0.297 | 0.7% |

## OOD: Success rate (%) by DOF lag
| lag | baseline | robust |
|-----|----------|--------|
| 5 | 94.7 | 97.8 |
| 15 | 91.2 | 96.8 |
| 40 | 84.5 | 97.8 |
| 50 | 74.7 | 97.2 |
| 60 | 46.8 | 95.7 |
| 80 | 0.0 | 90.3 |