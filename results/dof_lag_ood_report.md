# DOF Lag OOD Evaluation Summary

## In-Distribution (lag=0)
| Model | Success | lin_vel_mae | fall_rate |
|-------|---------|-------------|-----------|
| baseline | 97.4% | 0.124 | 2.6% |
| robust | 99.7% | 0.116 | 0.3% |

## OOD: Success rate (%) by DOF lag
| lag | baseline | robust |
|-----|--------|--------|
| 5 | 98.0 | 99.2 |
| 15 | 96.8 | 98.8 |
| 40 | 94.3 | 99.2 |
| 50 | 90.8 | 98.9 |
| 60 | 82.1 | 98.3 |
| 80 | 0.0 | 96.4 |
| 100 | 0.0 | 95.5 |