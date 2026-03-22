# DOF Lag OOD Evaluation

人形机器人 (X1) 站立与速度跟踪任务下，基于 **DOF lag 域随机化** 的鲁棒性实验。

## 实验设置

- **任务**：`x1_dh_stand` — 人形站立 + 线速度/角速度跟踪
- **Baseline**：无域随机化，训练时 `add_dof_lag=False`
- **Robust**：训练时 `add_dof_lag=True`，lag 随机采样于 `[0, 40]` timesteps
- **评估**：在未见过的 lag (0, 5, 15, 40, 50, 60, 80) 下做 OOD sweep，3 seeds

## 主要结论

- **Success rate**：robust 在高 lag (80) 下维持 ~90%，baseline 降为 0%
- **Tracking (lin_vel_mae)**：robust 随 lag 增长保持 ~0.30–0.35，baseline 在 lag≥40 时显著恶化
- **ID 性能**：robust 在 lag=0 时不劣于 baseline，甚至略优 (success 99.3% vs 93.3%)

## 文件说明

| 文件 | 说明 |
|------|------|
| `dof_lag_ood_success.png` | Success rate vs DOF lag |
| `dof_lag_ood_tracking.png` | 线速度跟踪误差 vs DOF lag |
| `dof_lag_ood_fall_rate.png` | Fall rate vs DOF lag |
| `dof_lag_ood_ang_vel_mae.png` | 角速度跟踪误差 vs DOF lag |
| `dof_lag_ood_report.md` | 数值汇总表 (ID + OOD) |
| `dof_lag_ood_summary.csv` | 完整统计 (mean/std/count) |
| `training_curves.png` | 训练曲线对比 (mean_reward, episode_length) |

## 复现

```bash
# OOD 评估
bash scripts/run_eval_dof_lag_ood.sh --load_runs <baseline_run> --checkpoint 3900 --out results/baseline.csv --overwrite
bash scripts/run_eval_dof_lag_ood.sh --load_runs <robust_run> --checkpoint 5000 --out results/robust.csv --overwrite

# 合并并画图
(head -1 results/dof_lag_ood_eval_baseline_1.csv; tail -n +2 results/dof_lag_ood_eval_baseline_1.csv; tail -n +2 results/dof_lag_ood_eval_robust_1.csv) > results/dof_lag_ood_eval_combined.csv
python scripts/analyze_dof_lag_ood.py --csv results/dof_lag_ood_eval_combined.csv --out_dir results --baseline_run <baseline_run> --robust_run <robust_run>
```
