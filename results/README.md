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
| `mujoco_sim2sim_baseline.gif` | MuJoCo sim2sim 录屏（baseline 策略，与主 README 内嵌一致） |

## 复现

```bash
# OOD 评估
bash scripts/run_eval_dof_lag_ood.sh --load_runs <baseline_run> --checkpoint 3900 --out results/baseline.csv --overwrite
bash scripts/run_eval_dof_lag_ood.sh --load_runs <robust_run> --checkpoint 5000 --out results/robust.csv --overwrite

# 合并并画图
(head -1 results/dof_lag_ood_eval_baseline_1.csv; tail -n +2 results/dof_lag_ood_eval_baseline_1.csv; tail -n +2 results/dof_lag_ood_eval_robust_1.csv) > results/dof_lag_ood_eval_combined.csv
python scripts/analyze_dof_lag_ood.py --csv results/dof_lag_ood_eval_combined.csv --out_dir results --baseline_run <baseline_run> --robust_run <robust_run>
```

### 多 run 合并图 (`dof_lag_ood_all_eval_*.png`)

```bash
python scripts/plot_all_eval_runs.py
```

会合并 `scripts/plot_all_eval_runs.py` 里 `MERGE_FILES` 列出的 CSV，写出 `eval_all_runs_merged.csv`，并生成 `dof_lag_ood_all_eval_{success,tracking,fall_rate,ang_vel_mae}.png` 与 `dof_lag_ood_all_eval_report.md`。默认 **`OMIT_DOF_LAG_MAX = [120]`**：横轴保留 **lag=100**，不画 **lag=120**（原始 CSV 仍可保留 120 行，仅作图/汇总时剔除）。若要画回 120，把该列表改为 `[]` 后重跑。

### 历史 CSV 与 `eval.py` 聚合 bug、行级缩放

旧版 `humanoid/scripts/eval.py` 在单步 `dones` 数大于「还缺几条 episode」时，把 `n * batch_mean` 全部加进分母为 `num_episodes` 的均值，会放大标量（常见约 `num_envs/num_episodes`）。**已修复**为只累加 `min(n, remaining)` 条。

对修复前导出的 CSV，可用 `humanoid/scripts/rescale_eval_csv_aggregation.py` 做**行级**校正：仅当 `num_envs > num_episodes` 且 **`timeout_rate > 1.01` 或 `active_cmd_ratio > 1.05`**（明显「被放大」）时，对该行各指标乘以 `num_episodes/num_envs`。多波次结束、rate 已在 `[0,1]` 的行**不缩放**，否则会误伤（例如 baseline 在 lag=80 全倒时 `lin_vel_mae` 等已是正确均值）。

原始快照在 `results/*.pre_agg_fix_bak`；行级处理前的副本在 `*.pre_rowwise_rescale_bak`（若存在）。

### 多 run 作图环境

`scripts/plot_all_eval_runs.py` / `scripts/analyze_dof_lag_ood.py` 需要 **pandas、matplotlib**（例如 conda 环境中的 Python）。
