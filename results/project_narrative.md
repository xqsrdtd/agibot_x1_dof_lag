# 项目一页叙事：DOF Lag 域随机 vs Baseline（X1 / Isaac Gym）

## 1. 问题与动机

人形机器人用强化学习做站立与速度跟踪时，仿真里往往假设**关节状态与控制指令几乎无延迟**；真实系统里存在通信、驱动与执行周期，**关节状态相对指令存在滞后**。若只在「零延迟」分布上训练，策略容易过拟合该假设，在**更大延迟**（训练时没见过或很少见）下成功率与跟踪精度会明显下降。本项目研究：**训练时随机化 DOF lag（域随机化的一种）** 能否在**不损害无延迟性能**的前提下，提升对**大延迟**的泛化。

## 2. 方法概要

- **算法与框架**：PPO（on-policy）+ Isaac Gym；任务 `x1_dh_stand`（站立 + 线速度/角速度跟踪）。
- **Baseline**：训练时关闭关节滞后，`add_dof_lag = False`。
- **Robust**：训练时开启 DOF lag，`add_dof_lag = True`，在训练中对 lag **随机采样**（本实验配置为 `[0, 40]` timesteps，与 `x1_dh_stand_config` 一致）。
- **对照原则**：两路使用相同任务与网络结构，**仅训练时是否引入 DOF lag 随机化**不同，便于归因。

## 3. 评估协议

- **指标**：成功率（success rate）、线速度跟踪误差（lin_vel_mae）、摔倒率（fall rate）、角速度跟踪误差（ang_vel_mae）等（详见 `dof_lag_ood_*.png`）。
- **延迟档位**：在 eval 中对 **固定或扫过的 DOF lag** 取值（本仓库结果含 lag = 0, 5, 15, 40, 50, 60, 80 等）；其中 **lag ≥ 训练上界** 可视为对 **OOD（分布外）** 的 stress test。
- **随机性**：**3 个随机种子**；报告均值 ± 标准差（见 `dof_lag_ood_summary.csv`）。

## 4. 主要结果（与仓库内数值一致）

- **无延迟（lag = 0，偏 ID）**：Robust 的 success **不低于** baseline，并略优（约 **99.3% vs 93.3%**）；lin_vel_mae 与 fall rate 亦略优（见 `dof_lag_ood_report.md`）。说明「为鲁棒性加随机 lag」在本设定下**未以明显牺牲 ID 性能为代价**。
- **极端 OOD（lag = 80）**：Baseline success 降至 **0%**；Robust 仍维持约 **90%** 成功率。Fall rate 与 tracking 误差随 lag 增大时，Robust 曲线明显更平稳。
- **中间档位**：随 lag 从 40→60→80，Baseline  success 与跟踪指标恶化更快；Robust 在多数档位上保持更高 success 与更低 fall rate。

## 5. 结论（面试可用的一句话）

在相同任务与算法下，**训练时对 DOF lag 做域随机**能显著提升策略在**未见过的更大延迟**下的成功率与跟踪表现，且在 **lag = 0** 上**未观察到**相对 baseline 的明显退步。

## 6. 局限与诚实边界（必说）

- 结论目前基于 **Isaac Gym 仿真**；**未在真实机器人上验证**。
- **Sim2sim（MuJoCo）** 若已跑通，可说明「跨另一套物理/接触栈仍能控制」，但 **MuJoCo 默认脚本未与 Isaac 的 DOF lag OOD sweep 一一对应**，**lag 的主结论仍以 Isaac eval 为准**。
- 未与 **其它** 抗延迟手段（如显式延迟建模、更长历史观测等）做系统对比；未在固定算力下做完整的 **样本效率**（wall-clock）对比。

## 7. 可复现与材料

- 图表与汇总：`results/dof_lag_ood_*.png`、`dof_lag_ood_report.md`、`dof_lag_ood_summary.csv`。
- 复现命令见同目录 [`README.md`](README.md)；完整安装与训练说明见 [`docs/INSTALL_AGIBOT.zh_CN.md`](../docs/INSTALL_AGIBOT.zh_CN.md)。

---

## 附录：英文简历 Bullet（可删改数字与仓库名）

- Trained PPO policies for **humanoid stand + velocity tracking** (`x1_dh_stand`) in **Isaac Gym**, comparing **baseline** (no DOF lag) vs **domain-randomized DOF lag** during training (random lag in `[0, 40]` timesteps).
- Evaluated **OOD actuator delay** across multiple lag levels (e.g., up to **80** steps) with **3 seeds**; reported success rate, velocity tracking MAE, and fall rate.
- **Robust policy** retained ~**90%** success at **lag=80** where **baseline** dropped to **0%**; at **lag=0**, robust reached ~**99.3%** vs **93.3%** success for baseline.
