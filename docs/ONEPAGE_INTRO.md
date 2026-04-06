# X1 双历史站立：PPO + 扩散先验（一页介绍）

## 项目目标

在并行仿真中训练 **AgiBot X1** 的 **站立/平衡** 策略，观测采用 **短历史 + CNN 长历史 + 状态估计**，算法主体为 **PPO（高斯策略）**。在此基础上引入 **horizon=1、无 lag token** 的 **扩散策略先验**（由 `diffusion_policy-main` 离线训练），用于 **正则、蒸馏、可选 rollout 混合与在线微调**，并针对 **执行器滞后（dof lag）** 等 sim2sim 差异做评估与叙事。

---

## 技术栈（实现层面）

| 模块 | 说明 |
|------|------|
| 环境 | `X1DHStandEnv` + `x1_dh_stand_config`，多环境并行 |
| 策略网络 | `ActorCriticDH`：短观测、长历史 CNN、state estimator，高斯 actor |
| RL 核心 | `DHPPO` + `RolloutStorage`，GAE、clip PPO、价值函数与线速度监督 |
| 扩散先验 | `DiffusionUnetLowdimPolicy`（325-d 短 obs × 1 步，18-d 动作），ckpt 可加载；可选冻结或 **单独 Adam 微调** |

---

## 训练管线（可配置开关）

1. **标准 PPO**：采集 → 优势估计 → 多 epoch minibatch 更新（surrogate clip、value、熵、state estimator MSE）。
2. **去噪正则**：对当前策略 **均值 μ** 与观测计算扩散 **`compute_loss`**，系数 `diffusion_reg_coef`；可选 **按 PPO advantage 加权**（多种权重模式）。
3. **Actor–扩散对齐**：`predict_action` 得 `a_diff`，**MSE(μ, a_diff)**（采样路径无梯度）；`diffusion_align_every` / `diffusion_align_inference_steps` 控制算力。
4. **Rollout 混合**：`diffusion_rollout_prob` 下，部分环境逐步执行 **扩散采样动作** 而非高斯采样；buffer 记录 **`rollout_from_diffusion`**。
5. **混合行为与 PPO**：ratio 仍用 **高斯 log π(a\|s)**；对扩散 rollout 行可用 **`diffusion_rollout_ppo_surrogate_scale`** 缩放 surrogate 与熵，削弱有偏项。若 **开启扩散微调**，可用 **`diffusion_rollout_denoise_coef`** 对 **实际执行的扩散动作** 做 **advantage 加权的 denoise**，使扩散目标与回报信号一致。

---

## 定位说明（与「完全 DPPO」）

- **主优化对象**仍是 **高斯 actor** 的 PPO；扩散是 **先验与可选行为分量**，不是闭式边缘似然下的单一扩散策略 RL。
- **混合 rollout** 时，PPO ratio **不是**严格意义下混合策略的精确似然比；当前通过 **标记分支 + 缩放 surrogate + 扩散子集 denoise 目标** 做工程上可辩护的折中。
- 适合表述为：**Gaussian PPO with optional horizon-1 diffusion prior**，而非端到端 **DPPO** 的严格数学形式。

---

## 实验与资产

- 配置入口：`humanoid/envs/x1/x1_dh_stand_config.py`（`algorithm` 中扩散相关超参）。
- 评估与 OOD：`dof_lag` 扫描、CSV/图与报告脚本（见 `results/` 与 `scripts/`）。
- 依赖：本仓库 RL 栈 + 独立路径下的 **`diffusion_policy-main`** 与对应 **ckpt**。

---

*文档版本：与当前 `DHPPO` / `rollout_storage` 行为一致；修改代码时请同步更新本页。*
