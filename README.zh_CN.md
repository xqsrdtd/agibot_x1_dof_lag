[English](README.md) | 中文

## 本仓库是什么

在 **智元 AgiBot X1** `x1_dh_stand` 强化学习训练栈（Isaac Gym、双历史策略 **`ActorCriticDH`**、runner **`DHOnPolicyRunner`**）上的扩展。在原有训练与导出流程之外，本仓库主要增加：

1. **关节 DOF 执行滞后（lag）** — 训练时可做域随机；评估时对 **更大 lag（OOD）** 做点扫并汇总 CSV/图（`results/`、`scripts/`）。
2. **可选的 horizon=1 扩散先验** — 从 `diffusion_policy-main` 的 checkpoint 加载，在 **`DHPPO`** 中作为 **正则/教师**，并可选在 **rollout 中作为混合行为策略**（见下表）。
3. **文档** — [`docs/ONEPAGE_INTRO.md`](docs/ONEPAGE_INTRO.md) 一页总览；[`docs/DIFFUSION_HEAVY_RECIPE.md`](docs/DIFFUSION_HEAVY_RECIPE.md) 偏扩散、仍可跑的超参建议。

**定位：** 主优化对象仍是 **高斯 PPO**（`humanoid/algo/ppo/dh_ppo.py`）。扩散 **不是** 严格数学意义上的完整 DPPO 似然；详见 one-pager。

---

## 扩散相关（摘要）

| 角色 | 行为 |
|------|------|
| **先验（更新阶段）** | 对短观测与策略均值 **μ** 做去噪损失（`diffusion_reg_coef`，可选按 advantage 加权）。**对齐：** `MSE(μ, predict_action)`，采样路径无梯度（`diffusion_actor_align_coef`）。可选 **UNet 在线微调**（单独 Adam）。 |
| **行为分量（采集阶段）** | 以概率 `diffusion_rollout_prob` 用 **`predict_action`** 的动作替换高斯采样；buffer 记录 **`rollout_from_diffusion`**。PPO 仍用 **高斯** `log π(a|s)`；`diffusion_rollout_ppo_surrogate_scale` 在扩散行上缩放 surrogate/熵。可选 **`diffusion_rollout_denoise_coef`**（需 finetune）：对 **实际执行的扩散动作** 做 advantage 加权 denoise。 |

**配置入口：** `humanoid/envs/x1/x1_dh_stand_config.py` → `class algorithm`。需有效 **`diffusion_prior_ckpt_path`** 及可导入的 **`diffusion_policy-main`**（与 `dh_ppo.py` 中加载逻辑一致）。

---

## 快速运行

```bash
pip install -e .
PYTHONPATH=. python humanoid/scripts/train.py --task=x1_dh_stand --headless --num_envs=<N>
```

- **完整安装、play、JIT/ONNX 导出、sim2sim、手柄说明** 见 **[`README.zh_CN.original.md`](README.zh_CN.original.md)**（原根目录中文 README）。
- **DOF lag 评估、合并 CSV、作图** 见 [`results/README.md`](results/README.md)、`scripts/run_eval_dof_lag_ood.sh`、`scripts/analyze_dof_lag_ood.py`。

---

## 目录要点

```
humanoid/algo/ppo/dh_ppo.py      # DHPPO + 扩散相关逻辑
humanoid/algo/ppo/actor_critic_dh.py
humanoid/algo/ppo/rollout_storage.py   # rollout_from_diffusion
humanoid/envs/x1/x1_dh_stand_config.py # 任务与 algorithm 超参
docs/                            # ONEPAGE_INTRO、DIFFUSION_HEAVY_RECIPE
results/、scripts/               # lag OOD 图表与评估脚本
README.original.md               # 归档：原英文长 README
README.zh_CN.original.md         # 归档：原中文长 README
```

---

## 致谢

基于 **智元 AgiBot X1** 开源 RL 代码；本仓库扩展包括 DOF lag 实验、评估工具链，以及在 **`DHPPO`** 中的 **可选扩散先验与混合 rollout**。上游引用见 [`README.original.md`](README.original.md)。
