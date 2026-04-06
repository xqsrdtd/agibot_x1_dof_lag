# 「偏扩散」参考参数方案（可跑版）

目标：在**能稳定跑完训练**的前提下，尽量拉高扩散的参与度（正则 + 对齐 + rollout 行为 + UNet 微调 + rollout denoise）。默认 `num_envs=9216` 下每步做大量 `predict_action` 不现实，**务必减并行环境数**。

---

## 1. 推荐启动命令（先跑通）

在 `algorithm` 按下面第二节改好后：

```bash
PYTHONPATH=. python humanoid/scripts/train.py --task=x1_dh_stand --headless \
  --num_envs=512 --rl_device cuda:0 --sim_device cuda:0
```

- 显存/算力紧：`--num_envs=256`。
- 仍慢：先把 `diffusion_rollout_prob` 降到 `0.15`，或 `diffusion_align_every=4`。

---

## 2. `x1_dh_stand_config.py` → `class algorithm` 建议值

把下面整块替换你当前的扩散相关行（**ckpt 路径改成你机器上的**）：

```python
        use_diffusion_reg = True
        diffusion_reg_coef = 3e-4
        diffusion_adv_weighted = True
        diffusion_adv_weight_mode = "positive"
        diffusion_prior_ckpt_path = "/path/to/your/latest.ckpt"

        diffusion_finetune = True
        diffusion_finetune_lr = 5e-7
        diffusion_finetune_max_grad_norm = 0.5

        diffusion_actor_align_coef = 1e-3
        diffusion_align_inference_steps = 8
        diffusion_align_every = 2

        diffusion_rollout_prob = 0.25
        diffusion_rollout_inference_steps = 8
        diffusion_rollout_ppo_surrogate_scale = 0.5
        diffusion_rollout_denoise_coef = 1e-4
```

含义简述：

| 参数 | 取值 | 说明 |
|------|------|------|
| `diffusion_reg_coef` | `3e-4` | 比 `1e-4` 更强拉 μ 贴近扩散流形；再大可先试 `5e-4`，不稳再退回 |
| `diffusion_finetune_lr` | `5e-7` | 比 `1e-6` 保守，配合强 rollout 更不易炸 |
| `diffusion_actor_align_coef` | `1e-3` | 对齐更强；发散时降到 `5e-4` |
| `diffusion_align_every` | `2` | 更常做 align，算力↑ |
| `diffusion_rollout_prob` | `0.25` | 约 1/4 步用扩散动作；`0.35` 更「扩散」但更慢、更难稳 |
| `diffusion_rollout_ppo_surrogate_scale` | `0.5` | 削弱扩散行上有偏的 Gaussian PPO 项，利于混合行为稳定 |
| `diffusion_rollout_denoise_coef` | `1e-4` | 对 **扩散 rollout 样本** 做 adv 加权 denoise，需 `finetune=True` 才有梯度进 UNet |

未改动的 PPO 基线（`learning_rate=1e-5`、`entropy_coef`、`gamma` 等）保持你当前 `x1_dh_stand` 默认即可；若仍震荡，可把 **`learning_rate` 降到 `8e-6`**。

---

## 3. 更激进（仅在你已跑稳上一档之后）

- `diffusion_rollout_prob = 0.35`
- `diffusion_reg_coef = 5e-4`
- `diffusion_rollout_denoise_coef = 2e-4`
- `diffusion_actor_align_coef = 1.5e-3`

同时建议 **`diffusion_rollout_ppo_surrogate_scale = 0.3`**，并优先 **加环境数/步数** 前先观察 loss 与回报曲线。

---

## 4. 若跑不动或不稳定

1. **`diffusion_rollout_prob = 0`**：先只开 reg + align + finetune，确认稳了再加 rollout。
2. **`diffusion_rollout_denoise_coef = 0`**：去掉 rollout 对 UNet 的额外项，只保留 `μ` 正则。
3. **`diffusion_align_every = 4`**、`**diffusion_actor_align_coef = 5e-4**`：降算力与对齐强度。
4. **`diffusion_finetune = False`**：先冻结扩散，只训高斯 + 正则。

---

## 5. 检查清单

- [ ] `diffusion_prior_ckpt_path` 存在且与 **325-d 短 obs、18-d 动作、horizon=1** 一致  
- [ ] `diffusion_policy-main` 在 Python 路径中（与 `dh_ppo._load_diffusion_prior_h1` 一致）  
- [ ] 混合 rollout 时 **`num_envs` 明显小于 9216**  

按上表属于**偏扩散的折中上限**；真正「全扩散策略」仍超出当前 PPO+高斯主干的设定，本方案是在现有实现下能训练的**最重扩散配方**。
