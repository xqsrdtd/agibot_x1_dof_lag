[English](README.md) | 中文

## 本项目：DOF 延迟鲁棒性实验

本仓库在智元 X1 **`x1_dh_stand`**（站立 + 速度跟踪）训练代码基础上，增加 **Baseline（训练时不加关节滞后）与 Robust（训练时对 DOF lag 做域随机，lag 采样于 `[0, 40]` timesteps）** 的对照实验，并在 Isaac Gym 中对 **更大延迟（OOD）** 做评估（例如 lag 最高到 80）。

**主要结论（3 个随机种子）：** 在 **lag=0** 时，Robust 成功率不低于 Baseline，甚至略优；在 **lag=80** 时，Baseline 成功率降至 **0%**，Robust 仍约 **~90%**。数值表见 [`results/dof_lag_ood_report.md`](results/dof_lag_ood_report.md)，一页叙事见 [`results/project_narrative.md`](results/project_narrative.md)。

| 成功率 vs DOF lag | 线速度跟踪误差 (lin_vel MAE) |
|:---:|:---:|
| ![](results/dof_lag_ood_success.png) | ![](results/dof_lag_ood_tracking.png) |

| 训练曲线（示例 run） |
|:---:|
| ![](results/training_curves.png) |

**复现图表与汇总：** 命令见 [`results/README.md`](results/README.md)（合并 CSV、`scripts/analyze_dof_lag_ood.py` 等）。

---

## 简介

[智元灵犀 X1](https://www.zhiyuan-robot.com/qzproduct/169.html) 是由智元研发并开源的模块化、高自由度人形机器人，X1 的软件系统基于智元开源组件 `AimRT` 作为中间件实现，并且采用强化学习方法进行运动控制。

本工程为智元灵犀 X1 所使用的强化学习训练代码，可配合智元灵犀 X1 配套的[推理软件](https://aimrt.org/)进行真机和仿真的行走调试，或导入其他机器人模型进行训练。
![](doc/id.jpg)

## 代码运行

### 安装依赖
1. 创建一个新的 Python 3.8 虚拟环境:
   - `conda create -n myenv python=3.8`.
2. 安装 pytorch 1.13 和 cuda-11.7:
   - `conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia`
3. 安装 numpy-1.23:
   - `conda install numpy=1.23`.
4. 安装 Isaac Gym:
   - 下载并安装 Isaac Gym Preview 4：https://developer.nvidia.com/isaac-gym.
   - `cd isaacgym/python && pip install -e .`
   - 用 `cd examples && python 1080_balls_of_solitude.py` 跑通示例。
   - 故障排查见 `isaacgym/docs/index.html`。
5. 安装训练代码依赖：
   - Clone 本仓库。
   - `pip install -e .`

### 使用

在**仓库根目录**执行脚本；入口脚本位于 `humanoid/scripts/`。

#### Train:
```bash
python humanoid/scripts/train.py --task=x1_dh_stand --run_name=<run_name> --headless
```
- 训练好的模型会保存在 `logs/<experiment_name>/exported_data/<date_time><run_name>/model_<iteration>.pt`，其中 `<experiment_name>` 在 config 中定义（默认 `x1_dh_stand`）。
![](doc/train.gif)

#### Play:
```bash
python humanoid/scripts/play.py --task=x1_dh_stand --load_run=<date_time><run_name>
```
![](doc/play.gif)

#### 生成 JIT 模型:
```bash
python humanoid/scripts/export_policy_dh.py --task=x1_dh_stand --load_run=<date_time><run_name>
```
- JIT 模型会保存在 `logs/<experiment_name>/exported_policies/<date_time>/`。

#### 生成 ONNX 模型:
```bash
python humanoid/scripts/export_onnx_dh.py --task=x1_dh_stand --load_run=<date_time>
```
- ONNX 模型会保存在 `logs/<experiment_name>/exported_policies/<date_time>/`。

#### 参数说明：
- task: 任务名
- resume: 是否从 checkpoint 恢复训练
- experiment_name: 实验名（与日志目录一致）
- run_name: 本次 run 名称
- load_run: 加载哪次 run；-1 表示最近一次
- checkpoint: checkpoint 编号；-1 表示最新
- num_envs: 并行环境数
- seed: 随机种子
- max_iterations: 最大训练迭代次数

### 添加新环境
1. 在 `envs/` 目录下创建一个新文件夹，在新文件夹下创建配置文件 `<your_env>_config.py` 和环境文件 `<your_env>_env.py`，这两个文件要分别继承 `LeggedRobotCfg` 和 `LeggedRobot`。

2. 将新机器的 urdf、mesh、mjcf 放到 `resources/` 文件夹下。
- 在 `<your_env>_config.py` 里配置新机器的 urdf path、PD gain、body name、default_joint_angles、experiment_name 等。

3. 在 `humanoid/envs/__init__.py` 里注册你的新机器。

### sim2sim
使用 MuJoCo 进行 sim2sim 验证（需先导出 JIT）：
```bash
python humanoid/scripts/sim2sim.py --task=x1_dh_stand --load_model <exported_policies 下的子文件夹名>
```
![](doc/mujoco.gif)

### 手柄使用
我们使用 Logitech F710 手柄。在启动 `play.py` 和 `sim2sim.py` 时，按住 4 的同时转动摇杆可以控制机器人前后、左右平移与旋转。
![](doc/joy_map.jpg)
|         按键          |         命令         |
| -------------------- |:--------------------:|
|         4 + 1-       |         前进          |
|         4 + 1+       |         后退          |
|         4 + 0-       |        左平移         |
|         4 + 0+       |        右平移         |
|         4 + 3-       |       逆时针旋转       |
|         4 + 3+       |       顺时针旋转       |


## 目录结构
```
.
|— humanoid           # 主要代码目录
|  |—algo             # 算法目录
|  |—envs             # 环境目录
|  |—scripts          # 脚本目录
|  |—utils            # 工具、功能目录
|— logs               # 训练输出（已 gitignore，体积大）
|— resources          # 资源库
|  |— robots          # 机器人 urdf、mjcf、mesh
|— results            # DOF lag 图表与汇总（本仓库扩展）
|— scripts            # 评估 / 分析脚本（本仓库扩展）
|— README.md          # 说明文档
```

## 说明与致谢

训练代码基于 **智元 AgiBot X1** 开源强化学习工程；本仓库在之上补充了 **DOF lag 实验、评估与作图脚本**，见 `results/`、`scripts/` 及上文「本项目」一节。

> 参考项目:
>
> * [GitHub - leggedrobotics/legged_gym: Isaac Gym Environments for Legged Robots](https://github.com/leggedrobotics/legged_gym)
> * [GitHub - leggedrobotics/rsl_rl: Fast and simple implementation of RL algorithms, designed to run fully on GPU.](https://github.com/leggedrobotics/rsl_rl)
> * [GitHub - roboterax/humanoid-gym: Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer https://arxiv.org/abs/2404.05695](https://github.com/roboterax/humanoid-gym)
