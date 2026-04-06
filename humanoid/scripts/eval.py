# SPDX-FileCopyrightText: Copyright (c) 2024, AgiBot Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import csv
import os
from dataclasses import dataclass
from typing import Dict, Any
from isaacgym import gymutil
import torch


from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import *  # noqa: F401,F403 (register tasks)
from humanoid.utils.task_registry import task_registry
from humanoid.utils.helpers import get_load_path


@dataclass
class EvalResult:
    num_episodes: int
    metrics_sum: Dict[str, float]

    def mean(self) -> Dict[str, float]:
        denom = max(1, int(self.num_episodes))
        return {k: v / denom for k, v in self.metrics_sum.items()}


def get_eval_args():
    custom_parameters = [
        # base (compatible with train/play/export)
        {"name": "--task", "type": str, "default": "x1_dh_stand", "help": "Task name."},
        {"name": "--experiment_name", "type": str, "help": "Overrides config experiment_name (for log path)."},
        {"name": "--load_run", "type": str, "help": "Run folder name under logs/<experiment_name>/exported_data/."},
        {
            "name": "--checkpoint",
            "type": int,
            "default": -1,
            "help": "Loads model_<checkpoint>.pt inside the run folder. Use -1 for the latest model_*.pt in that folder.",
        },
        {
            "name": "--model_path",
            "type": str,
            "default": "",
            "help": "Optional absolute or relative path to a .pt file. If set, overrides --load_run/--checkpoint for loading.",
        },
        {"name": "--headless", "action": "store_true", "default": True, "help": "Run without viewer."},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": "Device for policy inference."},
        {"name": "--seed", "type": int, "default": 0, "help": "Random seed."},
        {"name": "--num_envs", "type": int, "default": 256, "help": "Number of envs to use for evaluation."},
        # eval controls
        {"name": "--num_episodes", "type": int, "default": 200, "help": "Total number of episodes to evaluate."},
        {"name": "--out", "type": str, "default": "results/eval.csv", "help": "CSV output path (append)."},
        # dof lag override (focus of this project)
        {"name": "--enable_dof_lag", "action": "store_true", "default": False, "help": "Enable DOF lag during eval."},
        {"name": "--dof_lag_min", "type": int, "default": 0, "help": "Min DOF lag timesteps."},
        {"name": "--dof_lag_max", "type": int, "default": 0, "help": "Max DOF lag timesteps."},
        {
            "name": "--dof_lag_mode",
            "type": str,
            "default": "fixed",
            "help": "fixed: use dof_lag_max for all envs; random: sample in [min,max] at reset.",
        },
    ]

    args = gymutil.parse_arguments(description="Eval RL Policy", custom_parameters=custom_parameters)

    # Empty string means "not provided" (gymutil may not distinguish None)
    if not getattr(args, "model_path", None):
        args.model_path = None

    # Batch eval: no viewer by default. Isaac Gym's merged args can leave headless=False.
    # Set EVAL_WITH_VIEWER=1 to open the viewer for debugging.
    if os.environ.get("EVAL_WITH_VIEWER", "").strip().lower() in ("1", "true", "yes"):
        args.headless = False
    else:
        args.headless = True

    # align names with other scripts (Isaac Gym args)
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def _override_eval_cfg(env_cfg, train_cfg, args):
    # keep evaluation light and consistent
    env_cfg.env.num_envs = int(args.num_envs)
    env_cfg.env.episode_length_s = 20

    # make evaluation less confounded (we only want DOF lag effects)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_torque = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_motor_offset = False
    env_cfg.domain_rand.randomize_joint_friction = False
    env_cfg.domain_rand.randomize_joint_damping = False
    env_cfg.domain_rand.randomize_joint_armature = False

    # keep terrain simple & fast
    env_cfg.terrain.mesh_type = "plane"

    # DOF lag override
    if args.enable_dof_lag:
        env_cfg.domain_rand.add_dof_lag = True
        env_cfg.domain_rand.dof_lag_timesteps_range = [int(args.dof_lag_min), int(args.dof_lag_max)]
        if args.dof_lag_mode.lower() == "random":
            env_cfg.domain_rand.randomize_dof_lag_timesteps = True
        else:
            # fixed lag == always use max timesteps (see randomize_lag_props in LeggedRobot)
            env_cfg.domain_rand.randomize_dof_lag_timesteps = False
    else:
        env_cfg.domain_rand.add_dof_lag = False

    # pick model load path experiment name
    if args.experiment_name is not None:
        train_cfg.runner.experiment_name = args.experiment_name
    return env_cfg, train_cfg


def _resolve_model_path(train_cfg, args) -> str:
    if getattr(args, "model_path", None):
        raw = args.model_path
        candidates = [
            os.path.abspath(raw),
            os.path.abspath(os.path.join(LEGGED_GYM_ROOT_DIR, raw)),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(
            f"[eval] --model_path not found (tried cwd and repo root): {raw}"
        )

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported_data")
    load_run = args.load_run if args.load_run is not None else -1
    checkpoint = int(args.checkpoint) if args.checkpoint is not None else -1
    return get_load_path(log_root, load_run=load_run, checkpoint=checkpoint)


def evaluate(env, policy, num_episodes: int) -> EvalResult:
    keys = [
        "lin_vel_mae",
        "ang_vel_mae",
        "active_cmd_ratio",
        "tau2_mean",
        "power_abs_mean",
        "timeout_rate",
        "fall_rate",
    ]
    sums = {k: 0.0 for k in keys}

    obs = env.get_observations()
    finished = 0

    with torch.inference_mode():
        while finished < num_episodes:
            actions = policy(obs)
            obs, _, _, dones, infos = env.step(actions)
            n = int(torch.sum(dones).item())
            if n <= 0:
                continue
            remaining = num_episodes - finished
            # ep[k] is the mean over the n envs that finished this step; only count up to `remaining`
            # episodes toward the aggregate. Otherwise (e.g. n=512, remaining=200) we would add 512
            # copies into sums but divide by 200 in mean(), inflating metrics by n/remaining (~2.5x).
            m = min(n, remaining)
            finished += m
            ep = infos.get("episode", {})
            for k in keys:
                if k in ep:
                    sums[k] += float(ep[k].item()) * m

    return EvalResult(num_episodes=finished, metrics_sum=sums)


def append_csv(path: str, row: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = get_eval_args()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    train_cfg.seed = int(args.seed)
    env_cfg.seed = int(args.seed)
    env_cfg, train_cfg = _override_eval_cfg(env_cfg, train_cfg, args)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # build runner without logging, then load model explicitly
    train_cfg.runner.resume = False
    runner, _, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None)
    model_path = _resolve_model_path(train_cfg, args)
    print(f"[eval] Loading model from: {model_path}")
    runner.load(model_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    result = evaluate(env, policy, num_episodes=int(args.num_episodes))
    means = result.mean()

    row = {
        "task": args.task,
        "seed": int(args.seed),
        "experiment_name": train_cfg.runner.experiment_name,
        "load_run": args.load_run if args.load_run is not None else "-1",
        "checkpoint": int(args.checkpoint) if args.checkpoint is not None else -1,
        "model_path": args.model_path if args.model_path else "",
        "num_envs": int(args.num_envs),
        "num_episodes": int(result.num_episodes),
        "enable_dof_lag": bool(args.enable_dof_lag),
        "dof_lag_min": int(args.dof_lag_min),
        "dof_lag_max": int(args.dof_lag_max),
        "dof_lag_mode": str(args.dof_lag_mode),
        **means,
    }

    append_csv(args.out, row)
    print(f"[eval] Wrote results to: {args.out}")
    print("[eval] Means:")
    for k, v in means.items():
        print(f"  - {k}: {v:.6f}")


if __name__ == "__main__":
    main()

