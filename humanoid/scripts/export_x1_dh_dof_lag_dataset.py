# SPDX-FileCopyrightText: Copyright (c) 2024, AgiBot Inc. All rights reserved.
#
# Export offline dataset for diffusion-style action models.
# The dataset stores (obs_t, action_t) plus conditioning variables (command_t, dof_lag_t)
# and episode boundary info for sequence sampling.

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

from isaacgym import gymutil

import torch

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import *  # noqa: F401,F403 (register tasks)
from humanoid.utils.task_registry import task_registry
from humanoid.utils.helpers import get_load_path


@dataclass
class ExportCfg:
    task: str
    experiment_name: Optional[str]
    load_run: str
    checkpoint: int
    out_path: str
    seed: int
    num_envs: int
    num_episodes: int
    # dof lag
    enable_dof_lag: bool
    dof_lag_min: int
    dof_lag_max: int
    dof_lag_mode: str
    # randomization behavior
    randomize_dof_lag_timesteps: bool
    randomize_dof_lag_timesteps_perstep: bool
    # observation representation for diffusion training
    # "full": use env_cfg.env.frame_stack (default; very large obs dim)
    # "short": use last env_cfg.env.short_frame_stack frames only
    obs_history_mode: str


def get_export_args():
    custom_parameters = [
        # base (compatible with train/play/export)
        {"name": "--task", "type": str, "default": "x1_dh_stand", "help": "Task name."},
        {"name": "--experiment_name", "type": str, "help": "Overrides config experiment_name (log path)."},
        {"name": "--load_run", "type": str, "help": "Run folder name under logs/<experiment_name>/exported_data/."},
        {
            "name": "--checkpoint",
            "type": int,
            "default": -1,
            "help": "Loads model_<checkpoint>.pt inside the run folder. Use -1 for the latest model_*.pt.",
        },
        {"name": "--headless", "action": "store_true", "default": True, "help": "Run without viewer."},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": "Device for policy inference."},
        {"name": "--seed", "type": int, "default": 0, "help": "Random seed."},
        {"name": "--num_envs", "type": int, "default": 256, "help": "Number of envs to use for rollout."},
        {"name": "--num_episodes", "type": int, "default": 200, "help": "Total number of episodes to export (across all envs)."},
        {"name": "--out", "type": str, "default": "results/x1_dh_dof_lag_dataset.pt", "help": "Output file path (torch.save payload)."},
        # dof lag configuration
        {"name": "--enable_dof_lag", "action": "store_true", "default": False, "help": "Enable DOF lag during rollout."},
        {"name": "--dof_lag_min", "type": int, "default": 0, "help": "Min DOF lag timesteps."},
        {"name": "--dof_lag_max", "type": int, "default": 0, "help": "Max DOF lag timesteps."},
        {
            "name": "--dof_lag_mode",
            "type": str,
            "default": "fixed",
            "help": "fixed: use dof_lag_max for all envs; random: sample in [min,max] at reset.",
        },
        # randomization granularity (controls whether lag changes inside an episode)
        {"name": "--randomize_dof_lag_timesteps", "action": "store_true", "default": False, "help": "Sample lag at reset."},
        {
            "name": "--randomize_dof_lag_timesteps_perstep",
            "action": "store_true",
            "default": False,
            "help": "If set, re-sample lag every step (more challenging for learning).",
        },
        {
            "name": "--obs_history_mode",
            "type": str,
            "default": "full",
            "help": "full: use full frame_stack obs (large). short: use only last short_frame_stack frames.",
        },
    ]
    return gymutil.parse_arguments(description="Export offline dataset for diffusion/action models", custom_parameters=custom_parameters)


def _override_export_cfg(env_cfg, train_cfg, args):
    # keep rollout light and consistent: only DOF lag variation
    env_cfg.env.num_envs = int(args.num_envs)
    env_cfg.env.episode_length_s = 20

    env_cfg.noise.add_noise = False
    # disable other random confounds to isolate lag effect
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

    env_cfg.terrain.mesh_type = "plane"

    if args.enable_dof_lag:
        env_cfg.domain_rand.add_dof_lag = True
        env_cfg.domain_rand.dof_lag_timesteps_range = [int(args.dof_lag_min), int(args.dof_lag_max)]
        # control whether dof_lag_timestep is fixed or sampled
        if args.dof_lag_mode.lower() == "random":
            env_cfg.domain_rand.randomize_dof_lag_timesteps = True
            env_cfg.domain_rand.randomize_dof_lag_timesteps_perstep = bool(args.randomize_dof_lag_timesteps_perstep)
        else:
            env_cfg.domain_rand.randomize_dof_lag_timesteps = False
            env_cfg.domain_rand.randomize_dof_lag_timesteps_perstep = False
    else:
        env_cfg.domain_rand.add_dof_lag = False

    if args.experiment_name is not None:
        train_cfg.runner.experiment_name = args.experiment_name
    return env_cfg, train_cfg


def _resolve_model_path(train_cfg, args) -> str:
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported_data")
    load_run = args.load_run if args.load_run is not None else -1
    checkpoint = int(args.checkpoint) if args.checkpoint is not None else -1
    return get_load_path(log_root, load_run=load_run, checkpoint=checkpoint)


def export_dataset(cfg: ExportCfg, args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=cfg.task)
    train_cfg.seed = int(cfg.seed)
    env_cfg.seed = int(cfg.seed)

    # Override rollout settings for consistent data generation
    # Use parsed Isaac Gym args directly (must include physics_engine/sim_device/headless, etc.)
    args.num_envs = int(cfg.num_envs)
    args.seed = int(cfg.seed)
    if getattr(args, "experiment_name", None) is None:
        args.experiment_name = cfg.experiment_name
    env_cfg, train_cfg = _override_export_cfg(env_cfg, train_cfg, args)

    env, _ = task_registry.make_env(name=cfg.task, args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = False
    runner, _, _ = task_registry.make_alg_runner(
        env=env,
        name=cfg.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )

    model_path = _resolve_model_path(train_cfg, args)
    print(f"[export_dataset] Loading model from: {model_path}")
    runner.load(model_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    # rollout storage (flattened over vector envs, but keep episode_id and step_in_episode)
    # Note: we intentionally avoid numpy dependency here (use torch.save).
    obs_list, act_list = [], []
    lag_list, cmd_list = [], []
    done_list = []
    ep_id_list = []
    step_in_ep_list = []

    num_envs = env_cfg.env.num_envs
    obs = env.get_observations()
    # obs can be flattened (frame_stack * num_single_obs); optionally keep only short history.
    frame_stack = int(getattr(env_cfg.env, "frame_stack", 1))
    num_single_obs = int(getattr(env_cfg.env, "num_single_obs", env_cfg.env.num_observations))
    short_frame_stack = int(getattr(env_cfg.env, "short_frame_stack", frame_stack))
    if cfg.obs_history_mode not in ("full", "short"):
        raise ValueError("--obs_history_mode must be one of: full, short")

    def maybe_compress_obs(obs_tensor: torch.Tensor) -> torch.Tensor:
        if cfg.obs_history_mode == "full":
            return obs_tensor
        # obs_tensor: [N, frame_stack * num_single_obs]
        # keep last short_frame_stack frames => [N, short_frame_stack * num_single_obs]
        N = obs_tensor.shape[0]
        obs_tensor = obs_tensor.view(N, frame_stack, num_single_obs)
        obs_tensor = obs_tensor[:, -short_frame_stack:, :]
        return obs_tensor.reshape(N, short_frame_stack * num_single_obs)

    episode_id = torch.zeros(num_envs, dtype=torch.int64, device=env.device)
    step_in_episode = torch.zeros(num_envs, dtype=torch.int64, device=env.device)

    finished_episodes = 0
    # sample until we have enough episodes ended across all envs
    while finished_episodes < cfg.num_episodes:
        with torch.inference_mode():
            # conditioning at current time t (aligned with obs_t)
            lag_t = env.dof_lag_timestep.clone().detach().to(torch.int64) if cfg.enable_dof_lag else torch.zeros(num_envs, dtype=torch.int64, device=env.device)
            # x1 rewards/metrics use commands[:, :3] => (lin_x, lin_y, ang_yaw)
            cmd_t = env.commands[:, :3].clone().detach().to(torch.float32)

            actions = policy(obs)

        # env step produces obs_{t+1} and dones for transitions from obs_t -> obs_{t+1}
        obs_next, _, _, dones, infos = env.step(actions)

        # flatten current time-step transition for all envs
        obs_list.append(maybe_compress_obs(obs.detach().cpu()))
        act_list.append(actions.detach().cpu())
        lag_list.append(lag_t.detach().cpu())
        cmd_list.append(cmd_t.detach().cpu())
        done_list.append(dones.detach().cpu())
        ep_id_list.append(episode_id.detach().cpu())
        step_in_ep_list.append(step_in_episode.detach().cpu())

        # update episode counters (aligned so next obs belongs to updated episode_id)
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            finished_episodes += int(len(done_ids))
            episode_id[done_ids] += 1
            step_in_episode[done_ids] = 0
        # non-done envs advance step index
        non_done_ids = (~dones).nonzero(as_tuple=False).flatten()
        if len(non_done_ids) > 0:
            step_in_episode[non_done_ids] += 1

        obs = obs_next

    # concat and save
    obs_arr = torch.cat(obs_list, dim=0)
    act_arr = torch.cat(act_list, dim=0)
    lag_arr = torch.cat(lag_list, dim=0)
    cmd_arr = torch.cat(cmd_list, dim=0)
    done_arr = torch.cat(done_list, dim=0).to(torch.bool)
    ep_id_arr = torch.cat(ep_id_list, dim=0)
    step_in_ep_arr = torch.cat(step_in_ep_list, dim=0)

    os.makedirs(os.path.dirname(cfg.out_path) or ".", exist_ok=True)
    print(f"[export_dataset] Saving to: {cfg.out_path}")
    payload = {
        "obs": obs_arr,
        "action": act_arr,
        "lag": lag_arr,
        "command": cmd_arr,
        "done": done_arr,
        "episode_id": ep_id_arr,
        "step_in_episode": step_in_ep_arr,
        "meta": {
            "task": cfg.task,
            "seed": cfg.seed,
            "enable_dof_lag": cfg.enable_dof_lag,
            "dof_lag_min": cfg.dof_lag_min,
            "dof_lag_max": cfg.dof_lag_max,
            "dof_lag_mode": cfg.dof_lag_mode,
            "obs_shape": tuple(obs_arr.shape),
            "action_shape": tuple(act_arr.shape),
            "obs_history_mode": cfg.obs_history_mode,
            "frame_stack": frame_stack,
            "short_frame_stack": short_frame_stack,
            "num_single_obs": num_single_obs,
        },
    }
    torch.save(payload, cfg.out_path)
    print(
        "[export_dataset] Saved shapes: "
        f"obs={tuple(obs_arr.shape)}, action={tuple(act_arr.shape)}, lag={tuple(lag_arr.shape)}, command={tuple(cmd_arr.shape)}, done={tuple(done_arr.shape)}"
    )


def main():
    args = get_export_args()

    # Basic validation
    if args.load_run is None:
        raise ValueError("--load_run is required (use -1 for latest).")
    if args.enable_dof_lag:
        if args.dof_lag_min > args.dof_lag_max:
            raise ValueError("dof_lag_min must be <= dof_lag_max")

    out_path = args.out
    if not os.path.isabs(out_path):
        # default to repo root if relative
        out_path = os.path.join(LEGGED_GYM_ROOT_DIR, out_path)

    load_run_parsed: Any = args.load_run
    if str(load_run_parsed).strip() == "-1":
        load_run_parsed = -1

    cfg = ExportCfg(
        task=args.task,
        experiment_name=getattr(args, "experiment_name", None),
        load_run=load_run_parsed,
        checkpoint=int(args.checkpoint),
        out_path=out_path,
        seed=int(args.seed),
        num_envs=int(args.num_envs),
        num_episodes=int(args.num_episodes),
        enable_dof_lag=bool(args.enable_dof_lag),
        dof_lag_min=int(args.dof_lag_min),
        dof_lag_max=int(args.dof_lag_max),
        dof_lag_mode=str(args.dof_lag_mode),
        randomize_dof_lag_timesteps=bool(args.randomize_dof_lag_timesteps),
        randomize_dof_lag_timesteps_perstep=bool(args.randomize_dof_lag_timesteps_perstep),
        obs_history_mode=str(args.obs_history_mode),
    )
    # Important: keep args.load_run consistent with get_load_path behavior
    args.load_run = load_run_parsed
    export_dataset(cfg, args)


if __name__ == "__main__":
    main()

