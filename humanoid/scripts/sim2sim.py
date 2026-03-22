# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2024, AgiBot Inc. All rights reserved.

import math
import numpy as np
import mujoco, mujoco_viewer
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import *
from humanoid.utils import  Logger
import torch
import pygame
from threading import Thread
from humanoid.utils.helpers import get_load_path
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime
import csv

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.0, 0.0, 0.0
joystick_use = True
joystick_opened = False

if joystick_use:
    pygame.init()
    try:
        # get joystick
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
    except Exception as e:
        print(f"无法打开手柄：{e}")
    # joystick thread exit flag
    exit_flag = False

    def handle_joystick_input():
        global exit_flag, x_vel_cmd, y_vel_cmd, yaw_vel_cmd, head_vel_cmd
        
        
        while not exit_flag:
            # get joystick input
            pygame.event.get()
            # update robot command
            x_vel_cmd = -joystick.get_axis(1) * 1
            y_vel_cmd = -joystick.get_axis(0) * 1
            yaw_vel_cmd = -joystick.get_axis(3) * 1
            pygame.time.delay(100)

    if joystick_opened and joystick_use:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.start()

class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data,model):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('body-orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('body-angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    foot_positions = []
    foot_forces = []
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if '5_link' or 'ankle_roll' in body_name:  # according to model name
            foot_positions.append(data.xpos[i][2].copy().astype(np.double))
            foot_forces.append(data.cfrc_ext[i][2].copy().astype(np.double)) 
        if 'base_link' or 'waist_link' in body_name:  # according to model name
            base_pos = data.xpos[i][:3].copy().astype(np.double)
    return (q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces)

def pd_control(default_dof_pos,target_q, q, kp, target_dq, dq, kd, cfg):
    '''Calculates torques from position commands
    '''
    torque_out = (target_q + order_mujoco2isaacgym(default_dof_pos,env_cfg) - q ) * kp - dq * kd
    return torque_out

def get_dof_joint_names(model):
    '''
    获取所有关节名称（除浮动关节）
    '''
    dof_joint_names = []
    for i in range(model.njnt):
        dof_joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))
    return dof_joint_names[1:]

def get_default_dof_pos(model, env_cfg):
    '''
    获取初始关节角度,按mujoco顺序排列
    '''
    default_dof_pos = np.zeros(env_cfg.env.num_actions)
    dof_joint_names = get_dof_joint_names(model)
    keys = list(env_cfg.init_state.default_joint_angles.keys())
    # print(keys)
    for i in range(len(dof_joint_names)):
        index = [s for s in keys if dof_joint_names[i] in s]
        default_dof_pos[i] = env_cfg.init_state.default_joint_angles[index[0]]
    return default_dof_pos

def order_mujoco2isaacgym(mujoco_joints, env_cfg):
    '''
    将mujoco中关节信息数据排序更换为isaacgym中的关节排序
    '''
    '''
    isaacgym:
    ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_pitch_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_elbow_pitch_joint', 
     'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_pitch_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_elbow_pitch_joint']
    mujoco:
    ['left_shoulder_pitch', 'left_shoulder_roll', 'left_elbow_pitch', 'right_shoulder_pitch', 'right_shoulder_roll', 'right_elbow_pitch', 'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 'left_knee_pitch', 'left_ankle_pitch', 
    'left_ankle_roll', 'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw', 'right_knee_pitch', 'right_ankle_pitch', 'right_ankle_roll']
    '''
    if env_cfg.env.num_actions != len(mujoco_joints):
        raise ValueError(f"自由度维数错误:预期{env_cfg.env.num_actions}个关节,实际收到{len(mujoco_joints)}个")
    isaacgym_joints = np.zeros(len(mujoco_joints))
    isaacgym_joints[:6] = mujoco_joints[6:12]
    isaacgym_joints[6:9] = mujoco_joints[0:3]
    isaacgym_joints[9:15] = mujoco_joints[12:]
    isaacgym_joints[15:] = mujoco_joints[3:6]
    return isaacgym_joints

def run_mujoco(policy, cfg, env_cfg, save_plot=False):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    print("Load mujoco xml from:", cfg.sim_config.mujoco_model_path)
    # load model xml
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    # simulation timestep
    model.opt.timestep = cfg.sim_config.dt
    # model data
    data = mujoco.MjData(model)
    num_actuated_joints = env_cfg.env.num_actions  # This should match the number of actuated joints in your model
    default_dof_pos = get_default_dof_pos(model, env_cfg)
    #设定机器人初始角度
    # data.qpos[:3] = [0, 0, 0.6]
    data.qpos[-num_actuated_joints:] = default_dof_pos
    # print("data.qpos:", data.qpos)
    max_tau = np.zeros(env_cfg.env.num_actions)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)
    target_q = np.zeros((env_cfg.env.num_actions), dtype=np.double)
    action = np.zeros((env_cfg.env.num_actions), dtype=np.double)
    #历史观测
    hist_obs = deque()
    for _ in range(env_cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, env_cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 1
    logger = Logger(cfg.sim_config.dt)
    
    stop_state_log = 40000

    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    #初始化数据记录
    steps = int(cfg.sim_config.sim_duration / cfg.sim_config.dt)
    t = 0
    time_data = np.zeros(steps)
    torque_data = np.zeros((steps, model.nu))
    torque_sum = np.zeros(steps)

    for _ in range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)):
        # Obtain an observation
        q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces = get_obs(data,model)
        q = q[-env_cfg.env.num_actions:]
        dq = dq[-env_cfg.env.num_actions:]
        # q = order_mujoco2isaacgym(q)
        # dq = order_mujoco2isaacgym(dq)

        base_z = base_pos[2]
        foot_z = foot_positions
        foot_force_z = foot_forces
        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
            ####### for stand only #######
            if hasattr(env_cfg.commands,"sw_switch"):
                vel_norm = np.sqrt(x_vel_cmd**2 + y_vel_cmd**2 + yaw_vel_cmd**2)
                if env_cfg.commands.sw_switch and vel_norm <= env_cfg.commands.stand_com_threshold:
                    count_lowlevel = 0
                    
            obs = np.zeros([1, env_cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            #依序存入obs信息
            '''
            摘自x1_dh_stand_env.py:
            # obs q and dq
            q = (self.lagged_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
            dq = self.lagged_dof_vel * self.obs_scales.dof_vel  
            # 47+18
            obs_buf = torch.cat((
                self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
                q,    # 12->18
                dq,  # 12->18
                self.actions,   # 12->18
                self.lagged_base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.lagged_base_euler_xyz * self.obs_scales.quat,  # 3
            ), dim=-1)
            '''
            #将控制指令存入obs前num_commands项
            if env_cfg.env.num_commands == 5:
                obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / env_cfg.rewards.cycle_time)
                obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / env_cfg.rewards.cycle_time)
                obs[0, 2] = x_vel_cmd * env_cfg.normalization.obs_scales.lin_vel
                obs[0, 3] = y_vel_cmd * env_cfg.normalization.obs_scales.lin_vel
                obs[0, 4] = yaw_vel_cmd * env_cfg.normalization.obs_scales.ang_vel
            if env_cfg.env.num_commands == 3:
                obs[0, 0] = x_vel_cmd * env_cfg.normalization.obs_scales.lin_vel
                obs[0, 1] = y_vel_cmd * env_cfg.normalization.obs_scales.lin_vel
                obs[0, 2] = yaw_vel_cmd * env_cfg.normalization.obs_scales.ang_vel
            #当前角度与初始姿态之差(将mujoco顺序换为isaacgym顺序)
            obs[0, env_cfg.env.num_commands:env_cfg.env.num_commands+env_cfg.env.num_actions] = (order_mujoco2isaacgym((q - default_dof_pos), env_cfg)) * env_cfg.normalization.obs_scales.dof_pos
            #关节速度
            obs[0, env_cfg.env.num_commands+env_cfg.env.num_actions:env_cfg.env.num_commands+2*env_cfg.env.num_actions] = order_mujoco2isaacgym(dq, env_cfg) * env_cfg.normalization.obs_scales.dof_vel
            #
            obs[0, env_cfg.env.num_commands+2*env_cfg.env.num_actions:env_cfg.env.num_commands+3*env_cfg.env.num_actions] = action
            obs[0, env_cfg.env.num_commands+3*env_cfg.env.num_actions:env_cfg.env.num_commands+3*env_cfg.env.num_actions+3] = omega
            obs[0, env_cfg.env.num_commands+3*env_cfg. env.num_actions+3:env_cfg.env.num_commands+3*env_cfg.env.num_actions+6] = eu_ang
            
            ####### for stand only #######
            if env_cfg.env.add_stand_bool:
                vel_norm = np.sqrt(x_vel_cmd**2 + y_vel_cmd**2 + yaw_vel_cmd**2)
                stand_command = (vel_norm <= env_cfg.commands.stand_com_threshold)
                obs[0, -1] = stand_command
            
            # print("vel_cmd:",x_vel_cmd, y_vel_cmd, yaw_vel_cmd)
            # print("vel:", dq)
            # print("pos:", q)

            obs = np.clip(obs, -env_cfg.normalization.clip_observations, env_cfg.normalization.clip_observations)

            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, env_cfg.env.num_observations], dtype=np.float32)
            for i in range(env_cfg.env.frame_stack):
                policy_input[0, i * env_cfg.env.num_single_obs : (i + 1) * env_cfg.env.num_single_obs] = hist_obs[i][0, :]
            #计算动作
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, -env_cfg.normalization.clip_actions, env_cfg.normalization.clip_actions)
            target_q = action * env_cfg.control.action_scale
            # print("target_q:",target_q)

        target_dq = np.zeros((env_cfg.env.num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(default_dof_pos, target_q, order_mujoco2isaacgym(q, env_cfg), cfg.robot_config.kps,
                        target_dq, order_mujoco2isaacgym(dq, env_cfg), cfg.robot_config.kds, cfg)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        # print("tau:",tau)

        
        data.ctrl = tau
        applied_tau = data.actuator_force
        # mask_tau = np.abs(max_tau) < np.abs(applied_tau)
        # max_tau[mask_tau] = applied_tau[mask_tau]
        # if np.any(mask_tau):
        #     print(max_tau)
        
        mujoco.mj_step(model, data)
        #记录力矩数据
        time_data[t] = t
        torque_data[t, :] = applied_tau
        torque_sum[t] = sum(abs(torque_data[t, :]))
        t += 1

        viewer.render()

        count_lowlevel += 1
        idx = 5
        dof_pos_target = target_q + default_dof_pos
        if _ < stop_state_log:
            dict = {
                    'base_height': base_z,
                    'foot_z_l': foot_z[0],
                    'foot_z_r': foot_z[1],
                    'foot_forcez_l': foot_force_z[0],
                    'foot_forcez_r': foot_force_z[1],
                    'base_vel_x': v[0],
                    'command_x': x_vel_cmd,
                    'base_vel_y': v[1],
                    'command_y': y_vel_cmd,
                    'base_vel_z': v[2],
                    'base_vel_yaw': omega[2],
                    'command_yaw': yaw_vel_cmd,
                    'dof_pos_target': dof_pos_target[idx],
                    'dof_pos': q[idx],
                    'dof_vel': dq[idx],
                    'dof_torque': applied_tau[idx],
                    'cmd_dof_torque': tau[idx],
                }

            # add dof_pos_target
            for i in range(env_cfg.env.num_actions):
                dict[f'dof_pos_target[{i}]'] = dof_pos_target[i].item()

            # add dof_pos
            for i in range(env_cfg.env.num_actions):
                dict[f'dof_pos[{i}]'] = q[i].item()

            # add dof_torque
            for i in range(env_cfg.env.num_actions):
                dict[f'dof_torque[{i}]'] = applied_tau[i].item()

            # add dof_vel
            for i in range(env_cfg.env.num_actions):
                dict[f'dof_vel[{i}]'] = dq[i].item()
            logger.log_states(dict=dict)
        
        elif _== stop_state_log:
            logger.plot_states()

    viewer.close()
    motor_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_pitch_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_elbow_pitch_joint', 
     'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_pitch_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_elbow_pitch_joint']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    
    # 绘制折线图
    plt.figure(figsize=(12, 8))
    
    # 为每个电机绘制一条折线
    # for i in range(model.nu):
    #     plt.plot(time_data, torque_data[:, i], label=motor_names[i], linewidth=1.5)
    plt.plot(time_data, torque_data[:, 4], label=motor_names[4], linewidth=1.5)
    plt.plot(time_data, torque_data[:, 5], label=motor_names[5], linewidth=1.5)
    plt.plot(time_data, torque_data[:, 13], label=motor_names[13], linewidth=1.5)
    plt.plot(time_data, torque_data[:, 14], label=motor_names[14], linewidth=1.5)
    # plt.plot(time_data, torque_data[:, 0], label=motor_names[0], linewidth=1.5)
    # plt.plot(time_data, torque_data[:, 1], label=motor_names[1], linewidth=1.5)
    # plt.plot(time_data, torque_data[:, 9], label=motor_names[9], linewidth=1.5)
    # plt.plot(time_data, torque_data[:, 10], label=motor_names[10], linewidth=1.5)

    # 图表设置
    plt.title('Joint Torque - Time', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Torque (N·m)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧
    plt.tight_layout()  # 自动调整布局

    if save_plot:
        save_dir = f"/home/bblabhumanoid/agibot_x1_train-18DOFs/sim_data/{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        plot_filename = f"{save_dir}/{timestamp}_torque_plot.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"折线图已保存至: {plot_filename}")
        csv_filename = f"{save_dir}/{timestamp}_joint_torque_data.csv"
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time (s)'] + motor_names)
            for i in range(steps):
                writer.writerow([time_data[i]] + torque_data[i, :].tolist())
        print(f"数据已保存至: {csv_filename}")
    else:
        plt.show()


    # 画总力矩图
    plt.figure(figsize=(12, 8))
    plt.plot(time_data, torque_sum,label = "Total torch" ,linewidth =1.5)
    # 图表设置
    plt.title('Joint Torque - Time', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Torque (N·m)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧
    plt.tight_layout()  # 自动调整布局


    if save_plot:
        save_dir = f"/home/bblabhumanoid/agibot_x1_train-18DOFs/sim_data/{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        plot_filename = f"{save_dir}/{timestamp}_torque_sum.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"折线图已保存至: {plot_filename}")
        csv_filename = f"{save_dir}/{timestamp}_torque_sum.csv"
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time (s)','Total Torch(N.m)'])
            for i in range(steps):
                writer.writerow([[time_data[i]], torque_sum[i]])
        print(f"数据已保存至: {csv_filename}")
    else:
        plt.show()
    
    return 0

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str,
                        help='Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.')
    parser.add_argument('--task', type=str, required=True,
                        help='task name.')
    parser.add_argument('--data', action='store_true',help='输出力矩图表')
    args = parser.parse_args()
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    print(args.data)
    class Sim2simCfg():

        class sim_config:
            mujoco_model_path = env_cfg.asset.xml_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            sim_duration = 10.0
            dt = 0.001
            decimation = 10

        class robot_config:
            # get PD gain
            kps = np.array([env_cfg.control.stiffness[joint] for joint in env_cfg.control.stiffness.keys()]*2, dtype=np.double)
            kds = np.array([env_cfg.control.damping[joint] for joint in env_cfg.control.damping.keys()]*2, dtype=np.double)

            tau_limit = 500. * np.ones(env_cfg.env.num_actions, dtype=np.double)  # 定义关节力矩的限制

            default_dof_pos = np.array(list(env_cfg.init_state.default_joint_angles.values()))
            

    # load model
    root_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', args.task, 'exported_policies')
    if args.load_model == None:
        jit_path = os.listdir(root_path)
        jit_path.sort()
        model_path = os.path.join(root_path, jit_path[-1])
    else:
        model_path = os.path.join(root_path, args.load_model)
    jit_name = os.listdir(model_path)
    model_path = os.path.join(model_path,jit_name[-1])
    policy = torch.jit.load(model_path)
    print("Load model from:", model_path)

    run_mujoco(policy, Sim2simCfg(), env_cfg, args.data)

    