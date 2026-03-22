import mujoco
from mujoco import viewer
import numpy as np
import time
from humanoid.envs import *

env_cfg, _ = task_registry.get_cfgs(name='x1_dh_stand')
# env, _ = task_registry.make_env(name='x1_dh_stand')

model =  mujoco.MjModel.from_xml_path(r"resources/robots/x1/mjcf/xyber_x1_flat.xml")
data = mujoco.MjData(model)

body_name = []
for i in range(model.nbody):
    body_name.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))

# print("model.nbody:",model.nbody)
# print("body_name:",body_name)
print("model.njnt:",model.njnt)



def get_dof_joint_names(model):
    dof_joint_names = []
    for i in range(model.njnt):
        dof_joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))
    # print(dof_joint_names[1:])
    return dof_joint_names[1:]

# get_dof_joint_names(model)

def get_default_dof_pos(model, env_cfg):
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

print(env_cfg.env.num_actions)
q = get_default_dof_pos(model, env_cfg)
print(q)
print(env_cfg.init_state.default_joint_angles)
kps = np.array([env_cfg.control.stiffness[joint] for joint in env_cfg.control.stiffness.keys()]*2, dtype=np.double)
kds = np.array([env_cfg.control.damping[joint] for joint in env_cfg.control.damping.keys()]*2, dtype=np.double)
print("kps",kps)
print("kds",kds)
print("model.nu:",model.nu)

# print(len(q))
# q = order_mujoco2isaacgym(q, env_cfg)
# print(q)

# for motor_id in range(model.nu):  # model.nu 是电机总数
#     motor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MOTOR, motor_id)
#     print(f"motor_id: {motor_id} → name: {motor_name} → data.ctrl[{motor_id}]")