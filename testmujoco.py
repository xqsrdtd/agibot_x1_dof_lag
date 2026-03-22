import mujoco
from mujoco import viewer
import numpy as np
import time

model =  mujoco.MjModel.from_xml_path(r"resources/robots/x1/mjcf/xyber_x1_flat.xml")
model.opt.timestep = 0.005 #设置单步长时间
data = mujoco.MjData(model)


# dof_names: ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_pitch_joint', 
# 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_elbow_pitch_joint', 
# 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_pitch_joint', 'right_ankle_pitch_joint', 
# 'right_ankle_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_elbow_pitch_joint']


# 1. 查看所有关节名称和对应的qpos索引
print("所有关节信息:")
print(len(data.qpos))
# default_joint_angles = [0.0, -0.07, 0, 0, -0.07, 0, 0.4, 0.05, -0.31, 0.49, -0.21, 0, -0.4, -0.05 ,0.31, 0.49, -0.21, 0]
default_joint_angles = [ 0.14, -0.21, 0.72, 0.14, -0.21, 0.72, 0.4751, 0.0616, -0.3655, 0.6280, -0.2562,  0.0034, -0.4751, -0.0616,  0.3655, 0.6280, -0.2562,  0.0034]
final_swing_joint_delta_pos = [-0.5 , 0, 0., -0.5, 0, 0., 0.25, 0.05, -0.11, 0.35, -0.16, 0.0, -0.25, -0.05, 0.11, 0.35, -0.16, 0.0]




data.qpos[:3]=[0,0,0.7]

data.qpos[7:]=default_joint_angles


# for body_id in range(model.nbody):
#     link_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
#     print(f"link ID: {body_id}, 名称: {link_name}")

# link ID: 0, 名称: world
# link ID: 1, 名称: x1-body
# link ID: 2, 名称: body_yaw
# link ID: 3, 名称: body_roll
# link ID: 4, 名称: body_pitch
# link ID: 5, 名称: left_shoulder_pitch
# link ID: 6, 名称: left_shoulder_roll
# link ID: 7, 名称: left_shoulder_yaw
# link ID: 8, 名称: left_elbow_pitch
# link ID: 9, 名称: left_elbow_yaw
# link ID: 10, 名称: left_wrist_pitch
# link ID: 11, 名称: left_wrist_roll
# link ID: 12, 名称: right_shoulder_pitch
# link ID: 13, 名称: right_shoulder_roll
# link ID: 14, 名称: right_shoulder_yaw
# link ID: 15, 名称: right_elbow_pitch
# link ID: 16, 名称: right_elbow_yaw
# link ID: 17, 名称: right_wrist_pitch
# link ID: 18, 名称: right_wrist_roll
# link ID: 19, 名称: left_hip_pitch_link
# link ID: 20, 名称: left_hip_roll_link
# link ID: 21, 名称: left_hip_yaw_link
# link ID: 22, 名称: lleft_knee_pitch_link
# link ID: 23, 名称: left_ankle_pitch_link
# link ID: 24, 名称: left_ankle_roll_link
# link ID: 25, 名称: right_hip_pitch_link
# link ID: 26, 名称: right_hip_roll_link
# link ID: 27, 名称: right_hip_yaw_link
# link ID: 28, 名称: right_knee_pitch_link
# link ID: 29, 名称: right_ankle_pitch_link
# link ID: 30, 名称: right_ankle_roll_link
mujoco.mj_step(model, data)
print(data.xpos[8])
print(data.xpos[15])
dist =  np.linalg.norm(data.xpos[8] - data.xpos[15])
print(dist)
data.qpos[7:]=default_joint_angles
mujoco.mj_step(model, data)

def get_body_xaxis_world(model, data, body_name):
    """
    获取指定连杆的X轴在世界坐标系下的方向向量
    
    参数:
        model: MuJoCo模型对象
        data: MuJoCo数据对象
        body_name: 连杆名称（字符串）
    
    返回:
        numpy数组: 世界坐标系下X轴的方向向量（单位向量）
    """
    # 将连杆名称转换为ID
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"连杆名称 '{body_name}' 不存在")
    
    # 获取旋转矩阵（本地坐标系 → 世界坐标系）
    # data.xmat是形状为(nbody, 9)的数组，每个元素是3x3旋转矩阵的扁平化
    rot_mat = data.xmat[body_id].reshape(3, 3)  # 重塑为3x3矩阵
    
    # 旋转矩阵的第一列即为本地X轴在世界坐标系下的方向向量
    x_axis_world = rot_mat[:, 0]  # 取第一列
    
    print(x_axis_world) 

def run_mujoco(model, data, default_joint_angles, final_swing_joint_delta_pos):
    with mujoco.viewer.launch_passive(model, data) as v: 
        data.qpos[:3]=[0,0,0.7]
        # v.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
        ref_dof_pos=np.zeros(18)
        phase = 0
        while v.is_running(): 
            data.qpos[:3]=[0,0,0.7]
            
            phase += 0.01
            if phase > 1:
                phase -= 1
            sin_pos = np.sin(2 * np.pi * phase)
            
            sin_pos_l = sin_pos
            sin_pos_r = sin_pos
            if sin_pos_l > 0:
                sin_pos_l = 0
                
            sin_pos_l_arm = sin_pos
            sin_pos_r_arm = sin_pos

            #左腿周期步态
            ref_dof_pos[6] = -sin_pos_l * final_swing_joint_delta_pos[6]
            ref_dof_pos[7] = -sin_pos_l * final_swing_joint_delta_pos[7]
            ref_dof_pos[8] = -sin_pos_l * final_swing_joint_delta_pos[8]
            ref_dof_pos[9] = -sin_pos_l * final_swing_joint_delta_pos[9]
            ref_dof_pos[10] = -sin_pos_l * final_swing_joint_delta_pos[10]
            ref_dof_pos[11] = -sin_pos_l * final_swing_joint_delta_pos[11]
            if sin_pos_r < 0:
                sin_pos_r = 0

            #右腿周期步态
            ref_dof_pos[12] = sin_pos_r * final_swing_joint_delta_pos[12]
            ref_dof_pos[13] = sin_pos_r * final_swing_joint_delta_pos[13]
            ref_dof_pos[14] = sin_pos_r * final_swing_joint_delta_pos[14]
            ref_dof_pos[15] = sin_pos_r * final_swing_joint_delta_pos[15]
            ref_dof_pos[16] = sin_pos_r * final_swing_joint_delta_pos[16]
            ref_dof_pos[17] = sin_pos_r * final_swing_joint_delta_pos[17]
            
            #摆手
            if sin_pos_l_arm < 0:
                sin_pos_l_arm *= 0.5            
            ref_dof_pos[0] = sin_pos_l_arm * final_swing_joint_delta_pos[0]
            ref_dof_pos[1] = sin_pos_l_arm * final_swing_joint_delta_pos[1]

            ref_dof_pos[2] = sin_pos_l_arm * final_swing_joint_delta_pos[2]
            
            if sin_pos_r_arm > 0:
                sin_pos_r_arm *= 0.5           
            ref_dof_pos[3] = -sin_pos_r_arm * final_swing_joint_delta_pos[3]
            ref_dof_pos[4] = -sin_pos_r_arm * final_swing_joint_delta_pos[4]

            ref_dof_pos[5] = -sin_pos_r_arm * final_swing_joint_delta_pos[5]  
            if abs(sin_pos)<0.1:
                ref_dof_pos = np.zeros(len(default_joint_angles))
            ref_dof_pos += default_joint_angles
            # print(ref_dof_pos)
            # data.qpos[:3]=[0,0,0.8]
            data.qpos[7:]=ref_dof_pos
            # print(data.qpos)
            # print("刘哥牛逼！")
            
            mujoco.mj_step(model, data)
            time.sleep(0.01)
            v.sync()
            
run_mujoco(model, data, default_joint_angles, final_swing_joint_delta_pos)
# get_body_xaxis_world(model, data, "right_ankle_roll_link")