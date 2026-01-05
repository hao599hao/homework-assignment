import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# 1. 读取无人机姿态数据
df = pd.read_csv('tracking.csv')
t_list = df['t'].values
qx_b = df['qx'].values  # 无人机四元数虚部x
qy_b = df['qy'].values  # 无人机四元数虚部y
qz_b = df['qz'].values  # 无人机四元数虚部z
qw_b = df['qw'].values  # 无人机四元数实部w

# 2. 题目给定固定参数
omega = 0.5  # 角速度，单位rad/s
alpha = np.pi / 12  # 圆锥半顶角，单位rad
cos_alpha = np.cos(alpha)
sin_alpha = np.sin(alpha)

# 3. 初始化执行器四元数存储列表
exec_qx = []
exec_qy = []
exec_qz = []
exec_qw = []

# 4. 逐时刻计算执行器世界系姿态
for i in range(len(t_list)):
    t = t_list[i]
    # 4.1 无人机四元数转旋转矩阵（世界系→机体系 ^W R_B）
    # scipy库要求输入格式为 (qw, qx, qy, qz)，与csv数据对应
    q_drone = [qw_b[i], qx_b[i], qy_b[i], qz_b[i]]
    R_WB = R.from_quat(q_drone).as_matrix()
    
    # 4.2 计算机体系→执行器系旋转矩阵 ^B R_D
    cos_omega_t = np.cos(omega * t)
    sin_omega_t = np.sin(omega * t)
    R_BD = np.array([
        [cos_omega_t, -sin_omega_t * cos_alpha, sin_omega_t * sin_alpha],
        [sin_omega_t, cos_omega_t * cos_alpha, -cos_omega_t * sin_alpha],
        [0, sin_alpha, cos_alpha]
    ])
    
    # 4.3 计算世界系→执行器系旋转矩阵 ^W R_D = ^W R_B × ^B R_D
    R_WD = R_WB @ R_BD
    
    # 4.4 旋转矩阵转四元数（输出格式：qx, qy, qz, qw）
    q_exec = R.from_matrix(R_WD).as_quat()
    
    # 4.5 第一步：四元数归一化
    q_exec = q_exec / np.linalg.norm(q_exec)
    
    # 4.6 第二步：qw<0时整体取反
    if q_exec[3] < 0:
        q_exec = -q_exec # 整体取反：qx→-qx, qy→-qy, qz→-qz, qw→-qw
    
    # 存储全量四元数分量
    exec_qx.append(q_exec[0])
    exec_qy.append(q_exec[1])
    exec_qz.append(q_exec[2])
    exec_qw.append(q_exec[3])

# 5. 绘制四元数变化曲线
plt.figure(figsize=(12, 6))
plt.plot(t_list, exec_qx, label='qx', linewidth=1.2, color='#1f77b4')
plt.plot(t_list, exec_qy, label='qy', linewidth=1.2, color='#ff7f0e')
plt.plot(t_list, exec_qz, label='qz', linewidth=1.2, color='#2ca02c')
plt.plot(t_list, exec_qw, label='qw', linewidth=1.2, color='#d62728')

# 图表格式优化（满足报告清晰度要求）
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Quaternion Value', fontsize=12)
plt.title('Executor Quaternion in World Frame (qw ≥ 0)', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(alpha=0.3, linestyle='--')
plt.xticks(np.arange(t_list.min(), t_list.max() + 0.5, 0.5), rotation=45)  # 每0.5秒一个tick，清晰易读
plt.yticks(np.arange(-1.0, 1.1, 0.2))
plt.tight_layout()

# 保存图片
plt.savefig('executor_quaternion_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 保存结果数据
result_df = pd.DataFrame({
    't': t_list,
    'qx': exec_qx,
    'qy': exec_qy,
    'qz': exec_qz,
    'qw': exec_qw
})
result_df.to_csv('executor_quaternion_result.csv', index=False)