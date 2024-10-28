import numpy as np
import matplotlib.pyplot as plt

# 参数设置
step_time = 0.6            # 单步周期 (秒)
prediction_time = 0.6      # 预测时间 (秒)
time_step = 0.06           # 时间步长 (秒)
step_length = 0.3          # 每步长度 (米)
com_height = 0.3           # 倒立摆模型的重心高度 (米)
g = 9.81                   # 重力加速度

num_steps = int(prediction_time / time_step)  # 预测步数

# 初始条件
zmp_ref = np.zeros(num_steps)      # ZMP参考轨迹
com_position = np.zeros(num_steps) # 重心位置
com_velocity = np.zeros(num_steps) # 重心速度
com_acceleration = np.zeros(num_steps)  # 重心加速度

# 设置左右脚的ZMP位置交替
left_foot_pos = 0.0
right_foot_pos = step_length

for i in range(num_steps):
    t = i * time_step
    # 设置ZMP参考轨迹，左右脚交替踏步
    if t < step_time / 2:
        zmp_ref[i] = left_foot_pos
    else:
        zmp_ref[i] = right_foot_pos

# 计算预观增益
A = np.array([[1, time_step, 0.5 * time_step**2],
              [0, 1, time_step],
              [0, 0, 1]])
B = np.array([[time_step**3 / 6], [time_step**2 / 2], [time_step]])
C = np.array([1, 0, -com_height / g])

# 预观控制增益
Qe = 1.0  # ZMP误差权重
Qx = np.diag([0.01, 0.01, 0.01])  # 状态权重
R = np.array([[1e-6]])            # 控制输入权重

# 初始化误差累积和状态
e_sum = 0.0
x = np.array([[0], [0], [0]])  # 初始重心位置、速度、加速度

# 步态控制主循环
for i in range(num_steps):
    # 计算当前误差
    e = zmp_ref[i] - C @ x
    e_sum += e
    
    # 控制增益计算（简单的PID控制）
    u = Qe * e + Qx @ x + R @ e_sum

    # 状态更新
    x = A @ x + B * u
    com_position[i] = x[0, 0]  # 更新重心位置
    com_velocity[i] = x[1, 0]  # 更新重心速度
    com_acceleration[i] = x[2, 0]  # 更新重心加速度

# 可视化
time = np.arange(0, prediction_time, time_step)

plt.figure(figsize=(10, 6))
plt.plot(time, zmp_ref, label='ZMP Reference', linestyle='--')
plt.plot(time, com_position, label='COM Position')
plt.plot(time, com_velocity, label='COM Velocity')
plt.plot(time, com_acceleration, label='COM Acceleration')
plt.xlabel("Time (s)")
plt.ylabel("Position / Velocity / Acceleration")
plt.legend()
plt.title("ZMP-based Gait Planning with Preview Control")
plt.grid()
plt.show()

