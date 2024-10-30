from zmpEnv import zmpEnv
from stable_baselines3 import PPO
import torch
import matplotlib.pyplot as plt 
import torch.nn as nn
import numpy as np
import os

current_file_path = __file__
absolute_path = os.path.dirname(current_file_path)

np.random.seed(0)

def plot_rewards(rewards, title="Reward Over Time"):
    """
    绘制奖励曲线。
    
    参数:
    rewards (list or numpy array): 每个时间步的奖励列表或数组。
    title (str): 图表标题，默认为 "Reward Over Time"。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Reward")
    plt.xlabel("Time Steps")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


env = zmpEnv()
policy_kwargs = dict(
    net_arch=dict(pi=[256, 256], vf=[128, 128]), 
    activation_fn=nn.ReLU             
)

model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs,verbose=1)

# 加载 .pth 文件中的参数
load_path = absolute_path+"/../models/zmpModel.pth"
model.policy.load_state_dict(torch.load(load_path))



def step(action, obs):
    h  = 10
    T = 0.06
    zc = 0.93
    g = 9.81

    A = np.matrix([[1,0,T,0,T*T/2,0],
                    [0,1,0,T,0,T*T/2],
                    [0,0,1,0,T,0],
                    [0,0,0,1,0,T],
                    [0,0,0,0,1,0],
                    [0,0,0,0,0,1]])
        
    B = np.matrix([[T*T*T/6, 0],
                        [0, T*T*T/6],
                        [T*T/2, 0],
                        [0, T*T/2],
                        [T,0],
                        [0,T]])
    
    C = np.matrix([[1,0,0,0,-zc/g,0],
                        [0,1,0,0,0,-zc/g]])

    x0 = 5
    y0 = 0.0
    action = action*10
    stateTemp = np.zeros(6,dtype=np.float32)
    stateTemp[0] = x0
    stateTemp[1] = y0
    stateTemp[2:] = obs[:4]

    state_Horizon = np.zeros(6*h, dtype=np.float32)
    zmp_Horizon = np.zeros(2*h, dtype=np.float32)

    for i in range(0,h):
        zmpState = C * stateTemp.reshape(6,1)
        stateTemp = A * stateTemp.reshape(6,1) + B * action[i*2:i*2+2].reshape(2,1)
        state_Horizon[i*6:i*6+6] = stateTemp.squeeze()
        zmp_Horizon[i*2:i*2+2] = zmpState.squeeze()
    return state_Horizon


vx = 0.6
timepoint = 8

obs = np.array([0,0,0,0,vx,timepoint],dtype=np.float32)

action,_ = model.predict(obs)

state_horizon = step(action, obs)


x = state_horizon[0::6]
y = state_horizon[1::6]
x = np.concatenate([[state_horizon[6*i]] for i in range(10)])
y = np.concatenate([[state_horizon[6*i+1]] for i in range(10)])


plt.plot(x,y)
plt.show()



rewards = []
for _ in range(200):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    rewards.append(reward)
    print(action)

plot_rewards(rewards)

print(model.policy.action_net)

