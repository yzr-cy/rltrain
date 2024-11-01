from zmpEnv import zmpEnv
from stable_baselines3 import PPO
import torch
import matplotlib.pyplot as plt 
import torch.nn as nn
import numpy as np
import os
import argparse


if __name__ == "__main__":
    current_file_path = __file__
    absolute_path = os.path.dirname(current_file_path)
    render_enabled = True   
    parser = argparse.ArgumentParser(description="输入.pth文件的路径")
    parser.add_argument("path", type=str, help="输入.pth文件的路径")
    args = parser.parse_args()

    env = zmpEnv()
    env.scale = 3*env.scale
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[128, 128]), 
        activation_fn=nn.ReLU             
    )
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs,verbose=1)
    load_path = args.path
    model.policy.load_state_dict(torch.load(load_path))

    # 演示训练结果
    num_episodes = 5  # 设置要演示的回合数
    for episode in range(num_episodes):
        obs,_ = env.reset()
        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():  # 不计算梯度
                action, _ = model.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward

        print(f'Episode {episode + 1}: Total Reward: {total_reward}')

    env.close()