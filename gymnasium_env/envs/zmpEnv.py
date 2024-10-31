import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import matplotlib.pyplot as plt

import torch.nn as nn

import torch

import os

current_file_path = __file__
absolute_path = os.path.dirname(current_file_path)

class zmpEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        T = 0.06
        zc = 0.93
        g = 9.81
        self.h = 10
        self.hipWidth = 0.14
        self.trajT = T*self.h
        self.K_step = 0.1
                                #   1：表示左， -1：右
        self.timeArray = np.array([1,1,1,1,1,-1,-1,-1,-1,-1])
        self.timePoint = 0
        self.A = np.matrix([[1,0,T,0,T*T/2,0],
                            [0,1,0,T,0,T*T/2],
                            [0,0,1,0,T,0],
                            [0,0,0,1,0,T],
                            [0,0,0,0,1,0],
                            [0,0,0,0,0,1]])
        
        self.B = np.matrix([[T*T*T/6, 0],
                            [0, T*T*T/6],
                           [T*T/2, 0],
                           [0, T*T/2],
                           [T,0],
                           [0,T]])
        
        self.C = np.matrix([[1,0,0,0,-zc/g,0],
                            [0,1,0,0,0,-zc/g]])
                                                        #    xdot,xddot,ydot,yddot,vx_des,timePoint
        self.observation_space = spaces.Box(low = np.array([-10,-10,-10,-10,-1,0], dtype=np.float32),
                                            high=np.array([10,10,10,10,1,self.h],dtype=np.float32),
                                            dtype=np.float32)
                                                        #    ux0,uy0,...
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(20,), dtype=np.float32)
                            # 当前状态 x,y,xdot,ydot,xddot,yddot
        self._agent_state = np.array([[0],[0],[0],[0],[0],[0]],dtype=np.float32)
        self.state_Horizon = np.zeros(6*self.h, dtype=np.float32)
        self.zmp_Horizon = np.zeros(2*self.h, dtype=np.float32)
        self.vx_des = 0
        self.eposide = 500
        self.count = 0
        self.endEpisode = False

    def _get_obs(self):
        obs = np.zeros(6,dtype=np.float32)
        obs[0] = self._agent_state[1]
        obs[1] = self._agent_state[2]
        obs[2] = self._agent_state[4]
        obs[3] = self._agent_state[5]
        obs[4] = self.vx_des
        obs[5] = self.timePoint
        return obs
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self._agent_state = np.zeros((6,1),dtype=np.float32)
        self._agent_state[:2,0] = np.random.uniform(low=np.array([-100,-1], dtype=np.float32), 
                                                    high=np.array([100,1], dtype=np.float32), 
                                                    size=(2,))
        self._agent_state[2:,0] = np.random.uniform(low=np.array([-1,-1,-1,-1],dtype=np.float32), 
                                                  high=np.array([1,1,1,1], dtype=np.float32), 
                                                  size=(4,))
        
        self.timePoint = np.random.randint(0,self.h)

        self.vx_max = 1.0

        self.vy_max = 1.0

        self.vx_des = np.random.uniform(low=-self.vx_max,high=self.vx_max)

        self.count = 0

        self.endEpisode = False
        
        return self._get_obs(),{}
    
    def step(self, action):
        # 对action进行缩放
        action = action*10

        stateTemp = self._agent_state.reshape(6,1)
        zmpState = np.zeros(2,dtype=np.float32)
        for i in range(0,self.h):
            zmpState = self.C * stateTemp
            stateTemp = self.A * stateTemp + self.B * action[i*2:i*2+2].reshape(2,1)
            self.state_Horizon[i*6:i*6+6] = stateTemp.squeeze()
            self.zmp_Horizon[i*2:i*2+2] = zmpState.squeeze()
        
        self.generate_Foot()
        self.timePoint = self.timePoint + 1
        if self.timePoint == self.h:
            self.timePoint = 0
        
        self._agent_state = self.state_Horizon[:6]
        rewards = 0
        rewards += self._reward_zmpTrace()*0
        rewards += self._reward_action(action)
        rewards += self._reward_posFoot()*0

        TenState_reward = self._reward_faraway()
        rewards += TenState_reward*0

        rewards += self._reward_velTrace()


        self.count+=1
        if self.count > self.eposide:
            truncated = True
        else:
            truncated = False


        if -TenState_reward > 50:
            self.endEpisode = True
            rewards-=100
        else:
            self.endEpisode = False

        self.vx_des += np.random.uniform(-0.1,0.1)
        
        while self.vx_des > self.vx_max-0.2:
            self.vx_des -= 0.1
        while self.vx_des < -self.vx_max-0.2:
            self.vx_des += 0.1

        return self._get_obs(), rewards, self.endEpisode, truncated, {}


    def _reward_zmpTrace(self):
        return np.exp(-(np.linalg.norm(self.footStep - self.zmp_Horizon)))

    def _reward_action(self, action):
        return -np.linalg.norm(action)
    
    def _reward_posFoot(self):
        pos_horizon = np.concatenate([self.state_Horizon[6*i:6*i+2] for i in range(10)])
        return np.exp(-np.linalg.norm(self.footStep - pos_horizon))
        
    def _reward_faraway(self):
        TenState =np.concatenate([self._agent_state[:2] for _ in range(10)])
        pos_horizon = np.concatenate([self.state_Horizon[6*i:6*i+2] for i in range(10)])
        return -np.linalg.norm(pos_horizon - TenState)
    
    def _reward_velTrace(self):
        vel_des = np.concatenate([np.array([self.vx_des,0]) for _ in range(10)])
        vel_state = np.concatenate([self.state_Horizon[6*i+2:6*i+4] for i in range(10)])
        return np.exp(-np.linalg.norm(vel_des - vel_state))
    
    def generate_Foot(self):
        self.footStep = np.zeros(self.h*2, dtype=np.float32)
        k = 1
        for i in range(0,self.h):
            x = self._agent_state[0]
            y = self._agent_state[1]
            vx = self._agent_state[2]
            vy = self._agent_state[3]
            if np.abs(vx) > self.vx_max or np.abs(vy) > self.vy_max:
                self.endEpisode = True

            vx_des = self.vx_des
            if i > 0:
                if self.timeArray[(i+self.timePoint)%self.h] != self.timeArray[(i-1+self.timePoint)%self.h]:
                    k = k + 1
            self.footStep[i*2] = x + vx * self.trajT / 2 * k + self.K_step * (vx - vx_des)
            self.footStep[i*2+1] = self.hipWidth*(self.timeArray[(i+self.timePoint)%self.h]) + y + vy * self.trajT / 2 * k + self.K_step * (vy)

    def render(self):
        pass

    def close(self):
        pass





if __name__ == "__main__":

    env = zmpEnv()

    # check_env(env)

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[128, 128]), 
        activation_fn=nn.ReLU             
    )


    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs,verbose=1,tensorboard_log= absolute_path+"/../tensorboard/")

    model.learn(total_timesteps=6000000)

    save_path = absolute_path+"/../models/zmpModel.pth"

    torch.save(model.policy.state_dict(), save_path)
    
    print("策略网络结构：", model.policy.mlp_extractor.policy_net)
    print("价值网络结构：", model.policy.mlp_extractor.value_net)

