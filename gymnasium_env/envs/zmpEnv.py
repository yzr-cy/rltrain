import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import matplotlib.pyplot as plt

import torch.nn as nn

import torch

import os

import pygame

from pynput import keyboard

import threading

current_file_path = __file__
absolute_path = os.path.dirname(current_file_path)
render_enabled = True



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
        self.window_size_h = 1024
        self.window_size_w = 512*3

        self.render_mode = "human"
        self.window = None
        self.clock = None



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
        # self._agent_state[:2,0] = np.random.uniform(low=np.array([-100,-1], dtype=np.float32), 
        #                                             high=np.array([100,1], dtype=np.float32), 
        #                                             size=(2,))
        self._agent_state[:2,0] = np.array([0,0])
        self._agent_state[2:,0] = np.random.uniform(low=np.array([-1,-1,-1,-1],dtype=np.float32), 
                                                  high=np.array([1,1,1,1], dtype=np.float32), 
                                                  size=(4,))
        
        self.timePoint = np.random.randint(0,self.h)

        self.vx_max = 1.0

        self.vy_max = 1.0

        self.vx_des = np.random.uniform(low=-self.vx_max,high=self.vx_max)

        self.count = 0

        self.endEpisode = False

        self.updateVdes_count = 0

        
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
        self.pos_horizon = np.concatenate([self.state_Horizon[6*i:6*i+2] for i in range(10)])
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

        self.updateVdes_count+=1
        if self.updateVdes_count > 50:
            self.vx_des += np.random.uniform(-0.1,0.1)
            self.updateVdes_count = 0
        
        while self.vx_des > self.vx_max-0.2:
            self.vx_des -= 0.1
        while self.vx_des < -self.vx_max-0.2:
            self.vx_des += 0.1


        if self.render_mode == "human" and render_enabled:
            self._render_frame()




        return self._get_obs(), rewards, self.endEpisode, truncated, {}


    def _reward_zmpTrace(self):
        return np.exp(-(np.linalg.norm(self.footStep - self.zmp_Horizon)))

    def _reward_action(self, action):
        return -np.linalg.norm(action)
    
    def _reward_posFoot(self):
        return np.exp(-np.linalg.norm(self.footStep - self.pos_horizon))
        
    def _reward_faraway(self):
        TenState =np.concatenate([self._agent_state[:2] for _ in range(10)])
        return -np.linalg.norm(self.pos_horizon - TenState)
    
    def _reward_velTrace(self):
        vel_des = np.concatenate([np.array([self.vx_des,0]) for _ in range(10)])
        vel_state = np.concatenate([self.state_Horizon[6*i+2:6*i+4] for i in range(10)])
        return np.exp(-np.linalg.norm(vel_des - vel_state))
    
    def generate_Foot(self):
        self.footStep = np.zeros(self.h*2, dtype=np.float32)
        k = 0
        self.foot_draw = []
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
            self.footStep[i*2] = x + vx * self.trajT / 2 * k + self.K_step * (vx - vx_des) * k
            self.footStep[i*2+1] = self.hipWidth*(self.timeArray[(i+self.timePoint)%self.h]) + y + vy * self.trajT / 2 * k + self.K_step * (vy) * k
            self.foot_draw.append(np.array([self.footStep[i*2], self.footStep[i*2+1]]))

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()  
            pygame.quit()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size_w, self.window_size_h))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        self.bias = np.array([self.window_size_w / 2, self.window_size_h/2])
        # 216个像素表示1000mm
        self.scale = 10
        # 刷新画布
        self.canvas = pygame.Surface((self.window_size_w, self.window_size_h))
        self.canvas.fill((255, 255, 255))
        # 画坐标轴
        pygame.draw.line(self.canvas,(0,0,0),(0,self.bias[1]),(self.window_size_w,self.bias[1]),width=1)
        pygame.draw.line(self.canvas,(0,0,0),(self.bias[0],0),(self.bias[0],self.window_size_h),width=1)


        self._draw_rect(self.foot_draw)
        self.pos_draw = []
        for i in range(self.h):
            self.pos_draw.append(np.array([self.pos_horizon[i*2], self.pos_horizon[i*2+1]]))
        self._draw_circle(self.pos_draw)
        self.CurrPos_Draw =[self._agent_state[:2]]
        self._draw_circle(self.CurrPos_Draw,color=(0,0,255))
        self._draw_arrow(self.CurrPos_Draw[0], self.CurrPos_Draw[0] + np.array([self.vx_des,0]))

        if self.render_mode == "human":
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(4)
    
    def _draw_rect(self, pos: list):
        num = len(pos)
        for i in range(num):
            one_pos = pos[i]
            pygame.draw.rect(self.canvas, (255, 0 ,0), pygame.Rect(self.bias[0] + one_pos[0]*self.scale,self.bias[1] - one_pos[1]*self.scale, 10, 10))

    def _draw_circle(self, pos: list, color = (0,255,0)):
        num = len(pos)
        for i in range(num):
            one_pos = pos[i]
            pygame.draw.circle(self.canvas, color= color, center=(self.bias[0] + one_pos[0]*self.scale,self.bias[1] - one_pos[1]*self.scale), radius=2)

    def _draw_arrow(self, start, end, color = (0,255,255), width = 1):
        """绘制箭头，宽度根据输入值决定箭头的大小"""
        startLine = (self.bias[0] + start[0]*self.scale, self.bias[1] - start[1]*self.scale)
        endLine = (self.bias[0] + end[0]*self.scale, self.bias[1] - end[1]*self.scale)
        if startLine[0] > self.window_size_w or startLine[1] > self.window_size_h or startLine[0] < 0 or startLine[1] < 0:
            self.endEpisode = True
        pygame.draw.line(self.canvas, color, startLine, endLine, width)
        

def on_press(key):
    try:
        if key.char == 'r':  # 检测按下 'q' 键
            print("You pressed 'r'.")
            global render_enabled
            render_enabled = not render_enabled
            return False  # 返回 False 以停止监听
    except AttributeError:
        pass


def waitKey():
    while True:
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()


thread1 = threading.Thread(target=waitKey)

thread1.start()

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

