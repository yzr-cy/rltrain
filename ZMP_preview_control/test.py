import numpy as np
import gymnasium

from gymnasium_env.envs.zmpEnv import zmpEnv

gymnasium.make('gymnasium_env:gymnasium_env/Zmp-v0')

a = np.random.uniform(low=np.array([-2,-1,-1,-1],dtype=np.float32), high=np.array([1,1,1,1], dtype=np.float32), size=(4,))

print(a)


for i in range(0,10):
    print(i)

print(np.random.randint(0,1))