import sys
sys.path.insert(0, './')
from rlf.envs.dm_control_interface import DmControlInterface
from rlf.envs.env_interface import register_env_interface
# from gym import core
import gym

from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

class CartPoleEnv(gym.core.Wrapper):
    def __init__(self):
        env = gym.make("CartPole-v0")
        super().__init__(env)
        self.total_reward = 0
    # def reset(self):
    #     self.found_goal = False
    #     return super().reset()

    def step(self, a):
        obs, reward, done, info = super().step(a)
        # if reward == 1.0:
        #     self.total_reward += 1
        # # info['ep_found_goal'] = 0.0
        # if done:
        #     print(self.total_reward)
        #     self.total_reward = 0
            
        if reward == 1.0:
            self.total_reward += 1
        info['ep_found_goal'] = 0.0
        if done:
            if self.total_reward == 200:
                info['ep_found_goal'] = 1.0
            self.total_reward = 0
        return obs, reward, done, info
    
    def env_trans_fn(self, env, set_eval):
        return GoalCheckerWrapper(env)

    def get_special_stat_names(self):
        return ['ep_found_goal']

