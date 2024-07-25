from rlf.envs.env_interface import register_env_interface, EnvInterface
from gym import core
import numpy as np
import skimage.transform
import skimage.color
from gym import spaces


VIZ_RES = (30, 45)
class VizDoomWrapper(core.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
                high=self.observation_space.high[:VIZ_RES[0], :VIZ_RES[1], :1],
                low=self.observation_space.low[:VIZ_RES[0], :VIZ_RES[1], :1],
                dtype=np.float32)


    def reset(self):
        self.found_goal = False
        obs = super().reset()
        return self._proc_obs(obs)

    def _proc_obs(self, obs):
        obs = skimage.transform.resize(obs, VIZ_RES)
        obs = skimage.color.rgb2gray(obs)
        obs = np.expand_dims(obs, -1)
        obs = obs / 255.0
        obs = obs.astype(np.float32)
        return obs

    def step(self, a):
        obs, reward, done, info = super().step(a)
        obs = self._proc_obs(obs)
        if reward == 1.0:
            self.found_goal = True
            done = True
        info['ep_found_goal'] = float(self.found_goal)
        return obs, reward, done, info

class VizDoomInterface(EnvInterface):
    def env_trans_fn(self, env, set_eval):
        env = super().env_trans_fn(env, set_eval)
        return VizDoomWrapper(env)


register_env_interface("^Vizdoom", VizDoomInterface)
