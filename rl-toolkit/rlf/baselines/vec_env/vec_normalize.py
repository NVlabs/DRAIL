from . import VecEnvWrapper
from rlf.baselines.common.running_mean_std import RunningMeanStd
import numpy as np
import rlf.rl.utils as rutils
from gym import spaces


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10.,
            gamma=0.99, epsilon=1e-8, ret_raw_obs=False):
        VecEnvWrapper.__init__(self, venv)
        if ret_raw_obs:
            ospace = rutils.get_obs_space(self.observation_space)
            self.observation_space = rutils.combine_spaces(ospace,
                    'raw_obs',
                    spaces.Box(low=ospace.low, high=ospace.high,
                        dtype=ospace.dtype))
        if ob:
            self.ob_rms_dict = {k: RunningMeanStd(shape=shp)
                    for k,shp in rutils.get_ob_shapes(self.observation_space).items()}
        else:
            self.ob_rms_dict = None

        self.ret = np.zeros(self.num_envs)
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.gamma = gamma
        self.epsilon = epsilon
        self.ret_raw_obs = ret_raw_obs

    def step_wait(self):
        orig_obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        use_obs = orig_obs
        if self.ret_raw_obs:
            use_obs = rutils.clone_ob(orig_obs)
        obs = self._obfilt(use_obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.

        if isinstance(orig_obs, dict):
            orig_obs = rutils.get_def_obs(orig_obs)

        if self.ret_raw_obs:
            obs = rutils.combine_obs(obs, 'raw_obs',
                    rutils.get_def_obs(orig_obs))

        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms_dict:
            for k, ob_rms in self.ob_rms_dict.items():
                if k is None:
                    ob_rms.update(obs)
                    obs = np.clip((obs - ob_rms.mean) / np.sqrt(ob_rms.var + self.epsilon), -self.clipob, self.clipob)
                else:
                    ob_rms.update(obs[k])
                    obs[k] = np.clip((obs[k] - ob_rms.mean) / np.sqrt(ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        orig_obs = self.venv.reset()
        use_obs = orig_obs
        if self.ret_raw_obs:
            use_obs = rutils.clone_ob(orig_obs)
        obs = self._obfilt(use_obs)
        if self.ret_raw_obs:
            obs = rutils.combine_obs(obs, 'raw_obs',
                    rutils.get_def_obs(orig_obs))
        return obs