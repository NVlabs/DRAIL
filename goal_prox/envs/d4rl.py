from rlf.envs.env_interface import EnvInterface, register_env_interface
from gym import core
import rlf.rl.utils as rutils
from gym.spaces import Box
import numpy as np


class D4rlGoalCheckerWrapper(core.Wrapper):
    def step(self, a):
        obs, reward, done, info = super().step(a)
        info["ep_found_goal"] = float(info["goal_achieved"])
        return obs, reward, done, info


class FullObsWrapper(core.Wrapper):
    def __init__(self, env, dim):
        super().__init__(env)
        self.observation_space = rutils.reshape_obs_space(
            self.observation_space, (dim,)
        )

    def _trans_obs(self, obs):
        env_state = self.env.env.get_env_state()
        return np.concatenate([env_state["qpos"], env_state["qvel"], obs], axis=-1)

    def reset(self):
        obs = super().reset()
        return self._trans_obs(obs)

    def step(self, a):
        obs, reward, done, info = super().step(a)
        obs = self._trans_obs(obs)
        return obs, reward, done, info


class D4rlInterface(EnvInterface):
    def env_trans_fn(self, env, set_eval):
        if hasattr(env.env, "set_coverage"):
            env.env.set_coverage(self.args.d4rl_cover)
        if hasattr(env.env, "set_noise_ratio"):
            env.env.set_noise_ratio(self.args.noise_ratio, self.args.goal_noise_ratio)
        if self.args.pen_easy_obs:
            if self.args.env_name == 'pen-expert-v1':
                set_dim = 105
            else:
                raise ValueError('Easy obs in this env is not supported')
        else:
            set_dim = None
        env = D4rlGoalCheckerWrapper(env)
        if set_dim is not None:
            env = FullObsWrapper(env, set_dim)
        return env

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument("--d4rl-cover", type=float, default=1.0)
        parser.add_argument("--noise-ratio", type=float, default=1.0)
        parser.add_argument("--goal-noise-ratio", type=float, default=1.0)
        parser.add_argument("--pen-easy-obs", action='store_true')


D4RL_REGISTER_STR = "^(pen|hammer|door|relocate|maze2d)"
register_env_interface(D4RL_REGISTER_STR, D4rlInterface)
