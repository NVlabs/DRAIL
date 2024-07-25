import sys
sys.path.insert(0, './')
from gym import core
from gym import spaces
from rlf.envs.env_interface import EnvInterface, register_env_interface
from rlf.envs.fetch_interface import FETCH_REGISTER_STR, GymFetchInterface
from rlf.envs.image_obs_env import ImageObsWrapper
import rlf.algos.utils as autils
import rlf.rl.utils as rutils
import numpy as np

class GoalCheckerWrapper(core.Wrapper):
    def __init__(self, env, goal_check_cond_fn):
        super().__init__(env)
        self.goal_check_cond_fn = goal_check_cond_fn

    def reset(self):
        self.found_goal = False
        return super().reset()

    def step(self, a):
        obs, reward, done, info = super().step(a)
        self.found_goal = self.found_goal or self.goal_check_cond_fn(self.env, obs)
        if self.found_goal:
            done = True

        info['ep_found_goal'] = float(self.found_goal)
        return obs, reward, done, info



class BlockGripperActionWrapper(core.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._max_episode_steps = env._max_episode_steps
        self._is_success = env.env._is_success
        self.action_space = spaces.Box(
                high=self.action_space.high[:-1],
                low=self.action_space.low[:-1],
                dtype=self.action_space.dtype)

    def step(self, a):
        real_a = np.zeros(len(a) + 1)
        real_a[:-1] = a
        return super().step(real_a)


class EasyObsFetchWrapper(core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = self.observation_space.spaces['observation']
        self.observation_space.spaces['observation'] = spaces.Box(
                high=obs_space.high[:-12],
                low=obs_space.low[:-12],
                dtype=obs_space.dtype)
        try:
            self.max_episode_steps = env._max_episode_steps
        except AttributeError:
            pass

    def observation(self, obs):
        obs['observation'] = obs['observation'][:-15]
        obs['observation'] = np.concatenate([obs['observation'], obs['desired_goal']])
        return obs


class SingleFrameStack(core.Wrapper):
    def __init__(self, env, nstack, obs_key):
        super().__init__(env)
        self.okey = obs_key

        self.stacked_obs = autils.StackHelper(
                rutils.get_obs_shape(env.observation_space),
                nstack, None)

    def step(self, a):
        obs, reward, done, info = super().step(a)
        stacked_obs, infos = self.stacked_obs.update_obs(
                rutils.get_def_obs(obs, self.okey), done, info)
        obs = rutils.set_def_obs(obs, stacked_obs, self.okey)
        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        stacked_obs = self.stacked_obs.reset(
                rutils.get_def_obs(obs, self.okey))
        obs = rutils.set_def_obs(obs, stacked_obs, self.okey)
        return obs


class ControlPenaltyWrapper(core.Wrapper):
    def __init__(self, env, coef):
        super().__init__(env)
        self.coef = coef
        self.total_ep_pen = 0
        self.prev_obs = None
        self.prev_info = None

    def step(self, a):
        obs, reward, done, info = super().step(a)
        ctrl_pen = self.coef * np.linalg.norm(a)
        self.total_ep_pen += ctrl_pen
        reward -= ctrl_pen
        if done:
            info['ep_ctrl_pen'] = self.total_ep_pen
        self.prev_info = info
        self.prev_obs = obs
        return obs, reward, done, info

    def reset(self):
        self.total_ep_pen = 0
        return super().reset()

class GoalAntInterface(EnvInterface):
    def env_trans_fn(self, env, set_eval):
        env.env.spawn_noise = self.args.ant_noise
        env.env.is_expert = self.args.ant_is_expert
        env.env.coverage = self.args.ant_cover
        return env

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument("--ant-noise", type=float, default=0.0)
        parser.add_argument("--ant-cover", type=int, default=100)
        parser.add_argument('--ant-is-expert', action='store_true')


class GoalFetchInterface(GymFetchInterface):
    def env_trans_fn(self, env, set_eval):
        env = super().env_trans_fn(env, set_eval)

        if self.args.fetch_obj_range is not None:
            env.env.obj_range = self.args.fetch_obj_range
        if self.args.fetch_goal_range is not None:
            env.env.target_range = self.args.fetch_goal_range
        env.env.coverage = self.args.fetch_cover
        env.env.set_noise_ratio(self.args.noise_ratio, self.args.goal_noise_ratio)

        if env.env.block_gripper:
            env = BlockGripperActionWrapper(env)
            def check_goal(env, obs):
                return env._is_success(
                        obs['achieved_goal'],
                        obs['desired_goal'])
        else:
            def check_goal(env, obs):
                return env.env._is_success(
                        obs['achieved_goal'],
                        obs['desired_goal'])
        env = GoalCheckerWrapper(env, check_goal)
        if self.args.fetch_easy_obs:
            env = EasyObsFetchWrapper(env)
        if self.args.fetch_ctrl_pen != 0.0:
            env = ControlPenaltyWrapper(env, self.args.fetch_ctrl_pen)
        if self.args.img_dim is not None:
            env = ImageObsWrapper(env, self.args.img_dim)
        return env

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--fetch-obj-range', type=float, default=None)
        parser.add_argument('--fetch-goal-range', type=float, default=None)
        parser.add_argument('--fetch-ctrl-pen', type=float, default=0.0)
        parser.add_argument('--fetch-easy-obs', action='store_true',
                default=True)
        parser.add_argument('--fetch-cover', type=float, default=1.0)
        parser.add_argument('--noise-ratio', type=float, default=1.0)
        parser.add_argument('--goal-noise-ratio', type=float, default=1.0)
        parser.add_argument('--img-dim', type=int, default=None)

register_env_interface(FETCH_REGISTER_STR, GoalFetchInterface)
register_env_interface('AntGoal-v0', GoalAntInterface)
