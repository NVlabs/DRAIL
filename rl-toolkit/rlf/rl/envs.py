import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box

from rlf.baselines.monitor import Monitor
from rlf.baselines.common.atari_wrappers import make_atari, wrap_deepmind, WarpFrame
from rlf.baselines.vec_env import VecEnvWrapper
from rlf.baselines.vec_env.dummy_vec_env import DummyVecEnv
from rlf.baselines.vec_env.shmem_vec_env import ShmemVecEnv
from rlf.baselines.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_
import rlf.rl.utils as rutils
from functools import partial


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None

def make_env(rank, env_id, seed, allow_early_resets, env_interface,
        set_eval, alg_env_settings, args, immediate_call=False):
    """
    :param immediate_call: Whether to return the created environment or to return
      the lambda that creates the environment.
    """
    def _thunk():
        env = env_interface.create_from_id(env_id)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)

        env = env_interface.env_trans_fn(env, set_eval)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        env.seed(seed + rank)
        if hasattr(env.action_space, 'seed'):
            env.action_space.seed(seed+rank)

        env = Monitor(env, None,
                allow_early_resets=allow_early_resets)

        obs_space = env.observation_space

        if is_atari:
            if not rutils.is_dict_obs(obs_space) and len(obs_space.shape) == 3:
                env = wrap_deepmind(env)
        if args.warp_frame:
            env = WarpFrame(env, grayscale=True)

        keys = rutils.get_ob_keys(env.observation_space)
        transpose_keys = [k for k in keys
                if len(rutils.get_ob_shape(env.observation_space, k)) == 3]
        if len(transpose_keys) > 0 and args.transpose_frame:
            env = TransposeImage(env, op=[2, 0, 1], transpose_keys=transpose_keys)
        if set_eval and args.render_metric:
            def overall_render_mod(frames, **kwargs):
                frames = env_interface.mod_render_frames(frames, **kwargs)
                return alg_env_settings.mod_render_frames_fn(frames, **kwargs)
            env = RenderWrapper(env, overall_render_mod)

        env = env_interface.final_trans_fn(env)

        return env
    if immediate_call:
        return _thunk()

    return _thunk

def make_vec_envs_easy(env_name, num_processes, env_interface, alg_env_settings, args):
    return make_vec_envs(env_name, args.seed, num_processes,
                         args.gamma, args.device,
                         False, env_interface, args,
                         alg_env_settings)


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  device,
                  allow_early_resets,
                  env_interface,
                  args,
                  alg_env_settings,
                  num_frame_stack=None,
                  set_eval=False):

    if args.render_metric and set_eval and num_processes > 1:
        raise ValueError('Cannot create multiple processes when rendering metrics at the moment')

    envs = [
            make_env(i, env_name, seed, allow_early_resets, env_interface,
                set_eval, alg_env_settings, args)
            for i in range(num_processes)
            ]

    origin_env = env_interface.create_from_id(env_name)

    if len(envs) > 1 or args.force_multi_proc:
        custom_envs = env_interface.get_setup_multiproc_fn(make_env, env_name,
                seed, allow_early_resets, env_interface, set_eval,
                alg_env_settings, args)
        if custom_envs is None:
            envs = ShmemVecEnv(envs, context=args.context_mode)
        else:
            envs = custom_envs
    else:
        envs = DummyVecEnv(envs)

    ob_shapes = rutils.get_ob_shapes(envs.observation_space)

    single_shapes = {k:v for k,v in ob_shapes.items() if len(v) == 1}

    use_env_norm = not set_eval and len(single_shapes) > 0 and args.normalize_env

    if use_env_norm:
        if gamma is None:
            envs = VecNormalize(envs, ret=False,
                    ret_raw_obs=alg_env_settings.ret_raw_obs)
        else:
            envs = VecNormalize(envs, gamma=gamma,
                    ret_raw_obs=alg_env_settings.ret_raw_obs)

    if env_interface.requires_tensor_wrap():
        envs = VecPyTorch(envs, device)

        triple_shapes = {k:v for k,v in ob_shapes.items() if len(v) == 3}
        if num_frame_stack is not None and args.frame_stack:
            envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
        elif len(triple_shapes) > 0 and args.frame_stack:
            envs = VecPyTorchFrameStack(envs, 4, device)

    if (alg_env_settings.state_fn is not None) or (alg_env_settings.action_fn is not None):
        if use_env_norm and alg_env_settings.state_fn is not None:
            raise ValueError(('Cannot specify environment normalization at ',
                'the same time as action or state transformation. Specify ',
                '`--normalize-env False'))
        envs = EnvNormFnWrapper(envs, device, alg_env_settings.state_fn,
                alg_env_settings.action_fn)

    return origin_env, envs

class RenderWrapper(gym.Wrapper):
    def __init__(self, env, render_modify_fn):
        super().__init__(env)
        self.render_modify_fn = render_modify_fn
        self.last_obs = None
        self.last_reward = None

    def reset(self):
        obs = super().reset()
        self.env_cur_obs = obs
        self.env_cur_action = None
        self.env_next_obs = None
        self.env_cur_reward = None
        self.was_reset = True
        return obs

    def step(self, a):
        obs, reward, done, info = super().step(a)
        self.env_cur_action = a
        if self.was_reset:
            self.env_next_obs = obs
            self.was_reset = False
        else:
            self.env_cur_obs = self.env_next_obs
            self.env_next_obs = obs
        self.env_cur_reward = reward
        done = info['goal_achieved']
        return obs, reward, done, info

    def render(self, mode, **kwargs):
        frame = super().render(mode)
        return self.render_modify_fn(frame, env_cur_obs=self.env_cur_obs,
                env_next_obs=self.env_next_obs,
                env_cur_action=self.env_cur_action,
                env_cur_reward=self.env_cur_reward, **kwargs)

class EnvNormFnWrapper(VecEnvWrapper):
    def __init__(self, venv, device, state_fn, action_fn):
        super().__init__(venv)
        self.state_fn = state_fn
        self.action_fn = action_fn
        self.device = device

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if self.state_fn is not None:
            obs = self.state_fn(obs)
        if 'final_obs' in info:
            info['final_obs'] = self.state_fn(info['final_obs'])
        return obs, reward, done, info

    def step_async(self, actions):
        if self.action_fn is not None:
            actions = self.action_fn(actions.to(self.device)).cpu()
        self.venv.step_async(actions)

    def reset(self):
        return self.venv.reset()

    def close(self):
        self.venv.close()


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env, op, transpose_keys):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3
        self.op = op

        if rutils.is_dict_obs(env.observation_space):
            spaces = env.observation_space.spaces
        else:
            spaces = {}
        for k in transpose_keys:
            if k is not None:
                obs_space = self.observation_space.spaces[k]
                obs_shape = obs_space.shape
                spaces[k] = Box(
                    obs_space.low[0, 0, 0],
                    obs_space.high[0, 0, 0], [
                        obs_shape[self.op[0]], obs_shape[self.op[1]],
                        obs_shape[self.op[2]]
                    ],
                    dtype=obs_space.dtype)
        if len(spaces) == 0:
            obs_shape = self.observation_space.shape
            self.observation_space = Box(
                self.observation_space.low[0, 0, 0],
                self.observation_space.high[0, 0, 0], [
                    obs_shape[self.op[0]], obs_shape[self.op[1]],
                    obs_shape[self.op[2]]
                ],
                dtype=self.observation_space.dtype)
        else:
            self.observation_space = gym.spaces.Dict(spaces)

        self.transpose_keys = transpose_keys


    def observation(self, ob):
        for k in self.transpose_keys:
            if k is None:
                ob = ob.transpose(self.op[0], self.op[1], self.op[2])
            else:
                ob[k] = ob[k].transpose(self.op[0], self.op[1], self.op[2])
        return ob


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def _data_convert(self, arr):
        if isinstance(arr, np.ndarray) and arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        return arr

    def reset(self):
        obs = self.venv.reset()
        obs = self._trans_obs(obs)
        return obs

    def reset_model(self, states):
        obs = self.venv.reset_model(states)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def _trans_obs(self, obs):
        # Support for dict observations
        def _convert_obs(x):
            x = self._data_convert(x)
            x = torch.Tensor(x)
            x = x.to(self.device)
            return x
        if isinstance(obs, dict):
            for k in obs:
                obs[k] = _convert_obs(obs[k])
        else:
            obs = _convert_obs(obs)
        return obs

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = self._trans_obs(obs)

        reward = torch.Tensor(reward).unsqueeze(dim=1)
        # Reward is sometimes a Double. Observation is considered to always be
        # float32
        reward = reward.float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if not isinstance(obs, dict) and rutils.is_dict_obs(self.observation_space):
            obs = {'observation': obs}

        if self.ob_rms_dict:
            for k, ob_rms in self.ob_rms_dict.items():
                if k is None:
                    if self.training and update:
                        ob_rms.update(obs)
                    obs = np.clip((obs - ob_rms.mean) /
                                  np.sqrt(ob_rms.var + self.epsilon),
                                  -self.clipob, self.clipob)
                else:
                    if k not in obs:
                        continue
                    if self.training and update:
                        ob_rms.update(obs[k])
                    obs[k] = np.clip((obs[k] - ob_rms.mean) /
                                  np.sqrt(ob_rms.var + self.epsilon),
                                  -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    """
    For now, this will only stack the "observation" key in dictionary
    observation spaces.
    """

    def __init__(self, venv, nstack, device):
        self.venv = venv
        self.nstack = nstack

        ob_space = rutils.get_obs_space(venv.observation_space)

        self.stacked_obs = rutils.StackHelper(ob_space.shape, nstack, device, venv.num_envs)
        new_obs_space = rutils.update_obs_space(
                venv.observation_space,
                rutils.reshape_obs_space(ob_space, self.stacked_obs.get_shape()))

        VecEnvWrapper.__init__(self, venv, observation_space=new_obs_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()

        stacked_obs, infos = self.stacked_obs.update_obs(
                rutils.get_def_obs(obs), news, infos)

        obs = rutils.set_def_obs(obs, stacked_obs)
        return obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        stacked_obs = self.stacked_obs.reset(rutils.get_def_obs(obs))
        obs = rutils.set_def_obs(obs, stacked_obs)
        return obs

    def close(self):
        self.venv.close()