from typing import Any, Callable, Dict, Iterable

import attr
import numpy as np
import rlf.rl.utils as rutils
from rlf.rl.envs import get_vec_normalize
from rlf.storage import BaseStorage, RolloutStorage


@attr.s(auto_attribs=True, slots=True)
class AlgorithmSettings:
    """
    - ret_raw_obs: If this algorithm should have access to
        the raw observations in the `update` (via the storage object).
        A raw observation is before any preprocessing or normalization
        is applied. This raw observation is returned in the info
        dictionary of the environment.
    - mod_render_frames_fn: (frame, last_obs, last_reward, **kwargs ->
      updated_frame) Render algorithm information on the render output. Must
      specify `--render-metric` for this to be called.
    """

    ret_raw_obs: bool
    state_fn: Callable[[np.ndarray], np.ndarray]
    action_fn: Callable[[np.ndarray], np.ndarray]
    include_info_keys: list
    mod_render_frames_fn: Callable


def mod_render_frames_identity(cur_frame, obs):
    return cur_frame


class BaseAlgo(object):
    """
    Base class for all algorithms to derive from. Use this class as an empty
    updater (for policies that need no learning!).
    Updater classes typically look like:
    ```
    class MyUpdater([OnPolicy or OffPolicy]):
        def update(self, rollouts):
            # Calculate loss and then update:
            loss = ...
            self._standard_step(loss)
            return {
                    'logging_stat': 0.0
                    }

        def get_add_args(self, parser):
            super().get_add_args(parser)
            # Add additional arguments.
    ```
    """

    def __init__(self):
        pass

    def init(self, policy, args):
        self.update_i = 0
        self.policy = policy
        self.args = args

    def get_steps_generator(self, update_iter: int) -> Iterable[int]:
        """Generates an iterable for the number of rollout steps."""
        return range(self.args.num_steps)

    def set_env_ref(self, envs):
        self.envs = envs
        env_norm = get_vec_normalize(envs)

        def get_vec_normalize_fn():
            if env_norm is not None:
                obfilt = get_vec_normalize(envs)._obfilt

                def mod_env_ob_filt(state, update=True):
                    state = obfilt(state, update)
                    state = rutils.get_def_obs(state)
                    return state

                return mod_env_ob_filt
            return None

        self.get_env_ob_filt = get_vec_normalize_fn

    def first_train(self, log, eval_policy, env_interface):
        """
        Called before any RL training loop starts but after `self.init` is
        called.
        - log: logger object to log any statistics.
        - eval_policy: (policy: BasePolicy, total_num_steps: int, args) -> None
          function that evaluates the given policy with the args at timestep
          `total_num_steps`.
        """
        pass

    def get_num_updates(self) -> int:
        """
        Allows overridding the number of updates performed in the RL
        loop.Setting this is useful if an algorithm should
        dynamically calculate how many steps to take.
        """
        if self.args.num_steps == 0:
            return 0
        return (
            int(self.args.num_env_steps)
            // self.args.num_steps
            // self.args.num_processes
        )

    def get_completed_update_steps(self, num_updates: int) -> int:
        """
        num_updates: the number of times this updater has been called.
        Returns: (int) the number of environment frames processed.
        """
        return num_updates * self.args.num_processes * self.args.num_steps

    def get_env_settings(self, args):
        """
        Some updaters require specific things from the environment.
        """
        return AlgorithmSettings(False, None, None, [], mod_render_frames_identity)

    def set_get_policy(self, get_policy_fn, policy_args):
        """
        - get_policy_fn: (None -> rlf.BasePolicy)
        Sets the factory object for creating the policy.
        """
        self._get_policy_fn = get_policy_fn
        self._policy_args = policy_args

    def _copy_policy(self):
        """
        Creates a copy of the current policy.

        returns: (rlf.BasePolicy) with same params as `self.policy`
        """
        cp_policy = self._get_policy_fn()
        cp_policy.init(*self._policy_args)
        return cp_policy

    def load_resume(self, checkpointer) -> None:
        pass

    def load(self, checkpointer) -> None:
        pass

    def save(self, checkpointer) -> None:
        pass

    def pre_update(self, cur_update: int) -> None:
        pass

    def update(self, storage) -> Dict[str, Any]:
        self.update_i += 1
        return {}

    def on_traj_finished(self, traj):
        """
        done_trajs: A list of transitions where each transition is a tuple of form:
            (state,action,mask,info_dict,reward). The data is a bit confusing.
            mask[t] is technically the mask at t+1. The mask at t=0 is always
            1. The final state is NOT included and must be included through the
            info_dict if needed.
        """
        pass

    def get_add_args(self, parser):
        pass

    def get_storage_buffer(self, policy, envs, args) -> BaseStorage:
        return RolloutStorage(
            args.num_steps,
            args.num_processes,
            envs.observation_space,
            envs.action_space,
            args,
        )

    def get_requested_obs_keys(self):
        return ["observation"]
