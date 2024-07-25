import glob
import hashlib
import os
import os.path as osp
import pickle
import time
from abc import ABC
from collections import defaultdict
from contextlib import ContextDecorator, contextmanager
from timeit import default_timer
from typing import Any, Callable, Dict, List

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    import wandb
except:
    pass


def get_env_attr(env, attr_name, max_calls=10):
    try:
        return getattr(env, attr_name)
    except Exception as e:
        if max_calls == 0:
            raise e
        if hasattr(env, "env"):
            return get_env_attr(env.env, attr_name, max_calls - 1)
        elif hasattr(env, "_env"):
            return get_env_attr(env._env, attr_name, max_calls - 1)
        else:
            raise ValueError(f"Could not find property {attr_name} of {env}")


def get_save_dir(args) -> str:
    """Directory for saving images, videos or any random debug info."""
    return osp.join(args.vid_dir, args.env_name, args.prefix)


def plot_line(
    plot_vals,
    save_name,
    save_dir,
    to_wb,
    update_iter=None,
    x_vals=None,
    x_name=None,
    y_name=None,
    title=None,
    err=None,
    file_suffix="",
):
    """
    Plot a simple rough line.
    """
    if x_vals is None:
        x_vals = np.arange(len(plot_vals))
    save_path = osp.join(save_dir, f"{save_name}_{file_suffix}.png")

    if title is None:
        plt.title(save_name)
    else:
        plt.title(title)

    if x_name is not None:
        plt.xlabel(x_name)
    if y_name is not None:
        plt.ylabel(y_name)

    if err is None:
        plt.plot(x_vals, plot_vals)
    else:
        plt.errorbar(x_vals, plot_vals, err)
    plt.grid(b=True, which="major", color="lightgray", linestyle="--")
    plt_save(save_path)

    kwargs = {}
    if update_iter is not None:
        kwargs["step"] = update_iter

    if to_wb:
        wandb.log({save_name: [wandb.Image(save_path)]}, **kwargs)


def plt_save(*path_parts):
    save_name = osp.join(*path_parts)
    save_dir = osp.dirname(save_name)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_name)
    print(f"Saved fig to {save_name}")
    plt.clf()


def save_model(model, save_name, args):
    save_dir = osp.join(args.save_dir, args.env_name, args.prefix)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    save_path = osp.join(save_dir, save_name)
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")


#########################################
# FORMATTING / PRINTING
#########################################
def human_format_int(num, round_pos=2):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    format_str = "%." + str(round_pos) + "f"

    num_str = format_str % num
    num_str = num_str.rstrip("0").rstrip(".")

    return num_str + ["", "K", "M", "G", "T", "P"][magnitude]


def print_weights(m):
    for name, param in m.named_parameters():
        print(name)
        print(param)


def pstart_sep():
    print("")
    print("-" * 30)


def pend_sep():
    print("-" * 10)
    print("")


#########################################


#########################################
# OBS DICTIONARY ACTIONS
#########################################
def deep_dict_select(d, idx):
    ret_dict = {}
    for k in d:
        ret_dict[k] = d[k][idx]
    return ret_dict


def transpose_arr_dict(arr: List[Dict]) -> Dict[Any, Any]:
    keys = arr[0].keys()
    orig_format = arr[0][list(keys)[0]]
    ret_d = {k: [] for k in keys}
    for arr_ele in arr:
        for k in keys:
            ret_d[k].append(arr_ele[k])

    if isinstance(orig_format, torch.Tensor):
        for k in keys:
            ret_d[k] = torch.stack(ret_d[k])

    return ret_d


def transpose_dict_arr(d: Dict[Any, Any]) -> List[Dict]:
    keys = list(d.keys())
    lens = [len(d[k]) for k in d]
    if len(set(lens)) != 1:
        raise ValueError("All lists must have equal sizes")

    # Assumes that all the lists are equal length.
    ret = []
    for i in range(lens[0]):
        ret.append({k: d[k][i] for k in keys})
    return ret


def flatten_obs_dict(ob_shape, keep_keys):
    total_dim = 0
    low_val = None
    high_val = None
    for k in keep_keys:
        sub_space = ob_shape.spaces[k]
        assert len(sub_space.shape) == 1
        if low_val is None:
            low_val = sub_space.low.reshape(-1)[0]
            high_val = sub_space.high.reshape(-1)[0]
        else:
            low_val = min(sub_space.low.reshape(-1)[0], low_val)
            high_val = max(sub_space.high.reshape(-1)[0], high_val)
        total_dim += sub_space.shape[0]
    return gym.spaces.Box(
        shape=(total_dim,),
        low=np.float32(low_val),
        high=np.float32(high_val),
        dtype=np.float32,
    )


def obs_op(obs: Any, op: Callable[[Any], Any]) -> Any:
    """Apply an operation to every value in a dictionary."""
    if isinstance(obs, dict):
        return {k: op(obs[k]) for k in obs}
    return op(obs)


def obs_select(obs, idx):
    if isinstance(obs, dict):
        return {k: obs[k][idx] for k in obs}
    return obs[idx]


#########################################


def is_dict_obs(ob_space):
    return isinstance(ob_space, gym.spaces.Dict)


def get_ob_keys(ob_space):
    if isinstance(ob_space, gym.spaces.Dict):
        return list(ob_space.spaces.keys())
    else:
        return [None]


def ob_to_np(obs):
    if isinstance(obs, dict):
        for k in obs:
            obs[k] = obs[k].cpu().numpy()
        return obs
    else:
        return obs.cpu().numpy()


def clone_ob(obs):
    if isinstance(obs, dict):
        return {k: np.array(v) for k, v in obs.items()}
    return np.array(obs)


def ob_to_tensor(obs, device):
    if isinstance(obs, dict):
        for k in obs:
            obs[k] = torch.tensor(obs[k]).to(device)
        return obs
    else:
        return torch.tensor(obs).to(device)


def ob_to_cpu(obs):
    new_obs = {}
    if isinstance(obs, dict):
        for k in obs:
            new_obs[k] = obs[k].cpu()
        return new_obs
    elif obs is None:
        return None
    else:
        return obs.cpu()


def ac_space_to_tensor(action_space):
    return torch.tensor(action_space.low), torch.tensor(action_space.high)


def multi_dim_clip(val, low, high):
    return torch.max(torch.min(val, high), low)


def get_ob_shapes(ob_space):
    if isinstance(ob_space, gym.spaces.Dict):
        return {k: space.shape for k, space in ob_space.spaces.items()}
    else:
        return {None: ob_space.shape}


def get_ob_shape(obs_space, k):
    if k is None:
        return obs_space.shape
    else:
        return obs_space.spaces[k].shape


def get_obs_shape(ob_space, k="observation"):
    if isinstance(ob_space, gym.spaces.Dict):
        return ob_space.spaces[k].shape
    else:
        return ob_space.shape


def get_obs_space(ob_space):
    if isinstance(ob_space, gym.spaces.Dict):
        return ob_space.spaces["observation"]
    else:
        return ob_space


def get_def_obs(obs, k="observation"):
    if isinstance(obs, dict):
        return obs[k]
    else:
        return obs


def set_def_obs(obs, new_obs, k="observation"):
    if isinstance(obs, dict):
        obs[k] = new_obs
    else:
        obs = new_obs
    return obs


def obs_select(obs, idx):
    if isinstance(obs, dict):
        return {k: obs[k][idx] for k in obs}
    return obs[idx]


def deep_get_other_obs(obs):
    return [get_other_obs(o) for o in obs]


def deep_get_def_obs(obs):
    return [get_def_obs(o) for o in obs]


def get_other_obs(obs, maink="observation"):
    if isinstance(obs, dict):
        return {k: obs[k] for k in obs if k != maink}
    else:
        return {}


def combine_spaces(orig_space, new_space_key, new_space):
    if isinstance(orig_space, gym.spaces.Dict):
        return gym.spaces.Dict(
            {
                **orig_space.spaces,
                new_space_key: new_space,
            }
        )
    else:
        return gym.spaces.Dict(
            {
                "observation": orig_space,
                new_space_key: new_space,
            }
        )


def update_obs_space(cur_space, update_obs_space):
    if is_dict_obs(cur_space):
        new_obs_space = {**cur_space.spaces}
        new_obs_space["observation"] = update_obs_space
        new_obs_space = gym.spaces.Dict(new_obs_space)
    else:
        new_obs_space = update_obs_space
    return new_obs_space


def combine_obs(orig_obs, new_obs_key, new_obs):
    if isinstance(orig_obs, dict):
        return {**orig_obs, new_obs_key: new_obs}
    else:
        return {"observation": orig_obs, new_obs_key: new_obs}


def reshape_obs_space(obs_space, new_shape):
    assert isinstance(obs_space, gym.spaces.Box)
    return gym.spaces.Box(
        shape=new_shape,
        high=obs_space.low.reshape(-1)[0],
        low=obs_space.high.reshape(-1)[0],
        dtype=obs_space.dtype,
    )


def combine_states(state0, state1):
    if len(state0.shape) == 4:
        return torch.cat([state0, state1], dim=1)
    else:
        return torch.cat([state0, state1], dim=-1)


def get_ac_repr(ac, action):
    """
    Either returns the continuous value of the action or the one-hot encoded
    action
    """
    if isinstance(ac, gym.spaces.Box):
        return action
    elif isinstance(ac, gym.spaces.Discrete):
        y_onehot = torch.zeros(action.shape[0], ac.n).to(action.device)
        y_onehot = y_onehot.scatter(1, action.long(), 1)
        return y_onehot
    else:
        raise ValueError("Invalid action space type")


def get_ac_compact(ac, action):
    """
    Returns the opposite of `get_ac_repr`
    """
    if isinstance(ac, gym.spaces.Box):
        return action
    elif isinstance(ac, gym.spaces.Discrete):
        return torch.argmax(action, dim=-1).unsqueeze(-1)
    else:
        raise ValueError("Invalid action space type")


def get_ac_dim(ac):
    """
    Returns the dimensionality of the action space
    """
    if isinstance(ac, gym.spaces.Box):
        return ac.shape[0]
    elif isinstance(ac, gym.spaces.Discrete):
        return ac.n
    else:
        raise ValueError("Invalid action space type")


def is_discrete(ac):
    if ac.__class__.__name__ == "Discrete":
        return True
    elif ac.__class__.__name__ == "Box":
        return False
    else:
        raise ValueError(f"Action space {ac} not supported")


def agg_ep_log_stats(env_infos, alg_info):
    """
    Combine the values we want to log into one dictionary for logging.
    - env_info: (list[dict]) returns everything starting with 'ep_' and everything
      in the 'episode' key. There is a list of dicts for each environment
      process.
    - alg_info (dict) returns everything starting with 'alg_add_'
    """

    all_log_stats = defaultdict(list)
    for k in alg_info:
        if k.startswith("alg_add_"):
            all_log_stats[k].append(alg_info[k])
    #import ipdb; ipdb.set_trace()
    for inf in env_infos:
        if "episode" in inf:
            # Only log at the end of the episode
            for k in inf:
                if k.startswith("ep_"):
                    all_log_stats[k].append(inf[k])
            for k, v in inf["episode"].items():
                all_log_stats[k].append(v)
    return all_log_stats


# Get a render frame function (Mainly for transition)
def get_render_frame_func(venv):
    if hasattr(venv, "envs"):
        return venv.envs[0].unwrapped.render_frame
    elif hasattr(venv, "venv"):
        return get_render_frame_func(venv.venv)
    elif hasattr(venv, "env"):
        return get_render_frame_func(venv.env)

    return None


# Get a render function
def get_render_func(venv):
    if hasattr(venv, "envs"):
        return venv.envs[0].render
    elif hasattr(venv, "venv"):
        return get_render_func(venv.venv)
    elif hasattr(venv, "env"):
        return get_render_func(venv.env)

    return None


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
        for f in files:
            os.remove(f)


def update_args(args, update_dict, check_exist=False):
    args_dict = vars(args)
    for k, v in update_dict.items():
        if check_exist and k not in args_dict:
            raise ValueError("Could not set key %s" % k)
        args_dict[k] = v


CACHE_PATH = "./data/cache"


class CacheHelper:
    def __init__(self, cache_name, lookup_val, def_val=None, verbose=False, rel_dir=""):
        self.use_cache_path = osp.join(CACHE_PATH, rel_dir)
        if not osp.exists(self.use_cache_path):
            os.makedirs(self.use_cache_path)
        sec_hash = hashlib.md5(str(lookup_val).encode("utf-8")).hexdigest()
        cache_id = f"{cache_name}_{sec_hash}.pickle"
        self.cache_id = osp.join(self.use_cache_path, cache_id)
        self.def_val = def_val
        self.verbose = verbose

    def exists(self):
        return osp.exists(self.cache_id)

    def load(self, load_depth=0):
        if self.exists():
            try:
                with open(self.cache_id, "rb") as f:
                    if self.verbose:
                        print("Loading cache @", self.cache_id)
                    return pickle.load(f)
            except EOFError as e:
                if load_depth == 32:
                    raise e
                # try again soon
                print(
                    "Cache size is ", osp.getsize(self.cache_id), "for ", self.cache_id
                )
                time.sleep(1.0 + np.random.uniform(0.0, 1.0))
                return self.load(load_depth + 1)
            return self.def_val
        else:
            return self.def_val

    def save(self, val):
        with open(self.cache_id, "wb") as f:
            if self.verbose:
                print("Saving cache @", self.cache_id)
            pickle.dump(val, f)


@contextmanager
def elapsed_timer():
    """
    Measure time elapsed in a block of code. Used for debugging.
    Taken from:
    https://stackoverflow.com/questions/7370801/measure-time-elapsed-in-python
    """
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end - start


try:
    # For nvidia nsight profiling which is super helpful
    from habitat_sim.utils import profiling_utils
except:
    profiling_utils = None


class TimeProfiler(ContextDecorator):
    def __init__(self, timer_name, timee=None, timer_prop=None):
        """
        - timer_prop: str The code that is used to access `self` when using the
          this as a method decorator.
        """
        self.timer_name = timer_name
        self.timer_prop = timer_prop
        if timee is not None:
            self.add_time_f = timee.timer.add_time
        else:
            self.add_time_f = None

    def __enter__(self):
        if profiling_utils is not None:
            profiling_utils.range_push(self.timer_name)
        self.start_time = time.time()
        return self

    def __call__(self, f):
        def wrapper(*args, **kwargs):
            other_self = args[0]
            if self.timer_prop is not None:
                self.add_time_f = eval(f"other_self.{self.timer_prop}.timer.add_time")
            else:
                self.add_time_f = other_self.timer.add_time
            return f(*args, **kwargs)

        return super().__call__(wrapper)

    def __exit__(self, *exc):
        elapsed = time.time() - self.start_time
        self.add_time_f(self.timer_name, elapsed)
        if profiling_utils is not None:
            profiling_utils.range_pop()
        return False


class TimeProfilee:
    def __init__(self):
        self.clear()
        self._should_time = True

    def freeze(self):
        self._should_time = False

    def unfreeze(self):
        self._should_time = True

    def add_time(self, timer_name, timer_val):
        if self._should_time:
            self.timers[timer_name] += timer_val
            self.timer_call_count[timer_name] += 1

    def get_time(self, timer_name):
        return (self.timers[timer_name], self.timer_call_count[timer_name])

    def clear(self):
        self.timers = defaultdict(lambda: 0)
        self.timer_call_count = defaultdict(lambda: 0)


class StackHelper:
    """
    A helper for stacking observations.
    """

    def __init__(self, ob_shape, n_stack, device, n_procs=None):
        self.input_dim = ob_shape[0]
        self.n_procs = n_procs
        self.real_shape = (n_stack * self.input_dim, *ob_shape[1:])
        if self.n_procs is not None:
            self.stacked_obs = torch.zeros((n_procs, *self.real_shape))
            if device is not None:
                self.stacked_obs = self.stacked_obs.to(device)
        else:
            self.stacked_obs = np.zeros(self.real_shape)

    def update_obs(self, obs, dones=None, infos=None):
        """
        - obs: torch.tensor
        """
        if self.n_procs is not None:
            self.stacked_obs[:, : -self.input_dim] = self.stacked_obs[
                :, self.input_dim :
            ].clone()
            for (i, new) in enumerate(dones):
                if new:
                    self.stacked_obs[i] = 0
            self.stacked_obs[:, -self.input_dim :] = obs

            # Update info so the final observation frame stack has the final
            # observation as the final frame in the stack.
            for i in range(len(infos)):
                if "final_obs" in infos[i]:
                    new_final = torch.zeros(*self.stacked_obs.shape[1:])
                    new_final[:-1] = self.stacked_obs[i][1:]
                    new_final[-1] = torch.tensor(infos[i]["final_obs"]).to(
                        self.stacked_obs.device
                    )
                    infos[i]["final_obs"] = new_final
            return self.stacked_obs.clone(), infos
        else:
            self.stacked_obs[: -self.input_dim] = self.stacked_obs[
                self.input_dim :
            ].copy()
            self.stacked_obs[-self.input_dim :] = obs

            return self.stacked_obs.copy(), infos

    def reset(self, obs):
        if self.n_procs is not None:
            if torch.backends.cudnn.deterministic:
                self.stacked_obs = torch.zeros(self.stacked_obs.shape)
            else:
                self.stacked_obs.zero_()
            self.stacked_obs[:, -self.input_dim :] = obs
            return self.stacked_obs.clone()
        else:
            self.stacked_obs = np.zeros(self.stacked_obs.shape)
            self.stacked_obs[-self.input_dim :] = obs
            return self.stacked_obs.copy()

    def get_shape(self):
        return self.real_shape
