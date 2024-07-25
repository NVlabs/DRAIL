import torch
import numpy as np
from collections import defaultdict
import rlf.rl.utils as rutils

def select_idx_from_dict(batch_idx, sel_from):
    """
    - batch_idx: (torch.tensor [N])
    - sel_from: (dict(k -> torch.tensor))
    Use the indices specified in batch_idx to select from a list of
    dictionaries. This is like a "deep" index for dictionaries.
    """
    ret_d = {}
    for k in sel_from:
        ret_d[k] = sel_from[k][batch_idx]
    return ret_d

def convert_list_dict(arr, device):
    key_sets = [set(x.keys()) for x in arr]
    # Get the minimal set of keys.
    keys = set.intersection(*key_sets)

    ret_d = {k: [] for k in keys}
    for x in arr:
        for k,v in x.items():
            if k in ret_d:
                ret_d[k].append(v.to(device))
    for k in ret_d:
        ret_d[k] = torch.stack(ret_d[k])
    return ret_d

def dict_to_device(d, device):
    for k in d:
        d[k] = d[k].to(device)
    return d

def traj_to_tensor(trajs, device):
    """
    - trajs: [B, N, 5]. Batch size of B, trajectory length of N. 5 for state,
      action, mask, info, reward.
    """
    # Get the data into a format we can work with.
    max_traj_len = max([len(traj) for traj in trajs])
    n_trajs = len(trajs)
    ob_dim = rutils.get_def_obs(trajs[0][0][0]).shape
    ac_dim = trajs[0][0][1].shape
    other_obs_info = {k: x.shape
            for k, x in rutils.get_other_obs(trajs[0][0][0]).items()}

    obs = torch.zeros(n_trajs, max_traj_len, *ob_dim).to(device)
    obs_add = {k: torch.zeros(n_trajs, max_traj_len, *shp).to(device)
            for k, shp in other_obs_info.items()}
    actions = torch.zeros(n_trajs, max_traj_len, *ac_dim).to(device)
    masks = torch.zeros(n_trajs, max_traj_len, 1).to(device)
    rewards = torch.zeros(n_trajs, max_traj_len, 1).to(device)

    traj_mask = torch.zeros(n_trajs, max_traj_len).to(device)

    add_infos = {k: torch.zeros(len(trajs), max_traj_len, *v.shape)
            for k,v in trajs[0][-1][3].items()}

    for i in range(len(trajs)):
        traj_len = len(trajs[i])
        o, a, m, infos, r = list(zip(*trajs[i]))
        for j, inf in enumerate(infos):
            for k, v in inf.items():
                add_infos[k][i,j] = v

        traj_mask[i, :traj_len] = 1.0
        obs[i, :traj_len] = torch.stack([rutils.get_def_obs(o_i) for o_i in o]).to(device)
        for k in obs_add:
            obs_add[k][i, :traj_len] = torch.stack([o_i[k] for o_i in o]).to(device)
        actions[i, :traj_len] = torch.stack(a).to(device)
        masks[i, :traj_len] = torch.tensor(m).unsqueeze(-1).to(device)
        rewards[i, :traj_len] = torch.stack(r).to(device)

    for k in add_infos:
        add_infos[k] = add_infos[k].to(device)
    return obs, obs_add, actions, masks, add_infos, rewards

def mix_data(set0, set1, count, mix_factor):
    """
    - count (int): the number of elements to return
    - mix_factor (float): between 0 and 1. It's the fraction of count taken from
      set0.
    """
    # Always have the larger set be set 0
    if len(set0) < len(set1):
        set0, set1 = set1, set0
        mix_factor = 1.0 - mix_factor
    take_set0 = int(count * mix_factor)
    take_set1 = count - take_set0
    if take_set1 > len(set1):
        take_set0 = count - len(set1)
        take_set1 = len(set1)

    sel_set0 = np.random.choice(np.arange(0, len(set0)), take_set0, replace=False)
    sel_set1 = np.random.choice(np.arange(0, len(set1)), take_set1, replace=False)
    new_set0 = [set0[i] for i in sel_set0]
    new_set1 = [set1[i] for i in sel_set1]
    combined = [*new_set0, *new_set1]
    np.random.shuffle(combined)
    return combined


