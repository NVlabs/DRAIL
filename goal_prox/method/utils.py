import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import wandb
from collections import defaultdict
import os.path as osp
from goal_prox.envs.gw_helper import GwProxPlotter
from goal_prox.envs.debug_viz import DebugViz, LineDebugViz
import rlf.rl.utils as rutils
import torch

def get_success(add_data, masks):
    end_t = [len(m) for m in masks]
    is_success = [False for _ in range(len(masks))]
    if 'ep_found_goal' not in add_data:
        return is_success, end_t

    for i in range(len(masks)):
        for j, found_goal in enumerate(add_data['ep_found_goal'][i]):
            if found_goal == 1.0 or masks[i, j] == 0.0:
                end_t[i] = j+1
                is_success[i] = (found_goal == 1.0)
                break
    return is_success, end_t

def get_full_obs(obs, add_data, end_t):
    n_trajs = len(obs)
    final_state = add_data['final_obs'][:, -1]
    final_state = final_state.view(n_trajs, -1, *final_state.shape[1:])

    for i in range(n_trajs):
        traj_obs = obs[i][:end_t[i]]
        traj_obs = torch.cat([traj_obs, final_state[i]])
        yield traj_obs

def get_visualizer(args, policy, expert_dataset, viz_type):
    save_dir = osp.join(args.save_dir, args.env_name, args.prefix)
    if viz_type == 'gw':
        return GwProxPlotter(save_dir, args,
                rutils.get_obs_shape(policy.obs_space))
    elif viz_type == 'line':
        return LineDebugViz(save_dir, expert_dataset, args)
    elif viz_type is None:
        return DebugViz(save_dir, args)
    else:
        raise ValueError(f"Unexpected viz type {viz_type}")

def get_default_buff_size(args):
    return args.num_processes * args.num_steps

def simple_plot_loss(loss_vals, save_path, use_wb):
    plt.title('Proximity Function Loss')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss Value')
    plt.plot(np.arange(len(loss_vals)), loss_vals)
    plt.savefig(save_path)
    plt.clf()
    if use_wb:
        wandb.log({"pf_loss_curve":
            [wandb.Image(Image.open(save_path))]})
    print('Saved loss viz to %s' % save_path)

def trim_episodes_trans(trajs, orig_trajs):
    """
    Stops the episodes on the first episode the goal is found.
    """
    if 'ep_found_goal' not in trajs:
        return trajs
    done = trajs['done'].float()
    obs = trajs['obs'].float()
    next_obs = trajs['next_obs'].float()
    actions = trajs['actions'].float()

    found_goal = trajs['ep_found_goal'].float()

    real_obs = []
    real_done = []
    real_next_obs = []
    real_actions = []
    real_found_goal = []

    num_samples = done.shape[0]
    start_j = 0
    j = 0
    while j < num_samples:
        if found_goal[j] == 1.0:
            real_obs.extend(obs[start_j:j+1])
            real_done.extend(done[start_j:j+1])
            real_next_obs.extend(next_obs[start_j:j+1])
            real_actions.extend(actions[start_j:j+1])
            real_found_goal.extend(found_goal[start_j:j+1])

            # Force the episode to end now.
            real_done[-1] = torch.ones(done[-1].shape)

            # Move to where this episode ends
            while j < num_samples and not done[j]:
                j += 1
            start_j = j + 1

        if j < num_samples and done[j]:
            start_j = j + 1

        j += 1

    trajs['done'] = torch.stack(real_done)
    trajs['obs'] = torch.stack(real_obs)
    trajs['next_obs'] = torch.stack(real_next_obs)
    trajs['actions'] = torch.stack(real_actions)
    trajs['ep_found_goal'] = torch.stack(real_found_goal)
    return trajs




