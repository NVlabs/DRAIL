import os
import os.path as osp
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
from rlf.exp_mgr.viz_utils import save_mp4
from rlf.il.traj_mgr import TrajSaver
from rlf.policies.base_policy import get_empty_step_info
from rlf.rl import utils
from rlf.rl.envs import get_vec_normalize, make_vec_envs
from tqdm import tqdm
import wandb
import time
import math
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_density_division(
    envs,
    policy,
    algo,
    log,
    checkpointer,
    env_interface,
    args,
    alg_env_settings,
    create_traj_saver_fn,
    vec_norm,
):
    sth = plot(
        args,
        alg_env_settings,
        policy,
        algo,
        vec_norm,
        env_interface,
        0,
        "final",
        envs,
        log,
        create_traj_saver_fn,
    )
    envs.close()

    return sth

def plot(
    args,
    alg_env_settings,
    policy,
    algo,
    true_vec_norm,
    env_interface,
    num_steps,
    mode,
    eval_envs,
    log,
    create_traj_saver_fn,
    final=False
):

    if eval_envs is None:
        args.force_multi_proc = False
        _, eval_envs = make_vec_envs(
            args.env_name,
            args.seed + num_steps,
            num_processes,
            args.gamma,
            args.device,
            True,
            env_interface,
            args,
            alg_env_settings,
            set_eval=True,
        )
 

    ### sine
    min_x = 0
    max_x = 10
    min_y = -1.5
    max_y = 1.5
    interval = 0.1

    amplitude = 1.0  # Amplitude of the sine wave
    frequency = 0.1  # Frequency of the sine wave
    phase = 0.0  # Phase shift of the sine wave
    noise_std = 0.05  # Standard deviation of the Gaussian noise
    # Generate x values
    expert_x_data = np.linspace(0, 10, num=2000)
    expert_y_data = amplitude * np.sin(2 * np.pi * frequency * expert_x_data + phase) + np.random.normal(0, noise_std, size=len(expert_x_data))

    #env_name = args.env_name.lower().replace("-v0", "")
    #file_name = env_name + '_' + args.alg + '_.npz'
    file_name = args.png_name + '_' + args.alg + '_.npz'
    data = np.load(file_name)
    agent_x_data = data['state']
    agent_y_data = data['action']

    # Create a 2D histogram to represent the grid and calculate frequencies
    expert_hist, x_edges, y_edges = np.histogram2d(expert_x_data, expert_y_data, bins=[np.arange(min_x, max_x + interval, interval), np.arange(min_y, max_y + interval, interval)])
    expert_freq = expert_hist / len(expert_x_data)
    agent_hist, x_edges, y_edges = np.histogram2d(agent_x_data, agent_y_data, bins=[np.arange(min_x, max_x + interval, interval), np.arange(min_y, max_y + interval, interval)])
    agent_freq = agent_hist / len(agent_x_data)

    # Print the frequencies
    # for i in range(len(x_edges) - 1):
    #     for j in range(len(y_edges) - 1):
    #         print(f"Grid ({x_edges[i]:.1f}-{x_edges[i+1]:.1f}, {y_edges[j]:.1f}-{y_edges[j+1]:.1f}): Frequency = {frequencies[i, j]:.4f}")
    
    # Create a heatmap plot
    # agent_freq.T - expert_freq.T
    eps = 1e-20
    division_scalar = agent_freq.T / (expert_freq.T + eps)

    scalar = MinMaxScaler(feature_range=(0, 1))
    division_scalar = scalar.fit_transform(division_scalar)
    # derivative_division_scalar = -1/(division_scalar+eps)
    derivative_division_scalar = np.log(division_scalar + eps) - np.log(division_scalar + eps + 1)
    
    if args.derivative:
        plt.imshow(derivative_division_scalar, origin='lower', extent=[min_x, max_x, min_y, max_y], cmap='viridis', aspect='auto')
        plt.colorbar(label='Frequency')
        plt.axis('equal')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        #substraction
        plt.title('f\'(p/q)')
        plt.savefig('simple_env_plot/division/derivative-division-of-density-' + args.png_name + '.png')
        plt.close()
    else:
        plt.imshow(division_scalar, origin='lower', extent=[min_x, max_x, min_y, max_y], cmap='viridis', aspect='auto')
        plt.colorbar(label='Frequency')
        plt.axis('equal')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        #substraction
        plt.title('density of division')
        plt.savefig('simple_env_plot/division/division-of-density-' + args.png_name + '.png')
        plt.close()

    plt.imshow(agent_freq.T, origin='lower', extent=[min_x, max_x, min_y, max_y], cmap='viridis', aspect='auto')
    plt.colorbar(label='Frequency')
    plt.axis('equal')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    #substraction
    plt.title('density of agent')
    plt.savefig('simple_env_plot/division/agent-density-' + args.png_name + '.png')
    plt.close()  

    plt.imshow(expert_freq.T, origin='lower', extent=[min_x, max_x, min_y, max_y], cmap='viridis', aspect='auto')
    plt.colorbar(label='Frequency')
    plt.axis('equal')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    #substraction
    plt.title('density of expert')
    plt.savefig('simple_env_plot/division/expert-density-' + args.png_name + '.png')
    plt.close()  