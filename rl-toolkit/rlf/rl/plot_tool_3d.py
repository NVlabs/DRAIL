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

def plot_map_3d(
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

    min_x = 0
    max_x = 10
    min_y = -1.5
    max_y = 1.5

    # min_x = -10 #0
    # max_x = 40 #10
    # min_y = -10#-1.5
    # max_y = 5#1.5

    # numpy arrays
    x = np.linspace(min_x, max_x, 1001, dtype=np.float32)  # Divide x-axis into ten equal parts
    y = np.linspace(min_y, max_y, 1001, dtype=np.float32)  # Divide y-axis into ten equal parts

    # Create meshgrid for x and y coordinates
    X, Y = np.meshgrid(x, y)
    
    z = np.zeros((len(x), len(y)))

    # Assign z values to each (x, y) coordinate
    for i in tqdm(range(len(x))):
        for j in range(len(y)):
            obs = x[i]
            action = y[j]
            values = []
            for _ in range(100):  # Perform 10 iterations to obtain 10 values
                #import ipdb; ipdb.set_trace()
                d_val = algo.modules[0].eval_disc(torch.tensor([[obs]]), torch.tensor([[action]])).detach()
                values.append(d_val.item())
            avg_val = sum(values) / len(values)  # Calculate the average of the 10 values
            z[j, len(x)-i-1] = avg_val

    scalar = MinMaxScaler(feature_range=(0, 1))
    z = scalar.fit_transform(z)

    # Create a figure and axes for the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D surface
    surf = ax.plot_surface(X, Y, z, cmap='viridis')

    # Add a colorbar
    fig.colorbar(surf)

    # Set labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Value')
    ax.set_title('3D Heatmap')

    # Set the viewing angles
    elevation = 30  # Specify the desired elevation angle in degrees
    azimuth = 45  # Specify the desired azimuth angle in degrees
    ax.view_init(elevation, azimuth)

    plt.savefig('simple_env_plot/reward_map/' + args.png_name)
    plt.close()

    # Save x, y, and z into a compressed .npz file
    np.savetxt('plot_data/adm_x_mid.txt', x)
    np.savetxt('plot_data/adm_y_mid.txt', y)
    np.savetxt('plot_data/adm_z_midtxt', z)

    return obs

