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

def plot_map(
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
    if args.env_name == 'Sine-v0':
        min_x = 0
        max_x = 10
        min_y = -1.5
        max_y = 1.5

    ### scurve
    if args.env_name == 'SCurve-v0':
        min_x = -3
        max_x = 3
        min_y = -3
        max_y = 3

    # min_x = -10 #0
    # max_x = 40 #10
    # min_y = -10#-1.5
    # max_y = 5#1.5

    # numpy arrays
    x = np.linspace(min_x, max_x, 101, dtype=np.float32)  # Divide x-axis into ten equal parts
    y = np.linspace(min_y, max_y, 101, dtype=np.float32)  # Divide y-axis into ten equal parts
    
    z = np.zeros((len(x), len(y)))

    algo.modules[0].discrim_net.eval()
    with torch.no_grad():
        d_val1 = algo.modules[0].eval_disc(torch.tensor([[3.5]]), torch.tensor([[1]])).detach()
        print("d_val1:", d_val1)
        d_val1 = algo.modules[0].eval_disc(torch.tensor([[3.5]]), torch.tensor([[1]])).detach()
        print("d_val1:", d_val1)
        d_val2 = algo.modules[0].eval_disc(torch.tensor([[2.0]]), torch.tensor([[-1.0]])).detach()
        print("d_val2:", d_val2)
        d_val2 = algo.modules[0].eval_disc(torch.tensor([[2.0]]), torch.tensor([[-1.0]])).detach()
        print("d_val2:", d_val2)

    # Assign z values to each (x, y) coordinate
    for i in tqdm(range(len(x))):
        for j in range(len(y)):
            obs = x[i]
            action = y[j]
            #d_val = algo.modules[0]._compute_disc_val(torch.tensor([obs]), torch.tensor([predict_action])).detach()
            #z[i, j] = -1 - torch.log(-d_val).item()
            values = []
            for _ in range(100):  # Perform 10 iterations to obtain 10 values
                d_val = algo.modules[0].eval_disc(torch.tensor([[obs]]), torch.tensor([[action]])).detach()
                #d_val = algo.modules[0]._compute_disc_val(torch.tensor([action]), torch.tensor([obs])).detach()
                values.append(d_val.item())
            avg_val = sum(values) / len(values)  # Calculate the average of the 10 values
            z[j, len(x)-i-1] = avg_val

    scalar = MinMaxScaler(feature_range=(0, 1))
    z = scalar.fit_transform(z)

    import re
    parts = args.png_name.split('-')
    #import ipdb; ipdb.set_trace()
    if 'adm' in parts:
        if len(parts) == 6:
            title_name = f'{parts[5]} {parts[4]} (random step)'
        elif len(parts) == 7:
            title_name = f'{parts[5]} {parts[4]} (fix {parts[-1]})'
        else:
            title_name = 'else'
    else:
        title_name = f'{parts[5]} {parts[4]}'

    # Plot the heatmap
    #plt.imshow(z, interpolation='gaussian', cmap='hot', origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])
    #plt.axis('equal')
    plt.imshow(z, interpolation='gaussian', cmap='viridis', extent=[x.min(), x.max(), y.min(), y.max()])
    #plt.imshow(z, interpolation='gaussian', cmap='viridis', extent=[x.min(), x.max(), y.min(), y.max()], vmin=-7, vmax=1)
    plt.colorbar()  # Add a colorbar
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'reward map of {title_name}')
    plt.savefig('simple_env_plot/reward_map/' + args.png_name)
    plt.close()

    return obs

