import numpy as np
import gym
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os.path as osp
import wandb
from PIL import Image
from collections import defaultdict

from ..gym_minigrid.wrappers import *
from .debug_viz import DebugViz
import rlf.rl.utils as rutils
import random


def get_grid(N):
    return [(x, y)
            for x in range(1, N+1)
            for y in range(1, N+1)]


def get_universe(quads):
    quad_displacement = {
        '1': [0, 0],
        '2': [9, 0],
        '3': [0, 9],
        '4': [9, 9],
    }
    all_spots = []
    for quad in quads:
        offset = quad_displacement[quad]
        grid = get_grid(8)
        grid = [(x+offset[0], y+offset[1]) for x, y in grid]
        all_spots.extend(grid)

    # Set to a fixed seed so we have reproducable results
    rng = np.random.RandomState(42)
    rng.shuffle(all_spots)
    return all_spots


def get_universes(gw_agent_quad, gw_goal_quad, gw_cover, gw_compl):
    all_quads = '1,2,3,4'
    agent_quads = (gw_agent_quad or all_quads).split(',')
    goal_quads = (gw_goal_quad or all_quads).split(',')

    agent_u = get_universe(agent_quads)
    goal_u = get_universe(goal_quads)

    if gw_cover != 1.0:
        num_a = int(len(agent_u) * gw_cover)
        num_g = int(len(goal_u) * gw_cover)

        if gw_compl:
            agent_u = agent_u[num_a+1:]
            goal_u = goal_u[num_a+1:]
        else:
            agent_u = agent_u[:num_a]
            goal_u = goal_u[:num_a]
    return agent_u, goal_u


def choice_except(options, bad_val, rnd):
    choose_idx = rnd.choice([i for i, x in enumerate(options) if not (x == bad_val).all()])
    return options[choose_idx]


def convert_to_graph(grid_state):
    _, width, height = grid_state.shape
    graph = np.zeros((width, height))
    agent_pos = None
    goal_pos = None
    for i in range(width):
        for j in range(height):
            grid_val = grid_state[:, i, j]
            node_type = torch.argmax(grid_val).item()
            if node_type == 2:
                # Goal
                goal_pos = (i, j)
                node_val = 1
            elif node_type == 3:
                # Agent start
                agent_pos = (i, j)
                node_val = 1
            elif node_type == 1:
                # Wall
                node_val = 0
            elif node_type == 0:
                # Empty
                node_val = 1
            else:
                print(grid_val)
                raise ValueError('Unrecognized grid val')
            graph[j, i] = node_val
    return graph, agent_pos, goal_pos


def get_env_for_pos(agent_pos, goal_pos, args):
    return GoalCheckerWrapper(FullyObsWrapper(gym.make(args.env_name,
        agent_pos=agent_pos, goal_pos=goal_pos)), args)

def get_grid_obs_for_env(env):
    grid = env._get_obs(env.observation(env.gen_obs()))
    return grid

def plot_prox_heatmap(get_prox_func_fn, save_dir, iter_count,
                      name, args, obs_shape, with_compl):
    agent_u, goal_u = get_universes(args.gw_agent_quad, args.gw_goal_quad,
                                    args.gw_cover, with_compl)

    if args.gw_goal_pos is not None:
        fixed_goal = tuple([int(x) for x in args.gw_goal_pos.split(',')])
    else:
        fixed_goal = goal_u[0]
    agent_u = [x for x in agent_u if x != fixed_goal]

    states = []
    # The door positions
    agent_u = [(4, 9), (11, 9), (9, 4), (9, 14), *agent_u]
    for agent_pos in agent_u:
        env = get_env_for_pos(agent_pos, fixed_goal, args)
        # Need to get the partial observation, then get the full observation,
        # then get the one hot encoding
        grid = get_grid_obs_for_env(env)
        grid = grid.transpose(2, 0, 1)
        states.append(grid)

    states = torch.FloatTensor(states).to(args.device)
    proximities = get_prox_func_fn(states, action=None)
    p_vals = np.zeros((obs_shape[1], obs_shape[2]))
    for i, (x, y) in enumerate(agent_u):
        p_vals[x, y] = proximities[i].item()

    ax = sns.heatmap(p_vals, linewidth=0.5, annot=False, xticklabels=False,
            yticklabels=False)
    ax.tick_params(left=False, bottom=False)
    save_path = osp.join(save_dir, '%s_heat_%i.png' % (name, iter_count))
    print('Saved proximity heatmap to %s' % save_path)
    plt.savefig(save_path)
    plt.clf()

    if not args.no_wb:
        wandb.log({name: [wandb.Image(Image.open(save_path))]})

def disp_gw_state(state):
    for j in range(state.shape[2]):
        row_str = ''
        for i in range(state.shape[1]):
            grid_val = state[:, i, j]
            node_type = torch.argmax(grid_val).item()
            if node_type == 2:
                # Goal
                row_str += 'g'
            elif node_type == 3:
                # Agent start
                row_str += 'a'
            elif node_type == 1:
                # Wall
                row_str += 'x'
            elif node_type == 0:
                # Empty
                row_str += ' '
        print(row_str)

class GwProxPlotter(DebugViz):
    def __init__(self, save_dir, args, obs_shape):
        super().__init__(save_dir, args)
        self.obs_shape = obs_shape
        self.reset()

    def reset(self):
        self.sum_prox = defaultdict(lambda: np.zeros(self.obs_shape[1:]))
        self.count = defaultdict(lambda: np.zeros(self.obs_shape[1:]))

    def _should_process_batches(self):
        return (not self.args.gw_rand_pos) or (self.args.gw_goal_pos is not None)

    def add(self, batches):
        if not self._should_process_batches():
            return
        for name, batch in batches.items():
            state = batch['state'].cpu().numpy()
            proxs = batch['prox']
            _, agent_x, agent_y = np.where(state[:, 3] == 1)

            for x,y,prox in zip(agent_x, agent_y, proxs):
                self.sum_prox[name][x, y] += prox
                self.count[name][x, y] += 1

    def plot(self, iter_count, plot_names, plot_funcs):
        if self._should_process_batches():
            for plot_name in plot_names:
                avg_prox = self.sum_prox[plot_name] / self.count[plot_name]
                if np.isnan(avg_prox).all():
                    # We have nothing to render here.
                    continue

                #avg_prox[np.isnan(avg_prox)] = 0.0
                sns.heatmap(avg_prox)
                save_name = osp.join(self.save_dir, '%s_%i.png' % (plot_name, iter_count))
                plt.savefig(save_name)
                print(f"Saved to {save_name}")
                plt.clf()

        for func_name, plot_func in plot_funcs.items():
            plot_prox_heatmap(plot_func, self.save_dir, iter_count, func_name,
                              self.args, self.obs_shape, False)
            if self.args.gw_cover != 1.0:
                plot_prox_heatmap(plot_func, self.save_dir, iter_count, func_name + "_compl",
                                  self.args, self.obs_shape, True)

        self.reset()


def sample_range_no_replace(cache_vals, N, avoid_coords):
    if 'all_starts' not in cache_vals:
        cache_vals['all_starts'] = [(x, y)
                                    for x in range(1, N-1)
                                    for y in range(1, N-1) if (x, y) not in avoid_coords]
        cache_vals['sample_idx'] = list(range(len(cache_vals['all_starts'])))

    sample_idx = cache_vals['sample_idx']
    all_starts = cache_vals['all_starts']

    if len(sample_idx) == 0:
        cache_vals['sample_idx'] = list(range(len(all_starts)))
    idx = np.random.choice(sample_idx)
    del cache_vals['sample_idx'][sample_idx.index(idx)]
    start = all_starts[idx]
    return start


def gw_empty_spawn(env, cache_vals, args, N=8):
    if args.gw_rand_pos:
        start = sample_range_no_replace(cache_vals, N, [(6, 6)])
        env.env.agent_start_pos = start


def gw_room_spawn(env, cache_vals, args):
    if not args.gw_rand_pos:
        # Always start at the top left corner.
        env.env._agent_default_pos = [1, 1]
    else:
        if 'universe' not in cache_vals:
            cache_vals['universe'] = get_universes(args.gw_agent_quad,
                                                   args.gw_goal_quad, args.gw_cover, args.gw_compl)
        univ = cache_vals['universe']
        univ = np.array(univ)

        if args.gw_goal_pos is None:
            # Sample a position from both.
            agent_start_idx = env.np_random.choice(len(univ[0]))
            agent_start = univ[0][agent_start_idx]
            # Sample from anywhere except for agent position.
            goal_start = choice_except(univ[1], agent_start, env.np_random)
        else:
            goal_start = tuple([int(x) for x in args.gw_goal_pos.split(',')])
            agent_start = choice_except(univ[0], goal_start, env.np_random)

        env.env._agent_default_pos = agent_start
        env.env._goal_default_pos = goal_start


NODE_TO_ONE_HOT = {
    # Empty square
    (1, 0, 0): [1, 0, 0, 0],
    # Wall
    (2, 5, 0): [0, 1, 0, 0],
    # Goal
    (8, 1, 0): [0, 0, 1, 0],
    # Agent
    (10, 0, 0): [0, 0, 0, 1],
    (10, 0, 1): [0, 0, 0, 1],
    (10, 0, 2): [0, 0, 0, 1],
    (10, 0, 3): [0, 0, 0, 1],
}


class GoalCheckerWrapper(gym.Wrapper):
    def __init__(self, env, args):
        super().__init__(env)
        assert args.gw_img
        ob_s = env.observation_space.spaces['image'].shape

        ob_shape = (ob_s[0], ob_s[1], 4)

        low = 0.0
        high = 1.0
        self.observation_space = gym.spaces.Box(shape=ob_shape,
                                                low=np.float32(low),
                                                high=np.float32(high),
                                                dtype=np.float32)

        # Transform the action space to the cardinal directions
        self.action_space = gym.spaces.Discrete(4)
        self.cache_vals = {}
        self.args = args
        self.set_cond = {
            'MiniGrid-Empty-8x8-v0': gw_empty_spawn,
            'MiniGrid-FourRooms-v0': gw_room_spawn,
        }

    def _get_obs(self, obs_dict):
        obs = obs_dict['image']

        obs = obs.reshape(-1, 3)
        obs = np.array(list(map(lambda x: NODE_TO_ONE_HOT[tuple(x)], obs)))
        obs = obs.reshape(*rutils.get_obs_shape(self.observation_space))
        if self.args.gw_img:
            return obs
        else:
            return obs.reshape(-1)

    def reset(self):
        # Call whatever setup is specific to this version of grid world.
        self.set_cond[self.args.env_name](self.env, self.cache_vals, self.args)
        self.found_goal = False
        return self._get_obs(self.env.reset())

    def step(self, a):
        desired_dir = a
        self.env.env.agent_dir = desired_dir

        obs_dict, reward, done, info = self.env.step(self.env.actions.forward)

        obs = self._get_obs(obs_dict)

        if done and reward > 0.0:
            self.found_goal = True
        info['ep_found_goal'] = float(self.found_goal)
        return obs, reward, done, info
