"""
Stand alone script to generate the D4RL dataset in our format.
"""
import gym
import d4rl
import torch
import os.path as osp
import argparse
import numpy as np

ENVS = [
        # 'pen-expert-v1',
        # 'door-expert-v1',
        # 'hammer-expert-v1',
        # 'relocate-expert-v1',
        "hopper-expert-v0",
        "halfcheetah-expert-v0",
        "walker2d-expert-v0",
        "ant-expert-v0"
        ]
SAVE_DIR = 'expert_datasets'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--q-format', action='store_true')
    parser.add_argument('--simple-data', action='store_true')
    args = parser.parse_args()
    for env_name in ENVS:
        env = gym.make(env_name)
        save_name = env_name.replace('-', '_')+'_expert_dataset.pt'
        if args.simple_data:
            save_name = 'simple_' + save_name
        if args.q_format:
            data = d4rl.qlearning_dataset(env)
            dones = data['terminals']
            obs = data['observations']
            next_obs = data['next_observations']
            actions = data['actions']
        else:
            data = env.get_dataset()
            dones = data['timeouts'] | data['terminals']
            dones = dones[:-1]
            if args.simple_data:
                use_obs = np.concatenate([data['infos/qpos'],
                    data['infos/qvel'], data['observations']], axis=-1)
            else:
                use_obs = data['observations']
            obs = use_obs[:-1]
            next_obs = use_obs[1:]
            actions = data['actions'][:-1]
        torch.save({
            'done': torch.FloatTensor(dones),
            'obs': torch.tensor(obs),
            'next_obs': torch.tensor(next_obs),
            'actions': torch.tensor(actions)
            }, osp.join(SAVE_DIR, save_name))
        print('Saved to ', save_name)
