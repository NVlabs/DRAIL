# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import rl_utils
from traj import PPO
import wandb
import datetime

# Define the RL environment
class RLEnvironment:
    def __init__(self, data, x_coordinates):
        #self.x_min = -10
        #self.x_max = 10
        #self.y_min = 0
        #self.y_max = 20
        self.x_min = -1
        self.x_max = 1
        self.y_min = -2
        self.y_max = 2
        self.tolerance = 1e-3  # Set the tolerance threshold
        self.data = data
        self.x_coordinates = x_coordinates

    def get_true_y(self, x):
        if isinstance(x, np.ndarray):
            indice = [np.random.choice(np.where(np.abs(self.x_coordinates - x_val) < self.tolerance)[0]) for x_val in x]
        else:
            indices = np.where(np.abs(self.x_coordinates - x) < self.tolerance)[0]
            indice = [np.random.choice(indices)]
        y = data[indice, 1]

        return y

    def get_reward(self, predicted_y, true_y):
        # Calculate the reward based on the prediction error
        return -torch.abs(predicted_y - true_y)

    def reset(self):
        state = [np.random.choice(self.x_coordinates)]
        # Reset the environment
        return state

    def step(self, state, predict_y):
        # Get the true y-coordinate for the given x-coordinate
        true_y = self.get_true_y(state)
        next_state = [np.random.choice(self.x_coordinates)]

        return next_state, true_y

class Discriminator(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Discriminator, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        import ipdb; ipdb.set_trace()
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        return torch.sigmoid(self.fc2(x))

class GAIL:
    def __init__(self, agent, state_dim, action_dim, hidden_dim, lr_d):
        self.discriminator = Discriminator(state_dim, hidden_dim,
                                           action_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d)
        self.agent = agent

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s):
        expert_states = torch.tensor(expert_s, dtype=torch.float).to(device)
        expert_actions = torch.tensor(expert_a).to(device)
        agent_states = torch.tensor(agent_s, dtype=torch.float).to(device)
        agent_actions = torch.tensor(agent_a).to(device)
        expert_actions = F.one_hot(expert_actions.to(torch.int64), num_classes=2).float()
        agent_actions = F.one_hot(agent_actions.to(torch.int64), num_classes=2).float()

        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)
        discriminator_loss = nn.BCELoss()(
            agent_prob, torch.ones_like(agent_prob)) + nn.BCELoss()(
                expert_prob, torch.zeros_like(expert_prob))
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        wandb.log({'discriminator_loss':discriminator_loss})

        rewards = -torch.log(agent_prob).detach().cpu().numpy()
        wandb.log({'agent_prob':agent_prob[0]})
        wandb.log({'reward for policy training':rewards[0]})
        transition_dict = {
            'states': agent_s,
            'actions': agent_a,
            'rewards': rewards,
            'next_states': next_s,
        }
        self.agent.update(transition_dict)

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
wandb.init(project='gail_sine')

# Generate data from sine function
np.random.seed(42)  # Set random seed for reproducibility
# Define the parameters for the sine function
amplitude = 1.0  # Amplitude of the sine wave
frequency = 0.1  # Frequency of the sine wave
phase = 0.0  # Phase shift of the sine wave
noise_std = 0.05  # Standard deviation of the Gaussian noise
# Generate x values
x = np.linspace(0, 10, num=1000)
y = amplitude * np.sin(2 * np.pi * frequency * x + phase) + np.random.normal(0, noise_std, size=len(x))
expert_s = x_coordinates = x
expert_a = y
data = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)

# Initialize the environment
state_dim = 1
action_dim = 1
input_size = state_dim + action_dim
learning_rate = 0.01
env = RLEnvironment(data, x_coordinates)

# Initialize the RL agent
actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 250
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = 'cpu'

lr_d = 1e-3
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)
gail = GAIL(agent, state_dim, action_dim, hidden_dim, lr_d)
n_episode = 500
return_list = []

with tqdm(total=n_episode, desc="Progress") as pbar:
    for i in range(n_episode):
        state = env.reset()
        state_list = []
        action_list = []
        next_state_list = []
        #while not done:
        for i in range(100):
            action = agent.take_action(state)
            next_state, true_y = env.step(state, action)
            state_list.append(state)
            action_list.append(action)
            next_state_list.append(next_state)
            state = next_state
        gail.learn(expert_s, expert_a, state_list, action_list, next_state_list)
        pbar.update(1)

'''
# Plotting the results
plt.figure()
plt.scatter(state_list, true_test_y, label='Ground Truth')
plt.scatter(state_list, action_list, label='Predicted')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('result_sine.png')
'''