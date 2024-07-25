# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_s_curve

# Define the GAIL agent class
class GAILAgent(nn.Module):
    def __init__(self, input_size, g_lr=0.0005, d_lr=0.00005):
        super(GAILAgent, self).__init__()
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator(input_size)
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=g_lr)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=d_lr)
        self.loss_function = nn.BCELoss()

    def build_generator(self):
        model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            #nn.Sigmoid()
        )
        return model

    def build_discriminator(self, input_size):
        model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        return model

    def forward(self, x):
        #import ipdb; ipdb.set_trace()
        return self.generator(x)

    def train_step(self, x, true_y, d_epoch=5, g_epoch=1, g_entropy_weight=0.5):
        predict_y = self.forward(x)
        #import ipdb; ipdb.set_trace()
        generator_data = torch.concat([x, predict_y.detach()])
        expert_data = torch.concat([x, true_y])
        #print("predict_y:", predict_y.detach())
        #print("true_y:", true_y)

        # Train the discriminator
        for _ in range(d_epoch):
            self.optimizer_d.zero_grad()
            #import ipdb; ipdb.set_trace()
            expert_predictions = self.discriminator(expert_data)
            generator_predictions = self.discriminator(generator_data)
            #import ipdb; ipdb.set_trace()

            expert_loss = -torch.mean(torch.log(expert_predictions + 1e-8))
            generator_loss = -torch.mean(torch.log(1 - generator_predictions + 1e-8))

            d_loss = expert_loss + generator_loss
            d_loss.backward()
            self.optimizer_d.step()

        # Train the generator
        for _ in range(g_epoch):
            self.optimizer_g.zero_grad()

            generator_predictions = self.discriminator(generator_data)
            expert_labels = torch.ones_like(generator_predictions)  # Use ones as expert labels for MSE loss
            '''
            # Calculate entropy regularization
            generator_logits = self.discriminator(generator_data)
            generator_probabilities = torch.sigmoid(generator_logits)
            entropy = -torch.mean(generator_probabilities * torch.log(generator_probabilities + 1e-8) +
                                (1 - generator_probabilities) * torch.log(1 - generator_probabilities + 1e-8))

            # Update generator loss with entropy regularization
            g_loss += g_entropy_weight * entropy
            '''
            g_loss = torch.mean(torch.log(1 - generator_predictions + 1e-8))

            g_loss.backward()
            self.optimizer_g.step()

        return d_loss.item(), g_loss.item()

