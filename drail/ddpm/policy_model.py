# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve
import torch
import cv2
import os, sys
import argparse
import numpy as np
import scipy.stats
# from geomloss import SamplesLoss
import gym
import d4rl # Import required to register environments
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, SequentialSampler, BatchSampler
from tqdm import tqdm
from rlf.policies.basic_policy import BasicPolicy

import inspect
from functools import partial
import rlf.rl.utils as rutils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.03 #0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

########### decide the sample during definite diffusion process
# calculate x on given time based on x_0 and re-parameterization
def q_x(x_0,t):
    """based on x[0], get x[t] on any given time t"""
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return (alphas_t * x_0 + alphas_1_m_t * noise) # adding noise based on x[0]在x[0]

########### gaussian distribution in reverse diffusion process
import torch
import torch.nn as nn

class CreateAction():
    def __init__(self, action):
        self.action = action 
        self.hxs = {}
        self.extra = {}
        self.take_action = action

class MLPDiffusion(BasicPolicy, nn.Module):
    def __init__(self, 
            n_steps=1000,
            action_dim=2,
            state_dim=6,
            num_units=1024,
            depth=2,
            device='cuda', 
            is_stoch=False,
            fuse_states=[],
            use_goal=False,
            get_base_net_fn=None):
        input_dim = action_dim + state_dim
        super(MLPDiffusion, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        linears_list = []
        linears_list.append(nn.Linear(input_dim, num_units))
        linears_list.append(nn.ReLU())
        if depth > 1:
            for i in range(depth-1):
                linears_list.append(nn.Linear(num_units, num_units))
                linears_list.append(nn.ReLU())
        linears_list.append(nn.Linear(num_units, action_dim))
        self.linears = nn.ModuleList(linears_list).to(device)
        embed_list = []
        for i in range(depth-1):
            embed_list.append(nn.Embedding(n_steps, num_units))
        self.step_embeddings = nn.ModuleList(embed_list).to(device)

    def init(self, obs_space, action_space, args):
        self.action_space = action_space
        self.obs_space = obs_space
        self.args = args
        
        # if 'recurrent' in inspect.getfullargspec(self.get_base_net_fn).args:
            # self.get_base_net_fn = partial(self.get_base_net_fn,
                    # recurrent=self.args.recurrent_policy)
        if self.use_goal:
            use_obs_shape = rutils.get_obs_shape(obs_space, args.policy_ob_key)
            if len(use_obs_shape) != 1:
                raise ValueError(('Goal conditioning only ',
                    'works with flat state representation'))
            
            use_obs_shape = (use_obs_shape[0] + obs_space['desired_goal'].shape[0],)
        else:
            use_obs_shape = rutils.get_obs_shape(obs_space, args.policy_ob_key)

    def output(self, x, t):
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2*idx](x)
            x += t_embedding
            x = self.linears[2*idx+1](x)
        x = self.linears[-1](x)
        return x
    
    ########### training loss funciton
    # sample at any given time t, and calculate sampling loss
    def diffusion_loss_fn(self, x_0, condition, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
        batch_size = x_0.shape[0]
        # generate eandom t for a batch data
        t = torch.randint(0, n_steps, size=(batch_size//2,)).to(device)
        t = torch.cat([t, n_steps-1-t], dim=0) #[batch_size, 1]
        t = t.unsqueeze(-1)
        
        # coefficient of x0
        a = alphas_bar_sqrt[t]
        # coefficient of eps
        aml = one_minus_alphas_bar_sqrt[t]
        # generate random noise eps
        e = torch.randn_like(x_0)
        # model input
        x = x_0*a + e*aml
        # condition
        input_data = torch.cat((x, condition), 1)
        # get predicted randome noise at time t
        output = self.output(input_data, t.squeeze(-1))
        
        # calculate the loss between actual noise and predicted noise
        return (e - output).square().mean()

    def get_loss(self, x_0, condition):
        num_steps = 1000
        betas = cosine_beta_schedule(num_steps)
        betas = betas.to(self.args.device)
        
        alphas = 1-betas
        alphas_prod = torch.cumprod(alphas,0).to(self.args.device)
        alphas_prod_p = torch.cat((torch.tensor([1]).float().to(self.args.device),alphas_prod[:-1]),0)
        alphas_bar_sqrt = torch.sqrt(alphas_prod)
        one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
        predict_loss = self.diffusion_loss_fn(x_0, condition, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
        return predict_loss

    ########### reverse diffusion sample function（inference）
    def p_sample_loop(self, condition, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
        # generate[T-1]、x[T-2]|...x[0] from x[T]
        cur_x = torch.randn(shape)
        x_seq = [cur_x]
        for i in reversed(range(n_steps)):
            cur_x = self.p_sample(cur_x, condition, i, betas,one_minus_alphas_bar_sqrt)
            x_seq.append(cur_x)
        return x_seq

    def p_sample(self, x, condition, t, betas, one_minus_alphas_bar_sqrt):
        # sample reconstruction data at time t drom x[T]
        t = torch.tensor([t]).to(device)
        coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
        input_data = torch.cat((x, condition), 1)
        eps_theta = self.output(input_data, t)
        mean = (1/(1-betas[t]).sqrt())*(x-(coeff*eps_theta))
        z = torch.randn_like(x)
        sigma_t = betas[t].sqrt()
        sample = mean + sigma_t * z
        return (sample)


    def reconstruct(self, condition, shape, n_steps, betas, one_minus_alphas_bar_sqrt, device):
        # generate[T-1]、x[T-2]|...x[0] from x[T]
        num_process = condition.shape[0]
        x_T = torch.randn((num_process, shape)).to(device)
        # generate[T-1]、x[T-2]|...x[0] from x[T]
        for i in reversed(range(n_steps)):
            x_T = self.p_sample(x_T, condition, i, betas, one_minus_alphas_bar_sqrt)
        x_construct = x_T
        return x_construct

    '''
    def predict_action(self, noise_shape, condition, device):
        # decide beta
        num_steps = 100
        betas = torch.linspace(-6,6,num_steps)
        betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5
        betas = betas.to(self.args.device)
        
        alphas = 1-betas
        alphas_prod = torch.cumprod(alphas,0).to(self.args.device)
        alphas_prod_p = torch.cat((torch.tensor([1]).float().to(self.args.device),alphas_prod[:-1]),0)
        alphas_bar_sqrt = torch.sqrt(alphas_prod)
        one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
        
        predict = self.reconstruct(condition, noise_shape, num_steps, betas, one_minus_alphas_bar_sqrt, device)
        return predict
    '''

    def get_action(self, state, add_state, rnn_hxs, mask, step_info): 
        # decide beta
        num_steps = 1000
        betas = cosine_beta_schedule(num_steps)
        betas = betas.to(self.args.device)
        
        alphas = 1-betas
        alphas_prod = torch.cumprod(alphas,0).to(self.args.device)
        alphas_prod_p = torch.cat((torch.tensor([1]).float().to(self.args.device),alphas_prod[:-1]),0)
        alphas_bar_sqrt = torch.sqrt(alphas_prod)
        one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
        # import ipdb
        # ipdb.set_trace()
        predict_action = self.reconstruct(state, self.action_dim, num_steps, betas, one_minus_alphas_bar_sqrt, device)

        return CreateAction(predict_action)

########### start training, print loss and print the medium reconstrction result
seed = 1234

class EMA(): # Exponential Moving Average
    #EMA
    def __init__(self,mu=0.01):
        self.mu = mu
        self.shadow = {}
        
    def register(self,name,val):
        self.shadow[name] = val.clone()
        
    def __call__(self,name,x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0-self.mu)*self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average

if __name__ == '__main__':
    
    # Create the environment
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj-load-path')
    args = parser.parse_args()
    env = args.traj_load_path.split('/')[-1].split('.')[0]

    ########### hyper parameter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_steps = 1000
    batch_size = 128 #128
    num_epoch = 4000

    # decide beta
    betas = torch.linspace(1e-4, 0.02, num_steps).to(device)

    # calculate alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt
    alphas = 1-betas
    alphas_prod = torch.cumprod(alphas,0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float().to(device),alphas_prod[:-1]],0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    assert alphas.shape==alphas_prod.shape==alphas_prod_p.shape==\
    alphas_bar_sqrt.shape==one_minus_alphas_bar_log.shape\
    ==one_minus_alphas_bar_sqrt.shape
    print("all the same shape",betas.shape)

    data = torch.load(args.traj_load_path)
    obs = data['obs']
    actions = data['actions']
    next_obs = data['next_obs']
    dataset = torch.cat((obs, actions), 1)

    sample_num = dataset.size()[0]
    if  sample_num % 2 == 1:
        dataset = dataset[1:sample_num, :]
    print("after", dataset.size())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True)

    model = MLPDiffusion(num_steps, data_dim=actions.shape[1], condition_dim=obs.shape[1], num_units=128, device=device)
    optimizer = torch.optim.Adam(model.parameters(),lr=5*1e-5)
    
    train_loss_list = []
    val_loss_list = []
    for ep in tqdm(range(0, num_epoch)):
        epoch_loss = []
        for idx, batch_x in enumerate(dataloader):
            batch_x = batch_x.squeeze().to(device)
            action = batch_x[:,6:]
            state = batch_x[:,:6]
            loss = model.diffusion_loss_fn(action, state, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
            epoch_loss.append(loss.cpu().detach().item())
            #print("loss.cpu():", loss.cpu().detach().item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
        ave_loss = sum(epoch_loss) / len(dataloader)
        train_loss_list.append(ave_loss)
        if(ep % 10 == 0):
             print(ep, " epoch ", "train_loss:", ":", ave_loss)

        if (ep + 1) % 10 == 0:
            train_iteration_list = list(range(len(train_loss_list)))
            plt.plot(train_iteration_list, train_loss_list, color='r')
            #plt.plot(train_iteration_list, val_loss_list, color='b')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(env + '_model_loss.png')
            plt.savefig(env + '_model_loss_{}.png'.format(num_epoch))
            plt.close()

            with torch.no_grad():
                # random sample images
                print("len(dataloader):", len(dataloader))
                fig, axs = plt.subplots(1, 2, figsize=(28, 3))
                for idx, batch_x in tqdm(enumerate(dataloader)):
                    #batch_x = batch_x.squeeze().to(device)
                    action = dataset.squeeze().to(device)[:,6:]
                    state = dataset.squeeze().to(device)[:,:6]
                    out = model.reconstruct(state, action.size(), num_steps, betas, one_minus_alphas_bar_sqrt, device=device)
                    out = out.cpu().detach()

                    axs[0].scatter(dataset[:, 6], dataset[:, 7], color='blue', edgecolor='white')
                    axs[1].scatter(out[:, 0], out[:, 1], color='red', edgecolor='white')
                    file_name = env + 'curve-ddpm-condition-reconstruct-{}-{}.png'.format(num_epoch, ep)
                plt.savefig('results/' + file_name)
                plt.close()  
        
    torch.save(model.state_dict(), env + '_condition_ddpm_{}.pt'.format(num_epoch))
