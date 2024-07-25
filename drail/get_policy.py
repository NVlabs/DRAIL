# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from functools import partial
import torch.nn as nn
import drail.ddpm.policy_model as policy_model
from rlf.policies import BasicPolicy, DistActorCritic
from rlf.policies.actor_critic.dist_actor_q import (DistActorQ, get_sac_actor,
                                                    get_sac_critic)
from rlf.policies.actor_critic.reg_actor_critic import RegActorCritic
from rlf.rl.model import MLPBase, MLPBasic, TwoLayerMlpWithAction
from goal_prox.models import GwImgEncoder

def get_ppo_policy(env_name, args):
    if env_name.startswith("MiniGrid") and args.gw_img:
        return DistActorCritic(get_base_net_fn=lambda i_shape: GwImgEncoder(i_shape))

    return DistActorCritic(
        get_actor_fn=lambda _, i_shape: MLPBasic(
            i_shape[0], hidden_size=args.ppo_hidden_dim, num_layers=args.ppo_layers
        ),
        get_critic_fn=lambda _, i_shape, asp: MLPBasic(
            i_shape[0], hidden_size=args.ppo_hidden_dim, num_layers=args.ppo_layers
        ),
    )

# def get_deep_sac_policy(env_name, args):
#     return DistActorQ(
#         get_critic_fn=partial(get_sac_critic, hidden_dim=256),
#         get_actor_fn=partial(get_sac_actor, hidden_dim=256),
#     )

def get_deep_sac_policy(env_name, args):
    return DistActorQ(
        get_critic_fn=get_sac_critic,
        get_actor_fn=get_sac_actor,
    )

def get_deep_iqlearn_policy(env_name, args):
    return DistActorQ(
        get_critic_fn=get_sac_critic,
        get_actor_fn=get_sac_actor,
    )


def get_deep_ddpg_policy(env_name, args):
    def get_actor_head(hidden_dim, action_dim):
        return nn.Sequential(nn.Linear(hidden_dim, action_dim), nn.Tanh())

    return RegActorCritic(
        get_actor_fn=lambda _, i_shape: MLPBase(i_shape[0], False, (256, 256)),
        get_actor_head_fn=get_actor_head,
        get_critic_fn=lambda _, i_shape, a_space: TwoLayerMlpWithAction(
            i_shape[0], (256, 256), a_space.shape[0]
        ),
    )

def get_basic_policy(env_name, args, is_stoch):
    if env_name.startswith("MiniGrid") and args.gw_img:
        return BasicPolicy(
            is_stoch=is_stoch, get_base_net_fn=lambda i_shape: GwImgEncoder(i_shape)
        )
    else:
        return BasicPolicy(
            is_stoch=is_stoch,
            get_base_net_fn=lambda i_shape: MLPBasic(
                i_shape[0],
                hidden_size=args.hidden_dim,
                num_layers=args.depth
            ),
        )

    return BasicPolicy()

def get_diffusion_policy(env_name, args, is_stoch):    
    if env_name[:9] == 'FetchPush':
        state_dim = 16
        action_dim = 3
    if env_name[:9] == 'FetchPick':
        state_dim = 16
        action_dim = 4
    if env_name[:10] == 'CustomHand':
        state_dim = 68
        action_dim = 20
    if env_name[:4] == 'maze':
        state_dim = 6
        action_dim = 2
    if env_name[:6] == 'Walker':
        state_dim = 17
        action_dim = 6
    if env_name[:11] == 'HalfCheetah':
        state_dim = 17
        action_dim = 6
    if env_name[:7] == 'AntGoal':
        state_dim = 132
        action_dim = 8
    return policy_model.MLPDiffusion(
        n_steps = 1000,
        action_dim=action_dim, 
        state_dim=state_dim,
        num_units=args.hidden_dim,
        depth=args.depth,
        is_stoch=is_stoch,
        )

def get_deep_basic_policy(env_name, args):
    return BasicPolicy(
        get_base_net_fn=lambda i_shape: MLPBase(i_shape[0], False, (512, 512, 256, 128))
    )

