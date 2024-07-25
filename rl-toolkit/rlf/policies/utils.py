from rlf.rl.model import MLPBasic, MLPBase, CNNBase, IdentityBase, TwoLayerMlpWithAction, PassThroughBase
from rlf.rl.distributions import DiagGaussian, MixedDist, Categorical
from rlf.rl.model import def_mlp_weight_init, weight_init, reg_mlp_weight_init
import torch.nn as nn
import rlf.rl.utils as rutils

def is_image_obs(obs_shape):
    if len(obs_shape) == 3:
        return True
    elif len(obs_shape) == 1:
        return False
    else:
        raise NotImplementedError(
            'Observation space is %s' % str(obs_shape))

def get_def_critic_head(hidden_dim):
    critic_head = nn.Linear(hidden_dim, 1)
    critic_head = def_mlp_weight_init(critic_head)
    return critic_head

def get_def_actor_head(hidden_dim, action_dim):
    return def_mlp_weight_init(nn.Linear(hidden_dim, action_dim))

def get_mlp_net_out_fn(hidden_sizes):
    """
    Gives a function returning an MLP base with the specified architecture that
    takes as input the shape. This is an easy way to create default NN creation
    functions that can be later overriden.
    Returns: (i_shape -> MLPBase)
    """
    return lambda i_shape: MLPBase(i_shape[0], False, hidden_sizes,
            weight_init=reg_mlp_weight_init, no_last_act=True)

def get_mlp_net_var_out_fn(hidden_sizes):
    """
    Same as `get_mlp_net_fn` but you can specify a variable output size later
    after the function is returned.
    """
    return lambda i_shp, n_out: MLPBase(i_shp[0], False,
            [*hidden_sizes, n_out],
            weight_init=reg_mlp_weight_init, no_last_act=True)

def def_get_hidden_net(input_shape, hidden_size=64, num_layers=2,
        recurrent=False):
    if is_image_obs(input_shape):
        return CNNBase(input_shape[0], False, hidden_size)
    else:
        return MLPBasic(input_shape[0], hidden_size=hidden_size,
                num_layers=num_layers)

def get_img_encoder(obs_shape, recurrent, hidden_size=64):
    """
    Gets the base encoder.
    """
    if is_image_obs(obs_shape):
        return def_get_hidden_net(obs_shape, recurrent=recurrent)
    else:
        return PassThroughBase(obs_shape, recurrent, hidden_size)


def get_def_critic(obs_shape, input_shape, action_space):
    assert len(input_shape) == 1

    if is_image_obs(obs_shape):
        return IdentityBase(input_shape)
    else:
        return def_get_hidden_net(input_shape)

def get_def_ac_critic(obs_shape, input_shape, action_space, hidden_size=(64, 64)):
    assert len(input_shape) == 1
    ac_dim = rutils.get_ac_dim(action_space)
    return TwoLayerMlpWithAction(input_shape[0], hidden_size, ac_dim)


def get_def_actor(obs_shape, input_shape):
    if is_image_obs(obs_shape):
        return IdentityBase(input_shape)
    else:
        return def_get_hidden_net(input_shape)


def get_def_dist(input_shape, action_space):
    input_size = input_shape[0]
    if action_space.__class__.__name__ == "Discrete":
        return Categorical(input_size, action_space.n)
    elif action_space.__class__.__name__ == "Box":
        return DiagGaussian(input_size, action_space.shape[0])
    elif action_space.__class__.__name__ == 'Dict':
        keys = list(action_space.spaces.keys())
        return MixedDist(
                Categorical(input_size, action_space.spaces[keys[0]].n),
                DiagGaussian(input_size, action_space.spaces[keys[1]].shape[0]))
    else:
        raise NotImplementedError('Unrecognized environment action space')


