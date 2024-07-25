import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlf.rl.model import weight_init, no_bias_weight_init, reg_mlp_weight_init, MLPBase, MyMLP
from functools import partial
import numpy as np
from torch import distributions as pyd

#
# Standardize distribution interfaces
#

# Categorical
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(
    self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

# Normal
FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(
    self, actions).sum(
        -1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        init_ = lambda m: weight_init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)



class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        init_ = lambda m: weight_init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = nn.Parameter(torch.zeros(1, num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        action_logstd = self.logstd.expand_as(action_mean)
        return FixedNormal(action_mean, action_logstd.exp())



class DistWrapper(torch.distributions.distribution.Distribution):
    def __init__(self, disc,  cont):
        super().__init__()
        self.disc = disc
        self.cont = cont
        self.args = args
        self.cont_entropy_coef = args.cont_entropy_coef

    def mode(self):
        cont_sample = self.cont.mode().float()
        return torch.cat([self.disc.mode().float(), cont_sample], dim=-1)

    def sample(self):
        cont_sample = self.cont.sample().float()
        return torch.cat([self.disc.sample().float(), cont_sample], dim=-1)

    def log_probs(self, x):
        cont_prob = self.cont.log_probs(x[:, 1:]).float()

        log_probs = torch.cat([
            self.disc.log_probs(x[:, :1]).float(),
            cont_prob], dim=-1)
        return log_probs.sum(-1).unsqueeze(-1)

    def __str__(self):
        return 'Cont: %s, Disc: %s' % (self.cont, self.disc)

    def entropy(self):
        disc_ent = self.disc.entropy().float()
        cont_ent = self.cont.entropy().float()
        if len(disc_ent.shape) == 1:
            disc_ent = disc_ent.unsqueeze(-1)
            cont_ent = cont_ent.unsqueeze(-1)
        entropy = torch.cat([disc_ent, self.cont_entropy_coef * cont_ent], dim=-1)
        return entropy.sum(-1).unsqueeze(-1)


class MixedDist(nn.Module):
    def __init__(self, disc, cont):
        super().__init__()
        self.cont = cont
        self.disc = disc
        self.args = args

    def forward(self, x):
        cont_out = self.cont(x)
        disc_out = self.disc(x)
        return DistWrapper(disc_out, cont_out, args=self.args)


class TanhTransform(pyd.transforms.Transform):
    """
    Code from https://github.com/denisyarats/pytorch_sac.
    """
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """
    Code from https://github.com/denisyarats/pytorch_sac.
    """
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        
        # position
        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

class DiagGaussianActor(nn.Module):
    """
    Code from https://github.com/denisyarats/pytorch_sac.
    """
    def __init__(self, num_inputs, num_outputs, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()

        dims = [hidden_dim] * hidden_depth
        dims.append(2*num_outputs)

        self.log_std_bounds = log_std_bounds
        # self.trunk = MLPBase(num_inputs, False, dims,
        #         weight_init=reg_mlp_weight_init,
        #         get_activation=lambda: nn.ReLU(inplace=True),
        #         no_last_act=True)
        self.trunk = MLPBase(num_inputs, False, dims, weight_init=reg_mlp_weight_init,)

        self.apply(no_bias_weight_init)

    def forward(self, obs, hxs, masks):
        print("obs:", obs)
        print("hxs:", hxs)
        print("masks:", masks)
        x, _ = self.trunk(obs, hxs, masks)
        mu, log_std = x.chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        # [0, 1] -> [log_std_min, log_std_max]
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +1)

        std = log_std.exp()

        dist = SquashedNormal(mu, std)
        return dist

