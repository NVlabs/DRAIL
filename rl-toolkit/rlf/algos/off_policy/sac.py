"""
Code is heavily based off of https://github.com/denisyarats/pytorch_sac.
The license is at `rlf/algos/off_policy/denis_yarats_LICENSE.md`
"""
from collections import defaultdict

import numpy as np
import rlf.algos.utils as autils
import rlf.rl.utils as rutils
import torch
import torch.nn.functional as F
import torch.optim as optim
from rlf.algos.off_policy.actor_critic_updater import ActorCriticUpdater
from rlf.algos.off_policy.off_policy_base import OffPolicy
from rlf.args import str2bool


class SAC(OffPolicy):
    def init(self, policy, args):
        # Need to set up the parameter before we set up the optimizers
        self.log_alpha = torch.tensor(np.log(args.init_temperature)).to(args.device)
        self.log_alpha.requires_grad = True

        super().init(policy, args)

        # set target entropy to -|A|
        self.target_entropy = -rutils.get_ac_dim(policy.action_space)
        #import ipdb; ipdb.set_trace()
        """
        information: in drail/get_policy

        def get_deep_sac_policy(env_name, args):
            return DistActorQ(
                get_critic_fn=partial(get_sac_critic, hidden_dim=256),
                get_actor_fn=partial(get_sac_actor, hidden_dim=256),
            )
        """
        self.target_critic = self.policy.get_critic_fn(
            self.policy.obs_space,
            self.policy._get_base_out_shape(),
            self.policy.action_space,
            self.args.dist_q_hidden_dim,
        )
        self.target_critic = self.target_critic.to(self.args.device)
        autils.hard_update(self.policy.critic, self.target_critic)

    def _get_optimizers(self):
        opts = super()._get_optimizers()
        opts["alpha_opt"] = (
            optim.Adam(
                [self.log_alpha], lr=self._arg("alpha_lr"), eps=self._arg("eps")
            ),
            lambda: self.policy.parameters(),
            self._arg("alpha_lr"),
        )
        opts["actor_opt"] = (
            optim.Adam(
                self.policy.get_actor_params(), lr=self.args.lr, eps=self.args.eps
            ),
            self.policy.get_actor_params,
            self.args.lr,
        )
        opts["critic_opt"] = (
            optim.Adam(
                self.policy.get_critic_params(),
                lr=self.args.critic_lr,
                eps=self.args.eps,
            ),
            self.policy.get_critic_params,
            self.args.critic_lr,
        )
        return opts

    def update_critic(self, state, n_state, action, reward, not_done):

        dist = self.policy(n_state, None, None, None)
        # not_done = n_add_info["masks"]
        n_action = dist.rsample()
        log_prob = dist.log_prob(n_action).sum(-1, keepdim=True)

        target_Q1, target_Q2 = self.target_critic(n_state, n_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.args.gamma * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.policy.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Optimize the critic
        self._standard_step(critic_loss, "critic_opt")
        return {
            "critic_loss": critic_loss.item(),
            "Q1": current_Q1.clone().detach().cpu().view(-1),
            "Q2": current_Q2.clone().detach().cpu().view(-1),
        }

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update_actor_and_alpha(self, state):
        dist = self.policy(state, None, None, None)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.policy.critic(state, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self._standard_step(actor_loss, "actor_opt")

        log_vals = {}

        if self.args.learnable_temp:
            alpha_loss = (
                self.alpha * (-log_prob - self.target_entropy).detach()
            ).mean()
            self._standard_step(alpha_loss, "alpha_opt")
            log_vals["alpha_loss"] = alpha_loss.item()
            log_vals["alpha_value"] = self.alpha.item()

        return {
            **log_vals,
            "actor_loss": actor_loss.item(),
            "actor_target_entropy": self.target_entropy,
            "actor_entropy": -log_prob.detach().clone().mean(),
        }

    def update(self, storage, args, beginning=False, t=0):
        super().update(storage)
        if len(storage) < self.args.batch_size:
            return {}
        if self.update_i <= self.args.n_rnd_steps:
        #if self.update_i <= 100:
            return {}

        state, n_state, action, reward, add_info, n_add_info = self._sample_transitions(
            storage
        )
        not_done = n_add_info["mask"]

        all_log = {}

        critic_log = self.update_critic(state, n_state, action, reward, not_done)
        all_log.update(critic_log)

        if self.update_i % self.args.actor_update_freq == 0:
            actor_log = self.update_actor_and_alpha(state)
            all_log.update(actor_log)

        if self.update_i % self.args.critic_target_update_freq == 0:
            autils.soft_update(self.policy.critic, self.target_critic, self.args.tau)

        return all_log

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument("--actor-update-freq", type=int, default=1)
        parser.add_argument("--critic-target-update-freq", type=int, default=2)

        parser.add_argument("--critic-lr", type=float, default=1e-4)
        parser.add_argument("--alpha-lr", type=float, default=1e-4)

        parser.add_argument("--init-temperature", type=float, default=0.1)
        parser.add_argument("--learnable-temp", type=str2bool, default=True)

        # Override
        parser.add_argument("--tau", type=float, default=0.005)
