from rlf.policies.base_net_policy import BaseNetPolicy
import torch.nn as nn
import rlf.policies.utils as putils
import rlf.rl.utils as rutils
from rlf.rl.model import def_mlp_weight_init
from rlf.rl.model import weight_init


class ActorCritic(BaseNetPolicy):
    """
    Defines an actor and critic type policy
    """

    def __init__(self,
                 get_critic_fn=None,
                 get_critic_head_fn=None,
                 use_goal=False,
                 fuse_states=[],
                 get_base_net_fn=None):
        """
        - get_critic_fn: (obs_shape: (int), input_shape: (int),
          action_space: gym.spaces.space -> rlf.rl.model.BaseNet)
        - get_critic_head_fn: (hidden_dim: (int) -> nn.Module) should return a
          dimension 1, for the critic value.
        """

        super().__init__(use_goal, fuse_states, get_base_net_fn)

        if get_critic_fn is None:
            get_critic_fn = putils.get_def_critic
        if get_critic_head_fn is None:
            get_critic_head_fn = putils.get_def_critic_head

        self.get_critic_fn = get_critic_fn
        self.get_critic_head_fn = get_critic_head_fn

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)

        obs_shape = rutils.get_obs_shape(obs_space, args.policy_ob_key)

        self.critic = self.get_critic_fn(obs_shape, self._get_base_out_shape(),
                action_space)
        self.critic_head = self.get_critic_head_fn(self.critic.output_shape[0])

    def _get_value_from_features(self, base_features, hxs, masks):
        """
        - base_features: post fusion base features
        """
        critic_features, hxs = self.critic(base_features, hxs, masks)
        return self.critic_head(critic_features)

    def get_value(self, inputs, add_inputs, hxs, masks):
        base_features, hxs = self.base_net(inputs, hxs, masks)
        base_features = self._fuse_base_out(base_features, add_inputs)

        return self._get_value_from_features(base_features, hxs, masks)

    def get_critic_params(self):
        return list(self.base_net.parameters()) + \
                list(self.critic.parameters()) + \
                list(self.critic_head.parameters())

    def get_actor_params(self):
        return list(self.base_net.parameters()) + \
                list(self.actor_net.parameters())
