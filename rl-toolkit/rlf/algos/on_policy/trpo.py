import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
import rlf.policies.utils as putils
from rlf.algos.on_policy.on_policy_base import OnPolicy
from rlf.algos.base_policy import td_loss

##########################################
# A WORK IN PROGRESS. THIS IS NOT FINISHED
##########################################

class TRPO(OnPolicy):

    def update_critic(self, states, hxs, masks, returns):
        critic_params = self.policy.get_critic_params()

        # First fit the value function.
        def get_value_loss(flat_params):
            putils.set_flat_params_to(critic_params, flat_params)
            self.policy.zero_grad()

            pred = self.policy.get_value(states, hxs, masks)

            value_loss = F.mse_loss(pred, returns)
            for param in critic_params:
                value_loss += param.pow(2).sum() * self.args.w_l2_reg

            value_loss.backward()
            grads = torch.cat([param.grad.view(-1) if param.grad is None \
                    else torch.zeros(np.prod(param.view(-1).shape))
                    for param in critic_params])
            return value_loss.item(), grads

        flattened_params = torch.cat([param.view(-1) for param in critic_params])
        final_flat_params, _, _ = scipy.optimizer.fmin_l_bfgs_b(get_value_loss,
                flattened_params.detach().cpu().numpy(),
                maxiter=25)

        putils.set_flat_params_to(critic_params, final_flat_params)


    def update_actor(self, sample, advantages):
        ac_eval = self.policy.evaluate_actions(sample['state'],
                        sample['hxs'], sample['mask'],
                        sample['action'])
        action_loss = -advantages * torch.exp(ac_eval['log_prob'] - sample['prev_log_prob'])
        action_loss = action_loss.mean()

        actor_params = self.policy.get_actor_params()

        grads = torch.autograd.grad(action_loss, actor_params)
        flat_grads = torch.cat([grad.view(1) for grad in grads]).detach()


    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--w-l2-reg',
            type=float,
            default=0.2,
            help='L2 weight regularization')

