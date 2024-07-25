import torch
import torch.nn as nn
from rlf.algos.on_policy.on_policy_base import OnPolicy
import rlf.algos.utils as autils


def get_sarsa_rollout_data(rollouts):
    ### NOTE
    # This is completely incorrect. The data is not sequential, it's randomly
    # sampled when calling get_rollout_data. So cur_state and next_state are
    # not right!
    ret = rollouts.get_rollout_data()
    cur_state = ret['state'][:-1]
    next_state = ret['state'][1:]
    cur_action = ret['action'][:-1]
    next_action = ret['action'][1:]
    reward = ret['reward'][:-1]
    n_mask = ret['mask'][:-1]
    return cur_state, cur_action, reward, next_state, next_action, n_mask

class SARSA(OnPolicy):
    """
    Q(s_t, a_t) target:
    r_t + \gamma * Q(s_{t+1}, a_{t+1})
    """
    def update(self, rollouts):
        raise ValueError('Algorithm is incorrect. Fix it')
        ### NOTE
        # This should be changed to just predict the next action rather than
        # saving it from the rollout, this would enable you to set num_steps to
        # 1 and simply things.

        # n_ stands for "next"
        state, action, reward, n_state, n_action, n_mask = get_sarsa_rollout_data(rollouts)

        next_q_vals = self.policy(n_state).gather(1, n_action)
        next_q_vals *= n_mask

        target = reward + self.args.gamma * next_q_vals
        target = target.detach()

        loss = autils.td_loss(target, self.policy, state, action)
        self._standard_step(loss)

        return {
                'loss': loss.item()
                }

    def get_add_args(self, parser):
        super().get_add_args(parser)
        #########################################
        # Overrides
        parser.add_argument('--num-processes', type=int, default=1)
