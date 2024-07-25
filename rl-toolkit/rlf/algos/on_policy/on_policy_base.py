from rlf.storage.rollout_storage import RolloutStorage
from rlf.algos.base_net_algo import BaseNetAlgo
import torch
import rlf.rl.utils as rutils

class OnPolicy(BaseNetAlgo):
    def get_storage_buffer(self, policy, envs, args):
        return RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space, envs.action_space, args,
                              hidden_states=policy.get_storage_hidden_states())

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument(f"--{self.arg_prefix}num-epochs",
                            type=int,
                            default=4,
                            help='number of ppo epochs (default: 4)')
        parser.add_argument(f"--{self.arg_prefix}num-mini-batch",
                            type=int,
                            default=4,
                            help='number of batches for ppo (default: 4)')

    def _get_next_value(self, rollouts):
        """
        Gets the value of the final observations. Needed if you need to
        estimate the returns of any partial trajectories.
        """
        with torch.no_grad():
            next_value = self.policy.get_value(
                rutils.get_def_obs(rollouts.get_obs(-1), self.args.policy_ob_key),
                rutils.get_other_obs(rollouts.get_obs(-1), self.args.policy_ob_key),
                rollouts.get_hidden_state(-1),
                rollouts.masks[-1]).detach()
        return next_value

    def _compute_returns(self, rollouts):
        next_value = self._get_next_value(rollouts)
        rollouts.compute_returns(next_value)


