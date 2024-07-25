from rlf.algos.off_policy.off_policy_base import OffPolicy
import torch.optim as optim


class ActorCriticUpdater(OffPolicy):
    def init(self, policy, args):
        super().init(policy, args)
        self.target_policy = self._copy_policy()

    def _get_optimizers(self):
        return {
                'actor_opt': (
                    optim.Adam(self.policy.get_actor_params(), lr=self.args.lr,
                        eps=self.args.eps),
                    self.policy.get_actor_params,
                    self.args.lr
                ),
                'critic_opt': (
                    optim.Adam(self.policy.get_critic_params(), lr=self.args.critic_lr,
                        eps=self.args.eps),
                    self.policy.get_critic_params,
                    self.args.critic_lr
                )
            }

    def get_add_args(self, parser):
        super().get_add_args(parser)

        parser.add_argument('--tau',
            type=float,
            default=1e-3,
            help='Mixture for the target network weight update')

        parser.add_argument('--critic-lr',
            type=float,
            default=1e-3,
            help='')
