from rlf.algos.on_policy.on_policy_base import OnPolicy


class A2C(OnPolicy):
    def update(self, rollouts):
        self._compute_returns(storage)
        sample = rollouts.get_rollout_data()

        ac_eval = self.policy.evaluate_actions(sample['state'],
                sample['hxs'], sample['mask'],
                sample['action'])

        advantages = (sample['return'] - ac_eval['value']).detach()

        action_loss = -(advantages * ac_eval['log_prob']).mean()
        value_loss = (sample['return'] - ac_eval['value']).pow(2).mean()

        loss = (action_loss + self.args.value_loss_coef * value_loss)
        self._standard_step(loss)

        return {
                'loss': loss.item()
                }


    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--value-loss-coef',
            type=float,
            default=0.5,
            help='value loss coefficient (default: 0.5)')
