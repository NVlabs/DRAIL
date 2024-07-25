from rlf.algos.base_net_algo import BaseNetAlgo
from rlf.storage.transition_storage import TransitionStorage
from rlf.storage.rollout_storage import RolloutStorage
from rlf.storage.nested_storage import NestedStorage
import rlf.algos.utils as autils

class OptionCritic(BaseNetAlgo):
    def init(self, policy, args):
        super().init(policy, args)
        self.target_policy = self._copy_policy()

    def get_storage_buffer(self, policy, envs, args):
        dims = {
                'option': 1,
                'term': 1
                }
        return NestedStorage(
                {
                    'replay_buffer': TransitionStorage(args.trans_buffer_size, args,
                        hidden_states=dims),
                    'on_policy': RolloutStorage(args.num_steps,
                        args.num_processes, envs.observation_space,
                        envs.action_space, args, hidden_states=dims)
                    },
                'on_policy')

    def update_actor(self, storage):
        exp = storage.child_dict['on_policy']
        rollout_size = (exp.obs.shape[0]-1) * exp.obs.shape[1]

        # Get the data from the rollout buffer.
        state = exp.obs[:-1].view(rollout_size, *exp.obs.shape[2:])
        n_state = exp.obs[1:].view(rollout_size, *exp.obs.shape[2:])
        sel_option = exp.hidden_states['option'][1:].view(-1, 1).long()
        rewards = exp.rewards.view(-1,1)
        hxs = exp.hidden_states
        masks = exp.masks[1:].view(-1, 1)

        _, log_probs, ent = self.policy.get_actions(state, hxs, masks, sel_option)

        term_prob = self.policy.get_term_prob(state, hxs, masks).gather(1, sel_option)
        n_term_prob = self.policy.get_term_prob(n_state, hxs, masks).gather(1, sel_option)

        Q = self.policy.get_value(state, hxs, masks).detach()
        n_Q = self.target_policy.get_value(state, hxs, masks).detach()

        # From Theorem 2. Calculate the termination gradient
        term_loss = term_prob * \
                (Q.gather(1, sel_option) - Q.max(dim=-1)[0].unsqueeze(1) + self.args.term_reg) * \
                masks
        term_loss = term_loss.mean()

        # From Theorem 1. Calculate the policy gradient.
        gt = rewards + masks * self.args.gamma * \
                ((1 - n_term_prob) * n_Q.gather(1, sel_option) + \
                term_prob * n_Q.max(dim=-1)[0].unsqueeze(1))
        gt = gt.detach()
        policy_loss = -log_probs * (gt - Q.gather(1, sel_option)) - \
                self.args.entropy_reg * ent.unsqueeze(1)
        policy_loss = policy_loss.mean()

        loss = term_loss + policy_loss
        self._standard_step(loss)
        return {
                'policy_loss': policy_loss.item(),
                'term_loss': term_loss.item(),
                }

    def update_critic(self, storage):
        if len(storage.child_dict['replay_buffer']) < self.args.batch_size:
            return {}
        state, n_state, _, rewards, _, n_add = storage.child_dict['replay_buffer'].sample_tensors(self.args.batch_size)
        hxs = n_add['hxs']
        sel_option = hxs['option']
        masks = n_add['masks']

        term_prob = self.policy.get_term_prob(state, hxs, masks).gather(1, sel_option)
        n_term_prob = self.policy.get_term_prob(n_state, hxs, masks).gather(1, sel_option)

        Q = self.policy.get_value(state, hxs, masks)
        n_Q = self.target_policy.get_value(state, hxs, masks)

        gt = rewards + masks * self.args.gamma * \
                ((1 - n_term_prob) * n_Q.gather(1, sel_option) + \
                term_prob * n_Q.max(dim=-1)[0].unsqueeze(1))
        gt = gt.detach()
        critic_loss = (Q[sel_option] - gt.detach()).pow(2).mul(0.5).mean()
        self._standard_step(critic_loss)
        return {
                'critic_loss': critic_loss.item()
                }

    def update(self, storage):
        actor_log_vals = self.update_actor(storage)
        critic_log_vals = self.update_critic(storage)
        autils.soft_update(self.policy, self.target_policy, self.args.tau)
        return {**actor_log_vals, **critic_log_vals}

    def get_add_args(self, parser):
        super().get_add_args(parser)
        #########################################
        # Overrides

        #########################################
        # New args
        parser.add_argument('--trans-buffer-size', type=int, default=10000)
        parser.add_argument('--batch-size', type=int, default=32)
        parser.add_argument('--term-reg', type=float, default=0.01,
                help=("Bias added to the advantage calculation to discourage",
                    "option termination"))
        parser.add_argument('--entropy-reg', type=float, default=0.01)
        parser.add_argument('--tau', type=float, default=0.05)
