import torch
from rlf.algos.il.base_il import BaseILAlgo
from rlf.algos.il.base_irl import BaseIRLAlgo
from rlf.algos.off_policy.sac import SAC
from rlf.storage.transition_storage import TransitionStorage


class SqilTransitionStorage(TransitionStorage):
    def __init__(self, obs_space, action_space, capacity, args, il_algo):
        if args.traj_batch_size != args.batch_size:
            raise ValueError(
                """
                    Must sample an equal amount of expert and agent experience.
                    """
            )
        self.il_algo = il_algo
        self.expert_batch_iter = None
        super().__init__(obs_space, action_space, capacity, args)

    def get_next_expert_batch(self):
        batch = None
        if self.expert_batch_iter is not None:
            try:
                batch = next(self.expert_batch_iter)
            except StopIteration:
                pass

        if batch is None:
            self.expert_batch_iter = iter(self.il_algo.expert_train_loader)
            batch = next(self.expert_batch_iter)
        return batch

    def sample_tensors(self, sample_size):
        (
            states,
            next_states,
            actions,
            rewards,
            cur_add,
            next_add,
        ) = super().sample_tensors(sample_size)
        expert_sample = self.get_next_expert_batch()

        expert_states = self._norm_expert_state(expert_sample["state"])
        expert_next_states = self._norm_expert_state(expert_sample["next_state"])
        expert_actions = self.il_algo._adjust_action(expert_sample["actions"])

        next_add["masks"] = torch.cat(
            [
                next_add["masks"].to(self.args.device),
                expert_sample["done"].unsqueeze(-1),
            ],
            dim=0,
        )
        states = torch.cat([states, expert_states], dim=0)
        next_states = torch.cat([next_states, expert_next_states], dim=0)
        rewards = torch.cat(
            [
                torch.zeros(rewards.shape).to(rewards.device),
                torch.ones(rewards.shape).to(rewards.device),
            ],
            dim=0,
        )
        actions = torch.cat([actions, expert_actions], dim=0)

        return states, next_states, actions, rewards, cur_add, next_add

    def _norm_expert_state(self, state):
        """
        Applies normalization to the expert state, if we are using
        normalization.
        """
        obsfilt = self.il_algo.get_env_ob_filt()
        if obsfilt is None:
            return state
        state = state.cpu().numpy()
        state = obsfilt(state, update=False)
        state = torch.tensor(state).to(self.args.device)
        return state


class SQIL(SAC):
    def __init__(self):
        self.il_algo = BaseILAlgo()
        super().__init__()

    def get_storage_buffer(self, policy, envs, args):
        return SqilTransitionStorage(
            policy.obs_space,
            policy.action_space,
            args.trans_buffer_size,
            args,
            self.il_algo,
        )

    def init(self, policy, args):
        super().init(policy, args)
        self.il_algo.init(policy, args)

    def set_env_ref(self, envs):
        super().set_env_ref(envs)
        self.il_algo.set_env_ref(envs)

    def get_add_args(self, parser):
        super().get_add_args(parser)
        self.il_algo.get_add_args(parser)
