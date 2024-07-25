from rlf.policies.base_net_policy import BaseNetPolicy
from rlf.policies.base_policy import create_simple_action_data
from rlf.il.traj_dataset import TrajDataset
import torch
import gym

def get_env():
    env = gym.make('FetchPickAndPlaceCustom-v0')
    env.env.obj_range = 0.0
    return env

class ActionReplayPolicy(BaseNetPolicy):
    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)
        self.traj_dataset = TrajDataset(args.traj_load_path)
        # Indexed by if we are in evaluation mode or not.
        self.cur_ep = {
                True: -1,
                False: 0,
                }
        self.cur_idx = {
                True: -1,
                False: 0,
                }

    def get_action(self, state, hxs, mask, step_info):
        is_eval = step_info.is_eval

        if mask[0][0] == 0:
            self.cur_ep[is_eval] += 1
            self.cur_idx[is_eval] = 0

        ep_idx = self.cur_ep[is_eval]
        idx = self.cur_idx[is_eval]
        cur_traj = self.traj_dataset[ep_idx]
        cur_action = cur_traj[1][idx]
        cur_state = cur_traj[0][idx].view(1, -1).repeat(state.shape[0], 1)\
                .to(self.args.device)
        cur_action = cur_action.view(1, -1).repeat(state.shape[0], 1)

        if self.args.ensure_state_match and not torch.allclose(cur_state, state):
            print(state)
            print(cur_state)
            import ipdb; ipdb.set_trace()

        self.cur_idx[is_eval] += 1
        if self.cur_idx[is_eval] >= len(cur_traj[1]):
            self.cur_idx[is_eval] = 0
            self.cur_ep[is_eval] += 1

        return create_simple_action_data(cur_action)

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--traj-load-path', type=str, required=True)
        parser.add_argument('--ensure-state-match', action='store_true')
