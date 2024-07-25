from rlf.storage.transition_storage import TransitionStorage
import torch
import rlf.rl.utils as rutils
import numpy as np

def create_her_storage_buff(obs_space, action_space, buff_size, args):
    return HerStorage(obs_space, action_space, buff_size, args)

class HerStorage(TransitionStorage):
    """
    Uses the "final" HER strategy which uses the state achieved at the end of
    the trajectory.
    Observation should have format:
        {
        "achieved_goal": tensor
        "desired_goal": tensor
        "observation": tensor
        }
    Arguments are in `OffPolicy`
    """

    def _on_traj_done(self, done_trajs):
        for done_traj in done_trajs:
            for t in range(len(done_traj) - 1):
                state = done_traj[t][0].copy()
                next_state = done_traj[t+1][0].copy()

                if t == 0:
                    mask = 1.0
                else:
                    mask = done_traj[t-1][2]

                def push_trans(state, next_state):
                    if torch.allclose(next_state['desired_goal'],
                            next_state['achieved_goal'], 0.0001):
                        reward = 1.0
                        next_mask = 0.0
                    else:
                        reward = done_traj[t][4]
                        next_mask = done_traj[t][2]

                    self._push_transition({
                        'action': done_traj[t][1],
                        'state': state,
                        'mask': torch.tensor([mask]),
                        'hxs': {},
                        'reward': torch.tensor([reward]),
                        'next_state': next_state,
                        'next_mask': torch.tensor([next_mask]),
                        'next_hxs': {},
                        })

                # Augment with the HER style goal.
                if self.args.her_strat == 'future':
                    for k in range(self.args.her_K):
                        # Randomly choose a time step in the future.
                        future_t = np.random.randint(t, len(done_traj) - 1)
                        future_goal = done_traj[future_t+1][0]['achieved_goal']

                        state['desired_goal'] = future_goal
                        next_state['desired_goal'] = future_goal
                        push_trans(state, next_state)
                elif self.args.her_strat == 'final':
                    final_goal = done_traj[-1][0]['achieved_goal']
                    state['desired_goal'] = final_goal
                    next_state['desired_goal'] = final_goal
                    push_trans(state, next_state)
                else:
                    raise ValueError(f"Invalid HER strategy {self.args.her_strat}")
