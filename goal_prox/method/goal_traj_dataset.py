from rlf.il import TrajDataset


class GoalTrajDataset(TrajDataset):
    """
    The dataset does not necessarily need to have the `ep_found_goal` key. The
    goal proximities will be calculated fine based on the length of the
    episode. An `ep_found_goal` key is if the goal proximity should be
    calculated from an earlier state and the trajectory is trimmed once the
    expert reaches the goal.
    """
    def _setup(self, trajs):
        if 'ep_found_goal' in trajs:
            self.found_goal = trajs['ep_found_goal'].float()
        else:
            self.found_goal = None

    def should_terminate_traj(self, j, obs, next_obs, done, actions):
        if self.found_goal is None:
            return done[j]
        else:
            return self.found_goal[j] == 1.0
