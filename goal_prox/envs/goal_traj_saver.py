from rlf.il import TrajSaver

class GoalTrajSaver(TrajSaver):
    def __init__(self, save_dir, assert_saved):
        self.assert_saved = assert_saved
        super().__init__(save_dir)

    def should_save_traj(self, traj):
        last_info = traj[-1][-1]
        if 'ep_found_goal' in last_info:
            ret = last_info['ep_found_goal'] == 1.0
        else:
            ret = (last_info['episode']['r'] > 5000 and last_info['episode']['l'] == 1000)
            print(f"========== {last_info['episode']['r']} =========")
        if self.assert_saved and not ret:
            raise ValueError('Trajectory did not end successfully')
        return ret

