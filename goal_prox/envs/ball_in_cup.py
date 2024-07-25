import sys
sys.path.insert(0, './')
from rlf.envs.dm_control_interface import DmControlInterface
from rlf.envs.env_interface import register_env_interface
from gym import core

class GoalCheckerWrapper(core.Wrapper):
    def reset(self):
        self.found_goal = False
        return super().reset()

    def step(self, a):
        obs, reward, done, info = super().step(a)
        if reward == 1.0:
            self.found_goal = True
            done = True
        info['ep_found_goal'] = float(self.found_goal)
        return obs, reward, done, info

class BallInCupInterface(DmControlInterface):
    def env_trans_fn(self, env, set_eval):
        env = super().env_trans_fn(env, set_eval)
        return GoalCheckerWrapper(env)

    def get_special_stat_names(self):
        return ['ep_found_goal']


register_env_interface("^dm.ball_in_cup.catch$", BallInCupInterface)
