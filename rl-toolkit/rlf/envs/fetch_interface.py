from rlf.envs.env_interface import EnvInterface, register_env_interface
from rlf.args import str2bool
import numpy as np
import rlf.rl.utils as rutils
import gym


class GymFetchInterface(EnvInterface):
    def get_add_args(self, parser):
        parser.add_argument('--gf-dense', type=str2bool, default=True)

    def create_from_id(self, env_id):
        if self.args.gf_dense:
            reward_type = 'dense'
        else:
            reward_type = 'sparse'
        env = gym.make(env_id, reward_type=reward_type)
        return env


FETCH_REGISTER_STR = "^(FetchPickAndPlace|FetchPush|FetchReach|FetchSlide)"
register_env_interface(FETCH_REGISTER_STR, GymFetchInterface)
