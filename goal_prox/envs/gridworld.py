from rlf.envs.env_interface import EnvInterface, register_env_interface
import gym
from goal_prox.gym_minigrid.wrappers import *
from goal_prox.envs.gw_helper import *
from rlf.args import str2bool


class SpecGoalCheckerWrapper(GoalCheckerWrapper):
    def __init__(self, env, args):
        super().__init__(env, args)
        prev_obs_space = self.observation_space
        self.observation_space = gym.spaces.Dict({
            "observation": prev_obs_space,
            "goal": prev_obs_space})

    def reset(self):
        self.set_cond[self.args.env_name](self.env, self.cache_vals, self.args)
        self.found_goal = False

        goal_pos = self.env.env._goal_default_pos
        tmp_env = get_env_for_pos(goal_pos, goal_pos, self.args)
        # Check that goal obs is actually the correct observation.
        self.goal_obs = get_grid_obs_for_env(tmp_env)

        return self._get_obs(self.env.reset())

    def _get_obs(self, obs_dict):
        obs = super()._get_obs(obs_dict)
        return {
                "observation": obs,
                "goal": self.goal_obs,
                }

class GoalGridWorldInterface(EnvInterface):
    def env_trans_fn(self, env, set_eval):
        if self.args.gw_spec_goal:
            return SpecGoalCheckerWrapper(FullyObsWrapper(env), self.args)
        else:
            return GoalCheckerWrapper(FullyObsWrapper(env), self.args)

    def get_add_args(self, parser):
        parser.add_argument('--gw-rand-pos', type=str2bool,
                default=True)
        parser.add_argument('--gw-img', action='store_true',
                default=True,
                help=(
                    "If true the observations will be an image. If false they",
                    " will be flattened."
                    ))

        parser.add_argument('--gw-agent-quad', type=str,
                help=(
                    'Comma separated series of quadrants indices where agent spawning ',
                    'is allowed'
                    ),
                default=None)
        parser.add_argument('--gw-goal-quad', type=str,
                help=(
                    'Comma separated series of quadrants indices where goal spawning ',
                    'is allowed'
                    ),
                default=None)

        parser.add_argument('--gw-goal-pos', type=str, default=None)

        parser.add_argument('--gw-compl', action='store_true',
                default=False)
        parser.add_argument('--gw-cover', type=float,
                default=1.0)
        parser.add_argument('--gw-spec-goal', action='store_true',
                default=False)


register_env_interface("^MiniGrid", GoalGridWorldInterface)
