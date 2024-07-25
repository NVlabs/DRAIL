from rlf.envs.env_interface import EnvInterface, register_env_interface
import gym
import numpy as np
from rlf.args import str2bool
import rlf.rl.utils as rutils

try:
    import gym_minigrid
    from gym_minigrid.wrappers import *
    from gym_minigrid.minigrid import Goal
except:
    pass


class DirectionObsWrapper(gym.core.ObservationWrapper):
    """
    Provides the slope/angular direction to the goal with the observations as modeled by (y2 - y2 )/( x2 - x1)
    type = {slope , angle}
    """
    def __init__(self, env,type='slope'):
        super().__init__(env)
        self.goal_position = None
        self.type = type
        self.observation_space = rutils.reshape_obs_space(self.observation_space,
                (self.observation_space.shape[0]+1,))

    def reset(self):
        if not self.goal_position:
            self.goal_position = [x for x,y in enumerate(self.grid.grid) if isinstance(y,(Goal) ) ]
            if len(self.goal_position) >= 1: # in case there are multiple goals , needs to be handled for other env types
                self.goal_position = (int(self.goal_position[0]/self.height) , self.goal_position[0]%self.width)
        return super().reset()

    def observation(self, obs):
        use = np.linalg.norm(self.goal_position - self.agent_pos)
        return np.append(obs, use)


class MiniGridWrapper(gym.Wrapper):
    def __init__(self, env, use_cardinal_directions):
        super().__init__(env)
        self.card_dirs = use_cardinal_directions
        if use_cardinal_directions:
            # Transform the action space to the cardinal directions
            self.action_space = gym.spaces.Discrete(4)

    def step(self, a):
        use_action = a
        if self.card_dirs:
            desired_dir = a
            try:
                self.env.env.agent_dir = desired_dir
            except:
                self.env.agent_dir = desired_dir
            use_action = self.env.actions.forward
        obs_dict, reward, done, info = self.env.step(use_action)

        obs = self._get_obs(obs_dict)

        if done and reward > 0.0:
            self.found_goal = True
        info['ep_found_goal'] = float(self.found_goal)
        return obs, reward, done, info

    def reset(self):
        self.found_goal = False
        return self._get_obs(self.env.reset())

    def _get_obs(self, x):
        raise ValueError()


class FlatGrid(MiniGridWrapper):
    """
    Flattens the 7x7x3 region around the agent.
    """
    def _get_obs(self, x):
        return x['image'].reshape(-1).astype(np.float32)

    def __init__(self, env, card_dirs):
        super().__init__(env, card_dirs)
        ob_space = self.observation_space['image']
        self.observation_space = gym.spaces.Box(
                shape=(np.prod(ob_space.shape),),
                low=ob_space.low.reshape(-1)[0],
                high=ob_space.high.reshape(-1)[0],
                dtype=np.float32)


NODE_TO_ONE_HOT = {
    # Empty square
    (1, 0, 0): [1, 0, 0, 0],
    # Wall
    (2, 5, 0): [0, 1, 0, 0],
    # Goal
    (8, 1, 0): [0, 0, 1, 0],
    # Agent
    (10, 0, 0): [0, 0, 0, 1],
    (10, 0, 1): [0, 0, 0, 1],
    (10, 0, 2): [0, 0, 0, 1],
    (10, 0, 3): [0, 0, 0, 1],
}

class FullFlatGrid(MiniGridWrapper):
    def __init__(self, env, gw_card):
        super().__init__(env, gw_card)
        ob_s = env.observation_space.spaces['image'].shape

        ob_shape = (ob_s[0], ob_s[1], 4)
        low = 0.0
        high = 1.0
        self.observation_space = gym.spaces.Box(shape=ob_shape,
                                                low=np.float32(low),
                                                high=np.float32(high),
                                                dtype=np.float32)

    def _get_obs(self, obs_dict):
        obs = obs_dict['image']

        obs = obs.reshape(-1, 3)
        obs = np.array(list(map(lambda x: NODE_TO_ONE_HOT[tuple(x)], obs)))
        obs = obs.reshape(*rutils.get_obs_shape(self.observation_space))
        return obs



class MinigridInterface(EnvInterface):
    def create_from_id(self, env_id):
        env = gym.make(env_id)
        if self.args.gw_mode == 'flat':
            env = FlatGrid(env, self.args.gw_card_dirs)
        elif self.args.gw_mode == 'img':
            env = FullFlatGrid(FullyObsWrapper(env), self.args.gw_card_dirs)
        else:
            raise ValueError()

        if self.args.gw_goal_info:
            env = DirectionObsWrapper(env)
        return env

    def get_add_args(self, parser):
        parser.add_argument('--gw-mode', type=str, default='flat', help="""
                Options are: [flat,img]
        """)
        parser.add_argument('--gw-card-dirs', action='store_true',
                default=False)
        parser.add_argument('--gw-goal-info', action='store_true',
                default=False)

register_env_interface("^MiniGrid", MinigridInterface)
