import gym
from gym.spaces import Box, Discrete
from rlf.baselines.dict import Dict
from rlf.envs.env_interface import EnvInterface, register_env_interface
import numpy as np

BIT_FLIP_ID = 'BitFlip-v0'

class BitFlipEnv(gym.Env):
    """
    From https://gist.github.com/MishaLaskin/f2e76ba1d6171ecdecbe7f6a5431e0a8
    """

    def __init__(self, n=5, reward_type='sparse'):
        self.n = n # number of bits
        self.reward_type = reward_type
        self.observation_space = Dict({
            'observation': Box(shape=(n,), low=0, high=1, dtype=np.float32),
            'achieved_goal': Box(shape=(n,), low=0, high=1, dtype=np.float32),
            'desired_goal': Box(shape=(n,), low=0, high=1, dtype=np.float32),
            })
        self.action_space = Discrete(n)
        self.n_steps = 0

    def seed(self, sd):
        self.rng = np.random.RandomState(sd)

    def reset(self):
        self.goal = self.rng.randint(2, size=(self.n)) # a random sequence of 0's and 1's
        self.state = self.rng.randint(2, size=(self.n)) # another random sequence of 0's and 1's as initial state
        self.n_steps = 0
        return {
                'observation': np.copy(self.state),
                'achieved_goal': np.copy(self.state),
                'desired_goal': np.copy(self.goal),
                }

    def step(self, action):
        self.n_steps += 1
        self.state[action] = 1-self.state[action] # flip this bit
        done = np.array_equal(self.state, self.goal)
        if self.reward_type == 'sparse':
            reward = 0 if done else -1
        else:
            reward = -np.sum(np.square(self.state-self.goal))
        obs = {
                'observation': np.copy(self.state),
                'achieved_goal': np.copy(self.state),
                'desired_goal': np.copy(self.goal),
                }

        if self.n_steps >= self.n:
            done = True

        info = {}
        if done:
            info['ep_success'] = float(np.array_equal(self.state, self.goal))
        return obs, reward, done, info

    def render(self):
        print("\rstate :", np.array_str(self.state), end=' '*10)

class BitFlipInterface(EnvInterface):
    def get_add_args(self, parser):
        parser.add_argument('--bit-flip-n', type=int, default=5)
        parser.add_argument('--bit-flip-reward', type=str, default='sparse')

    def create_from_id(self, env_id):
        return BitFlipEnv(self.args.bit_flip_n, self.args.bit_flip_reward)

# Match any version
register_env_interface(BIT_FLIP_ID.split('-')[0], BitFlipInterface)
