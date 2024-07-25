import gym
from rlf.envs.env_interface import EnvInterface, register_env_interface
BIT_FLIP_ID = 'BitFlip-v0'


class BlackJackWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(shape=(3,), high=100, low=0)

    def observation(self, observation):
        return [*observation[:2], int(observation[2])]


class BlackJackInterface(EnvInterface):
    def create_from_id(self, env_id):
        env = super().create_from_id(env_id)
        return BlackJackWrapper(env)

# Match any version
register_env_interface('Blackjack', BlackJackInterface)
