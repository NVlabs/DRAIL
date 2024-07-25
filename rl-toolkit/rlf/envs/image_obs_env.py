import gym
import numpy as np
import rlf.rl.utils as rutils
import copy

class ImageObsWrapper(gym.Wrapper):
    """
    Returns the render of the environment as the observation. Still returns the
    state observation in the observation dictionary under "state".
    Note that the image is in [0,255]
    """
    def __init__(self, env, render_dim):
        super().__init__(env)
        self.render_dim = render_dim

        if rutils.is_dict_obs(env.observation_space):
            prev_spaces = copy.copy(env.observation_space.spaces)
            prev_spaces['state'] = prev_spaces.pop('observation')
            self.observation_space = gym.spaces.Dict({
                'observation': gym.spaces.Box(shape=(render_dim, render_dim,
                    3), low=0, high=255, dtype=np.uint8),
                **prev_spaces})
        else:
            self.observation_space = gym.spaces.Dict({
                'observation': gym.spaces.Box(shape=(render_dim, render_dim, 3),
                    low=0.0, high=255.0, dtype=np.uint8),
                'state': env.observation_space})

    def reset(self):
        state =  self.env.reset()
        return self._trans_obs(state)

    def step(self, a):
        state, reward, done, info = self.env.step(a)
        obs = self._trans_obs(state)
        return obs, reward, done, info

    def _trans_obs(self, state):
        img = self.env.render('rgb_array', width=self.render_dim,
                height=self.render_dim)

        state['state'] = state.pop('observation')

        return {
            'observation': img,
            **state
            }
