from rlf.envs.env_interface import EnvInterface, register_env_interface
from gym import core, spaces
try:
    from dm_control import suite
    from dm_env import specs
except ImportError:
    pass
from gym.utils import seeding
import gym
import numpy as np
import sys
import matplotlib.pyplot as plt
import pyglet

# From
# https://github.com/martinseilair/dm_control2gym/blob/master/dm_control2gym/viewer.py
class DmControlViewer:
    def __init__(self, width, height, depth=False):
        self.window = pyglet.window.Window(width=width, height=height, display=None)
        self.width = width
        self.height = height

        self.depth = depth

        if depth:
            self.format = 'RGB'
            self.pitch = self.width * -3
        else:
            self.format = 'RGB'
            self.pitch = self.width * -3

    def update(self, pixel):
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        if self.depth:
            pixel = np.dstack([pixel.astype(np.uint8)] * 3)
        pyglet.image.ImageData(self.width, self.height, self.format, pixel.tobytes(), pitch=self.pitch).blit(0, 0)
        self.window.flip()

    def close(self):
        self.window.close()




# From
# https://github.com/martinseilair/dm_control2gym/blob/master/dm_control2gym/wrapper.py
class DmcDiscrete(gym.spaces.Discrete):
    def __init__(self, _minimum, _maximum):
        super().__init__(_maximum - _minimum)
        self.offset = _minimum

def convertSpec2Space(spec, clip_inf=False):
    if spec.dtype == np.int:
        # Discrete
        return DmcDiscrete(spec.minimum, spec.maximum)
    else:
        # Box
        if type(spec) is specs.Array:
            return spaces.Box(-np.inf, np.inf, shape=spec.shape)
        elif type(spec) is specs.BoundedArray:
            _min = spec.minimum
            _max = spec.maximum
            if clip_inf:
                _min = np.clip(spec.minimum, -sys.float_info.max, sys.float_info.max)
                _max = np.clip(spec.maximum, -sys.float_info.max, sys.float_info.max)

            if np.isscalar(_min) and np.isscalar(_max):
                # same min and max for every element
                return spaces.Box(_min, _max, shape=spec.shape)
            else:
                # different min and max for every element
                return spaces.Box(_min + np.zeros(spec.shape),
                                  _max + np.zeros(spec.shape))
        else:
            raise ValueError('Unknown spec!')

def convertOrderedDict2Space(odict):
    if len(odict.keys()) == 1:
        # no concatenation
        return convertSpec2Space(list(odict.values())[0])
    else:
        # concatentation
        numdim = sum([np.int(np.prod(odict[key].shape)) for key in odict])
        return spaces.Box(-np.inf, np.inf, shape=(numdim,))


def convertObservation(spec_obs):
    if len(spec_obs.keys()) == 1:
        # no concatenation
        return list(spec_obs.values())[0]
    else:
        # concatentation
        numdim = sum([np.int(np.prod(spec_obs[key].shape)) for key in spec_obs])
        space_obs = np.zeros((numdim,))
        i = 0
        for key in spec_obs:
            space_obs[i:i+np.prod(spec_obs[key].shape)] = spec_obs[key].ravel()
            i += np.prod(spec_obs[key].shape)
        return space_obs

class DmControlWrapper(core.Env):
    def __init__(self, env, step_limit):
        self.dmcenv = env

        # convert spec to space
        self.action_space = convertSpec2Space(self.dmcenv.action_spec(), clip_inf=True)
        self.observation_space = convertOrderedDict2Space(self.dmcenv.observation_spec())

        self.render_mode_list = {}
        self.create_render_mode('rgb_array', show=False, return_pixel=True)

        self.metadata['render.modes'] = list(self.render_mode_list.keys())
        self.viewer = {key:None for key in self.render_mode_list.keys()}
        self.step_count = 0
        self.step_limit = step_limit

        # set seed
        self.seed()

    def getObservation(self):
        return convertObservation(self.timestep.observation)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.step_count = 0
        self.timestep = self.dmcenv.reset()
        return self.getObservation()

    def step(self, a):
        self.step_count += 1
        done = False
        if self.step_limit is not None and self.step_count > self.step_limit:
            done = True
        if type(self.action_space) == DmcDiscrete:
            a += self.action_space.offset
        self.timestep = self.dmcenv.step(a)

        return self.getObservation(), self.timestep.reward, (self.timestep.last() or done), {}


    def create_render_mode(self, name, show=True, return_pixel=False, height=240, width=320, camera_id=-1, overlays=(), depth=False, scene_option=None):
        render_kwargs = { 'height': height, 'width': width, 'camera_id': camera_id,
                                  'overlays': overlays, 'depth': depth, 'scene_option': scene_option}
        self.render_mode_list[name] = {'show': show, 'return_pixel': return_pixel, 'render_kwargs': render_kwargs}


    def render(self, mode='rgb_array', close=False):
        self.pixels = self.dmcenv.physics.render(**self.render_mode_list[mode]['render_kwargs'])
        if close:
            if self.viewer[mode] is not None:
                self._get_viewer(mode).close()
                self.viewer[mode] = None
            return
        elif self.render_mode_list[mode]['show']:
            self._get_viewer(mode).update(self.pixels)

        if self.render_mode_list[mode]['return_pixel']:
            return self.pixels

    def _get_viewer(self, mode):
        if self.viewer[mode] is None:
            self.viewer[mode] = DmControlViewer(self.pixels.shape[1], self.pixels.shape[0], self.render_mode_list[mode]['render_kwargs']['depth'])
        return self.viewer[mode]


class DmControlInterface(EnvInterface):
    def env_trans_fn(self, env, set_eval):
        return DmControlWrapper(env, self.args.time_limit)

    def create_from_id(self, env_id):
        # Must be in the format dm.domain.task
        _, domain, task = env_id.split('.')
        try:
            task_kwargs = None
            if self.args.time_limit is not None:
                task_kwargs = {'time_limit': self.args.time_limit}
            env = suite.load(domain_name=domain, task_name=task,
                    task_kwargs=task_kwargs)
            return env
        except NameError as e:
            print('DeepMind Control Suite is not installed')
            raise e

register_env_interface("^dm\.", DmControlInterface)
