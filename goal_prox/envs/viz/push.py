import os
from gym import utils
#from gym.envs.robotics import fetch_env
from goal_prox.envs.viz.base_fetch import FetchEnv


# Ensure we get the path separator correct on windows
# MODEL_XML_PATH = 'push.xml'
MODEL_XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'viz_push.xml')
# MODEL_XML_PATH = '/home/mecoli/adm/goal_prox_il/goal_prox/envs/viz/viz_push.xml'


class VizFetchPushEnv(FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False,
            target_offset=[0.0,0,-0.3],
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        self.max_episode_steps = 60


    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        #lookat = self.sim.data.body_xpos[body_id]
        lookat = [1.34193362, 0.74910034, 0.6]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 0.8
        #self.viewer.cam.azimuth = 132.
        self.viewer.cam.azimuth = 165
        self.viewer.cam.elevation = -2
