import os
from gym import utils
from gym.envs.robotics import fetch_env


# Ensure we get the path separator correct on windows
# MODEL_XML_PATH = 'pick_and_place.xml'
MODEL_XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'viz_pick_and_place.xml')
# MODEL_XML_PATH = '/home/mecoli/adm/goal_prox/goal_prox/envs/viz/viz_pick_and_place.xml'

class VizFetchPickAndPlaceEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        # To manually control things for rendering.
        #fetch_env.FetchEnv.__init__(
        #    self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
        #    gripper_extra_height=0.2, target_in_the_air=True,
        #    # You should change the secon coordinate
        #    target_offset=[0.1, -0.1, 0.1],
        #    obj_range=0.15, target_range=0.3, distance_threshold=0.05,
        #    initial_qpos=initial_qpos, reward_type=reward_type)

        # For actual policy eval / training.
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        self.max_episode_steps = 100

    def set_noise_ratio(self, x,y):
        pass

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
