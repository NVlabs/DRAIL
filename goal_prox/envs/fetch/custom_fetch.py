from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from goal_prox.envs.viz.pick import VizFetchPickAndPlaceEnv
import numpy as np
from goal_prox.envs.holdout_sampler import HoldoutSampler
from goal_prox.envs.old_holdout_sampler import OldHoldoutSampler

class FetchPickAndPlaceCustom(FetchPickAndPlaceEnv):
    def __init__(self, reward_type='sparse'):
        super().__init__(reward_type)

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        object_xpos = self.initial_gripper_xpos[:2]
        object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        goal = self.initial_gripper_xpos[:3] + np.array([0.05, 0.05, 0])
        goal += self.target_offset
        goal[2] = self.height_offset
        goal[2] += 0.2
        return goal.copy()


import os
from gym import utils
MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')
from gym.envs.robotics import fetch_env

class FetchPickAndPlaceHarder(FetchPickAndPlaceEnv):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
                'robot0:slide0': 0.405,
                'robot0:slide1': 0.48,
                'robot0:slide2': 0.0,
                'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
                }
        fetch_env.FetchEnv.__init__(
                self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
                gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
                obj_range=0.15, target_range=0.15, distance_threshold=0.05,
                initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        obs = super()._get_obs()
        obs['observation'] = np.concatenate([obs['observation'], obs['desired_goal']])
        return obs



Y_NOISE = 0.02
X_NOISE = 0.05
# OBJ_X_NOISE = 0.02
OBJ_X_NOISE = 0.05
OFFSET = 0.1

class FetchPickAndPlaceDiff(FetchPickAndPlaceEnv):
    def __init__(self, reward_type='sparse'):
        super().__init__(reward_type)

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2] + np.array([0.0, OFFSET])
            object_xpos += self.np_random.uniform([-1*OBJ_X_NOISE, -1*Y_NOISE],
                    [OBJ_X_NOISE, Y_NOISE], size=2)

            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        goal = self.initial_gripper_xpos[:3] + np.array([0.0, -1*OFFSET, 0.0])
        goal[:2] += self.np_random.uniform([-1*X_NOISE, -1*Y_NOISE],
                [X_NOISE, Y_NOISE], size=2)

        goal += self.target_offset
        goal[2] = self.height_offset
        goal[2] += 0.15
        return goal.copy()

    def _get_obs(self):
        obs = super()._get_obs()
        obs['observation'] = np.concatenate([obs['observation'], obs['desired_goal']])
        return obs


class FetchPickAndPlaceDiffHoldout(VizFetchPickAndPlaceEnv):
    def __init__(self, reward_type='sparse'):
        self.coverage = 1.0
        self.rnd_gen = False
        self.set_noise_ratio(1.0, 1.0)

        super().__init__(reward_type)
        self.coverage = 1.0

    def set_noise_ratio(self, noise_ratio, goal_noise_ratio):
        self.obj_sampler = OldHoldoutSampler([-noise_ratio * OBJ_X_NOISE, 0],
                [noise_ratio * OBJ_X_NOISE, noise_ratio * Y_NOISE * 2], 4)
        self.goal_sampler = OldHoldoutSampler(
                [-goal_noise_ratio*X_NOISE, -goal_noise_ratio*Y_NOISE * 2],
                [goal_noise_ratio*X_NOISE, 0], 4)
        # self.obj_sampler = OldHoldoutSampler([-noise_ratio * OBJ_X_NOISE, -noise_ratio * Y_NOISE],
        #         [noise_ratio * OBJ_X_NOISE, noise_ratio * Y_NOISE], 4)
        # self.goal_sampler = OldHoldoutSampler(
        #         [-goal_noise_ratio*X_NOISE, -goal_noise_ratio*Y_NOISE],
        #         [goal_noise_ratio*X_NOISE, goal_noise_ratio*Y_NOISE], 4)

    def _reset_sim(self):
        #import ipdb; ipdb.set_trace()
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2] + np.array([0.0, OFFSET])
            object_xpos += self.obj_sampler.sample(self.coverage, self.np_random)

            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        goal = self.initial_gripper_xpos[:3] + np.array([0.0, -1*OFFSET, 0.0])
        goal[:2] += self.goal_sampler.sample(self.coverage, self.np_random)

        goal += self.target_offset
        goal[2] = self.height_offset
        goal[2] += 0.15
        return goal.copy()

    def _get_obs(self):
        obs = super()._get_obs()
        obs['observation'] = np.concatenate([obs['observation'], obs['desired_goal']])
        return obs

    def relabel_ob(self, ob_current, ob_future):
        import torch

        if isinstance(ob_current, torch.Tensor):
            return torch.cat([ob_current[:-3], ob_future[-3:]])
        return np.concatenate([ob_current[:-3], ob_future[-3:]])

    def is_reached(self, ob):
        import torch

        if isinstance(ob, torch.Tensor):
            ob = ob.cpu()
        dist = np.linalg.norm(ob[-3:] - ob[3:6])
        return float(dist < self.distance_threshold)


from gym.envs.robotics import rotations, robot_env
from gym.envs.robotics import utils as robot_utils
class FetchViz(FetchPickAndPlaceEnv):
    def __init__(self, reward_type='sparse'):
        super().__init__(reward_type)

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2] + np.array([0.0, OFFSET])
            object_xpos += self.np_random.uniform([-1*OBJ_X_NOISE, -1*Y_NOISE],
                    [OBJ_X_NOISE, Y_NOISE], size=2)

            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        goal = self.initial_gripper_xpos[:3] + np.array([0.0, -1*OFFSET, 0.0])
        goal[:2] += self.np_random.uniform([-1*X_NOISE, -1*Y_NOISE],
                [X_NOISE, Y_NOISE], size=2)

        goal += self.target_offset
        goal[2] = self.height_offset
        goal[2] += 0.15
        return goal.copy()

    def _get_obs(self):
        obs = super()._get_obs()
        obs['observation'] = np.concatenate([obs['observation'], obs['desired_goal']])
        return obs

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        lookat = [1.34193362, 0.74910034, 0.55472272]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.3
        self.viewer.cam.azimuth = 132
        self.viewer.cam.elevation = -14.


    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        robot_utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]
