import os
from gym import utils
from gym.envs.robotics import fetch_env
import numpy as np
from goal_prox.envs.holdout_sampler import HoldoutSampler, LineHoldoutSampler
from goal_prox.envs.old_holdout_sampler import OldHoldoutSampler


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push.xml')

Y_NOISE = 0.02
X_NOISE = 0.05
OBJ_X_NOISE = 0.05
OFFSET = 0.10


class FetchPushEnvCustom(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='dense', random_start=True):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        self.coverage = 1.0
        self.goal_noise = True
        self.rnd_gen = False
        self.set_noise_ratio(1.0, 1.0)
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=random_start, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0,
            # The ranges shouldn't matter because we sample ourselves
            obj_range=0.1, target_range=0, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        self.max_episode_steps = 60

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

    def _get_obs(self):
        obs = super()._get_obs()
        obs['observation'] = np.concatenate([obs['observation'],
            obs['desired_goal']])
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

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2] + np.array([0.0, OFFSET])
            object_xpos += self.obj_sampler.sample(self.coverage,
                    self.np_random)

            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        goal = self.initial_gripper_xpos[:3] + np.array([0.0, -1*OFFSET, 0.0])
        if self.goal_noise:
            goal[:2]+= self.goal_sampler.sample(self.coverage, self.np_random)

            goal += self.target_offset
        goal[2] = self.height_offset
        return goal.copy()

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

class FetchDebugPushEnv(FetchPushEnvCustom):
    def set_noise_ratio(self, noise_ratio, goal_noise_ratio):
        noise_ratio *= 1
        y_noise_scale = 0.15 / (noise_ratio * Y_NOISE)
        #y_noise_scale = 1.0
        self.obj_sampler = LineHoldoutSampler(
                [-noise_ratio * OBJ_X_NOISE, -y_noise_scale*noise_ratio * Y_NOISE],
                [noise_ratio * OBJ_X_NOISE, y_noise_scale*noise_ratio * Y_NOISE])

        self.goal_sampler = HoldoutSampler(
                [-goal_noise_ratio*X_NOISE, -goal_noise_ratio*Y_NOISE],
                [goal_noise_ratio*X_NOISE, goal_noise_ratio*Y_NOISE], 1, True)


