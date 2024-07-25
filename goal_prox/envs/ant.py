import os
import numpy as np
from gym import spaces
from gym import utils
from gym.envs.mujoco import mujoco_env
from mujoco_py.builder import MujocoException


class AntGoalEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, angle_range=(0, np.pi), distance=5):
        self.goal = np.array([10, 0])
        self.angle_range = angle_range
        self.distance = distance
        self.spawn_noise = 0.0
        self.is_expert = False
        self.found_goal = False
        self.coverage = 100

        mujoco_env.MujocoEnv.__init__(
            self, os.path.join(os.path.abspath(os.path.dirname(__file__)), "ant.xml"), 5
        )
        utils.EzPickle.__init__(self)

    def step(self, a):
        info = {}
        self.do_simulation(a, self.frame_skip)

        # forward reward
        current_position = self.get_body_com("torso")
        dist = np.linalg.norm((current_position[:2] - self.goal)) / self.distance
        forward_reward = 1 - dist

        # control and contact costs
        ctrl_cost = 0.5 * 1e-2 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )

        reward = forward_reward - ctrl_cost - contact_cost

        state = self.state_vector()
        if self.is_expert:
            notdone = all(
                [
                    np.isfinite(state).all(),
                    not self.touching("torso_geom", "floor"),
                    state[2] >= 0.2,
                    state[2] <= 1.0,
                ]
            )
        notdone = True
        done = not notdone
        if dist < 0.1:
            # if not self.found_goal:
            #    print("Success")
            self.found_goal = True

        ob = self._get_obs()

        info["ep_found_goal"] = float(self.found_goal)
        info["reward_forward"] = forward_reward
        info["reward_ctrl"] = -ctrl_cost
        info["reward_contact"] = -contact_cost
        return ob, reward, done, info

    def _get_obs(self):
        current_position = self.get_body_com("torso")
        return np.concatenate(
            [
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
                self.goal - current_position[:2],
            ]
        )

    def relabel_ob(self, ob_current, ob_future):
        import torch

        if isinstance(ob_current, torch.Tensor):
            return torch.cat([ob_current[:-2], ob_current[-2:] - ob_future[-2:]])
        return np.concatenate([ob_current[:-2], ob_current[-2:] - ob_future[-2:]])

    def is_reached(self, ob):
        import torch

        if isinstance(ob, torch.Tensor):
            ob = ob.cpu()
        dist = np.linalg.norm(ob[-2:]) / self.distance
        return float(dist < 0.1)

    def reset_model(self):
        # qpos = self.init_qpos.copy() + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        # qvel = self.init_qvel.copy() + self.np_random.randn(self.model.nv) * .1
        qpos = self.model.numeric_data.copy()
        qvel = self.init_qvel.copy() + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )

        # For expert
        if self.is_expert:
            qvel[9:12] = 0  # need to be disabled during IL

        # For learner: need to enable one of the following during IL
        qpos = qpos + self.np_random.uniform(
            low=-self.spawn_noise, high=self.spawn_noise, size=self.model.nq
        )
        # qpos = qpos + self.np_random.uniform(low=-0.00, high=0.00,
        #        size=self.model.nq)

        # set goal
        self.goal = self.propose_original()
        qpos[-7:-5] = self.goal

        # goal_prox
        self.found_goal = False

        self.set_state(qpos, qvel)
        return self._get_obs()

    def propose_original(self):
        magnitude = self.distance

        num_division = 20
        if self.coverage == 100:
            exclude_range = []  # 100% coverage
        elif self.coverage == 75:
            exclude_range = range(2, 20, 4)  # 75% coverage
        elif self.coverage == 50:
            exclude_range = range(1, 20, 2)  # 50% coverage
        elif self.coverage == 25:
            exclude_range = [
                x for x in range(20) if x not in range(2, 20, 4)
            ]  # 25% coverage
        elif self.coverage == 99:
            print("Using 99")
            exclude_range = [x for x in range(num_division) if x not in [2, 6, 10]]
        elif self.coverage == 1:
            print("Using 1")
            exclude_range = [x for x in range(num_division) if x not in [0]]
        elif self.coverage == 2:
            print("Using 2")
            exclude_range = [x for x in range(num_division) if x not in [6]]
        else:
            raise ValueError("Invalid coverage")

        l = (self.angle_range[1] - self.angle_range[0]) / num_division
        while True:
            angle = self.angle_range[0] + (
                self.np_random.rand() * (self.angle_range[1] - self.angle_range[0])
            )
            exclude = False
            for i in exclude_range:
                s = self.angle_range[0] + i * l
                e = s + l
                if s < angle and angle < e:
                    exclude = True
                    break
            if not exclude:
                break

        return np.array([magnitude * np.cos(angle), magnitude * np.sin(angle)])

    def viewer_setup(self):
        # body_id = self.sim.model.body_name2id('torso')
        # lookat = self.sim.data.body_xpos[body_id]
        # for idx, v in enumerate(lookat):
        #     self.viewer.cam.lookat[idx] = v
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 12.0
        self.viewer.cam.azimuth = +90.0
        self.viewer.cam.elevation = -20

    def touching(self, geom1_name, geom2_name):
        idx1 = self.model.geom_name2id(geom1_name)
        idx2 = self.model.geom_name2id(geom2_name)
        for c in self.data.contact:
            if (c.geom1 == idx1 and c.geom2 == idx2) or (
                c.geom1 == idx2 and c.geom2 == idx1
            ):
                return True
        return False
