import sys
sys.path.insert(0, './')
import gym
import goal_prox.envs.viz
from PIL import Image


if __name__ == '__main__':
    env = gym.make('VizFetchPickAndPlaceCustom-v0')
    init_gripper = env.sim.data.get_mocap_pos('robot0:mocap')
    init_block = env.sim.data.get_site_xpos('object0')
    init_goal = env.goal
    for _ in range(10):
        env.sim.step()
    frame = env.render(mode='rgb_array', width=128, height=128)
    Image.fromarray(frame).save('data/plot/test.png')
