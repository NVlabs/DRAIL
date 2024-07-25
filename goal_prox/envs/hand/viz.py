import sys
sys.path.insert(0, './')
import goal_prox.envs.hand
import gym
from PIL import Image

env = gym.make('HandReachCustom-v0')
obs = env.reset()

for i in range(10):
    env.step(env.action_space.sample())
    frame = env.render('rgb_array')
    Image.fromarray(frame).save('data/vids/hand_viz_%i.png' % i)
