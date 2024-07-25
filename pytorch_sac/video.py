import imageio
import os
import numpy as np
import sys

import utils

class VideoRecorder(object):
    def __init__(self, save_dir, should_spec_dims, height=256, width=256, camera_id=0, fps=30):
        self.save_dir = save_dir
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.should_spec_dims = should_spec_dims
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env):
        if self.enabled:
            kwargs = {}
            if self.should_spec_dims:
                kwargs = {
                        'width': self.width,
                        'height': self.height
                        }
            frame = env.render(mode='rgb_array',**kwargs)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            print('Saved to ', path)
            imageio.mimsave(path, self.frames, fps=self.fps)
