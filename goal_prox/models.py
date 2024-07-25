import torch.nn as nn
from rlf.rl.model import BaseNet, Flatten, MLPBase


class GwImgEncoder(BaseNet):
    """
    Custom image encoder to support the Grid World environment with image
    observations (rather than flattened observations). This is important for
    large grids.
    """

    def __init__(self, obs_shape, hidden_size=64):
        super().__init__(False, hidden_size, hidden_size)

        # Network architecture inspired by https://github.com/lcswillems/rl-starter-files/blob/master/model.py
        n = obs_shape[1]
        m = obs_shape[2]
        image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        self.net = nn.Sequential(
            nn.Conv2d(obs_shape[0], 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            Flatten(),
            nn.Linear(image_embedding_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self, inputs, rnn_hxs, masks):
        return self.net(inputs), rnn_hxs
