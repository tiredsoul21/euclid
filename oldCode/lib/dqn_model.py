import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        # 84 is arbitrary, but it is the size of the current frame
        self.conv = nn.Sequential(
            # 84x84xN -> 20x20x32
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # 20x20x32 -> 9x9x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # 9x9x64 -> 7x7x64
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # 7x7x64 -> 3136
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    # This is a helper function to get the output size of the conv layer
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    # This method is to flatten the conv output
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
