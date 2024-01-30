import copy
import numpy as np

import torch
import torch.nn as nn

class DQNConv1D(nn.Module):
    def __init__(self, shape, actionCount):
        """
        Create a DQN model for 1D convolutional input
        :param shape: shape of input (channels, in)
        """
        super(DQNConv1D, self).__init__()

        # Create convolutional layers - transform time series into features
        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], 128, 5),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5),
            nn.ReLU(),
        )

        convOutSize = self._getConvOut(shape)

        # Create fully connected layers - transform features into value
        self.fcValue = nn.Sequential(
            nn.Linear(convOutSize, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Create fully connected layers - transform features into advantage for each action
        self.fcAdvantage = nn.Sequential(
            nn.Linear(convOutSize, 512),
            nn.ReLU(),
            nn.Linear(512, actionCount)
        )

    def _getConvOut(self, shape):
        """
        Calculate the output size of the convolutional layers
        :param shape: shape of the input
        :return: size of the output
        """
        o = self.conv(torch.zeros(1, *shape))
        return o.view(1, -1).size(1)

    def forward(self, x):
        """
        Forward pass of the model
        :param x: input
        :return: value and advantage
        """
        convOut = self.conv(x).view(x.size()[0], -1)
        val = self.fcValue(convOut)
        adv = self.fcAdvantage(convOut)
        return val + (adv - adv.mean(dim=1, keepdim=True))

class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__(self, model):
        """
        Create a target net from a model (deep copy)
        """
        self.model = model
        self.targetModel = copy.deepcopy(model)

    def sync(self):
        """
        Copy weights from model to target model
        """
        self.targetModel.load_state_dict(self.model.state_dict())

    def alphaSync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """

        # Check input parameters
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0

        state = self.model.state_dict()
        targetState = self.targetModel.state_dict()

        # Blend the parameters with a weighted average
        for k, v in state.items():
            targetState[k] = targetState[k] * alpha + (1 - alpha) * v

        # Load the blended parameters into the target model
        self.targetModel.load_state_dict(targetState)