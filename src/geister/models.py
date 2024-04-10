""" Models for Geister game. """

import numpy as np
import torch
from torch import device, nn

class GeisterModel(nn.Module):
    """
    Model for Geister game
    """
    def __init__(self, input_shapes, action_size):
        """
        Initialize the model
        There are 108 possilbe directions to move. (output size)
                  32 movable spaces
                  3 unique piece types as a player
                  
        :param input_shapes: list of input shapes
        :param action_size: size of the action space
        """
        super(GeisterModel, self).__init__()

        
