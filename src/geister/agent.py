""" The agent players of the game """

import numpy as np
import torch
from torch import device

from ..lib.agents import BaseAgent
from .environment import GeisterState

class RandomAgent(BaseAgent):
    """
    Random agent
    """
    def __init__(self, seed: int = 42):
        self.seed = seed

    def initial_state(self) -> any:
        """ Initialize the state """
        return None

    def __call__(self, states: list, agent_states) -> tuple:
        actions = []
        for state in states:
            actions.append(np.random.choice(state.get_posssible_actions(1)))
        return actions, agent_states

class GeisterAgent(BaseAgent):
    """
    Geister agent
    """
    def __init__(self, model, postprocessor, hardware: device = "cpu", preprocessor=None):
        """
        Create Geister agent
        :param model: model to use for action and or value calculation
        :param postprocessor: function to convert model output into actions
        :param hardware: device to use for calculations (cpu or cuda)
        :param preprocessor: function to process states batch before feeding it into model
        """
        self.model = model
        self.postprocessor = postprocessor
        self.preprocessor = preprocessor
        self.device = hardware

    @torch.no_grad()
    def __call__(self, states: list[GeisterState] , agent_states = None) -> tuple:
        if agent_states is None:
            # Create initial empty state
            agent_states = [None] * len(states)

        if self.preprocessor is not None:
            states = self.preprocessor(states, self.device)

        actions = self.model(states).cpu().numpy()
        actions = self.postprocessor(actions)

        return actions, agent_states

    def set_preprocessor(self, preprocessor):
        """
        Allows for ad-hoc preprocessor change
        In relation to the agent, the input to the model must remain unchanged
        :param preprocessor: function to process states batch before feeding it into model
        """
        self.preprocessor = preprocessor

    def set_postprocessor(self, postprocessor):
        """
        Allows for ad-hoc postprocessor change
        In relation to the agent, the output of the model is unchanged
        """
        self.postprocessor = postprocessor
