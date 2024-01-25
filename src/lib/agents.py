"""
Agent is something which converts states into actions and has state
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F

from . import actions


class BaseAgent:
    """
    Abstract Agent interface
    """
    def initialState(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent wants to remember
        """
        return None

    def __call__(self, states, agentStates):
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :param agentStates: list of states with the same length as observations
        :return: tuple of actions, states
        """
        assert isinstance(states, list)
        assert isinstance(agentStates, list)
        assert len(agentStates) == len(states)

        raise NotImplementedError


def defaultStatesPreprocessor(states):
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    """
    if len(states) == 1:
        # Turn single state into a batch
        npStates = np.expand_dims(states[0], 0)
    else:
        # Transpose list of arrays into array of arrays
        npStates = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(npStates)

class DQNAgent(BaseAgent):
    """
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and  converts them into the actions using actionSelector
    """
    def __init__(self,dqnModel, actionSelector, device="cpu", preprocessor=defaultStatesPreprocessor):
        """
        Create DQN-based agent
        :param dqnModel: DQN model to use for action calculation
        :param actionSelector: selector to choose actions from Q-values
        :param device: device to use for calculations (cpu or cuda)
        :param preprocessor: function to process states batch before feeding it into DQN
        """
        self.dqnModel = dqnModel
        self.actionSelector = actionSelector
        self.preprocessor = preprocessor
        self.device = device

    # Skip gradient calculation on forward pass
    @torch.no_grad()
    def __call__(self, states, agentStates=None):
        if agentStates is None:
            # Create initial empty state
            agentStates = [None] * len(states)
        if self.preprocessor is not None:
            # Preprocess states if needed
            states = self.preprocessor(states)
            # Checks if states is a tensor before moving it to the device
            if torch.is_tensor(states):
                states = states.to(self.device)
            # May need to catch if states is not a tensor?
        # Calculate Q values - forward pass of the model
        qValues = self.dqnModel(states)
        # Convert Q values into numpy array and move to CPU
        q = qValues.data.cpu().numpy()
        # Select actions using the selector
        actions = self.actionSelector(q)
        return actions, agentStates
