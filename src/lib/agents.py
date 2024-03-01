""" Agent classes for the reinforcement learning algorithms """
import numpy as np
import torch

from lib.utils import arrayStateToTensor

class BaseAgent:
    """
    Abstract Agent interface
    """
    def initial_state(self) -> any:
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent wants to remember
        """
        return None

    def __call__(self, states: list, agent_states) -> tuple:
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :param agent_states: list of states with the same length as observations
        :return: tuple of actions, states
        """

        # Check input parameters
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)

        raise NotImplementedError

def default_states_preprocessor(states: list) -> torch.Tensor:
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    """
    if len(states) == 1:
        # Turn single state into a batch
        np_states = np.expand_dims(states[0], 0)
    else:
        # Transpose list of arrays into array of arrays
        np_states = np.array(states)
    return torch.from_numpy(np_states)

class DQNAgent(BaseAgent):
    """
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and converts them into the actions using action_selctor
    """
    def __init__(self, dqn_model, action_selctor, device="cpu", preprocessor=arrayStateToTensor):
        """
        Create DQN-based agent
        :param dqnModel: DQN model to use for action calculation
        :param action_selctor: selector to choose actions from Q-values
        :param device: device to use for calculations (cpu or cuda)
        :param preprocessor: function to process states batch before feeding it into DQN
        """
        self.dqn_model = dqn_model
        self.action_selctor = action_selctor
        self.preprocessor = preprocessor
        self.device = device

    # Skip gradient calculation on agent's call of the model
    @torch.no_grad()
    def __call__(self, states: list, agent_states=None) -> tuple:
        if agent_states is None:
            # Create initial empty state
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states, self.device)

        # Calculate Q values - forward pass of the model
        q_values = self.dqn_model(states)
        # Convert Q values into numpy array and move to CPU
        q = q_values.data.cpu().numpy()
        # Select actions using the selector
        actions = self.action_selctor(q)
        return actions, agent_states
