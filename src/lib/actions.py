""" Action Mechanics """
from typing import Union
import numpy as np

class ActionSelector:
    """ Abstract class which converts scores to the actions """
    def __call__(self, scores):
        raise NotImplementedError

class ArgmaxActionSelector(ActionSelector):
    """ Selects actions using argmax """
    def __call__(self, scores: np.ndarray):
        assert isinstance(scores, np.ndarray)
        return np.argmax(scores, axis=1)

class EpsilonGreedyActionSelector(ActionSelector):
    """ Selects actions using epsilon-greedy policy """
    def __init__(self,
                 epsilon: float = 0.05,
                 selector=None):

        assert 0.0 <= epsilon <= 1.0, "Epsilon should be in [0, 1]"

        # Set initial parameters
        self.epsilon = epsilon
        # Set selector if given, otherwise use argmax
        self.selector = selector if selector is not None else ArgmaxActionSelector()

    def __call__(self, scores: np.ndarray):
        # Check input parameters
        assert isinstance(scores, np.ndarray)

        # Get batch size and number of actions
        batch_size, action_count = scores.shape

        # Select and return actions
        actions = self.selector(scores)
        mask = np.random.random(size=batch_size) < self.epsilon
        random_actions = np.random.choice(action_count, size=sum(mask))
        actions[mask] = random_actions
        return actions

class EpsilonTracker:
    """ Updates epsilon according to linear schedule """
    def __init__(self,
                 selector: EpsilonGreedyActionSelector,
                 eps_start: Union[int, float],
                 eps_final: Union[int, float],
                 eps_frames: int):

        # Check input parameters
        assert isinstance(selector, EpsilonGreedyActionSelector)
        assert isinstance(eps_start, (int, float))
        assert isinstance(eps_final, (int, float))
        assert isinstance(eps_frames, int)

        # Set initial parameters
        self.selector = selector
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_frames = eps_frames
        self.frame(0)

    def frame(self, frame: int):
        """
        Update epsilon value
        :param frame: frame number
        """
        # Check input parameters
        assert isinstance(frame, int)

        # Set epsilon value to linear decay or final epsilon (max)
        eps = self.eps_start - frame / self.eps_frames
        self.selector.epsilon = max(self.eps_final, eps)

    def __call__(self, frame: int):
        self.frame(frame)
        return self.selector.epsilon
