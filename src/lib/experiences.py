""" Library for experience sources and replay buffers """
from collections import namedtuple, deque
from typing import Optional, Union
import math

import numpy as np
import gym

from .agents import BaseAgent
from .utils import MinSegmentTree, SumSegmentTree

# Element containing a state, chosen action, reward, and whether the episode is done
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])
# Element containing a state, chosen action, reward, and the last state
ExperienceFirstLast = namedtuple('ExperienceFirstLast', ('state', 'action', 'reward', 'last_state'))

def _partition_list(items, lengths):
    """
    Partitions the list of items by lengths
    :param items: list of items
    :param lengths: list of integers
    :return: list of list of items partitioned by lengths
    """
    # Iterate through the lengths and partition the items
    result = [items[offset:offset+length] for offset, length in enumerate(lengths)]
    return result

class ExperienceSource:
    """
    An n-step experience source that can handle single or multiple environments.
    Each experience consists of a sequence of n Experience entries.
    These entries the agent's n interactions with the environment.
    """
    def __init__(self,
                 env: Union[gym.Env, list, tuple],
                 agent: BaseAgent,
                 step_count: int = 2,
                 exp_interval: int = 1,
                 vectorized: bool = False):
        """
        Create simple experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions to take
        :param step_count: count of steps to track for every experience chain
        :param exp_interval: how many steps to do between experience items
        :param vectorized: support of vectorized envs from OpenAI universe
        """

        # Check input parameters
        assert isinstance(env, (gym.Env, list, tuple))
        assert isinstance(agent, BaseAgent)
        assert isinstance(step_count, int) and step_count >= 1
        assert isinstance(exp_interval, int)
        assert isinstance(vectorized, bool)

        # Initialize variables
        self.pool = env if isinstance(env, (list, tuple)) else [env]
        self.agent = agent
        self.step_count = step_count
        self.exp_interval = exp_interval
        self.vectorized = vectorized
        self.total_rewards = []
        self.total_steps = []

    def __iter__(self):
        """
        Iterate over the source
        :return: experience batch
        """

        # Initialize variables
        states, agent_states, histories, current_reward, current_steps = [], [], [], [], []
        env_lengths = []

        # Loop through the environments
        for env in self.pool:
            obs = env.reset()

            # Store the states for each environment
            obs_lengths = len(obs) if self.vectorized else 1
            _ = states.extend(obs) if self.vectorized else states.append(obs)
            env_lengths.append(obs_lengths)

            # Initialize starting values
            histories = [deque(maxlen=self.step_count) for _ in range(obs_lengths)]
            current_reward.extend([0.0] * obs_lengths)
            current_steps.extend([0] * obs_lengths)
            agent_states.extend([self.agent.initial_state() for _ in range(obs_lengths)])

        while True:
            # Initialize variables
            # Pick an environment and get the action space
            actions = [self.pool[0].actionSpace.sample() \
                       if state is None else None for state in states]
            # Filter out None states and their indices
            state_input = [state for state in states if state is not None]
            states_indices = [idx for idx, state in enumerate(states) if state is not None]
            global_offset = 0

            # Check if state_input is empty
            if state_input:
                # Get actions and new agent states
                states_actions, next_agent_states = self.agent(state_input, agent_states)

                # Store the actions and new agent states
                for idx, action in enumerate(states_actions):
                    state_index = states_indices[idx]
                    actions[state_index] = action
                    agent_states[state_index] = next_agent_states[idx]

            # Partition the actions by environment
            grouped_actions = _partition_list(actions, env_lengths)

            # Loop through the environments and...
            for _, (env, actions) in enumerate(zip(self.pool, grouped_actions)):
                if self.vectorized:
                    # Receive the step results as an array
                    next_states, rewards, is_dones, _, _ = env.step(actions)
                else:
                    # Receive the step results and cast to an array
                    next_state, r, is_done, _, _ = env.step(actions[0])
                    next_states, rewards, is_dones = [next_state], [r], [is_done]

                # Loop through the step results
                for offset, (action, next_state, r, is_done) in \
                        enumerate(zip(actions, next_states, rewards, is_dones)):
                    idx = global_offset + offset
                    state = states[idx]
                    history = histories[idx]

                    # Update the reward and step count
                    current_reward[idx] += r
                    current_steps[idx] += 1

                    # Add the experience to the history
                    if state is not None:
                        history.append(Experience(state=state, \
                                                  action=action, reward=r, done=is_done))

                    # if enough steps collected, yield the item
                    if len(history) == self.step_count \
                            and current_steps[idx] % self.exp_interval == 0:
                        yield tuple(history)

                    states[idx] = next_state

                    # if is_done, clear the history
                    if is_done:
                        # Yield all the remaining history
                        if 0 < len(history) < self.step_count:
                            yield tuple(history)

                        # Pop from the history until it is empty
                        while len(history) > 1:
                            history.popleft()
                            yield tuple(history)

                        # Store the total reward and step count
                        self.total_rewards.append(current_reward[idx])
                        self.total_steps.append(current_steps[idx])

                        # Clear current reward and step count
                        current_reward[idx] = 0.0
                        current_steps[idx] = 0

                        # Reset everything
                        states[idx] = env.reset() if not self.vectorized else None
                        agent_states[idx] = self.agent.initial_state()
                        history.clear()

                # Update the global offset
                global_offset += len(actions)

    def pop_total_rewards(self):
        """
        Returns the total rewards and clears the list
        :return: total rewards
        """
        reward = self.total_rewards.copy()
        self.total_rewards.clear()
        self.total_steps.clear()
        return reward

    def pop_rewards_steps(self):
        """
        Returns the total steps and total rewards and clears the lists
        :return: total steps
        """
        result = list(zip(self.total_rewards, self.total_steps))
        self.total_rewards.clear()
        self.total_steps.clear()
        return result

class ExperienceSourceFirstLast(ExperienceSource):
    """
    This is a wrapper around ExperienceSource to prevent storing full trajectory
    in replay buffer when we need only first and last states. For every trajectory
    piece it calculates discounted reward and emits only first and last states and
    action taken in the first state. If we have partial trajectory at the end
    of episode, last_state will be None
    :param ExperienceSource: source to take trajectories from
    :param agent: callable to convert batch of states into actions to take
    :param gamma: discount for reward calculation
    :param step_count: count of steps in trajectory piece
    :param exp_interval: how many steps to do between experience items
    :param vectorized: support of vectorized envs from OpenAI universe
    """
    def __init__(self,
                 env : Union[gym.Env, list, tuple],
                 agent: BaseAgent,
                 gamma: float,
                 step_count: int = 1,
                 exp_interval: int = 1,
                 vectorized:bool = False):

        # Check input parameters - rest are checked in ExperienceSource
        assert isinstance(gamma, float)
        super().__init__(env, agent, step_count+1, exp_interval, vectorized=vectorized)

        # Initialize variables
        self.gamma = gamma
        self.steps = step_count

    def __iter__(self):
        """
        Modify the iterator to return ExperienceFirstLast
        """
        for exp in super(ExperienceSourceFirstLast, self).__iter__():
            # If the last state is None, then the episode is done
            if exp[-1].done and len(exp) <= self.steps:
                last_state = None
                elems = exp
            # else, last state is the last state in the experience
            else:
                last_state = exp[-1].state
                elems = exp[:-1]

            # Calculate the total reward
            total_reward = sum(e.reward * (self.gamma ** i) for i, e in enumerate(reversed(elems)))

            yield ExperienceFirstLast(state=exp[0].state, action=exp[0].action,
                                      reward=total_reward, last_state=last_state)

class ExperienceReplayBuffer:
    """ Buffer to store experiences and sample them randomly """
    def __init__(self,
                 exp_source: Optional[ExperienceSource],
                 buffer_size: int):
        """
        Create a replay buffer object
        :param exp_source: an iterable source of experience to replay
        :param buffer_size: maximum number of elements in buffer
        """

        # Check input parameters
        assert isinstance(buffer_size, int)
        assert isinstance(exp_source, (ExperienceSource, type(None)))

        # Initialize variables
        self.buffer = []
        self.capacity = buffer_size
        self.pos = 0

        # Point to the experience source iterator method
        self.exp_source_iter = None if exp_source is None else iter(exp_source)


    def __len__(self):
        """
        :return: length of the buffer
        """
        return len(self.buffer)

    def __iter__(self):
        """
        Calls base iter
        :return: iterator over the buffer
        """
        return iter(self.buffer)

    def sample(self, batch_size):
        """
        Get one random batch from experience replay
        :param batch_size:
        :return:
        """
        if len(self.buffer) <= batch_size:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def _add(self, sample):
        """
        Add a sample to the buffer
        :param sample: sample to add
        """

        # If the buffer is not full, append the sample
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        # Else, replace the sample at the current position
        else:
            self.buffer[self.pos] = sample
        # Increment the position
        self.pos = (self.pos + 1) % self.capacity

    def populate(self, samples: int):
        """
        Populates samples into the buffer
        :param samples: how many samples to populate
        """
        # Loops through the samples and add them to the buffer
        for _ in range(samples):
            entry = next(self.exp_source_iter)
            self._add(entry)

class PrioritizedReplayBuffer(ExperienceReplayBuffer):
    """ Replay buffer with prioritized experience replay """
    @property
    def _max_priority(self):
        return 1.0

    def __init__(self,
                 exp_source: ExperienceSource,
                 buffer_size: int,
                 alpha: float = 0.6):
        """
        Create prioritized replay buffer
        :param exp_source: source to generate experience to store
        :param buffer_size: maximum number of transitions to store
        :param alpha: how much prioritization is used, 
            0 - no prioritization,
            1 - full prioritization
        """
        super(PrioritizedReplayBuffer, self).__init__(exp_source, buffer_size)

        # Check input parameters
        assert isinstance(exp_source, ExperienceSource)
        assert isinstance(buffer_size, int) and buffer_size > 0
        assert isinstance(alpha, float) and alpha > 0

        # Initialize variables
        self._alpha = alpha
        capacity = 2 ** math.ceil(math.log2(buffer_size))
        self._sum_seg_tree = SumSegmentTree(capacity)
        self._min_seg_tree = MinSegmentTree(capacity)

    def _add(self, *args, **kwargs):
        super()._add(*args, **kwargs)

        idx = self.pos
        priority = self._max_priority ** self._alpha

        # Add the priority to the sum and min segment trees
        self._sum_seg_tree[idx] = priority
        self._min_seg_tree[idx] = priority

    def _sample_proportional(self, batch_size: int):
        """
        Generates random masses, scales them based on the sum of priorities.
        Uses these scaled masses to sample indices from the sum tree.
        This sampling process gives higher probabilities to transitions with higher priorities.
        :param batch_size: size of the batch
        :return: list of indices
        """
        mass = np.random.random(batch_size) * self._sum_seg_tree.sum(0, len(self) - 1)
        idx = np.array([self._sum_seg_tree.find_prefixsum_idx(m) for m in mass])
        return idx.tolist()

    def sample(self, batch_size: int, beta: float = 0.4):
        """
        Sample a batch of experiences
        :param batch_size: size of the batch
        :param beta: beta parameter for prioritized replay buffer
        :return: list of experiences
        """

        # Check input parameters
        assert isinstance(beta, float) and beta > 0
        assert isinstance(batch_size, int) and 0 < batch_size <= len(self)

        # Sample indices
        indices = self._sample_proportional(batch_size)

        # Compute total priority and minimum priority
        total_priority = self._sum_seg_tree.sum()
        p_min = self._min_seg_tree.min() / total_priority

        # Calculate weights
        max_weight = (p_min * len(self)) ** (-beta)
        weights = ((self._sum_seg_tree[idx] / total_priority \
                    * len(self)) ** (-beta) / max_weight for idx in indices)

        # Get the samples
        samples = [self.buffer[idx] for idx in indices]
        return samples #, indices, weights

    def update_priorities(self, indices, priorities):
        """
        Update priorities of sampled transitions
        :param indices: list of sample indices
        :param priorities: list of sample priorities
        """
        # Check input parameters
        assert len(indices) == len(priorities)
        assert all(priority > 0 for priority in priorities)
        assert all(0 <= idx < len(self) for idx in indices)

        # Vectorized priority update
        idx = np.arange(len(self))
        self._sum_seg_tree[idx] = np.where(np.isin(idx, indices), \
                                           priorities ** self._alpha, self._sum_seg_tree[idx])
        self._min_seg_tree[idx] = np.where(np.isin(idx, indices), \
                                           priorities ** self._alpha, self._min_seg_tree[idx])
        self._max_priority = max(self._max_priority, priorities)
