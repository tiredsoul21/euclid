import gym
import random
import collections

import numpy as np

from collections import namedtuple, deque

from .agents import BaseAgent

# Element containing a state, chosen action, reward, and whether the episode is done
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])
# those entries are emitted from ExperienceSourceFirstLast. Reward is discounted over the trajectory piece
ExperienceFirstLast = collections.namedtuple('ExperienceFirstLast', ('state', 'action', 'reward', 'lastState'))

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
                 env: (gym.Env, list, tuple),
                 agent: BaseAgent,
                 stepsCount: int = 2,
                 experienceInterval: int = 1,
                 vectorized: bool = False):
        """
        Create simple experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions to take
        :param stepsCount: count of steps to track for every experience chain
        :param experienceInterval: how many steps to do between experience items
        :param vectorized: support of vectorized envs from OpenAI universe
        """

        # Check input parameters
        assert isinstance(env, (gym.Env, list, tuple))
        assert isinstance(agent, BaseAgent)
        assert isinstance(stepsCount, int) and stepsCount >= 1
        assert isinstance(experienceInterval, int)
        assert isinstance(vectorized, bool)

        # Initialize variables
        self.pool = env if isinstance(env, (list, tuple)) else [env]
        self.agent = agent
        self.stepsCount = stepsCount
        self.experienceInterval = experienceInterval
        self.vectorized = vectorized
        self.totalRewards = []
        self.totalSteps = []

    def __iter__(self):
        """
        Iterate over the source
        :return: experience batch
        """

        # Initialize variables
        states, agentStates, histories, currentReward, currentSteps = [], [], [], [], []
        envLengths = []

        # Loop through the environments
        for env in self.pool:
            obs = env.reset()

            # Store the states for each environment
            obsLengths = len(obs) if self.vectorized else 1
            states.extend(obs) if self.vectorized else states.append(obs)
            envLengths.append(obsLengths)

            # Initialize starting values
            histories = [deque(maxlen=self.stepsCount) for _ in range(obsLengths)]
            currentReward.extend([0.0] * obsLengths)
            currentSteps.extend([0] * obsLengths)
            agentStates.extend([self.agent.initialState() for _ in range(obsLengths)])

        while True:
            # Initialize variables
            # Pick an environment and get the action space
            actions = [self.pool[0].actionSpace.sample() if state is None else None for state in states]
            # Filter out None states and their indices
            statesInput = [state for state in states if state is not None]
            statesIndices = [idx for idx, state in enumerate(states) if state is not None]
            globalOffset = 0

            # Check if statesInput is empty
            if statesInput:
                # Get actions and new agent states
                statesActions, nextAgentStates = self.agent(statesInput, agentStates)
                
                # Store the actions and new agent states
                for idx, action in enumerate(statesActions):
                    stateIndex = statesIndices[idx]
                    actions[stateIndex] = action
                    agentStates[stateIndex] = nextAgentStates[idx]
            
            # Partition the actions by environment
            groupedActions = _partition_list(actions, envLengths)

            # Loop through the environments and...
            for envIndex, (env, actions) in enumerate(zip(self.pool, groupedActions)):
                if self.vectorized:
                    # Receive the step results as an array
                    nextStates, rewards, isDones, _, _ = env.step(actions)
                else:
                    # Receive the step results and cast to an array
                    nextState, r, isDone, _, _ = env.step(actions[0])
                    nextStates, rewards, isDones = [nextState], [r], [isDone]

                # Loop through the step results
                for offset, (action, nextState, r, isDone) in enumerate(zip(actions, nextStates, rewards, isDones)):
                    idx = globalOffset + offset
                    state = states[idx]
                    history = histories[idx]

                    # Update the reward and step count
                    currentReward[idx] += r
                    currentSteps[idx] += 1

                    # Add the experience to the history
                    if state is not None:
                        history.append(Experience(state=state, action=action, reward=r, done=isDone))

                    # if enough steps collected, yield the item
                    if len(history) == self.stepsCount and currentSteps[idx] % self.experienceInterval == 0:
                        yield tuple(history)

                    states[idx] = nextState

                    # if isDone, clear the history
                    if isDone:
                        # in case of very short episode (shorter than our steps count), send gathered history
                        if 0 < len(history) < self.stepsCount:
                            yield tuple(history)

                        # Pop from the history until it is empty
                        while len(history) > 1:
                            history.popleft()
                            yield tuple(history)

                        # Store the total reward and step count
                        self.totalRewards.append(currentReward[idx])
                        self.totalSteps.append(currentSteps[idx])
                        
                        # Clear current reward and step count
                        currentReward[idx] = 0.0
                        currentSteps[idx] = 0

                        # Reset everything
                        states[idx] = env.reset() if not self.vectorized else None
                        agentStates[idx] = self.agent.initialState()
                        history.clear()

                # Update the global offset
                globalOffset += len(actions)

    def popTotalRewards(self):
        """
        Returns the total rewards and clears the list
        :return: total rewards
        """
        reward = self.totalRewards.copy()
        self.totalRewards.clear()
        self.totalSteps.clear()
        return reward

    def popRewardsSteps(self):
        """
        Returns the total steps and total rewards and clears the lists
        :return: total steps
        """
        result = list(zip(self.totalRewards, self.totalSteps))
        self.totalRewards.clear()
        self.totalSteps.clear()
        return result

class ExperienceSourceFirstLast(ExperienceSource):
    """
    This is a wrapper around ExperienceSource to prevent storing full trajectory in replay buffer when we need
    only first and last states. For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.
    If we have partial trajectory at the end of episode, lastState will be None
    :param ExperienceSource: source to take trajectories from
    :param agent: callable to convert batch of states into actions to take
    :param gamma: discount for reward calculation
    :param stepsCount: count of steps in trajectory piece
    :param experienceInterval: how many steps to do between experience items
    :param vectorized: support of vectorized envs from OpenAI universe
    """
    def __init__(self,
                 env : (gym.Env, list, tuple),
                 agent: BaseAgent,
                 gamma: float,
                 stepsCount: int = 1,
                 experienceInterval: int = 1,
                 vectorized:bool = False):

        # Check input parameters - rest are checked in ExperienceSource
        assert isinstance(gamma, float)
        super(ExperienceSourceFirstLast, self).__init__(env, agent, stepsCount+1, experienceInterval, vectorized=vectorized)

        # Initialize variables
        self.gamma = gamma
        self.steps = stepsCount

    def __iter__(self):
        """
        Modify the iterator to return ExperienceFirstLast
        """
        for exp in super(ExperienceSourceFirstLast, self).__iter__():
            # If the last state is None, then the episode is done
            if exp[-1].done and len(exp) <= self.steps:
                lastState = None
                elems = exp
            # else, last state is the last state in the experience
            else:
                lastState = exp[-1].state
                elems = exp[:-1]

            # Calculate the total reward
            totalReward = sum(e.reward * (self.gamma ** i) for i, e in enumerate(reversed(elems)))

            yield ExperienceFirstLast(state=exp[0].state, action=exp[0].action,
                                      reward=totalReward, lastState=lastState)

class ExperienceReplayBuffer:
    def __init__(self,
                 experienceSource: (ExperienceSource, None),
                 bufferSize: int):
        """
        Create a replay buffer object
        :param experienceSource: an iterable source of experience to replay
        :param bufferSize: maximum number of elements in buffer
        """

        # Check input parameters
        assert isinstance(bufferSize, int)
        assert isinstance(experienceSource, (ExperienceSource, type(None)))

        # Initialize variables
        self.buffer = []
        self.capacity = bufferSize
        self.pos = 0

        # Point to the experience source iterator method
        self.experienceSourceIter = None if experienceSource is None else iter(experienceSource)


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

    def sample(self, batchSize):
        """
        Get one random batch from experience replay
        :param batchSize:
        :return:
        """
        if len(self.buffer) <= batchSize:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batchSize, replace=True)
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
            entry = next(self.experienceSourceIter)
            self._add(entry)

class PrioritizedReplayBuffer(ExperienceReplayBuffer):
    @property
    def _maxPriority(self):
        return 1.0

    def __init__(self,
                 experienceSource: ExperienceSource,
                 bufferSize: int,
                 alpha: float):
        super(PrioritizedReplayBuffer, self).__init__(experienceSource, bufferSize)

        # Check input parameters
        assert isinstance(experienceSource, ExperienceSource)
        assert isinstance(bufferSize, int) and bufferSize > 0
        assert isinstance(alpha, float) and alpha > 0

        # Initialize variables
        self._alpha = alpha
        capacity = 2 ** math.ceil(math.log2(bufferSize))
        self._sumSegTree = utils.SumSegmentTree(capacity)
        self._minSegTree = utils.MinSegmentTree(capacity)

    def _add(self, *args, **kwargs):
        super()._add(*args, **kwargs)

        idx = self.pos
        priority = self._maxPriority ** self._alpha
        
        # Add the priority to the sum and min segment trees
        self._sumSegTree[idx] = priority
        self._minSegTree[idx] = priority

    def _sampleProportional(self, batchSize: int):
        """
        Generates random masses, scales them based on the sum of priorities.
        Uses these scaled masses to sample indices from the sum tree.
        This sampling process gives higher probabilities to transitions with higher priorities.
        :param batchSize: size of the batch
        :return: list of indices
        """
        # Generate a random mass and find the indices
        mass = np.random.random(batchSize) * self._sumSegTree.sum(0, len(self) - 1)
        idx = self._sumSegTree.find_prefixsum_idx(mass)
        return idx.tolist()

    def sample(self, batchSize: int, beta: float):
        """
        Sample a batch of experiences
        :param batchSize: size of the batch
        :param beta: beta parameter for prioritized replay buffer
        :return: list of experiences
        """

        # Check input parameters
        assert isinstance(beta, float) and beta > 0
        assert isinstance(batchSize, int) and 0 < batchSize <= len(self)

        # Sample indices
        indices = self._sampleProportional(batchSize)
        totalPriority = self._sumSegTree.sum()
        pMin = self._minSegTree.min() / totalPriority
        maxWeight = (pMin * len(self)) ** (-beta)

        # Calculate the weights
        weights = [(self._sumSegTree[idx] / totalPriority * len(self)) ** (-beta) / maxWeight for idx in indices]
        weights = np.array(weights, dtype=np.float32)

        # Get the samples
        samples = [self.buffer[idx] for idx in indices]
        return samples, indices, weights

    def updatePriorities(self, indices, priorities):
        """
        Update priorities of sampled transitions
        :param indices: list of sample indices
        :param priorities: list of sample priorities
        """
        # Check input parameters
        assert len(indices) == len(priorities)
        assert all(priority > 0 for priority in priorities)
        assert all(0 <= idx < len(self) for idx in indices)

        self._sumSegTree = [priority ** self._alpha if idx in indices else self._sumSegTree[idx] for idx, priority in enumerate(self._sumSegTree)]
        self._minSegTree = [priority ** self._alpha if idx in indices else self._minSegTree[idx] for idx, priority in enumerate(self._minSegTree)]
        self._maxPriority = max(self._maxPriority, max(priorities))
