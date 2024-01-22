#!/usr/bin/env python3
import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
ITERATION_TYPE = "q" # "v" or "q"
#ENV_NAME = "FrozenLake8x8-v0"      # uncomment for larger version
GAMMA = 0.9
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        # My environment
        self.env = gym.make(ENV_NAME)
        # Current state
        self.state = self.env.reset()
        # Rewards lookup table
        self.rewards = collections.defaultdict(float)
        # Transitions lookup table
        self.transits = collections.defaultdict(collections.Counter)
        # Values lookup table
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        for _ in range(count):
            # Get a random action
            action = self.env.action_space.sample()
            # Get the new state, reward for the action
            new_state, reward, is_done, _ = self.env.step(action)
            # Update the transition-reward matrix
            self.rewards[(self.state, action, new_state)] = reward
            # Update the transition matrix
            self.transits[(self.state, action)][new_state] += 1
            # Update the current state (reset or new state)
            self.state = self.env.reset() if is_done else new_state

    # Only used in v-iteration
    def calc_action_value(self, state, action):
        # Get the counters for the state-action pair
        target_counts = self.transits[(state, action)]
        # Get the total number of transitions
        total = sum(target_counts.values())
        action_value = 0.0

        for tgt_state, count in target_counts.items():
            # Get the reward from the transition-reward matrix
            reward = self.rewards[(state, action, tgt_state)]
            # Calculate the bellman equation
            val = reward + GAMMA * self.values[tgt_state]
            # Calculate the action value
            action_value += (count / total) * val
        return action_value

    # Select the best action based on the state
    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            # Calculate action_Value based on type
            # Get the action value
            if ITERATION_TYPE == "v":
                action_value = self.calc_action_value(state, action)
            elif ITERATION_TYPE == "q":
                action_value = self.values[(state, action)]
            # Update the best action and value if necessary
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    # Go through an episode
    def play_episode(self, env):
        total_reward = 0.0
        # Get the initial state
        state = env.reset()
        while True:
            # Select the best action based on the state
            action = self.select_action(state)
            # Act on the environment
            new_state, reward, is_done, _ = env.step(action)
            # Log the reward and transition
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def value_v_iteration(self):
        for state in range(self.env.observation_space.n):
            # Add the action values to the values table
            state_values = [
                self.calc_action_value(state, action)
                for action in range(self.env.action_space.n)
            ]
            # Use the max action value as the state value
            self.values[state] = max(state_values)

    def value_q_iteration(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                # Get the counters for the state-action pair
                target_counts = self.transits[(state, action)]
                # Get the total number of transitions
                total = sum(target_counts.values())
                for tgt_state, count in target_counts.items():
                    key = (state, action, tgt_state)
                    # Get the reward from the transition-reward matrix
                    reward = self.rewards[key]
                    # Get the best action for the target state
                    best_action = self.select_action(tgt_state)
                    # Calculate the q-value
                    val = reward + GAMMA * self.values[(tgt_state, best_action)]
                    # Calculate the action value
                    action_value += (count / total) * val
                self.values[(state, action)] = action_value

if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    if ITERATION_TYPE == "v":
        writer = SummaryWriter(comment="-v-iteration")
    elif ITERATION_TYPE == "q":
        writer = SummaryWriter(comment="-q-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        # Play 100 random steps to fill the transition matrix
        agent.play_n_random_steps(100)
        # Update the values table
        if ITERATION_TYPE == "v":
            agent.value_v_iteration()
        elif ITERATION_TYPE == "q":
            agent.value_q_iteration()

        # Test the agent
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES

        # Log the reward
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
