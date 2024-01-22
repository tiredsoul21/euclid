#!/usr/bin/env python3
import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        # My environment
        self.env = gym.make(ENV_NAME)
        # Current state
        self.state = self.env.reset()
        # Values lookup table
        self.values = collections.defaultdict(float)

    def sample_env(self):
        # Get a random action
        action = self.env.action_space.sample()
        # Get the new state, reward for the action
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        # Update the current state (reset or new state)
        self.state = self.env.reset() if is_done else new_state
        return old_state, action, reward, new_state

    def best_value_and_action(self, state):
        best_value, best_action = None, None
        # Get the best action based on the state/action value
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, s, a, r, next_s):
        # Get the best action/value pair for the next state
        best_v, _ = self.best_value_and_action(next_s)
        # Calculate the new value based v-iteration
        new_v = r + GAMMA * best_v
        old_v = self.values[(s, a)]
        # Update the state/action value with moving average
        self.values[(s, a)] = old_v * (1-ALPHA) + new_v * ALPHA

    # Run an episode and return the total reward
    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            # Get the best action/value pair for the state
            _, action = self.best_value_and_action(state)
            # Get the new state, reward for the action
            new_state, reward, is_done, _ = env.step(action)
            # Update the total reward
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1

        # Sample the environment once
        s, a, r, next_s = agent.sample_env()
        # Update the state/action value
        agent.value_update(s, a, r, next_s)

        # Test the agent every 1000 iterations
        if iter_no % 1000 == 0:
            reward = 0.0
            for _ in range(TEST_EPISODES):
                reward += agent.play_episode(test_env)
            reward /= TEST_EPISODES
            writer.add_scalar("reward", reward, iter_no)
            if reward > best_reward:
                print("Best reward updated %.3f -> %.3f" % (
                    best_reward, reward))
                best_reward = reward
            if reward > 0.80:
                print("Solved in %d iterations!" % iter_no)
                break
    writer.close()
