import random
from typing import List


class Environment:
    # Constructor
    def __init__(self):
        self.steps_left = 10

    # Returns state of the environment (observations)
    def get_observation(self) -> List[float]:
        return [0.0, 0.0, 0.0]

    # Returns list of possible actions in this state
    def get_actions(self) -> List[int]:
        return [0, 1]

    # Returns if end of episode
    def is_done(self) -> bool:
        return self.steps_left == 0

    # Returns reward for action (currently random -- not implemented)
    def action(self, action: int) -> float:
        if self.is_done():
            raise Exception("Game is over")
        self.steps_left -= 1
        return random.random()


class Agent:
    # Constructor
    def __init__(self):
        self.total_reward = 0.0

    # Performs one step of the agent (currently disregards observation)
    def step(self, env: Environment):
        # Get observations about the state of the environment
        current_obs = env.get_observation()
        # Get list of possible actions
        actions = env.get_actions()
        # Perform action and get reward
        reward = env.action(random.choice(actions))
        self.total_reward += reward


if __name__ == "__main__":
    # Create environment and agent
    env = Environment()
    agent = Agent()

    # Run agent in environment until end of episode
    while not env.is_done():
        agent.step(env)

    # Print total reward from episode
    print("Total reward got: %.4f" % agent.total_reward)
