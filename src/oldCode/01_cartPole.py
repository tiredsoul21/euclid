import gym

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    # Set up recording of environment
    env = gym.wrappers.Monitor(env, "recording", force=True)

    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    # Run random agent until end of episode
    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break

    # Print results
    print("Episode done in %d steps, total reward %.2f" % (
        total_steps, total_reward))
    
    # Close environment and recording
    env.close()
    env.env.close()

# ############################## Random action wrapper

# import gym
# from typing import TypeVar
# import random

# Action = TypeVar('Action')

# class RandomActionWrapper(gym.ActionWrapper):
#     # Constructor with random action probability (epsilon)
#     # Useful for exploration -- synonymous with action instability
#     def __init__(self, env, epsilon=0.1):
#         super(RandomActionWrapper, self).__init__(env)
#         self.epsilon = epsilon

#     # Override action method adding random action
#     def action(self, action: Action) -> Action:
#         if random.random() < self.epsilon:
#             print("Random!")
#             # Return random action
#             return self.env.action_space.sample()
#         return action


# if __name__ == "__main__":
#     env = RandomActionWrapper(gym.make("CartPole-v0"))

#     obs = env.reset()
#     total_reward = 0.0

#     # Always take action 0 by default, but sometimes acts unpredictably by design
#     while True:
#         obs, reward, done, _, _ = env.step(0)
#         total_reward += reward
#         if done:
#             break

#     # Print results
#     print("Reward got: %.2f" % total_reward)

# ############################## Pure random agent

# import gym

# if __name__ == "__main__":
#     # import the environment
#     env = gym.make("CartPole-v0")

#     # initialize variables
#     total_reward = 0.0
#     total_steps = 0
#     obs = env.reset()

#     # A random agent with no learning
#     while True:
#         # Make a random action
#         action = env.action_space.sample()
#         obs, reward, done, _, _  = env.step(action)
#         total_reward += reward
#         total_steps += 1
#         if done:
#             break

#     # Print results
#     print("Episode done in %d steps, total reward %.2f" % (
#         total_steps, total_reward))
