#!/usr/bin/env python3
import random
import gym
import gym.spaces
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_SIZE = 128
BATCH_SIZE = 100
PERCENTILE = 30
GAMMA = 0.9

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)

        # Convert obs to be 0/1 but float
        shape = (env.observation_space.n, )
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape, dtype=np.float32)

    # Convert from int to one-hot at index
    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res

# Same NN as 04 cartPole
class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            # hidden layer of variable size & ReLU activation function
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            # output layer with n_actions outputs raw scores -- must be converted to probabilities
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)

def iterate_batches(env, net, batch_size):
    # initialize
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    # Softmax for probabilities -- used only locally
    sm = nn.Softmax(dim=1)

    # Generate batches of episodes
    while True:
        # Convert observation to tensor
        obs_v = torch.FloatTensor([obs])
        # Get probabilities from network
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]

        # Act wrt probabilities
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)

        # Log episode steps
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))

        # If episode is done, log episode and reset
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    # Discount to deprioritize long runs
    discount = lambda s: s.reward * (GAMMA ** len(s.steps))
    # Generate reward for XX percentile
    rewards = list(map(discount, batch))
    reward_bound = np.percentile(rewards, percentile)

    # Filter out episodes with reward < XX percentile
    train_obs = []
    train_act = []
    elite_batch = []
    for example, discounted_reward in zip(batch, rewards):
        if discounted_reward > reward_bound:
            train_obs.extend(map(lambda step: step.observation,
                                 example.steps))
            train_act.extend(map(lambda step: step.action,
                                 example.steps))
            elite_batch.append(example)

    return elite_batch, train_obs, train_act, reward_bound


if __name__ == "__main__":
    random.seed(12345)
    env = DiscreteOneHotWrapper(gym.make("FrozenLake-v0"))
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Create the NN
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    # Create the loss function and optimizer
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)

    # Create the writer
    writer = SummaryWriter(comment="-frozenlake")

    full_batch = []
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))
        # Run batch
        full_batch, obs, acts, reward_bound = filter_batch(full_batch + batch, PERCENTILE)
        if not full_batch:
            continue
        obs_v = torch.FloatTensor(obs)
        acts_v = torch.LongTensor(acts)
        full_batch = full_batch[-500:]

        # Clear gradients
        optimizer.zero_grad()
        # Get output from network
        action_scores_v = net(obs_v)
        # Calculate loss
        loss_v = objective(action_scores_v, acts_v)
        # Gradient & train
        loss_v.backward()
        optimizer.step()

        # Log the results
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            iter_no, loss_v.item(), reward_mean, reward_bound))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_bound, iter_no)
        writer.add_scalar("reward_mean", reward_mean, iter_no)
        if reward_mean > 0.8:
            print("Solved!")
            break
    writer.close()
