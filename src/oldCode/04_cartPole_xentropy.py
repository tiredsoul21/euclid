#!/usr/bin/env python3
import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            # Variable hidden layer with ReLU activation function
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            # Output layer with n_actions outputs raw scores -- must be converted to probabilities
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
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)

        # If episode is done, log episode and reset
        if is_done:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    # Get the rewards, XX% and 50% percentile
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    # Take the top XX% of the batch & add to training data
    train_obs = []
    train_act = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))

    # Convert to tensors and return
    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    env = gym.wrappers.Monitor(env, directory="mon", force=True)

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Create network
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    # Create loss function and optimizer
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)

    # Create tensorboard writer
    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        # Run batch
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        # Clear gradients
        optimizer.zero_grad()
        # Get output from network
        action_scores_v = net(obs_v)
        # Calculate loss
        loss_v = objective(action_scores_v, acts_v)
        # Gradient & train
        loss_v.backward()
        optimizer.step()

        # Log results
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 499:
            print("Solved!")
            break
    writer.close()
