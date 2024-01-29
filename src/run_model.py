#!/usr/bin/env python3
import argparse
import numpy as np

from lib import data
from lib import models
from lib import environments

import torch

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

EPSILON = 0.02

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True, help="CSV file with quotes to run the model")
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-b", "--bars", type=int, default=50, help="Count of bars to feed into the model")
    parser.add_argument("-n", "--name", required=True, help="Name to use in output images")
    parser.add_argument("--commission", type=float, default=0.0, help="Commission size in percent, default=0.1")
    parser.add_argument("--conv", default=False, action="store_true", help="Use convolution model instead of FF")
    args = parser.parse_args()

    # Load our data into the environment
    prices = data.loadRelative(args.data)
    env = environments.StocksEnv({"TEST": prices},
                                 barCount=args.bars,
                                 resetOnClose=False,
                                 commission=args.commission,
                                 randomOffset=False,
                                 rewardOnClose=False,
                                 volumes=False)

    # Load our model
    net = models.DQNConv1D(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))

    # Iitialize everything
    obs = env.reset()
    startPrice = env._state._currentClose()
    totalReward = 0.0
    stepIndex = 0
    rewards = []

    while True:
        stepIndex += 1
        observationVector = torch.tensor([obs])
        outputVector = net(observationVector)
        actionIndex = outputVector.max(dim=1)[1].item()
        if np.random.random() < EPSILON:
            actionIndex = env.action_space.sample()
        action = environments.Actions(actionIndex)

        obs, reward, done, _, _ = env.step(actionIndex)
        totalReward += reward
        rewards.append(totalReward)
        if stepIndex % 100 == 0:
            print("%d: reward=%.3f" % (stepIndex, totalReward))
        if done:
            break

    # Generate the plot
    plt.clf()
    plt.plot(rewards)
    plt.title("Total reward, data=%s" % args.name)
    plt.ylabel("Reward, %")
    plt.savefig("rewards-%s.png" % args.name)
