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
RUNS = 100

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

    allRewards = []

    for _ in range(RUNS):
        # Iitialize everything
        obs = env.reset()
        startPrice = env._state._currentClose()
        totalReward = startPrice # Starting with 1 share
        stepIndex = 0
        rewards = []
        prices = []

        while True:
            stepIndex += 1
            observationVector = torch.tensor([obs])
            outputVector = net(observationVector)
            actionIndex = outputVector.max(dim=1)[1].item()
            if np.random.random() < EPSILON:
                actionIndex = env.action_space.sample()
            action = environments.Actions(actionIndex)

            hasPosition = observationVector[0][3][1] > 0

            rewardMultiplier = 1.0
            if hasPosition:
                rewardMultiplier = env._state._currentClose() / startPrice
            if action == environments.Actions.Buy and not hasPosition:
                startPrice = env._state._currentClose()

            rewards.append(totalReward * rewardMultiplier)
            prices.append(env._state._currentClose())

            if action == environments.Actions.Close and hasPosition:
                totalReward *= rewardMultiplier
                totalReward -= args.commission
                totalReward -= args.commission
                startPrice = 0.0

            obs, reward, done, _, _ = env.step(actionIndex) 
                
            if done:
                print("Run %d, final reward: %.2f" % (len(allRewards), totalReward))
                allRewards.append(rewards)
                break

    # Get mean position over time for all runs
    meanReward = np.mean(allRewards, axis=0)
    upperCILimit = np.percentile(allRewards, 95, axis=0)
    lowerCILimit = np.percentile(allRewards, 5, axis=0)

    # Market Performance
    performance = meanReward[-1] / prices[-1] * 100

    # Generate the plot
    plt.clf()
    plt.plot(meanReward, label=" Mean Position")
    plt.fill_between(range(len(meanReward)), lowerCILimit, upperCILimit, color="blue", alpha=0.3)
    plt.plot(prices, label="Price")
    plt.title("Total reward, data=%s" % args.name)
    plt.ylabel("Reward, %")

    # Add the market performance
    plt.text(len(meanReward) - 1, meanReward[-1], "Market\n%.2f" % performance, ha="center", va="bottom")

    plt.legend()
    plt.savefig("rewards-%s.png" % args.name)
    

