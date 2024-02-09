#!/usr/bin/env python3
import json
import argparse
import pathlib
import numpy as np
from scipy.stats import ttest_1samp

from lib import data
from lib import models
from lib import environments

import torch

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

EPSILON = 0.02
RUNS = 100

# python3 src/run_model.py -d /home/derrick/data/daily_price_data/test/ -m valReward-22.174.data -n test-01
# python3 src/run_model.py -d /home/derrick/data/daily_price_data/test/FOX.csv -m valReward-22.174.data -n test-01
# python3 src/run_model.py -d results-test-01.json -o "plot" -n test1

def createPlots(data, name, significance=0.05):
    tStat, pValue = ttest_1samp(data, 0)

    print("Null hypothesis: the mean is equal to zero")
    print("Probability that the null hypothesis is true: %.4f" % pValue)
    print("Significance level: %.4f" % significance)
    print("p-value: %.4f, t-statistic: %.4f" % (pValue, tStat))
    print("Mean: %.8f" % np.mean(data))
    print("Standard deviation: %.4f" % np.std(data))

    with open("norm-performance-%s.txt" % name, "w") as file:
        file.write("Null hypothesis: the mean is equal to zero\n")
        file.write("Probability that the null hypothesis is true: %.4f\n" % pValue)
        file.write("Significance level: %.4f\n" % significance)
        file.write("p-value: %.4f, t-statistic: %.4f\n" % (pValue, tStat))
        file.write("Mean: %.8f\n" % np.mean(data))
        file.write("Standard deviation: %.4f\n" % np.std(data))

    if pValue < significance:
        print("The mean is statistically significantly different from zero.")
    else:
        print("The mean is not statistically significantly different from zero.")

    # Plot bar whisker plot of the normalized performance
    plt.clf()
    plt.hist(data, bins=20)
    plt.title("Normalized performance, data=%s" % name)
    plt.ylabel("Frequency")
    plt.xlabel("Normalized performance")
    plt.savefig("norm-performance-%s.png" % name)

    # Plot histogram of the normalized performance
    plt.clf()
    plt.boxplot(data)
    plt.title("Normalized performance, data=%s" % name)
    plt.ylabel("Normalized performance")
    plt.savefig("norm-performance-box-%s.png" % name)

def createPlotsFromFile(file, name):
    # Load the results
    with open(file, "r") as file:
        results = json.load(file)
    
    normPerformance = []
    for result in results:
        for run in result:
            normPerformance.append(run[4])
    print(normPerformance)

    # Plot histogram of the normalized performance
    createPlots(normPerformance, name)

def runTest(args, file):

    performance = []
    normPerformance = []

    # Load our data into the environment
    prices = data.loadRelative(file)
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
            observationVector = torch.tensor(np.array([obs]))
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
                # print("Run %d, final reward: %.2f" % (len(allRewards), totalReward))
                allRewards.append(rewards)
                break

    # Get mean position over time and 95% confidence interval ready for output to file
    meanReward = np.mean(allRewards, axis=0).tolist()
    upperCILimit = np.percentile(allRewards, 97.5, axis=0).tolist()
    lowerCILimit = np.percentile(allRewards, 2.5, axis=0).tolist()
    

    # Market Performance
    performance = meanReward[-1] / prices[-1] - 1
    normPerformance = performance/ stepIndex 

    return meanReward, upperCILimit, lowerCILimit, performance, normPerformance

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--op",   default="test", help="Operation to run")
    parser.add_argument("-d", "--data", required=True, help="CSV file with quotes to run the model")
    parser.add_argument("-m", "--model", help="Model file to load")
    parser.add_argument("-b", "--bars", type=int, default=50, help="Count of bars to feed into the model")
    parser.add_argument("-n", "--name", required=True, help="Name to use in outputs")
    parser.add_argument("--commission", type=float, default=0.0, help="Commission size in percent, default=0.1")
    args = parser.parse_args()

    dataPath = pathlib.Path(args.data)

    if args.op == "plot":
        createPlotsFromFile(args.data, args.name)
        exit(0)
    
    results = []

    if dataPath.is_file():
        # Import data from file to dictionary
        results.append([runTest(args, str(dataPath))])
    elif dataPath.is_dir():
        # Run test for each item in the data folder
        for file in dataPath.iterdir():
            if file.suffix == ".csv":
                results.append([runTest(args, str(file))])
    else:
        raise RuntimeError("No data to train on")

    # Save the results
    with open("results-%s.json" % args.name, "w") as file:
        json.dump(results, file)
    
