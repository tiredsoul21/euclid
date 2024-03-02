""" Test the stock model on a given dataset """
import json
import argparse
import pathlib
import numpy as np

import torch
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("Agg")

from lib.utils import dict_state_to_tensor
from lib import data
from lib import models
from lib.environments import StocksEnv, StockActions as Actions

P_MASS = True
RUNS = 100

# python3 src/test_stock_model.py -d ~/data/daily_price_data/test/ -m <model> -n test-01
# python3 src/test_stock_model.py -d ~/data/daily_price_data/test/FOX.csv -m <model> -n test-01
# python3 src/test_stock_model.py -d results-test-01.json -o "plot" -n test1

def create_plots(walk_data, det_data, name, significance=0.05):
    t_stat, p_val = ttest_1samp(data, 0)
    t_stat_det, p_val_det = ttest_1samp(det_data, 0)

    print("----------------P-Mass walk-------------")
    print("Null hypothesis: the mean is equal to zero")
    print("Probability that the null hypothesis is true: %.4f" % p_val)
    print("Significance level: %.4f" % significance)
    print("p-value: %.4f, t-statistic: %.4f" % (p_val, t_stat))
    print("Mean: %.8f" % np.mean(walk_data))
    print("Standard deviation: %.4f" % np.std(walk_data))

    print("--------------Deterministic-------------")
    print("Null hypothesis: the mean is equal to zero")
    print("Probability that the null hypothesis is true: %.4f" % p_val_det)
    print("Significance level: %.4f" % significance)
    print("p-value: %.4f, t-statistic: %.4f" % (p_val_det, t_stat_det))
    print("Mean: %.8f" % np.mean(det_data))
    print("Standard deviation: %.4f" % np.std(det_data))

    with open("norm-performance-%s.txt" % name, "w") as file:
        file.write("----------------P-Mass walk-------------\n")
        file.write("Null hypothesis: the mean is equal to zero\n")
        file.write("Probability that the null hypothesis is true: %.4f\n" % p_val)
        file.write("Significance level: %.4f\n" % significance)
        file.write("p-value: %.4f, t-statistic: %.4f\n" % (p_val, t_stat))
        file.write("Mean: %.8f\n" % np.mean(walk_data))
        file.write("Standard deviation: %.4f\n" % np.std(walk_data))
        file.write("----------------Deterministic-------------\n")
        file.write("Null hypothesis: the mean is equal to zero\n")
        file.write("Probability that the null hypothesis is true: %.4f\n" % p_val_det)
        file.write("Significance level: %.4f\n" % significance)
        file.write("p-value: %.4f, t-statistic: %.4f\n" % (p_val_det, t_stat_det))
        file.write("Mean: %.8f\n" % np.mean(det_data))
        file.write("Standard deviation: %.4f\n" % np.std(det_data))

    if p_val < significance:
        print("The mean is statistically significantly different from zero.")
    else:
        print("The mean is not statistically significantly different from zero.")

    # Plot histogram plot of the normalized performance
    plt.clf()
    plt.hist(walk_data, bins=20)
    plt.title("Normalized performance, data=%s" % name)
    plt.ylabel("Frequency")
    plt.xlabel("Normalized performance")
    plt.savefig("norm-performance-%s.png" % name)

    # Plot histogram plot of the deterministic normalized performance
    plt.clf()
    plt.hist(det_data, bins=20)
    plt.title("Deterministic normalized performance, data=%s" % name)
    plt.ylabel("Frequency")
    plt.xlabel("Normalized performance")
    plt.savefig("norm-det-performance-%s.png" % name)

    # Plot bar whisker of the normalized performance and deterministic in the same plot
    plt.clf()
    positions = [1, 2]
    plt.boxplot(walk_data, positions=[positions[0]], widths=0.6)
    plt.boxplot(det_data, positions=[positions[1]], widths=0.6)
    plt.title("Normalized performance")
    plt.ylabel("Normalized performance")
    plt.xlabel("Walks")
    plt.xticks(positions, ["P-Mass", "Deterministic"])
    plt.savefig("norm-performance-box-%s.png" % name)
    
def create_plots_from_file(file, name):
    # Load the results
    with open(file, "r") as file:
        results = json.load(file)
    
    norm_perfomance = []
    det_norm_perfomance = []
    for result in results:
        for run in result:
            norm_perfomance.append(run[4])
            det_norm_perfomance.append(run[6])

    # Plot histogram of the normalized performance
    print(norm_perfomance)
    create_plots(norm_perfomance, det_norm_perfomance, name)

def run_test(args, file):

    performance = []
    norm_perfomance = []

    # Load our data into the environment
    prices = data.load_relative(file)
    env = StocksEnv({"TEST": prices},
                    bar_count=args.bars,
                    reset_on_close=False,
                    commission=args.commission,
                    random_offset=False,
                    reward_on_close=False,
                    volumes=False)

    # Load our model
    net = models.DQNConv2D(env.state_shape(), env.action_space.n)
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))

    # Iitialize Observaions
    obs = env.reset()
    obsWalks = [obs for _ in range(RUNS)]
    obsWalks = dict_state_to_tensor(obsWalks)

    # Initialize start prices
    detHasPosition = False
    detStartPrice = env._state._current_close()
    detPosition = [detStartPrice]
    walkHasPosition = [False for _ in range(RUNS)]
    walkStartPrice = [detStartPrice for _ in range(RUNS)]
    walkPosition = [[detStartPrice for _ in range(RUNS)]]

    # Initialize everything else
    stepIndex = 0
    prices = []
    
    # Run the model
    # print(detStartPrice)
    while True:
        stepIndex += 1
        closePrice = env._state._current_close()

        # Get the actions
        walkOutput = net(obsWalks)

        # Get the action by probability mass
        walkActionIndex = torch.distributions.Categorical(torch.nn.functional.softmax(walkOutput, dim=1)).sample().tolist()
        walkActionIndex = [Actions(actionIndex) for actionIndex in walkActionIndex]

        # Calculate the reward multiplier 
        walkRewardMultiplier = [closePrice / startPrice if hasPosition else 1.0 for hasPosition, startPrice in zip(walkHasPosition, walkStartPrice)]

        # Update the start price if we buy or keep the same price if we have a position
        walkStartPrice = [startPrice if hasPosition else closePrice for hasPosition, startPrice in zip(walkHasPosition, walkStartPrice)]

        # Update the position
        walkPosition.append([startPrice * rewardMultiplier - args.commission if actionIndex == Actions.SELL and hasPosition
                                                                           else position
                                                                           for startPrice, position, actionIndex, rewardMultiplier, hasPosition
                                                                           in zip(walkStartPrice, walkPosition[-1], walkActionIndex, walkRewardMultiplier, walkHasPosition)])
        walkHasPosition = [actionIndex == Actions.BUY and not hasPosition
                           or actionIndex == Actions.SELL and hasPosition
                           for actionIndex, hasPosition in zip(walkActionIndex, walkHasPosition)]

        # Same as above but for deterministic for one run
        detOutput = net(dict_state_to_tensor([obs]))
        detActionIndex = Actions(torch.argmax(torch.nn.functional.softmax(detOutput, dim=1)).item())
        detRewardMultiplier = closePrice / detStartPrice if detHasPosition else 1.0

        # # print if we sold
        # if detActionIndex == Actions.SELL and detHasPosition:
        #     print("Sold at %.14f, step %d, multiplier %.14f, openPrice %.14f" % (closePrice, stepIndex, detRewardMultiplier, detStartPrice))

        detStartPrice = detStartPrice if detHasPosition else closePrice

        # Update position only if a buy or sell action is taken
        detPosition.append(detStartPrice * detRewardMultiplier - args.commission if detActionIndex == Actions.SELL and detHasPosition else detPosition[-1])
        # Update position flag
        detHasPosition = detActionIndex == Actions.BUY or (detActionIndex == Actions.SELL and detHasPosition)

        # Update the prices
        prices.append(closePrice)

        obs, _, done, _, _ = env.step(0)
        obsWalks = dict_state_to_tensor([obs for _ in range(RUNS)])

        if done:
            break

    # Get mean position over time and 95% confidence interval ready for output to file
    meanReward = [np.percentile(walk, 50) for walk in walkPosition]
    upperCILimit = [np.percentile(walk, 97.5) for walk in walkPosition]
    lowerCILimit = [np.percentile(walk, 2.5) for walk in walkPosition]

    # Market Performance
    performance = meanReward[-1] / prices[-1] - 1
    norm_perfomance = performance/ stepIndex
    detPerformance = detPosition[-1] / prices[-1] - 1
    det_norm_perfomance = detPerformance / stepIndex

    #plot price and mean reward and confidence interval
    file_name = file.split("/")[-1].split(".")[0]
    plt.clf()
    plt.plot(prices, label="Price")
    plt.plot(meanReward, label="Mean Reward")
    plt.plot(detPosition, label="Deterministic Run")
    plt.fill_between(range(len(meanReward)), lowerCILimit, upperCILimit, color='gray', alpha=0.5)
    plt.title(file)
    plt.legend()
    plt.savefig("price-%s-%s.png" % (file_name, args.name))

    return meanReward, upperCILimit, lowerCILimit, performance, norm_perfomance, detPerformance, det_norm_perfomance

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
        create_plots_from_file(args.data, args.name)
        exit(0)
    
    results = []

    if dataPath.is_file():
        # Import data from file to dictionary
        results.append([run_test(args, str(dataPath))])
    elif dataPath.is_dir():
        # Run test for each item in the data folder
        for file in dataPath.iterdir():
            if file.suffix == ".csv":
                results.append([run_test(args, str(file))])
    else:
        raise RuntimeError("No data to train on")

    # Save the results
    with open("results-%s.json" % args.name, "w") as file:
        json.dump(results, file)
    
