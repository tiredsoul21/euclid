""" Test the stock model on a given dataset """
import sys
import argparse
import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")

import torch
from scipy.stats import ttest_1samp

from ..lib import data
from ..lib.environments import StocksEnv, StockActions as Actions
from ..lib import models
from ..lib.utils import dict_state_to_tensor

P_MASS = True
RUNS = 100

# python3 -m src.rl_stock_model.test -d ~/data/daily_price_data/test/ -m <model> -n test-01
# python3 -m src.rl_stock_model.test -d ~/data/daily_price_data/test/FOX.csv -m <model> -n test-01
# python3 -m src.rl_stock_model.test -d results-test-01.json -o "plot" -n test1

def create_plots(walk_data, det_data, name, significance=0.05):
    """ Create plots and run t-test on the normalized performance """
    t_stat, p_val = ttest_1samp(data, 0)
    t_stat_det, p_val_det = ttest_1samp(det_data, 0)

    print("----------------P-Mass walk-------------")
    print("Null hypothesis: the mean is equal to zero")
    print(f"Probability that the null hypothesis is true: {p_val:.4f}")
    print(f"Significance level: {significance:.4f}")
    print(f"p-value: {p_val:.4f}, t-statistic: {t_stat:.4f}")
    print(f"Mean: {np.mean(walk_data):.8f}")
    print(f"Standard deviation: {np.std(walk_data):.4f}")

    print("--------------Deterministic-------------")
    print("Null hypothesis: the mean is equal to zero")
    print(f"Probability that the null hypothesis is true: {p_val_det:.4f}")
    print(f"Significance level: {significance:.4f}")
    print(f"p-value: {p_val_det:.4f}, t-statistic: {t_stat_det:.4f}")
    print(f"Mean: {np.mean(det_data):.8f}")
    print(f"Standard deviation: {np.std(det_data):.4f}")


    with open(f"norm-performance-{name}.txt", "w", encoding="utf-8") as fout:
        fout.write("----------------P-Mass walk-------------\n")
        fout.write("Null hypothesis: the mean is equal to zero\n")
        fout.write(f"Probability that the null hypothesis is true: {p_val:.4f}\n")
        fout.write(f"Significance level: {significance:.4f}\n")
        fout.write(f"p-value: {p_val:.4f}, t-statistic: {t_stat:.4f}\n")
        fout.write(f"Mean: {np.mean(walk_data):.8f}\n")
        fout.write(f"Standard deviation: {np.std(walk_data):.4f}\n")

        fout.write("----------------Deterministic-------------\n")
        fout.write("Null hypothesis: the mean is equal to zero\n")
        fout.write(f"Probability that the null hypothesis is true: {p_val_det:.4f}\n")
        fout.write(f"Significance level: {significance:.4f}\n")
        fout.write(f"p-value: {p_val_det:.4f}, t-statistic: {t_stat_det:.4f}\n")
        fout.write(f"Mean: {np.mean(det_data):.8f}\n")
        fout.write(f"Standard deviation: {np.std(det_data):.4f}\n")

    if p_val < significance:
        print("The mean is statistically significantly different from zero.")
    else:
        print("The mean is not statistically significantly different from zero.")

    # Plot histogram plot of the normalized performance
    plt.clf()
    plt.hist(walk_data, bins=20)
    plt.title(f"Normalized performance, data=%{name}")
    plt.ylabel("Frequency")
    plt.xlabel("Normalized performance")
    plt.savefig(f"norm-performance-{name}.png")

    # Plot histogram plot of the deterministic normalized performance
    plt.clf()
    plt.hist(det_data, bins=20)
    plt.title(f"Deterministic normalized performance, data={name}")
    plt.ylabel("Frequency")
    plt.xlabel("Normalized performance")
    plt.savefig(f"norm-det-performance-{name}.png")

    # Plot bar whisker of the normalized performance and deterministic in the same plot
    plt.clf()
    positions = [1, 2]
    plt.boxplot(walk_data, positions=[positions[0]], widths=0.6)
    plt.boxplot(det_data, positions=[positions[1]], widths=0.6)
    plt.title("Normalized performance")
    plt.ylabel("Normalized performance")
    plt.xlabel("Walks")
    plt.xticks(positions, ["P-Mass", "Deterministic"])
    plt.savefig(f"norm-performance-box-{name}.png")

def create_plots_from_file(data_file, name):
    """ Create plots from a file """
    # Load the results
    with open(data_file, "r", encoding="utf-8") as fin:
        results_array = json.load(fin)

    norm_perfomance = []
    det_norm_perfomance = []
    for result in results_array:
        for run in result:
            norm_perfomance.append(run[4])
            det_norm_perfomance.append(run[6])

    # Plot histogram of the normalized performance
    print(norm_perfomance)
    create_plots(norm_perfomance, det_norm_perfomance, name)

def run_test(kargs, fin):
    """ Run the test on the given file """
    performance = []
    norm_perfomance = []

    # Load our data into the environment
    prices = data.load_relative(fin)
    env = StocksEnv({"TEST": prices},
                    bar_count=kargs.bars,
                    reset_on_close=False,
                    commission=kargs.commission,
                    random_offset=False,
                    reward_on_close=False,
                    volumes=False)

    # Load our model
    net = models.DQNConv2D(env.state_shape(), env.action_space.n)
    net.load_state_dict(torch.load(kargs.model, map_location=lambda storage, loc: storage))

    # Iitialize Observaions
    obs = env.reset()
    obs_walks = [obs for _ in range(RUNS)]
    obs_walks = dict_state_to_tensor(obs_walks)

    # Initialize start prices
    det_has_position = False
    det_start_price = env.get_close()
    det_position = [det_start_price]
    walk_has_position = [False for _ in range(RUNS)]
    walk_start_price = [det_start_price for _ in range(RUNS)]
    walk_position = [[det_start_price for _ in range(RUNS)]]

    # Initialize everything else
    step_index = 0
    prices = []

    # Run the model
    # print(det_start_price)
    while True:
        step_index += 1
        close_price = env.get_close()

        # Get the actions
        walk_output = net(obs_walks)

        # Get the action by probability mass
        walk_action_idx = torch.distributions.Categorical(
            torch.nn.functional.softmax(walk_output, dim=1)).sample().tolist()
        walk_action_idx = [Actions(action_idx) for action_idx in walk_action_idx]

        # Calculate the reward multiplier
        walk_reward_mult = [close_price / start_price if has_pos else 1.0
                            for has_pos, start_price in zip(walk_has_position, walk_start_price)]

        # Update the start price if we buy or keep the same price if we have a position
        walk_start_price = [start_price if has_pos else close_price
                            for has_pos, start_price in zip(walk_has_position, walk_start_price)]

        # Update the position
        walk_position.append([start_price * rewardMultiplier - kargs.commission
                              if action_idx == Actions.SELL and has_pos
                              else position
                              for start_price, position, action_idx, rewardMultiplier, has_pos
                              in zip(walk_start_price, walk_position[-1],
                                     walk_action_idx, walk_reward_mult, walk_has_position)])

        walk_has_position = [action_idx == Actions.BUY and not has_pos
                           or action_idx == Actions.SELL and has_pos
                           for action_idx, has_pos in zip(walk_action_idx, walk_has_position)]

        # Same as above but for deterministic for one run
        det_output = net(dict_state_to_tensor([obs]))
        det_action_idx = Actions(torch.argmax(
            torch.nn.functional.softmax(det_output, dim=1)).item())
        det_reward_mult = close_price / det_start_price if det_has_position else 1.0

        det_start_price = det_start_price if det_has_position else close_price

        # Update position only if a buy or sell action is taken
        det_position.append(det_start_price * det_reward_mult - kargs.commission
                            if det_action_idx == Actions.SELL and det_has_position
                            else det_position[-1])

        # Update position flag
        det_has_position = (det_action_idx == Actions.BUY or
                           (det_action_idx == Actions.SELL and det_has_position))

        # Update the prices
        prices.append(close_price)

        obs, _, done, _, _ = env.step(0)
        obs_walks = dict_state_to_tensor([obs for _ in range(RUNS)])

        if done:
            break

    # Get mean position over time and 95% confidence interval ready for output to file
    mean_reward = [np.percentile(walk, 50) for walk in walk_position]
    upper_conf = [np.percentile(walk, 97.5) for walk in walk_position]
    lower_conf = [np.percentile(walk, 2.5) for walk in walk_position]

    # Market Performance
    performance = mean_reward[-1] / prices[-1] - 1
    norm_perfomance = performance/ step_index
    det_performance = det_position[-1] / prices[-1] - 1
    det_norm_perfomance = det_performance / step_index

    #plot price and mean reward and confidence interval
    file_name = fin.split("/")[-1].split(".")[0]
    plt.clf()
    plt.plot(prices, label="Price")
    plt.plot(mean_reward, label="Mean Reward")
    plt.plot(det_position, label="Deterministic Run")
    plt.fill_between(range(len(mean_reward)), lower_conf, upper_conf, color='gray', alpha=0.5)
    plt.title(file_name)
    plt.legend()
    plt.savefig(f"price-{file_name}-{kargs.name}.png")

    return mean_reward, upper_conf, lower_conf, performance, \
           norm_perfomance, det_performance, det_norm_perfomance

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--op",   default="test", help="Operation to run")
    parser.add_argument("-d", "--data", required=True, help="CSV file with quotes to run the model")
    parser.add_argument("-m", "--model", help="Model file to load")
    parser.add_argument("-b", "--bars", type=int, default=50, help="Model bar count, default=50")
    parser.add_argument("-n", "--name", required=True, help="Name to use in outputs")
    parser.add_argument("--commission", type=float, default=0.0, help="Commission size, default=0")
    args = parser.parse_args()

    dataPath = pathlib.Path(args.data)

    if args.op == "plot":
        create_plots_from_file(args.data, args.name)
        sys.exit(0)

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
    with open(f"results-{args.name}.json", "w", encoding="utf-8") as fout_json:
        json.dump(results, fout_json)
        fout_json.close()
