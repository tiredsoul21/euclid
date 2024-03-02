#!/usr/bin/env python3
""" This script performs training of the stock model """
import pathlib
import argparse
import numpy as np

import gym.wrappers

from torch import device as hardware, save, optim

from ignite.engine import Engine, Events
from ignite.contrib.handlers import tensorboard_logger as tb_logger

from lib import data
from lib import common
from lib import models
from lib import agents
from lib import actions
from lib import validation
from lib import experiences
from lib import environments
from lib import ignite as local_ignite
from lib.utils import dict_state_to_tensor

# python3 src/train_stock_model.py -p /home/derrick/data/daily_price_data -r test --cuda
# python3 src/train_stock_model.py -p /home/derrick/data/daily_price_data/other -r test --cuda

SAVES_DIR = pathlib.Path("output")

# How many bars to feed into the model
BAR_COUNT = 50

BATCH_SIZE = 64

# EPSILON GREEDY - for exploration
EPS_START = 1.0
# Final value of epsilon
EPS_FINAL = 0.1
# How many steps to decay the epsilon to EPS_FINAL
EPS_STEPS = 1000000

# Learning rate for neural network training
LEARNING_RATE = 0.0005
# Discount factor for future rewards
GAMMA = 0.99

# How far in the future to look for rewards
REWARD_STEPS = 2

# Size of the experience replay buffer
REPLAY_SIZE = 500000
REPLAY_INITIAL = 10000

# How often to run validation and sync the target network
VALIDATION_INTERVAL = 1000
TARGETNET_SYNC_INTERNVAL = 1000

# Number of states to evaluate when syncing the target network
STATES_TO_EVALUATE = 3000

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(      "--cuda", default=False, help="Enable cuda",  action="store_true")
    parser.add_argument("-p", "--path", required=True, help="Directory or file of price data")
    parser.add_argument("-v", "--val",                 help="Validation data, default=path/val/")
    parser.add_argument("-t", "--test",                help="Test data, default=path/test/")
    parser.add_argument("-r", "--run",  required=True, help="Run name")
    args = parser.parse_args()
    device = hardware("cuda" if args.cuda else "cpu")

    # Create output directory
    savesPath = SAVES_DIR / f"{args.run}"
    savesPath.mkdir(parents=True, exist_ok=True)

    # Set data paths
    data_path = pathlib.Path(args.path)
    dataFolder = data_path

    # If data_path is a file, use fetch containing directory
    if data_path.is_file():
        dataFolder = data_path.parent

    # Set validation path
    if args.val is None:
        valPath = dataFolder / "val"
    else:
        valPath = pathlib.Path(args.val)

    # Set test path
    if args.test is None:
        testPath = dataFolder / "test"
    else:
        testPath = pathlib.Path(args.test)

    # Create Environment
    if data_path.is_file():
        # Import data from file to dictionary
        index = data_path.stem
        price_data = {index: data.read_csv(str(data_path), sep=',', fix_open_price = True) }
        env = environments.StocksEnv(price_data, bar_count=BAR_COUNT)
    elif data_path.is_dir():
        env = environments.StocksEnv.from_directory(data_path, bar_count=BAR_COUNT, sep=',',
                                                   fix_open_price = True)
    else:
        raise RuntimeError("No data to train on")

    env.use_moving_avg()

    # Create validation environmentstart:stop
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    envTest = environments.StocksEnv.from_directory(testPath, bar_count=BAR_COUNT)
    envVal = environments.StocksEnv.from_directory(valPath, bar_count=BAR_COUNT)

    # Create the networks
    net = models.DQNConv2D(env.state_shape(), env.action_space.n).to(device)
    target_net = models.TargetNet(net)

    # Create the action selector
    selector = actions.EpsilonGreedyActionSelector(EPS_START)
    epsilon_tracker = actions.EpsilonTracker(selector, EPS_START, EPS_FINAL, EPS_STEPS)

    # Create the agent
    agent = agents.DQNAgent(net, selector, device=device, preprocessor=dict_state_to_tensor)

    # Create the experience source
    exp_source = experiences.ExperienceSourceFirstLast(env, agent, GAMMA, step_count=REWARD_STEPS)
    buffer = experiences.ExperienceReplayBuffer(exp_source, REPLAY_SIZE)

    # Create the optimizer
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    def process_batch(engine, batch):
        """
        Process a batch of data
        :param engine: engine to process batch
        :param batch: batch to process
        :return: loss and epsilon
        """

        # Zero the gradients
        optimizer.zero_grad()

        # Calculate the loss
        validation_loss = common.calculate_loss(batch, net, target_net.target_model, 
                                                gamma=GAMMA ** REWARD_STEPS, device=device)

        # Backpropagate the loss
        validation_loss.backward()

        # Update the weights
        optimizer.step()

        # Update the epsilon
        epsilon_tracker.frame(engine.state.iteration)

        # If eval_states is not set...
        if getattr(engine.state, "eval_states", None) is None:
            # Get a sample of states to evaluate
            eval_states = buffer.sample(STATES_TO_EVALUATE)
            eval_states = [transition.state for transition in eval_states]
            engine.state.eval_states = np.array(eval_states, copy=False)

        # Return the loss and epsilon
        return {
            "loss": validation_loss.item(),
            "epsilon": selector.epsilon,
        }

    # Create the engine to process the batch
    engine = Engine(process_batch)

    # Attach the tensorboard logger
    tb = common.setup_ignite(engine, exp_source, f"{args.run}", extra_metrics=('MeanValue',))

    # Set the TargetNet Sync engine
    @engine.on(Events.ITERATION_COMPLETED)
    def sync_eval(engine: Engine):
        """Sync the target_net with the net"""
        # Run every TARGETNET_SYNC_INTERNVAL iterations (Default: 1000)
        if engine.state.iteration % TARGETNET_SYNC_INTERNVAL == 0:
            target_net.sync()

            # Calculate the mean value of the states
            mean_value = common.calculate_states_values(engine.state.eval_states, \
                                                        net, device=device)
            engine.state.metrics["MeanValue"] = mean_value

            # If bestMeanValue is not set set it to mean_value
            if getattr(engine.state, "bestMeanValue", None) is None:
                engine.state.bestMeanValue = mean_value

            # If mean_value is greater than bestMeanValue save the model
            if engine.state.bestMeanValue < mean_value:
                print(f"{engine.state.iteration}: Best mean value updated \
                      {engine.state.bestMeanValue:.3f} -> {mean_value:.3f}")
                path = savesPath / ("mean_value-{mean_value:.3f}.data")
                save(net.state_dict(), path)
                engine.state.bestMeanValue = mean_value

    # Set the validation engine
    @engine.on(Events.ITERATION_COMPLETED)
    def validate(engine: Engine):
        """Runs validation periodically."""
        # Run every VALIDATION_INTERVAL iterations (Default: 10000)
        if engine.state.iteration % VALIDATION_INTERVAL == 0:
            # Test: Get/print the mean: reward, steps, order profits, order steps
            res = validation.validation_run(envTest, net, device=device)
            print(f"{engine.state.iteration}: tst: {res}")
            # Add the metrics to the engine
            for key, val in res.items():
                engine.state.metrics[key + "_tst"] = val

            # Val: Get/print the mean: reward, steps, order profits, order steps
            res = validation.validation_run(envVal, net, device=device)
            print(f"{engine.state.iteration}: val: {res}")
            # Add the metrics to the engine
            for key, val in res.items():
                engine.state.metrics[key + "_val"] = val

            # If bestValReward is not set set it to mean episode reward
            val_reward = res['episodeReward']
            if getattr(engine.state, "bestValReward", None) is None:
                engine.state.bestValReward = val_reward

            # If val_reward is greater than bestValReward save the model
            if engine.state.bestValReward < val_reward:
                print(f"Best validation reward updated: \
                      {engine.state.bestValReward:.3f} -> {val_reward:.3f}, \
                        model saved")
                engine.state.bestValReward = val_reward
                path = savesPath / f"val_reward-{val_reward:.3f}.data"
                save(net.state_dict(), path)

    # Log event and metrics
    event = local_ignite.PeriodEvents.ITERS_10000_COMPLETED

    # Create the logger for the test data
    metrics = [metric + "_tst" for metric in validation.METRICS]
    tst_handler = tb_logger.OutputHandler(tag="test", metric_names=metrics)
    tb.attach(engine, log_handler=tst_handler, event_name=event)

    # Create the logger for the validation data
    metrics = [metric + "_val" for metric in validation.METRICS]
    val_handler = tb_logger.OutputHandler(tag="validation", metric_names=metrics)
    tb.attach(engine, log_handler=val_handler, event_name=event)

    # Run the engine
    engine.run(common.batch_generator(buffer, REPLAY_INITIAL, BATCH_SIZE))
