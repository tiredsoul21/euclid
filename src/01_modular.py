#!/usr/bin/env python3
import pathlib
import argparse
import numpy as np

import gym.wrappers

import torch
import torch.optim as optim

from ignite.engine import Engine
from ignite.engine import Events
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
from lib.utils import dictionaryStateToTensor

# python3 src/01_modular.py -p /home/derrick/data/daily_price_data -r test --cuda

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
LEARNING_RATE = 0.0001
# Discount factor for future rewards
GAMMA = 0.99

# How far in the future to look for rewards
REWARD_STEPS = 2

# Size of the experience replay buffer
REPLAY_SIZE = 500000
REPLAY_INITIAL = 10000

# How often to run validation and sync the target network
VALIDATION_INTERVAL = 10000
TARGETNET_SYNC_INTERNVAL = 1000

# Number of states to evaluate when syncing the target network
STATES_TO_EVALUATE = 3000

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(      "--cuda",                help="Enable cuda", default=False, action="store_true")
    parser.add_argument("-p", "--path", required=True, help="Directory or file of price data")
    parser.add_argument("-v", "--val",                 help="Validation data, default=path/val/")
    parser.add_argument("-t", "--test",                help="Test data, default=path/test/")
    parser.add_argument("-r", "--run",  required=True, help="Run name")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    # Create output directory
    savesPath = SAVES_DIR / f"{args.run}"
    savesPath.mkdir(parents=True, exist_ok=True)

    # Set data paths
    dataPath = pathlib.Path(args.path)
    dataFolder = dataPath
    
    # If dataPath is a file, use fetch containing directory
    if dataPath.is_file():
        dataFolder = dataPath.parent

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
    if dataPath.is_file():
        # Import data from file to dictionary
        index = dataPath.stem
        priceData = {index: data.readCSV(str(dataPath)) }
        env = environments.StocksEnv(priceData, barCount=BAR_COUNT)
    elif dataPath.is_dir():
        env = environments.StocksEnv.fromDirectory(dataPath, barCount=BAR_COUNT)
    else:
        raise RuntimeError("No data to train on")

    # Create validation environment
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    envTest = environments.StocksEnv.fromDirectory(testPath, barCount=BAR_COUNT)
    envVal = environments.StocksEnv.fromDirectory(valPath, barCount=BAR_COUNT)

    # Create the networks
    net = models.DQNConv1D(env.observation_space.shape, env.action_space.n).to(device)
    targetNet = models.TargetNet(net)

    # Create the action selector
    selector = actions.EpsilonGreedyActionSelector(EPS_START)
    epsilonTracker = actions.EpsilonTracker(selector, EPS_START, EPS_FINAL, EPS_STEPS)

    # Create the agent
    agent = agents.DQNAgent(net, selector, device=device, preprocessor=dictionaryStateToTensor)

    # Create the experience source
    expSource = experiences.ExperienceSourceFirstLast(env, agent, GAMMA, stepsCount=REWARD_STEPS)
    buffer = experiences.PrioritizedReplayBuffer(expSource, REPLAY_SIZE)

    # Create the optimizer
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    def processBatch(engine, batch):
        """
        Process a batch of data
        :param engine: engine to process batch
        :param batch: batch to process
        :return: loss and epsilon
        """

        # Zero the gradients
        optimizer.zero_grad()

        # Calculate the loss
        validationLoss = common.calculateLoss(batch, net, targetNet.targetModel, gamma=GAMMA ** REWARD_STEPS, device=device)

        # Backpropagate the loss
        validationLoss.backward()

        # Update the weights
        optimizer.step()

        # Update the epsilon
        epsilonTracker.frame(engine.state.iteration)

        # If evalStates is not set...
        if getattr(engine.state, "evalStates", None) is None:
            # Get a sample of states to evaluate
            evalStates = buffer.sample(STATES_TO_EVALUATE)
            evalStates = [transition.state for transition in evalStates]
            engine.state.evalStates = np.array(evalStates, copy=False)

        # Return the loss and epsilon
        return {
            "loss": validationLoss.item(),
            "epsilon": selector.epsilon,
        }

    # Create the engine to process the batch
    engine = Engine(processBatch)

    # Attach the tensorboard logger
    tb = common.setupIgnite(engine, expSource, f"{args.run}", extraMetrics=('MeanValue',))

    # Set the TargetNet Sync engine
    @engine.on(Events.ITERATION_COMPLETED)
    def sync_eval(engine: Engine):
        # Run every TARGETNET_SYNC_INTERNVAL iterations (Default: 1000)
        if engine.state.iteration % TARGETNET_SYNC_INTERNVAL == 0:
            # Sync the targetNet with the net
            targetNet.sync()

            # Calculate the mean value of the states
            meanValue = common.calculateStatesValues(engine.state.evalStates, net, device=device)
            engine.state.metrics["MeanValue"] = meanValue

            # If bestMeanValue is not set set it to meanValue
            if getattr(engine.state, "bestMeanValue", None) is None:
                engine.state.bestMeanValue = meanValue

            # If meanValue is greater than bestMeanValue save the model
            if engine.state.bestMeanValue < meanValue:
                print("%d: Best mean value updated %.3f -> %.3f" % (engine.state.iteration, engine.state.bestMeanValue, meanValue))
                path = savesPath / ("meanValue-%.3f.data" % meanValue)
                torch.save(net.state_dict(), path)
                engine.state.bestMeanValue = meanValue

    # Set the validation engine
    @engine.on(Events.ITERATION_COMPLETED)
    def validate(engine: Engine):
        # Run every VALIDATION_INTERVAL iterations (Default: 10000)
        if engine.state.iteration % VALIDATION_INTERVAL == 0:
            # Test: Get/print the mean: reward, steps, order profits, order steps
            res = validation.validationRun(envTest, net, device=device)
            print("%d: tst: %s" % (engine.state.iteration, res))
            # Add the metrics to the engine
            for key, val in res.items():
                engine.state.metrics[key + "_tst"] = val

            # Val: Get/print the mean: reward, steps, order profits, order steps
            res = validation.validationRun(envVal, net, device=device)
            print("%d: val: %s" % (engine.state.iteration, res))
            # Add the metrics to the engine
            for key, val in res.items():
                engine.state.metrics[key + "_val"] = val

            # If bestValReward is not set set it to mean episode reward
            valReward = res['episodeReward']
            if getattr(engine.state, "bestValReward", None) is None:
                engine.state.bestValReward = valReward

            # If valReward is greater than bestValReward save the model
            if engine.state.bestValReward < valReward:
                print("Best validation reward updated: %.3f -> %.3f, model saved" % (engine.state.bestValReward, valReward))
                engine.state.bestValReward = valReward
                path = savesPath / ("valReward-%.3f.data" % valReward)
                torch.save(net.state_dict(), path)

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
    engine.run(common.batchGenerator(buffer, REPLAY_INITIAL, BATCH_SIZE))
