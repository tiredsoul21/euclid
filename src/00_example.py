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

SAVES_DIR = pathlib.Path("output")

# How many bars to feed into the model
BAR_COUNT = 50

BATCH_SIZE = 32

# EPSILON GREEDY - for exploration
EPS_START = 1.0
EPS_FINAL = 0.1
EPS_STEPS = 1000000

LEARNING_RATE = 0.0001
GAMMA = 0.99

REWARD_STEPS = 2

REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000

VALIDATION_INTERVAL = 10000
TARGETNET_SYNC_INTERNVAL = 1000

STATES_TO_EVALUATE = 1000

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
    savesPath = SAVES_DIR / f"00-{args.run}"
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
        env._state.barCount = BAR_COUNT
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
    agent = agents.DQNAgent(net, selector, device=device)

    # Create the experience source
    exp_source = experiences.ExperienceSourceFirstLast(env, agent, GAMMA, stepsCount=REWARD_STEPS)
    buffer = experiences.ExperienceReplayBuffer(exp_source, REPLAY_SIZE)

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
        loss_v = common.calculateLoss(batch, net, targetNet.targetModel, gamma=GAMMA ** REWARD_STEPS, device=device)

        # Backpropagate the loss
        loss_v.backward()

        # Update the weights
        optimizer.step()

        # Update the epsilon
        epsilonTracker.frame(engine.state.iteration)

        if getattr(engine.state, "evalStates", None) is None:
            evalStates = buffer.sample(STATES_TO_EVALUATE)
            evalStates = [np.array(transition.state, copy=False)
                           for transition in evalStates]
            engine.state.evalStates = np.array(evalStates, copy=False)

        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }

    engine = Engine(processBatch)
    tb = common.setupIgnite(engine, exp_source, f"conv-{args.run}", extra_metrics=('values_mean',))

    @engine.on(Events.ITERATION_COMPLETED)
    def sync_eval(engine: Engine):
        if engine.state.iteration % TARGETNET_SYNC_INTERNVAL == 0:
            targetNet.sync()

            mean_val = common.calculateStatesValues(
                engine.state.evalStates, net, device=device)
            engine.state.metrics["values_mean"] = mean_val
            if getattr(engine.state, "best_mean_val", None) is None:
                engine.state.best_mean_val = mean_val
            if engine.state.best_mean_val < mean_val:
                print("%d: Best mean value updated %.3f -> %.3f" % (
                    engine.state.iteration, engine.state.best_mean_val,
                    mean_val))
                path = savesPath / ("mean_value-%.3f.data" % mean_val)
                torch.save(net.state_dict(), path)
                engine.state.best_mean_val = mean_val

    @engine.on(Events.ITERATION_COMPLETED)
    def validate(engine: Engine):
        if engine.state.iteration % VALIDATION_INTERVAL == 0:
            res = validation.validation_run(envTest, net, device=device)
            print("%d: tst: %s" % (engine.state.iteration, res))
            for key, val in res.items():
                engine.state.metrics[key + "_tst"] = val
            res = validation.validation_run(envVal, net, device=device)
            print("%d: val: %s" % (engine.state.iteration, res))
            for key, val in res.items():
                engine.state.metrics[key + "_val"] = val
            val_reward = res['episode_reward']
            if getattr(engine.state, "best_val_reward", None) is None:
                engine.state.best_val_reward = val_reward
            if engine.state.best_val_reward < val_reward:
                print("Best validation reward updated: %.3f -> %.3f, model saved" % (
                    engine.state.best_val_reward, val_reward
                ))
                engine.state.best_val_reward = val_reward
                path = savesPath / ("val_reward-%.3f.data" % val_reward)
                torch.save(net.state_dict(), path)


    event = local_ignite.PeriodEvents.ITERS_10000_COMPLETED
    tst_metrics = [m + "_tst" for m in validation.METRICS]
    tst_handler = tb_logger.OutputHandler(tag="test", metric_names=tst_metrics)
    tb.attach(engine, log_handler=tst_handler, event_name=event)

    val_metrics = [m + "_val" for m in validation.METRICS]
    val_handler = tb_logger.OutputHandler(tag="validation", metric_names=val_metrics)
    tb.attach(engine, log_handler=val_handler, event_name=event)

    engine.run(common.batchGenerator(buffer, REPLAY_INITIAL, BATCH_SIZE))
