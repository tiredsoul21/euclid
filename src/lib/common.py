import numpy as np

import torch
import torch.nn as nn

import warnings
from typing import Iterable
from datetime import datetime, timedelta

from lib import experiences
from lib import ignite as local_ignite

from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger

@torch.no_grad()
def calculateStatesValues(states,
                          net, 
                          device: str = "cpu"):
    """
    Calculate the values of the states
    :param states: states to calculate the values of
    :param net: network to use
    :param device: device to use
    """
    meanValues = []
    for batch in np.array_split(states, 64):
        # Run the batch through the network
        statesV = torch.tensor(batch).to(device)
        actionValuesV = net(statesV)
        # Get the best action values
        bestActionValuesV = actionValuesV.max(1)[0]
        # Get the mean of the best action values
        meanValues.append(bestActionValuesV.mean().item())
    # Return the mean of the mean values
    return np.mean(meanValues)

def unpackBatch(batch):
    """
    Unpack a batch of experiences
    :param batch: batch to unpack
    :return: unpacked batch
    """

    # Initialize lists
    states, actions, rewards, dones, lastStates = [], [], [], [], []

    # For each experience in the batch...
    for exp in batch:
        # Add the experience to the lists
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.lastState is None)

        # Account for the last state
        if exp.lastState is None:
            # Will be masked anyway
            lastStates.append(state)
        else:
            lastStates.append(np.array(exp.lastState, copy=False))
    # Return the array of states, actions, rewards, dones, and last states
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(lastStates, copy=False)

def calculateLoss(batch,
                  net,
                  targetNet,
                  gamma,
                  device="cpu"):
    """
    Calculate the loss of a batch
    :param batch: batch to calculate the loss of
    :param net: network to use
    :param targetNet: target network to use
    :param gamma: gamma to use
    :param device: device to use
    :return: loss of the batch
    """

    # Unpack the batch
    states, actions, rewards, dones, next_states = unpackBatch(batch)

    # Convert the batch to tensors
    statesV = torch.tensor(states).to(device)
    nextStateV = torch.tensor(next_states).to(device)
    actionsV = torch.tensor(actions).to(device)
    rewardsV = torch.tensor(rewards).to(device)
    doneMask = torch.BoolTensor(dones).to(device)

    # Calculate the state action values
    stateActionValues = net(statesV).gather(1, actionsV.unsqueeze(-1)).squeeze(-1)
    nextStateActions = net(nextStateV).max(1)[1]
    nextStateValues = targetNet(nextStateV).gather(1, nextStateActions.unsqueeze(-1)).squeeze(-1)
    nextStateValues[doneMask] = 0.0

    #Calculate the reward + discounted next state value
    meanStateActionValues = rewardsV + nextStateValues.detach() * gamma

    # Calculate & return the loss
    return nn.MSELoss()(stateActionValues, meanStateActionValues)

def batchGenerator(buffer: experiences.ExperienceReplayBuffer,
                   initial: int,
                   batch_size: int):
    """
    Generate a batch of experiences
    :param buffer: buffer to use
    :param initial: initial size of the buffer
    :param batch_size: batch size to use
    :return: batch of experiences
    """

    # Populate the buffer
    buffer.populate(initial)
    while True:
        # Populate the buffer with one more experience and yield
        buffer.populate(1)
        yield buffer.sample(batch_size)

def setupIgnite(engine: Engine,
                experienceSource,
                runName: str,
                extraMetrics: Iterable[str] = ()):
    """
    Setup the ignite engine
    :param engine: engine to setup
    :param experienceSource: experience source to use
    :param runName: name of the run
    :param extraMetrics: extra metrics to use
    :return: tensorboard logger
    """
    # get rid of missing metrics warning
    warnings.simplefilter("ignore", category=UserWarning)

    # Attach the end of episode handler
    handler = local_ignite.EndOfEpisodeHandler(experienceSource, subSampleEndOfEpisode=100)
    handler.attach(engine)
    
    # Attach the episode events
    local_ignite.EpisodeFPSHandler().attach(engine)

    # On episode completed event...
    @engine.on(local_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def episodeCompleted(trainer: Engine):
        # Print episode stats
        passed = trainer.state.metrics.get('timePassed', 0)
        print("Episode %d: reward=%.0f, steps=%s, " "speed=%.1f f/s, elapsed=%s" % (
            trainer.state.episode, trainer.state.episode_reward,
            trainer.state.episode_steps,
            trainer.state.metrics.get('avgFps', 0),
            timedelta(seconds=int(passed))))

    # Create the tensorboard logger
    now = datetime.now().isoformat(timespec='minutes')
    logdir = f"runs/{now}-{runName}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)

    # Attach the running average
    runningAverage = RunningAverage(output_transform=lambda v: v['loss'])
    runningAverage.attach(engine, "avgLoss")

    # Log episode metrics
    metrics = ['reward', 'steps', 'avgReward']
    handler = tb_logger.OutputHandler(tag="episodes", metric_names=metrics)
    event = local_ignite.EpisodeEvents.EPISODE_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)

    # Log training metrics
    local_ignite.PeriodicEvents().attach(engine)
    metrics = ['avgLoss', 'avgFps']
    metrics.extend(extraMetrics)
    handler = tb_logger.OutputHandler(tag="train", metric_names=metrics, output_transform=lambda a: a)
    event = local_ignite.PeriodEvents.ITERS_1000_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)

    # Return the tensorboard logger
    return tb
