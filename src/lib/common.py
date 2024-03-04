""" Common functions for the training """
from typing import Iterable
from datetime import datetime, timedelta
import warnings
import numpy as np

import torch
from torch import nn

from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger

from ..lib import experiences
from ..lib import ignite as local_ignite
from ..lib.utils import dict_state_to_tensor

@torch.no_grad()
def calculate_states_values(states, net, device: str = "cpu"):
    """
    Calculate the values of the states
    :param states: states to calculate the values of
    :param net: network to use
    :param device: device to use
    """
    mean_values = []
    states_v = dict_state_to_tensor(states.tolist(), device)

    # Assuming all tensors in states_v have the same length
    batch_size = next(iter(states_v.values())).size(0)

    # Split each tensor separately and process batches
    for i in range(0, batch_size, 64):
        batch = {key: val[i:i+64] for key, val in states_v.items()}
        # Run the batch through the network
        action_values_v = net(batch)
        # Get the best action values
        best_action_values_v = action_values_v.max(1)[0]
        # Get the mean of the best action values
        mean_values.append(torch.mean(best_action_values_v).item())

    # Return the mean of the mean values
    return np.mean(mean_values)

def unpack_batch(batch):
    """
    Unpack a batch of experiences
    :param batch: batch to unpack
    :return: unpacked batch
    """

    # Initialize lists
    states, actions, rewards, dones, last_states = [], [], [], [], []

    # For each experience in the batch...
    for exp in batch:
        # Add the experience to the lists
        state = exp.state
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)

        # Account for the last state
        if exp.last_state is None:
            # Will be masked anyway
            last_states.append(state)
        else:
            last_states.append(exp.last_state)
    # Return the array of states, actions, rewards, dones, and last states
    return states, np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), last_states

def calculate_loss(batch,
                  net,
                  target_net,
                  gamma,
                  device="cpu"):
    """
    Calculate the loss of a batch
    :param batch: batch to calculate the loss of
    :param net: network to use
    :param target_net: target network to use
    :param gamma: gamma to use
    :param device: device to use
    :return: loss of the batch
    """

    # Unpack the batch
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    # Convert the batch to tensors
    states_v = dict_state_to_tensor(states, device)
    next_state_v = dict_state_to_tensor(next_states, device)
    actions_v = torch.tensor(actions).to(device)
    reward_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    # Get the Q values for the action historically taken
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        # Get the Q values for the next state with current policy
        next_state_actions = net(next_state_v).max(1)[1]
        # Get the Q values for the next state with target policy
        next_state_values = target_net(next_state_v).gather(1, \
                            next_state_actions.unsqueeze(-1)).squeeze(-1)
        next_state_values[done_mask] = 0.0

    #Calculate the reward + discounted next state value
    mean_state_action_values = reward_v + next_state_values.detach() * gamma

    # Calculate & return the loss
    return nn.MSELoss()(state_action_values, mean_state_action_values)

def batch_generator(buffer: experiences.ExperienceReplayBuffer,
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

def setup_ignite(engine: Engine,
                exp_source,
                run_name: str,
                extra_metrics: Iterable[str] = ()):
    """
    Setup the ignite engine
    :param engine: engine to setup
    :param exp_source: experience source to use
    :param run_name: name of the run
    :param extra_metrics: extra metrics to use
    :return: tensorboard logger
    """
    # get rid of missing metrics warning
    warnings.simplefilter("ignore", category=UserWarning)

    # Attach the end of episode handler
    handler = local_ignite.EndOfEpisodeHandler(exp_source, sub_sample_end_of_episode=100)
    handler.attach(engine)

    # Attach the episode events
    local_ignite.EpisodeFPSHandler().attach(engine)

    # On episode completed event...
    @engine.on(local_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def episodeCompleted(trainer: Engine):
        # Print episode stats
        passed = trainer.state.metrics.get('timePassed', 0)
        print(f"Episode {trainer.state.episode}: "
              f"reward={trainer.state.episode_reward:.0f}, "
              f"steps={trainer.state.episode_steps}, "
              f"speed={trainer.state.metrics.get('avgFps', 0):.1f} f/s, "
              f"elapsed={timedelta(seconds=int(passed))}")


    # Create the tensorboard logger
    now = datetime.now().isoformat(timespec='minutes')
    logdir = f"runs/{run_name}-{now}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)

    # Attach the running average
    running_average = RunningAverage(output_transform=lambda v: v['loss'])
    running_average.attach(engine, "avgLoss")

    # Log episode metrics
    metrics = ['reward', 'steps', 'avgReward']
    handler = tb_logger.OutputHandler(tag="episodes", metric_names=metrics)
    event = local_ignite.EpisodeEvents.EPISODE_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)

    # Log training metrics
    local_ignite.PeriodicEvents().attach(engine)
    metrics = ['avgLoss', 'avgFps']
    metrics.extend(extra_metrics)
    handler = tb_logger.OutputHandler(tag="train", metric_names=metrics, \
                                      output_transform=lambda a: a)
    event = local_ignite.PeriodEvents.ITERS_1000_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)

    # Return the tensorboard logger
    return tb
