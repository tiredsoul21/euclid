""" Defines the ValidationRun methods """
import numpy as np

from lib.environments import StockActions
from lib.utils import dictionaryStateToTensor

METRICS = (
    'episodeReward',
    'episodeSteps',
    'orderProfits',
    'orderSteps',
)


def validationRun(env, net,
                  episodes: int = 100,
                  device: str = "cpu",
                  epsilon: float = 0.02,
                  comission: float = 0.1):
    """
    Run the model on the environment and return the metrics
    :param env: Environment to run the model on
    :param net: Model to run
    :param episodes: Number of episodes to run
    :param device: Device to run the model on
    :param epsilon: Epsilon value for exploration
    :param comission: Commission rate
    :return: Dictionary of metrics
    """

    # Initialize the metrics
    stats = { metric: [] for metric in METRICS }

    # Run the episodes
    for _ in range(episodes):
        # Initialize the variables
        obs = env.reset()
        total_reward = 0.0
        position = None
        holdDuration = None
        episodeSteps = 0

        # Runs until episode is done
        while True:
            # Process the observation and get the actions
            observationVector = dictionaryStateToTensor([obs], device)
            outputVector = net(observationVector)
            actionInext = outputVector.max(dim=1)[1].item()

            # If we allow exploration
            if np.random.random() < epsilon:
                actionInext = env.action_space.sample()
            action = StockActions(actionInext)

            # Process Buy action
            closePrice = env._state._current_close()
            if action == StockActions.BUY and position is None:
                position = closePrice
                holdDuration = 0
            # Process Sell action
            elif action == StockActions.SELL and position is not None:
                # Price difference minus commission
                profit = closePrice - position - (closePrice + position) * comission / 100
                # Convert to percentage
                profit = 100.0 * profit / position

                # Save the metrics
                stats['orderProfits'].append(profit)
                stats['orderSteps'].append(holdDuration)
                position = None
                holdDuration = None

            # Process the step
            obs, reward, done, _, _ = env.step(actionInext)
            total_reward += reward

            # Update the episode steps and hold duration
            episodeSteps += 1
            if holdDuration is not None:
                holdDuration += 1

            # If the episode is done
            if done:
                if position is not None:
                    # Price difference minus commission
                    profit = closePrice - position - (closePrice + position) * comission / 100
                    # Convert to percentage
                    profit = 100.0 * profit / position

                    # Save the metrics
                    stats['orderProfits'].append(profit)
                    stats['orderSteps'].append(holdDuration)
                break

        # Save the metrics
        stats['episodeReward'].append(total_reward)
        stats['episodeSteps'].append(episodeSteps)

    # Return the mean of the metrics
    return { key: np.mean(vals) for key, vals in stats.items() }
