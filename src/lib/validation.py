""" Defines the ValidationRun methods """
import numpy as np

from lib.environments import StockActions
from lib.utils import dict_state_to_tensor

METRICS = (
    'episodeReward',
    'episode_steps',
    'orderProfits',
    'orderSteps',
)


def validation_run(env, net,
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
        hold_duration = None
        episode_steps = 0

        # Runs until episode is done
        while True:
            # Process the observation and get the actions
            obs_vector = dict_state_to_tensor([obs], device)
            output_vector = net(obs_vector)
            action_index = output_vector.max(dim=1)[1].item()

            # If we allow exploration
            if np.random.random() < epsilon:
                action_index = env.action_space.sample()
            action = StockActions(action_index)

            # Process Buy action
            close_price = env.get_close()
            if action == StockActions.BUY and position is None:
                position = close_price
                hold_duration = 0
            # Process Sell action
            elif action == StockActions.SELL and position is not None:
                # Price difference minus commission
                profit = close_price - position - (close_price + position) * comission / 100
                # Convert to percentage
                profit = 100.0 * profit / position

                # Save the metrics
                stats['orderProfits'].append(profit)
                stats['orderSteps'].append(hold_duration)
                position = None
                hold_duration = None

            # Process the step
            obs, reward, done, _, _ = env.step(action_index)
            total_reward += reward

            # Update the episode steps and hold duration
            episode_steps += 1
            if hold_duration is not None:
                hold_duration += 1

            # If the episode is done
            if done:
                if position is not None:
                    # Price difference minus commission
                    profit = close_price - position - (close_price + position) * comission / 100
                    # Convert to percentage
                    profit = 100.0 * profit / position

                    # Save the metrics
                    stats['orderProfits'].append(profit)
                    stats['orderSteps'].append(hold_duration)
                break

        # Save the metrics
        stats['episodeReward'].append(total_reward)
        stats['episode_steps'].append(episode_steps)

    # Return the mean of the metrics
    return { key: np.mean(vals) for key, vals in stats.items() }
