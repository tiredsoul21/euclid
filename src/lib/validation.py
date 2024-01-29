import numpy as np

import torch

from lib import environments

METRICS = (
    'episodeReward',
    'episodeSteps',
    'orderProfits',
    'orderSteps',
)


def validationRun(env, net, episodes=100, device="cpu", epsilon=0.02, comission=0.1):
    stats = { metric: [] for metric in METRICS }

    for episode in range(episodes):
        obs = env.reset()

        total_reward = 0.0
        position = None
        position_steps = None
        episodeSteps = 0

        while True:
            obs_v = torch.tensor([obs]).to(device)
            out_v = net(obs_v)

            action_idx = out_v.max(dim=1)[1].item()
            if np.random.random() < epsilon:
                action_idx = env.action_space.sample()
            action = environments.Actions(action_idx)

            close_price = env._state._currentClose()

            if action == environments.Actions.Buy and position is None:
                position = close_price
                position_steps = 0
            elif action == environments.Actions.Close and position is not None:
                profit = close_price - position - (close_price + position) * comission / 100
                profit = 100.0 * profit / position
                stats['orderProfits'].append(profit)
                stats['orderSteps'].append(position_steps)
                position = None
                position_steps = None

            obs, reward, done, _, _ = env.step(action_idx)
            total_reward += reward
            episodeSteps += 1
            if position_steps is not None:
                position_steps += 1
            if done:
                if position is not None:
                    profit = close_price - position - (close_price + position) * comission / 100
                    profit = 100.0 * profit / position
                    stats['orderProfits'].append(profit)
                    stats['orderSteps'].append(position_steps)
                break

        stats['episodeReward'].append(total_reward)
        stats['episodeSteps'].append(episodeSteps)

    return { key: np.mean(vals) for key, vals in stats.items() }
