#!/usr/bin/env python3
from lib import wrappers
from lib import dqn_model

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19               # Training exit

GAMMA = 0.99                         # Discount factor
BATCH_SIZE = 32                      # Batch size
REPLAY_SIZE = 10000                  # Replay buffer size
LEARNING_RATE = 1e-4                 # Learning rate
SYNC_TARGET_FRAMES = 1000            # Sync target net with main net
REPLAY_START_SIZE = 10000            # Start training after this many frames

EPSILON_DECAY_LAST_FRAME = 150000    # Epsilon decay rate?
EPSILON_START = 1.0                  # Epsilon start value
EPSILON_FINAL = 0.01                 # Epsilon end value


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceBuffer:
    def __init__(self, capacity):
        # Degue to store the experiences
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        # Return the length of the buffer
        return len(self.buffer)

    def append(self, experience):
        # Add the experience to the buffer
        self.buffer.append(experience)

    def sample(self, batch_size):
        # Sample a batch of experiences from the buffer
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        # Unzip the experiences and return
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)

class Agent:
    def __init__(self, env, exp_buffer):
        # Add environment
        self.env = env
        # Add experience buffer
        self.exp_buffer = exp_buffer
        # Reset the environment (initialize the state and total reward)
        self._reset()

    def _reset(self):
        # Reset the environment (initialize the state and total reward)
        self.state = env.reset()
        self.total_reward = 0.0

    # This statement is used to disable the gradient calculation
    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        # If random (explor / exploit)
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        # Else, get the action from the network
        else:
            # Get the state as a numpy array and convert to tensor
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            # Get the Q values for the state (feed through the network)
            q_vals_v = net(state_v)
            # Get the action with the highest Q value
            _, act_v = torch.max(q_vals_v, dim=1)
            # Get the action as an integer
            action = int(act_v.item())

        # Apply the action to the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        # Record the experience
        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)

        # Update the state and return
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu"):
    # Unzip the batch
    states, actions, rewards, dones, next_states = batch

    # Convert to tensors
    states_v =      torch.tensor(np.array(states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
    actions_v =     torch.tensor(actions).to(device)
    rewards_v =     torch.tensor(rewards).to(device)
    done_mask =     torch.BoolTensor(dones).to(device)

    # Extracts predicted Q-values for the actions that were actually taken
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    # Ensures no gradients are calculated for the target network
    with torch.no_grad():
        # Get the next state values find the max Q value for each state
        next_state_values = tgt_net(next_states_v).max(1)[0]
        # No next state values for the terminal states -- won't converge otherwise
        next_state_values[done_mask] = 0.0
        # Detach the next state values from the graph - no gradients
        next_state_values = next_state_values.detach()

    # Calculate the bellman approximation and return the loss
    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = wrappers.make_env(args.env)

    # Create the network and target network
    net =     dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    print(net)

    # Create the writer
    writer = SummaryWriter(comment="-" + args.env)

    # Create the experience buffer
    buffer = ExperienceBuffer(REPLAY_SIZE)

    # Create the agent
    agent = Agent(env, buffer)

    # Create the optimizer
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Initialize values
    epsilon = EPSILON_START
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    while True:
        frame_idx += 1

        # Decay the epsilon value
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        # Play the game
        reward = agent.play_step(net, epsilon, device=device)

        # If the game is done
        if reward is not None:
            # calculate stats
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])

            # Print the results
            print("%d: done %d games, reward %.3f, "
                  "eps %.2f, speed %.2f f/s" % (
                  frame_idx, len(total_rewards), m_reward, epsilon, speed))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), args.env +
                           "-best_%.0f.dat" % m_reward)
                if best_m_reward is not None:
                    print("Best reward updated %.3f -> %.3f" % (
                        best_m_reward, m_reward))
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break

        # If buffer not full go back to the start of the loop
        if len(buffer) < REPLAY_START_SIZE:
            continue

        # Sync the target network with the main network every SYNC_TARGET_FRAMES
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        # Train the network
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
        
    # Done training
    writer.close()
