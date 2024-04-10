""" Trainer for Geister Competition """
import os
import argparse

from torch import device
from .environment import GeisterEnv
from .agent import GeisterAgent

SEED = 42

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    args = parser.parse_args()

    # Set device for torch
    DEVICE = device("cuda" if args.cuda else "cpu")

    # Create output directory
    SAVES_DIR = f"saves/{args.name}"
    os.makedirs(SAVES_DIR, exist_ok=True)

    env = GeisterEnv(seed=SEED)
    state = env.reset()
    actions = state.get_posssible_actions(1)
    for action in actions:
        print(action)
    board = env.get_board(1)
    print(board)
    print(actions)

    agent = GeisterAgent(None, None, DEVICE, None)
    env.print_board(1)
    env.step(((4, 1), (3, 1),1))

    print("-----------------")
    board = env.get_board(1)
    env.print_board(1)
    print(board)
    board = env.get_board(2)
    env.print_board(2)
    print(board)
    env.step(((3, 1), (2, 1),1))

    print("-----------------")
    board = env.get_board(1)
    env.print_board(1)
    print(board)
    board = env.get_board(2)
    env.print_board(2)
    print(board)
    state = env.step(((2, 1), (1, 1),1))

    print("-----------------")
    board = env.get_board(1)
    env.print_board(1)
    print(board)
    board = env.get_board(2)
    env.print_board(2)
    print(board)

    print(state.ghosts)
    state = env.step(((1, 1), (1, 2),1))
    state = env.step(((1, 2), (1, 3),1))
    state = env.step(((1, 3), (1, 4),1))
    state = env.step(((1, 4), (0, 4),1))
    state = env.step(((0, 4), (0, 3),1))
    state = env.step(((0, 3), (0, 2),1))
    print(state.done)
    state = env.step(((0, 2), (0, 1),1))

    print("-----------------")
    board = env.get_board(1)
    env.print_board(1)
    print(board)
    board = env.get_board(2)
    env.print_board(2)
    print(board)
    print(state.player1_reward)
    print(state.done)
    print(state.ghosts)





