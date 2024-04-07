""" Trainer for Geister Competition """
import os
import argparse

from torch import device
from .environment import GeisterEnv

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
    env.reset()
    # board = env.get_board(1)
    # print(board)
    # env.print_board(1)
    # env.step(((4, 1), (3, 1),1))

    # print("-----------------")
    # board = env.get_board(1)
    # env.print_board(1)
    # print(board)
    # board = env.get_board(2)
    # env.print_board(2)
    # print(board)

    # env.step(((3, 1), (3, 0),1))

    # print("-----------------")
    # board = env.get_board(1)
    # env.print_board(1)
    # print(board)
    # board = env.get_board(2)
    # env.print_board(2)
    # print(board)

    # env.step(((5, 1), (5, 0),1))

    # print("-----------------")
    # board = env.get_board(1)
    # env.print_board(1)
    # print(board)
    # board = env.get_board(2)
    # env.print_board(2)
    # print(board)
