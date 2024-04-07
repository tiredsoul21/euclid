""" Build the environment for the game of Geister """
import sys
import random
import numpy as np
import torch

import gym

class GeisterEnv(gym.Env):
    """
    Environment for the game of Geister
    This is a 6x6 board game where each player has 8 ghosts.
    You have 4 good ghosts and 4 bad ghosts.
    Three ways to win:
        1. Escape one ghost to the opponent's side
        2. You capture the opponent's good ghosts
        3. Opponent captures all of your bad ghosts
    """
    def __init__(self, seed=None):
        self.board = np.zeros((6, 6), dtype=np.int32)
        self.masked_board = np.zeros((6, 6), dtype=np.int32)
        self.seed = seed if seed is not None else random.randint(0, sys.maxsize)
        self.player_start_pos = None
        self.opponent_start_pos = None
        self.reset()

    def reset(self, **kwargs):
        """
        Resets the board to initial random state
        0: Empty
        1-4: Player Good Ghost
        5-8: Player Bad Ghost
        9-12: Opponent Good Ghost
        13-16: Opponent Bad Ghost
        17: Opponent's Ghosts masked
        18: Player's Escape Zones
        19: Opponent's Escape Zones
        """
        super().reset(**kwargs)
        self.board = np.zeros((6, 6), dtype=np.int32)

        # Set player's ghosts
        if self.player_start_pos is None:
            player_good = np.random.permutation(4) + 1
            player_bad = np.random.permutation(4) + 5
            self.player_start_pos = np.concatenate([player_good, player_bad]).reshape(2, 4)

        # Set opponent's ghosts
        if self.opponent_start_pos is None:
            opponent_good = np.random.permutation(4) + 9
            opponent_bad = np.random.permutation(4) + 13
            self.opponent_start_pos = np.concatenate([opponent_good, opponent_bad]).reshape(2, 4)

        # Build the board
        # Set players' ghosts
        for i in range(2):
            for j in range(4):
                self.board[i, j + 1] = self.opponent_start_pos[i, j]
                self.board[i + 4, j + 1] = self.player_start_pos[i, j]
        # Set escape zones
        self.board[0, 0] = 18
        self.board[0, 5] = 18
        self.board[5, 0] = 19
        self.board[5, 5] = 19

        # Mask the opponent's ghosts
        self.masked_board = self.board.copy()
        self.masked_board[0:2, 1:5] = 17

        print(self.masked_board)
        self.player_start_pos = None
        
        return self.masked_board

    def step(self, action):
        """
        Take an action in the environment
        action: a 2-tuple of (from, to) positions
        """
        assert len(action) == 2 and len(action[0]) == 2 and len(action[1]) == 2

        # Check if the action is valid
        from_pos, to_pos = action
        assert 0 <= from_pos[0] < 6 and 0 <= from_pos[1] < 6
        assert 0 <= to_pos[0] < 6   and 0 <= to_pos[1] < 6
        assert self.board[from_pos[0], from_pos[1]] != 0
        assert self.board[to_pos[0], to_pos[1]] == 0


    def render(self):
        """
        Not implemented -- no visualization yet implemented
        """

    def set_board(self, player_start_pos):
        """
        Player's choice of ghost positions
        board: a 2x4 array of positions for the player's ghosts
            board must consist of all integers from 1 to 8
            1-4: Player Good Ghost
            5-8: Player Bad Ghost
        """
        assert player_start_pos.shape == (2, 4)
        seen_numbers = set()
        for i in range(2):
            for j in range(4):
                assert 1 <= player_start_pos[i, j] <= 8
                assert player_start_pos[i, j] not in seen_numbers
                seen_numbers.add(player_start_pos[i, j])

        self.player_start_pos = player_start_pos