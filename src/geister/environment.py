""" Build the environment for the game of Geister """
import sys
import random
from typing import Tuple
import numpy as np

import gym

# Defind the tuple for the action
Action = Tuple[Tuple[int, int], Tuple[int, int], int]

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
        # Player 1's board
        self.player_1_start_pos = None
        self._board_1 = np.zeros((6, 6), dtype=np.int32)
        self._mask_board_1 = np.zeros((6, 6), dtype=np.int32)

        # Player 2's board
        self.player_2_start_pos = None
        self._board_2 = np.zeros((6, 6), dtype=np.int32)
        self._mask_board_2 = np.zeros((6, 6), dtype=np.int32)

        # Set the number of steps of the game
        self._current_state = { 
            "step": 0, 
            "done": False, 
            "player1_reward": 0, 
            "player2_reward": 0,
            "player_turn": 1,
            "player1_win": False,
            "player2_win": False
        }

        # Set the seed
        np.random.seed(seed)
        self.reset()

    def reset(self, **kwargs):
        """
        Resets the board to initial random state
        0: Empty
        1: Player's Good Ghost
        2: Player's Bad Ghost
        3: Opponent's Good Ghost
        4: Opponent's Bad Ghost
        5: Opponent's Ghosts masked
        6: Player's Escape Zones
        7: Opponent's Escape Zones
        8-9: Reserved for swaps
        """
        super().reset(**kwargs)
        self._board_1 = np.zeros((6, 6), dtype=np.int32)

        # Set player's ghosts
        if self.player_1_start_pos is None:
            player_array = np.array([1, 1, 1, 1, 2, 2, 2, 2])
            np.random.shuffle(player_array)
            self.player_1_start_pos = player_array.reshape(2, 4)

        # Set opponent's ghosts
        if self.player_2_start_pos is None:
            player_array = np.array([1, 1, 1, 1, 2, 2, 2, 2])
            np.random.shuffle(player_array)
            self.player_2_start_pos = player_array.reshape(2, 4)

        # Set players1 board view
        for i in range(2):
            for j in range(4):
                self._board_1[i, j + 1] = self.player_2_start_pos[i, j] + 2
                self._board_1[i + 4, j + 1] = self.player_1_start_pos[i, j]

        # Set Escape Zones
        self._board_1[0, 0] = 6
        self._board_1[0, 5] = 6
        self._board_1[5, 0] = 7
        self._board_1[5, 5] = 7

        # Mask the opponent's ghosts
        self._mask_board_1 = np.where(self._board_1 == 3, 5, self._board_1)
        self._mask_board_1 = np.where(self._mask_board_1 == 4, 5, self._mask_board_1)

        self.reflect_board()

    def step(self, action):
        """
        Take an action in the environment
        action: a tuple of (from, to, player_num)
        """
        assert len(action) == 3 and len(action[0]) == 2 and len(action[1]) == 2
        assert action[2] in [1, 2]

        _, to_pos, player_num = action
        board = self._board_1 if player_num == 1 else self._board_2
        self.valid_moves(action, board)

        # Make the move on the board
        target_space = board[to_pos[0], to_pos[1]]
        self.make_move(action)

        # Reflect the move on the opponent's board
        self.reflect_move(action)

        # Check if the game is over
        # Escape one ghost to the opponent's side
        if (target_space == 6 or         # Player escape zone
            np.sum(board == 3) == 0 or   # Opponent's good ghosts captured
            np.sum(board == 2) == 0):    # Player's bad ghosts captured
            self._current_state["done"] = True
            self._current_state["player1_win"] =  player_num == 1
            self._current_state["player2_win"] =  player_num == 2
            self._current_state["player1_reward"] = 1 if player_num == 1 else -1
            self._current_state["player2_reward"] = 1 if player_num == 2 else -1

        if self._current_state["step"] == 100 and not self._current_state["done"]:
            self._current_state["done"] = True
            # Count the number of good ghosts on the board
            p1_good_capture = 4 - np.sum(self._board_1 == 3)
            p2_good_capture = 4 - np.sum(self._board_2 == 3)
            p1_bad_capture = 4 - np.sum(self._board_1 == 4)
            p2_bad_capture = 4 - np.sum(self._board_2 == 4)
            self._current_state["player1_reward"] = (p1_good_capture - p1_bad_capture) / 4
            self._current_state["player2_reward"] = (p2_good_capture - p2_bad_capture) / 4
            self._current_state["player1_win"] = self._current_state["player1_reward"] > 0
            self._current_state["player2_win"] = self._current_state["player2_reward"] > 0

        self._current_state["step"] += 1
        self._current_state["player_turn"] = 2 if player_num == 1 else 1

        return self._current_state

    def render(self):
        """
        Not implemented -- no visualization yet implemented
        """

    def set_positions(self, player_start_pos, player_num):
        """
        Player's choice of ghost positions
        board: a 2x4 array of positions for the player's ghosts
            board must consist of 4 1s and 4 2s
            1: Player Good Ghost
            2: Player Bad Ghost
        """
        assert player_num in [1, 2]
        assert player_start_pos.shape == (2, 4)
        assert np.sum(player_start_pos == 1) == 4
        assert np.sum(player_start_pos == 2) == 4

        if player_num == 1:
            self.player_1_start_pos = player_start_pos
        else:
            # Player 2's board is flipped
            player_start_pos = np.flip(player_start_pos, axis=0)
            player_start_pos = np.flip(player_start_pos, axis=1)
            self.player_2_start_pos = player_start_pos

    def make_move(self, action):
        """
        Make the move on the board
        action: a tuple of (from, to, player_num) positions
        """
        # Get the player's board
        from_pos, to_pos, player_num = action
        board = self._board_1 if player_num == 1 else self._board_2
        mask_board = self._mask_board_1 if player_num == 1 else self._mask_board_2

        # Check if the action is valid
        board[to_pos[0], to_pos[1]] = board[from_pos[0], from_pos[1]]
        mask_board[to_pos[0], to_pos[1]] = mask_board[from_pos[0], from_pos[1]]
        board[from_pos[0], from_pos[1]] = 0
        mask_board[from_pos[0], from_pos[1]] = 0

    def reflect_move(self, action):
        """
        Reflect the move on the opponent's board
          up -> down, right -> left, etc.
        action: a tuple of (from, to, player_num) positions
        """
        # Get the opponent's board
        from_pos, to_pos, player_num = action
        opp_board = self._board_2 if player_num == 1 else self._board_1
        opp_mask = self._mask_board_2 if player_num == 1 else self._mask_board_1

        opp_board[5 - to_pos[0], 5 - to_pos[1]] = opp_board[5 - from_pos[0], 5 - from_pos[1]]
        opp_mask[5 - to_pos[0], 5 - to_pos[1]] = opp_mask[5 - from_pos[0], 5 - from_pos[1]]
        opp_board[5 - from_pos[0], 5 - from_pos[1]] = 0
        opp_mask[5 - from_pos[0], 5 - from_pos[1]] = 0

    def reflect_board(self):
        """
        Create a reflection of the board for player 2
        The intent of this is to each player with the same 'view' of the board.
        This is intended to aid in training the model. (universal model for both players)
        This should happen once at the start of the game.
        """
        # Flip the board
        self._board_2 = np.flip(self._board_1, axis=0)
        self._board_2 = np.flip(self._board_2, axis=1)

        # Swap the player's ghosts (numbering)
        self._board_2 = np.where(self._board_2 == 1, 8, self._board_2)
        self._board_2 = np.where(self._board_2 == 2, 9, self._board_2)
        self._board_2 = np.where(self._board_2 == 3, 1, self._board_2)
        self._board_2 = np.where(self._board_2 == 4, 2, self._board_2)
        self._board_2 = np.where(self._board_2 == 8, 3, self._board_2)
        self._board_2 = np.where(self._board_2 == 9, 4, self._board_2)

        # Set Escape Zones
        self._board_2[0, 0] = 6
        self._board_2[0, 5] = 6
        self._board_2[5, 0] = 7
        self._board_2[5, 5] = 7

        # Mask the opponent's ghosts
        self._mask_board_2 = self._board_2.copy()
        self._mask_board_2 = np.where(self._board_2 == 3, 5, self._board_2)
        self._mask_board_2 = np.where(self._mask_board_2 == 4, 5, self._mask_board_2)

    def valid_moves(self, action, board):
        """
        Check if the action is valid
        This is intended to break the code if the action is invalid. As agent 
        will be responsible for checking the validity of the action.
        action: a tuple of (from, to, player_num) positions
        board: the board to check the action on
        """
        # Check if the action in board
        from_pos, to_pos, _ = action
        assert 0 <= from_pos[0] < 6 and 0 <= from_pos[1] < 6
        assert 0 <= to_pos[0]   < 6 and 0 <= to_pos[1]   < 6

        # Check if moving the player's ghost
        assert 1 <= board[from_pos[0], from_pos[1]] <= 2

        # Check if valid move
        assert (board[to_pos[0], to_pos[1]] == 0 or           # Empty space or
                board[to_pos[0], to_pos[1]] == 3 or           # Opponent's ghost or
                board[to_pos[0], to_pos[1]] == 4 or           # Opponent's ghost or
                board[to_pos[0], to_pos[1]] == 6)             # Player's escape zone

        # Check move size is 1
        assert ((abs(from_pos[0] - to_pos[0]) == 1 and from_pos[1] == to_pos[1]) or # Up / down 1
               ( abs(from_pos[1] - to_pos[1]) == 1 and from_pos[0] == to_pos[0]))   # Left / right 1

    def print_board(self, player_num):
        """
        Print the board
        player_num: the player to print the board for (1 or 2)
        """
        assert player_num in [1, 2]
        if player_num == 1:
            print("Player 1's Board")
            print(self._board_1)
        else:
            print("Player 2's Board")
            print(self._board_2)

    def get_board(self, player_num):
        """
        Get the board for the player
        player_num: the player to get the board for (1 or 2)
        """
        assert player_num in [1, 2]
        return self._mask_board_1 if player_num == 1 else self._mask_board_2
