"""Gymnasium environment for Pferdeäpfel."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.game.board import Board
from src.game.game import Game
from src.game.rules import Rules
from src.players.random import RandomPlayer

logger = logging.getLogger(__name__)


class PferdeapfelEnv(gym.Env):
    """
    Gymnasium environment for Pferdeäpfel.

    Observation Space:
        Box(low=0, high=4, shape=(8, 8), dtype=np.int8)
        Represents the board grid.

    Action Space:
        MultiDiscrete([8, 65])
        - 0: Knight Move Index (0-7) corresponding to Rules.KNIGHT_MOVES
        - 1: Apple Placement Index (0-63 for board squares, 64 for "No Apple")
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, mode: int = 3, opponent_policy: str = "random") -> None:
        """
        Initialize the environment.

        Args:
            mode: Game mode (1, 2, or 3)
            opponent_policy: Strategy for the opponent ("random" only for now)
        """
        super().__init__()
        self.mode = mode
        self.opponent_policy = opponent_policy

        # Define action and observation space
        self.action_space = spaces.MultiDiscrete([8, 65])
        self.observation_space = spaces.Box(low=0, high=4, shape=(8, 8), dtype=np.int8)

        # Initialize game components
        # We use placeholder players since the env controls the game flow
        self.white_player = RandomPlayer("White")
        self.black_player = RandomPlayer("Black")
        self.game: Optional[Game] = None
        self.agent_color = "white"

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to a new game state."""
        super().reset(seed=seed)

        # Create new game
        self.game = Game(self.white_player, self.black_player, mode=self.mode)
        self.agent_color = "white"

        # If Black starts, make a move for Black
        if self.game.current_player == "black":
            self._opponent_move()

        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: [move_index, apple_index]
        """
        if self.game is None:
            raise RuntimeError("Call reset() before step()")

        # 1. Decode Action
        move_idx, apple_idx = action

        # Get current position
        current_pos = self.game.board.get_horse_position(self.agent_color)

        # Calculate target move
        dr, dc = Rules.KNIGHT_MOVES[move_idx]
        move_to = (current_pos[0] + dr, current_pos[1] + dc)

        # Calculate apple placement
        extra_apple = None
        if apple_idx < 64:
            extra_apple = (apple_idx // 8, apple_idx % 8)

        # 2. Validate and Execute Move
        # We use Rules.make_move directly or game.make_move
        # game.make_move handles turn switching and win checking

        # Check legality first to give penalty for invalid moves
        legal_moves = Rules.get_legal_knight_moves(self.game.board, self.agent_color)

        if move_to not in legal_moves:
            # Invalid move penalty
            return self._get_observation(), -10.0, True, False, {"error": "Invalid move"}

        # Try to make the move
        # Note: extra_apple might be invalid too (e.g. on occupied square)
        # game.make_move returns False if move is invalid (including apple placement)
        success = self.game.make_move(move_to, extra_apple)

        if not success:
            # Invalid apple placement or other rule violation
            return self._get_observation(), -10.0, True, False, {"error": "Invalid action (apple/rules)"}

        # 3. Check Game End (Agent Win)
        if self.game.game_over:
            reward = 1.0 if self.game.winner == self.agent_color else -1.0
            return self._get_observation(), reward, True, False, {"winner": self.game.winner}

        # 4. Opponent Turn
        # The agent made a valid move, now it's the opponent's turn
        self._opponent_move()

        # 5. Check Game End (Opponent Win)
        if self.game.game_over:
            reward = 1.0 if self.game.winner == self.agent_color else -1.0
            return self._get_observation(), reward, True, False, {"winner": self.game.winner}

        # Game continues
        return self._get_observation(), 0.0, False, False, {}

    def _opponent_move(self) -> None:
        """Make a move for the opponent."""
        if self.game.game_over:
            return

        opponent_color = "black" if self.agent_color == "white" else "white"

        # Get legal moves
        legal_moves = Rules.get_legal_knight_moves(self.game.board, opponent_color)
        if not legal_moves:
            return

        # 1. Select Random Move
        move_idx = self.np_random.integers(0, len(legal_moves))
        move_to = legal_moves[move_idx]

        extra_apple = None

        # 2. Select Random Apple (Mode 3 Logic)
        if self.game.board.mode == 3:
            if self.game.board.brown_apples_remaining > 0:
                # 50% chance to place apple
                if self.np_random.random() < 0.5:
                    # Find valid placements
                    # We need to simulate to ensure we don't block White (if opponent is Black)
                    # For simplicity in this env wrapper, we will try to find a valid placement
                    # by checking empty squares and verifying the rule.

                    # Get all empty squares
                    empty_squares = []
                    for r in range(8):
                        for c in range(8):
                            if self.game.board.is_empty(r, c):
                                empty_squares.append((r, c))

                    # Shuffle to pick randomly
                    self.np_random.shuffle(empty_squares)

                    # Try to find a valid one
                    for r, c in empty_squares:
                        # Check if this placement is valid (doesn't block White's last move)
                        # Just pick a random empty square. If game.make_move fails, we retry or skip apple.
                        extra_apple = (r, c)
                        break

        # Execute move
        success = self.game.make_move(move_to, extra_apple)

        # If failed (e.g. invalid apple), try again without apple
        if not success and extra_apple is not None:
            self.game.make_move(move_to, None)

    def _get_observation(self) -> np.ndarray:
        """Get the current board state as observation."""
        if self.game is None:
            return np.zeros((8, 8), dtype=np.int8)
        return self.game.board.grid.copy()

    def _get_info(self) -> Dict[str, Any]:
        """Get auxiliary info."""
        if self.game is None:
            return {}
        return {
            "turn": self.game.current_player,
            "white_pos": self.game.board.white_pos,
            "black_pos": self.game.board.black_pos,
        }

    def render(self) -> Optional[List[str]]:
        """Render the environment."""
        if self.game is None:
            return None

        # Simple ANSI render
        grid = self.game.board.grid
        output = []
        output.append("  0 1 2 3 4 5 6 7")
        for r in range(8):
            row_str = f"{r} "
            for c in range(8):
                val = grid[r, c]
                char = "."
                if val == Board.WHITE_HORSE:
                    char = "W"
                elif val == Board.BLACK_HORSE:
                    char = "B"
                elif val == Board.BROWN_APPLE:
                    char = "o"
                elif val == Board.GOLDEN_APPLE:
                    char = "G"
                row_str += f"{char} "
            output.append(row_str)

        if self.render_mode == "human":
            for line in output:
                print(line)

        return output
