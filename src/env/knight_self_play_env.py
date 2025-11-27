"""
Self-play Gymnasium environment for Pferdeäpfel.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.game.board import Board
from src.game.game import Game
from src.game.rules import Rules
from src.players.random import RandomPlayer

logger = logging.getLogger(__name__)


class KnightSelfPlayEnv(gym.Env):
    """
    Gymnasium environment for Pferdeäpfel self-play.

    Observation Space:
        Box(low=0, high=1, shape=(7, 8, 8), dtype=np.float32)
        - Channel 0: Current Player Position (1.0 at pos, 0.0 elsewhere)
        - Channel 1: Opponent Player Position (1.0 at pos, 0.0 elsewhere)
        - Channel 2: Blocked Squares (1.0 if blocked/apple/visited, 0.0 if empty)
        - Channel 3: Mode ID (Normalized: 1->0.0, 2->0.5, 3->1.0)
        - Channel 4: Current Role (1.0 if White, 0.0 if Black)
        - Channel 5: Brown Apples Remaining (Normalized: count / 28.0)
        - Channel 6: Golden Apples Remaining (Normalized: count / 12.0)

    Action Space:
        MultiDiscrete([8, 65])
        - 0: Knight Move Index (0-7) corresponding to Rules.KNIGHT_MOVES
        - 1: Apple Placement Index (0-63 for board squares, 64 for "No Apple")
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, mode: Union[int, str] = "random") -> None:
        """
        Initialize the environment.

        Args:
            mode: Game mode (1, 2, 3) or "random"
        """
        super().__init__()
        self.mode_config = mode

        # Define action and observation space
        self.action_space = spaces.MultiDiscrete([8, 65])
        # (7, 8, 8) tensor for CNN/MLP
        self.observation_space = spaces.Box(low=0, high=1, shape=(7, 8, 8), dtype=np.float32)

        # Initialize game components
        # We use placeholder players since the env controls the game flow
        self.white_player = RandomPlayer("White")
        self.black_player = RandomPlayer("Black")
        self.game: Optional[Game] = None

        # Track who is the "agent" currently acting (switches every step)
        self.current_agent_color = "white"

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to a new game state."""
        super().reset(seed=seed)

        # Determine mode for this episode
        if self.mode_config == "random":
            mode = int(self.np_random.integers(1, 4))  # 1, 2, or 3
        else:
            mode = int(self.mode_config)

        # Create new game
        self.game = Game(self.white_player, self.black_player, mode=mode)

        # Randomly decide who starts (Game does this, but we need to sync)
        self.current_agent_color = self.game.current_player

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
        current_pos = self.game.board.get_horse_position(self.current_agent_color)

        # Calculate target move
        dr, dc = Rules.KNIGHT_MOVES[move_idx]
        move_to = (current_pos[0] + dr, current_pos[1] + dc)

        # Calculate apple placement
        extra_apple = None
        if apple_idx < 64:
            extra_apple = (apple_idx // 8, apple_idx % 8)

        # 2. Validate and Execute Move

        # Check legality first to give penalty for invalid moves
        legal_moves = Rules.get_legal_knight_moves(self.game.board, self.current_agent_color)

        if move_to not in legal_moves:
            # Invalid move penalty
            # We punish illegal moves heavily and end the episode to teach rules
            return self._get_observation(), -1.0, True, False, {"error": "Invalid move"}

        # Try to make the move
        # game.make_move handles turn switching and win checking
        success = self.game.make_move(move_to, extra_apple)

        if not success:
            # Invalid apple placement or other rule violation
            return self._get_observation(), -1.0, True, False, {"error": "Invalid action (apple/rules)"}

        # 3. Check Game End
        if self.game.game_over:
            # If the game ended, check if the current agent won
            # self.game.winner is set by Game
            if self.game.winner == self.current_agent_color:
                reward = 1.0
            elif self.game.winner == "draw":
                reward = 0.0
            else:
                # Opponent won (or we lost by getting stuck, though legal_moves check handles that mostly)
                reward = -1.0

            return self._get_observation(), reward, True, False, {"winner": self.game.winner}

        # 4. Switch Perspective
        self.current_agent_color = self.game.current_player

        # Return observation for the NEW current player
        return self._get_observation(), 0.0, False, False, {}

    def _get_observation(self) -> np.ndarray:
        """
        Get the current board state as observation from current player's perspective.
        Shape: (7, 8, 8)
        """
        if self.game is None:
            return np.zeros((7, 8, 8), dtype=np.float32)

        obs = np.zeros((7, 8, 8), dtype=np.float32)

        my_pos = self.game.board.get_horse_position(self.current_agent_color)
        opp_color = "black" if self.current_agent_color == "white" else "white"
        opp_pos = self.game.board.get_horse_position(opp_color)

        # Channel 0: My Position
        obs[0, my_pos[0], my_pos[1]] = 1.0

        # Channel 1: Opponent Position
        obs[1, opp_pos[0], opp_pos[1]] = 1.0

        # Channel 2: Blocked Squares
        for r in range(8):
            for c in range(8):
                if not self.game.board.is_empty(r, c):
                    obs[2, r, c] = 1.0

        # Channel 3: Mode ID (Normalized)
        # 1 -> 0.0, 2 -> 0.5, 3 -> 1.0
        mode_val = 0.0
        if self.game.board.mode == 2:
            mode_val = 0.5
        elif self.game.board.mode == 3:
            mode_val = 1.0
        obs[3, :, :] = mode_val

        # Channel 4: Current Role
        # White -> 1.0, Black -> 0.0
        role_val = 1.0 if self.current_agent_color == "white" else 0.0
        obs[4, :, :] = role_val

        # Channel 5: Brown Apples Remaining (Normalized)
        # Max 28
        brown_val = self.game.board.brown_apples_remaining / 28.0
        obs[5, :, :] = brown_val

        # Channel 6: Golden Apples Remaining (Normalized)
        # Max 12
        golden_val = self.game.board.golden_apples_remaining / 12.0
        obs[6, :, :] = golden_val

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get auxiliary info."""
        if self.game is None:
            return {}
        return {
            "turn": self.current_agent_color,
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
