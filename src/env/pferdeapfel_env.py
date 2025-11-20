"""Gymnasium-compatible environment for the Pferdeäpfel trailing mode (mode 2)."""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from src.game.board import Board
from src.game.rules import Rules

logger = logging.getLogger(__name__)


def board_to_observation(board: Board, current_player: str) -> np.ndarray:
    """Flatten the board and append metadata for RL consumption."""
    board_flat = board.grid.flatten().astype(np.float32)
    extra = np.array(
        [
            board.white_pos[0],
            board.white_pos[1],
            board.black_pos[0],
            board.black_pos[1],
            1.0 if current_player == "white" else 0.0,
            float(board.brown_apples_remaining),
            1.0 if board.golden_phase_started else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([board_flat, extra])


class PferdeapfelEnv(gym.Env[np.ndarray, int]):
    """
    Single-agent environment for mode 2 (trailing mode).

    The agent controls one side (default: Black). The opponent plays random legal
    moves. Each step corresponds to the agent move; the environment then plays
    the opponent's move before returning the next observation.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        agent_color: str = "black",
        opponent_policy: str = "random",
        illegal_move_penalty: float = -1.0,
        valid_move_reward: float = 0.01,
        win_reward: float = 1.0,
        loss_reward: float = -1.0,
    ) -> None:
        super().__init__()
        if agent_color not in ("white", "black"):
            raise ValueError("agent_color must be 'white' or 'black'")
        self.agent_color = agent_color
        self.opponent_color = "white" if agent_color == "black" else "black"
        self.opponent_policy = opponent_policy
        self.illegal_move_penalty = illegal_move_penalty
        self.valid_move_reward = valid_move_reward
        self.win_reward = win_reward
        self.loss_reward = loss_reward

        # Observation: 64 squares + positions + turn + counts/flags
        high = np.array([4] * (Board.BOARD_SIZE * Board.BOARD_SIZE), dtype=np.float32)
        extra_high = np.array([7, 7, 7, 7, 1, 28, 1], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.concatenate([high, extra_high]),
            shape=(Board.BOARD_SIZE * Board.BOARD_SIZE + extra_high.size,),
            dtype=np.float32,
        )
        # Action: target square index (row * 8 + col)
        self.action_space = spaces.Discrete(Board.BOARD_SIZE * Board.BOARD_SIZE)

        self.board: Board | None = None
        self.current_player: str = self.agent_color
        self._np_random, _ = seeding.np_random()
        self.done = False
        self.info: dict[str, Any] = {}

    def seed(self, seed: Optional[int] = None) -> list[int]:
        """Seed RNG for reproducibility."""
        self._np_random, seed_out = seeding.np_random(seed)
        return [int(seed_out)]

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self.board = Board(mode=2)
        self.done = False
        self.info = {}
        self.current_player = "white"  # White always starts

        # If the opponent is White, let them move first so agent acts next.
        if self.agent_color == "black":
            self._play_opponent_turn()

        obs = board_to_observation(self.board, self.agent_color)
        return obs, self.info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Perform an agent move, then opponent reply."""
        assert self.board is not None, "Call reset() before step()."
        if self.done:
            return board_to_observation(self.board, self.agent_color), 0.0, True, False, self.info

        info: dict[str, Any] = {}
        legal_moves = Rules.get_legal_knight_moves(self.board, self.agent_color)
        info["legal_moves"] = legal_moves
        move = self._action_to_coord(int(action))

        if move not in legal_moves:
            self.done = True
            info["invalid_action"] = True
            return board_to_observation(self.board, self.agent_color), self.illegal_move_penalty, True, False, info

        Rules.make_move(self.board, self.agent_color, move)
        winner = Rules.check_win_condition(self.board, last_mover=self.agent_color)
        if winner:
            self.done = True
            reward = self._reward_from_winner(winner)
            info["winner"] = winner
            return board_to_observation(self.board, self.agent_color), reward, True, False, info

        # Opponent move
        opp_winner = self._play_opponent_turn()
        if opp_winner:
            self.done = True
            info["winner"] = opp_winner
            reward = self._reward_from_winner(opp_winner)
            return board_to_observation(self.board, self.agent_color), reward, True, False, info

        # Standard progress reward
        reward = self.valid_move_reward
        return board_to_observation(self.board, self.agent_color), reward, False, False, info

    def render(self) -> str:
        """Return a string representation of the board."""
        if self.board is None:
            return "Environment not initialized."

        symbols = {
            Board.EMPTY: ".",
            Board.WHITE_HORSE: "W",
            Board.BLACK_HORSE: "B",
            Board.BROWN_APPLE: "o",
            Board.GOLDEN_APPLE: "g",
        }
        lines = []
        grid = self.board.grid
        for r in range(Board.BOARD_SIZE):
            row = "".join(symbols[int(grid[r, c])] for c in range(Board.BOARD_SIZE))
            lines.append(row)
        return "\n".join(lines)

    def _action_to_coord(self, action: int) -> tuple[int, int]:
        row = action // Board.BOARD_SIZE
        col = action % Board.BOARD_SIZE
        return row, col

    def _reward_from_winner(self, winner: str) -> float:
        if winner == self.agent_color:
            return self.win_reward
        if winner == "draw":
            return 0.0
        return self.loss_reward

    def _play_opponent_turn(self) -> Optional[str]:
        """Execute a random opponent move. Returns winner if game ends."""
        assert self.board is not None
        if self.opponent_policy == "none":
            return None

        legal_moves = Rules.get_legal_knight_moves(self.board, self.opponent_color)
        if not legal_moves:
            # Agent wins by immobilization
            winner = Rules.check_win_condition(self.board, last_mover=self.agent_color)
            return winner

        if self.opponent_policy == "random":
            idx = int(self._np_random.integers(low=0, high=len(legal_moves)))
            move = legal_moves[idx]
        else:
            move = legal_moves[0]

        Rules.make_move(self.board, self.opponent_color, move)
        winner = Rules.check_win_condition(self.board, last_mover=self.opponent_color)
        return winner
