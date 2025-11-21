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


def legal_action_mask(env: "PferdeapfelEnv") -> np.ndarray:
    """Return a boolean mask over actions for sb3-contrib's ActionMasker."""
    if hasattr(env, "get_action_mask"):
        return env.get_action_mask()
    return np.ones(env.action_space.n, dtype=bool)


class PferdeapfelEnv(gym.Env[np.ndarray, int]):
    """
    Single-agent environment for mode 2 (trailing mode).

    The agent controls one side (default: Black). The opponent plays random legal
    moves (or mirrors the agent in self-play). Each step corresponds to the
    agent move; the environment then plays the opponent's move before returning
    the next observation.

    Set agent_color to "random" to sample the agent side each episode so the
    same policy learns both roles.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        agent_color: str = "black",
        opponent_policy: str = "random",
        opponent_model: Any | None = None,
        opponent_deterministic: bool = True,
        random_opponent_chance: float = 0.0,
        illegal_move_penalty: float = -1.0,
        valid_move_reward: float = 0.01,
        win_reward: float = 1.0,
        loss_reward: float = -1.0,
    ) -> None:
        super().__init__()
        if agent_color not in ("white", "black", "random"):
            raise ValueError("agent_color must be 'white', 'black', or 'random'")
        if opponent_policy not in ("random", "none", "self_play", "self"):
            raise ValueError("opponent_policy must be 'random', 'none', 'self_play', or 'self'")

        self._agent_color_pref = agent_color
        # Initialized placeholder; actual color chosen in reset()
        self.agent_color = "black" if agent_color == "random" else agent_color
        self.opponent_color = "white" if self.agent_color == "black" else "black"
        self.opponent_policy = opponent_policy
        self.opponent_model = opponent_model
        self.opponent_deterministic = opponent_deterministic
        if not 0.0 <= random_opponent_chance <= 1.0:
            raise ValueError("random_opponent_chance must be in [0, 1].")
        self.random_opponent_chance = random_opponent_chance
        self.opponent_best_model: Any | None = None
        self.opponent_old_models: list[Any] = []
        self.opponent_best_prob: float = 0.0
        self.opponent_old_prob: float = 0.0
        self.illegal_move_penalty = illegal_move_penalty
        self.valid_move_reward = valid_move_reward
        self.win_reward = win_reward
        self.loss_reward = loss_reward
        self._opponent_use_random = False
        self._opponent_model_for_episode = opponent_model
        self._opponent_source = "current"

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
        self.agent_color = self._choose_agent_color()
        self.opponent_color = "white" if self.agent_color == "black" else "black"
        self._opponent_use_random = False
        if self.opponent_policy in ("self_play", "self") and self.random_opponent_chance > 0.0:
            self._opponent_use_random = bool(self._np_random.random() < self.random_opponent_chance)
            if self._opponent_use_random:
                logger.debug(
                    "Using random opponent this episode (p=%.2f).", self.random_opponent_chance
                )
        self._choose_opponent_for_episode()
        self.current_player = "white"  # White always starts

        # If the opponent is White, let them move first so agent acts next.
        if self.agent_color == "black":
            self._play_opponent_turn()

        self.current_player = self.agent_color
        obs = board_to_observation(self.board, self.agent_color)
        return obs, self.info

    def get_action_mask(self, player: Optional[str] = None) -> np.ndarray:
        """Compute a mask of legal destination squares for the given player."""
        target_player = player or self.agent_color
        mask = np.zeros(self.action_space.n, dtype=bool)

        if self.board is None or self.done:
            mask[:] = True
            return mask

        legal_moves = Rules.get_legal_knight_moves(self.board, target_player)
        for row, col in legal_moves:
            idx = row * Board.BOARD_SIZE + col
            mask[idx] = True

        # Avoid all-False masks which would break MaskablePPO sampling
        if not np.any(mask):
            mask[:] = True
        return mask

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
        """Execute an opponent move. Returns winner if game ends."""
        assert self.board is not None
        if self.opponent_policy == "none":
            return None

        legal_moves = Rules.get_legal_knight_moves(self.board, self.opponent_color)
        if not legal_moves:
            # Agent wins by immobilization
            winner = Rules.check_win_condition(self.board, last_mover=self.agent_color)
            return winner

        move = self._select_opponent_move(legal_moves)
        Rules.make_move(self.board, self.opponent_color, move)
        winner = Rules.check_win_condition(self.board, last_mover=self.opponent_color)
        self.current_player = self.agent_color
        return winner

    def set_opponent_model(self, model: Any, deterministic: bool = True) -> None:
        """Register a model used for opponent moves in self-play."""
        self.opponent_model = model
        self.opponent_deterministic = deterministic

    def set_opponent_pool(
        self,
        best_model: Any | None,
        old_models: list[Any] | None,
        best_prob: float,
        old_prob: float,
        deterministic: bool = True,
    ) -> None:
        """Configure self-play with a best model and a pool of older snapshots."""
        self.opponent_best_model = best_model
        self.opponent_old_models = old_models or []
        clamped_best = max(0.0, min(1.0, best_prob))
        clamped_old = max(0.0, min(1.0 - clamped_best, old_prob))
        self.opponent_best_prob = clamped_best
        self.opponent_old_prob = clamped_old
        self.opponent_deterministic = deterministic

    def _choose_agent_color(self) -> str:
        """Draw the agent color for this episode."""
        if self._agent_color_pref == "random":
            return "white" if int(self._np_random.integers(low=0, high=2)) == 0 else "black"
        return self._agent_color_pref

    def _choose_opponent_for_episode(self) -> None:
        """Pick which opponent to face this episode."""
        self._opponent_model_for_episode = self.opponent_model
        self._opponent_source = "current"

        if self._opponent_use_random:
            self._opponent_model_for_episode = None
            self._opponent_source = "random"
            return

        r = float(self._np_random.random())
        best_prob = self.opponent_best_prob
        old_prob = self.opponent_old_prob
        if self.opponent_best_model is not None and r < best_prob:
            self._opponent_model_for_episode = self.opponent_best_model
            self._opponent_source = "best"
            return

        if (
            self.opponent_old_models
            and r < best_prob + old_prob
            and len(self.opponent_old_models) > 0
        ):
            idx = int(self._np_random.integers(low=0, high=len(self.opponent_old_models)))
            self._opponent_model_for_episode = self.opponent_old_models[idx]
            self._opponent_source = "old_pool"
            return

        # Default: current ongoing policy
        self._opponent_source = "current"

    def _select_opponent_move(self, legal_moves: list[tuple[int, int]]) -> tuple[int, int]:
        """Choose opponent move based on configured policy."""
        use_random = self._opponent_use_random
        opponent_model = self._opponent_model_for_episode
        if self.opponent_policy in ("self_play", "self") and not use_random and opponent_model is not None:
            assert self.board is not None
            obs = board_to_observation(self.board, self.opponent_color).reshape(1, -1)
            try:
                mask = self.get_action_mask(player=self.opponent_color)
                action, _ = opponent_model.predict(
                    obs, deterministic=self.opponent_deterministic, action_masks=mask
                )
                move = self._action_to_coord(int(action))
                if move in legal_moves:
                    return move
                logger.debug("Opponent predicted illegal move %s; falling back to random.", move)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Self-play opponent predict failed (%s); using random.", exc)

        if self.opponent_policy == "random" or self.opponent_policy in ("self_play", "self"):
            idx = int(self._np_random.integers(low=0, high=len(legal_moves)))
            return legal_moves[idx]

        return legal_moves[0]
