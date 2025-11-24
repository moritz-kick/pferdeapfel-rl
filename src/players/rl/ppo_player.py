"""PPO Player implementation."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional, Tuple, Union

from src.game.board import Board
from src.game.rules import Rules
from src.players.base import Player

try:  # pragma: no cover - optional dependency guard
    from stable_baselines3 import PPO
except Exception as exc:  # pragma: no cover - guard import
    PPO = None  # type: ignore[assignment]
    _PPO_IMPORT_ERROR = exc
else:  # pragma: no cover - guard import
    _PPO_IMPORT_ERROR = None

logger = logging.getLogger(__name__)


class PPOPlayer(Player):
    """Player that uses a trained PPO model."""

    DISPLAY_NAME = "PPO"

    def __init__(
        self,
        color: str,
        model_path: Union[str, Path, None],
        *,
        deterministic: bool = True,
    ) -> None:
        """
        Initialize PPO player and load the provided model.

        Args:
            color: Which side this player controls ("white" / "black")
            model_path: Path to the trained model zip file
            deterministic: Whether to use deterministic actions
        """
        if PPO is None:
            raise RuntimeError(f"stable-baselines3 is not available: {_PPO_IMPORT_ERROR}")
        if model_path is None:
            raise ValueError("model_path must be provided for PPOPlayer")

        super().__init__(color.capitalize())
        self.side = color.lower()
        self.model_path = Path(model_path).expanduser()
        self.deterministic = deterministic
        self.model: Optional[PPO] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the PPO model from disk."""
        try:
            self.model = PPO.load(str(self.model_path))
            logger.info("Loaded PPO model from %s", self.model_path)
        except Exception as exc:
            self.model = None
            raise RuntimeError(f"Failed to load PPO model at {self.model_path}: {exc}") from exc

    def get_move(
        self, board: Board, legal_moves: list[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
        """Get move from PPO model."""
        if not legal_moves:
            raise ValueError("No legal moves available for PPO player")

        if self.model is None:
            logger.warning("PPO model missing, falling back to random move")
            return self._fallback_move(board, legal_moves)

        obs = board.grid.copy()
        action, _states = self.model.predict(obs, deterministic=self.deterministic)
        move_idx = int(action[0])
        apple_idx = int(action[1]) if len(action) > 1 else 64

        move_to = self._decode_move(board, move_idx, legal_moves)
        extra_apple = self._decode_apple(board, apple_idx)

        if move_to not in legal_moves:
            logger.warning("PPO predicted illegal move %s, choosing fallback", move_to)
            return self._fallback_move(board, legal_moves)

        return move_to, extra_apple

    def _decode_move(self, board: Board, move_idx: int, legal_moves: list[Tuple[int, int]]) -> Tuple[int, int]:
        """Translate PPO move index into a board coordinate."""
        current_pos = board.get_horse_position(self.side)
        dr, dc = Rules.KNIGHT_MOVES[move_idx % len(Rules.KNIGHT_MOVES)]
        return current_pos[0] + dr, current_pos[1] + dc

    def _decode_apple(self, board: Board, apple_idx: int) -> Optional[Tuple[int, int]]:
        """Translate PPO apple index into coordinates if possible."""
        if apple_idx >= Board.BOARD_SIZE * Board.BOARD_SIZE:
            return None

        row, col = divmod(apple_idx, Board.BOARD_SIZE)
        if not board.is_empty(row, col):
            return None
        return (row, col)

    def _fallback_move(
        self, board: Board, legal_moves: list[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
        """Fallback random policy if model prediction is not usable."""
        move = random.choice(legal_moves)
        extra_apple = None
        if board.mode in (1, 3):
            empty_squares = [(r, c) for r in range(8) for c in range(8) if board.is_empty(r, c)]
            if empty_squares:
                extra_apple = random.choice(empty_squares)
        return move, extra_apple
