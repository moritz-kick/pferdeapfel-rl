"""PPO Player implementation with Action Masking support."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np

from src.game.board import Board
from src.game.rules import Rules
from src.players.base import Player

# Try to import MaskablePPO first (preferred), then fall back to PPO
try:  # pragma: no cover - optional dependency guard
    from sb3_contrib import MaskablePPO
    _MASKABLE_PPO_AVAILABLE = True
except Exception:  # pragma: no cover - guard import
    MaskablePPO = None
    _MASKABLE_PPO_AVAILABLE = False

try:  # pragma: no cover - optional dependency guard
    from stable_baselines3 import PPO
    _PPO_AVAILABLE = True
except Exception as exc:  # pragma: no cover - guard import
    PPO = None  # type: ignore[assignment]
    _PPO_AVAILABLE = False
    _PPO_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)


class PPOPlayer(Player):
    """Player that uses a trained PPO or MaskablePPO model with action masking support."""

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
        if not _MASKABLE_PPO_AVAILABLE and not _PPO_AVAILABLE:
            raise RuntimeError("Neither sb3-contrib nor stable-baselines3 is available")
        if model_path is None:
            raise ValueError("model_path must be provided for PPOPlayer")

        super().__init__(color.capitalize())
        self.side = color.lower()
        self.model_path = Path(model_path).expanduser()
        self.deterministic = deterministic
        self.model: Optional[Any] = None
        self.is_maskable = False
        self._load_model()

    def _load_model(self) -> None:
        """Load the PPO model from disk. Tries MaskablePPO first, then PPO."""
        # Try MaskablePPO first
        if _MASKABLE_PPO_AVAILABLE:
            try:
                self.model = MaskablePPO.load(str(self.model_path))
                self.is_maskable = True
                logger.info("Loaded MaskablePPO model from %s", self.model_path)
                return
            except Exception as exc:
                logger.debug("Failed to load as MaskablePPO: %s", exc)
        
        # Fall back to regular PPO
        if _PPO_AVAILABLE:
            try:
                self.model = PPO.load(str(self.model_path))
                self.is_maskable = False
                logger.info("Loaded PPO model from %s", self.model_path)
                return
            except Exception as exc:
                self.model = None
                raise RuntimeError(f"Failed to load PPO model at {self.model_path}: {exc}") from exc
        
        raise RuntimeError(f"Could not load model at {self.model_path}")

    def _get_action_masks(self, board: Board) -> np.ndarray:
        """Compute flattened action mask (length 73) for MaskablePPO.

        Action space = MultiDiscrete([8, 65]) -> flattened ordering:
        [move_0 .. move_7, apple_0 .. apple_63, apple_no_choice]

        Returns:
            1D boolean numpy array of length 73.
        """
        current_pos = board.get_horse_position(self.side)
        
        # --- Mask for knight moves (8 possible moves) ---
        move_mask = np.zeros(8, dtype=bool)
        legal_moves = Rules.get_legal_knight_moves(board, self.side)
        legal_move_set = set(legal_moves)
        
        for idx, (dr, dc) in enumerate(Rules.KNIGHT_MOVES):
            target = (current_pos[0] + dr, current_pos[1] + dc)
            if target in legal_move_set:
                move_mask[idx] = True
        
        # --- Mask for apple placement (64 squares + 1 for "no apple") ---
        apple_mask = np.zeros(65, dtype=bool)
        
        if board.mode == 1:
            # Mode 1: Apple placement is REQUIRED before moving
            for r in range(8):
                for c in range(8):
                    if board.is_empty(r, c):
                        apple_mask[r * 8 + c] = True
        elif board.mode == 2:
            # Mode 2: Apple is automatic (trail), only "no apple" valid
            apple_mask[64] = True
        elif board.mode == 3:
            # Mode 3: Apple placement is OPTIONAL
            # Restriction: Cannot block White's last remaining escape route
            white_legal_moves = Rules.get_legal_knight_moves(board, "white")
            white_legal_set = set(white_legal_moves)
            
            for r in range(8):
                for c in range(8):
                    if board.is_empty(r, c):
                        # Check if this placement would block White's only escape
                        if (r, c) in white_legal_set and len(white_legal_moves) == 1:
                            # This would block White's last escape - not allowed
                            continue
                        apple_mask[r * 8 + c] = True
            apple_mask[64] = True
        
        # Safety: ensure at least one action is valid
        if not move_mask.any():
            move_mask[:] = True
        if not apple_mask.any():
            apple_mask[64] = True
        
        return np.concatenate([move_mask, apple_mask])

    def get_move(
        self, board: Board, legal_moves: list[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
        """Get move from PPO model with action masking support."""
        if not legal_moves:
            raise ValueError("No legal moves available for PPO player")

        if self.model is None:
            logger.warning("PPO model missing, falling back to random move")
            self.last_move_metadata = {"source": "fallback", "reason": "no_model"}
            return self._fallback_move(board, legal_moves)

        # Construct observation in (7, 8, 8) format
        obs = np.zeros((7, 8, 8), dtype=np.float32)

        my_pos = board.get_horse_position(self.side)
        opp_side = "black" if self.side == "white" else "white"
        opp_pos = board.get_horse_position(opp_side)

        obs[0, my_pos[0], my_pos[1]] = 1.0
        obs[1, opp_pos[0], opp_pos[1]] = 1.0

        for r in range(8):
            for c in range(8):
                if not board.is_empty(r, c):
                    obs[2, r, c] = 1.0

        # Channel 3: Mode ID (Normalized)
        mode_val = 0.0
        if board.mode == 2:
            mode_val = 0.5
        elif board.mode == 3:
            mode_val = 1.0
        obs[3, :, :] = mode_val

        # Channel 4: Current Role
        role_val = 1.0 if self.side == "white" else 0.0
        obs[4, :, :] = role_val

        # Channel 5: Brown Apples Remaining (Normalized)
        brown_val = board.brown_apples_remaining / 28.0
        obs[5, :, :] = brown_val

        # Channel 6: Golden Apples Remaining (Normalized)
        golden_val = board.golden_apples_remaining / 12.0
        obs[6, :, :] = golden_val

        # Predict with action masks if using MaskablePPO
        if self.is_maskable:
            action_masks = self._get_action_masks(board)
            action, _states = self.model.predict(
                obs, deterministic=self.deterministic, action_masks=action_masks
            )
        else:
            action, _states = self.model.predict(obs, deterministic=self.deterministic)
        
        move_idx = int(action[0])
        apple_idx = int(action[1]) if len(action) > 1 else 64

        move_to = self._decode_move(board, move_idx, legal_moves)
        extra_apple = self._decode_apple(board, apple_idx)

        # Validate move and extra apple by simulating it
        if move_to not in legal_moves:
            logger.warning(
                "PPO predicted illegal move %s (legal: %s). Falling back to random.",
                move_to,
                legal_moves,
            )
            self.last_move_metadata = {"source": "fallback", "reason": "illegal_move"}
            return self._fallback_move(board, legal_moves)

        # Simulate to check if extra apple placement is valid (especially for Mode 3)
        board_copy = board.copy()
        if not Rules.make_move(board_copy, self.side, move_to, extra_apple):
            logger.warning(
                "PPO predicted invalid move/placement (move=%s, apple=%s). Falling back to random.",
                move_to,
                extra_apple,
            )
            self.last_move_metadata = {"source": "fallback", "reason": "invalid_placement"}
            return self._fallback_move(board, legal_moves)

        self.last_move_metadata = {"source": "model", "reason": "prediction"}
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
