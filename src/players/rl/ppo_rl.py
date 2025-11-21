"""Maskable PPO-based RL player for mode 2 games."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np

from src.env.pferdeapfel_env import board_to_observation
from src.game.board import Board
from src.players.base import Player

try:  # pragma: no cover - optional dependency guard
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.base_class import BaseAlgorithm
except Exception:  # pragma: no cover - guard import so tests can run without SB3
    MaskablePPO = None
    BaseAlgorithm = Any

logger = logging.getLogger(__name__)


class PPOPlayer(Player):
    """SB3 MaskablePPO-backed player that uses the latest saved PPO model."""

    DISPLAY_NAME = "ppo"

    def __init__(
        self,
        side: str,
        model: BaseAlgorithm | None = None,
        model_path: str | Path | None = None,
        deterministic: bool = True,
        search_roots: Iterable[str | Path] | None = None,
    ) -> None:
        """
        Create a PPO player.

        Args:
            side: "white" or "black".
            model: Optional pre-loaded SB3 model (useful for testing).
            model_path: Explicit path to a .zip model; if omitted, the newest model under
                data/models/ppo_pferdeapfel (or data/models) is used.
            deterministic: Whether to use deterministic actions.
            search_roots: Extra directories/files to scan for the latest model when model_path is omitted.
        """
        side_clean = side.lower()
        if side_clean not in ("white", "black"):
            raise ValueError("side must be 'white' or 'black'")

        display_name = f"{side_clean.capitalize()} PPO"
        super().__init__(display_name)
        self.side = side_clean
        self.deterministic = deterministic
        self.model_path: Path | None = None

        if model is not None:
            self.model = model
            self.model_path = Path(model_path) if model_path else None
        else:
            if MaskablePPO is None:
                msg = "sb3-contrib is required to load PPO models."
                raise ImportError(msg)

            resolved = self._resolve_model_path(model_path, search_roots)
            self.model_path = resolved
            self.model = MaskablePPO.load(str(resolved))
            logger.info("Loaded PPO model from %s", resolved)

        # Emit a warning if the player is used outside the trailing mode this policy was trained on.
        self._warned_mode_mismatch = False

    def get_move(
        self, board: Board, legal_moves: list[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
        """Predict a move using the PPO policy."""
        if not legal_moves:
            raise ValueError("No legal moves available for PPO player.")

        if board.mode != 2 and not self._warned_mode_mismatch:
            logger.warning("PPOPlayer was trained for mode 2; current mode is %s.", board.mode)
            self._warned_mode_mismatch = True

        obs = board_to_observation(board, self.side).reshape(1, -1)
        mask = self._legal_action_mask(legal_moves)

        try:
            action, _ = self.model.predict(obs, deterministic=self.deterministic, action_masks=mask)
            move = self._action_to_coord(int(action))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("PPO predict failed (%s); choosing random legal move.", exc)
            move = random.choice(legal_moves)

        if move not in legal_moves:
            logger.debug("PPO predicted illegal move %s; choosing random legal move.", move)
            move = random.choice(legal_moves)

        return move, None

    @staticmethod
    def _action_to_coord(action: int) -> tuple[int, int]:
        row = action // Board.BOARD_SIZE
        col = action % Board.BOARD_SIZE
        return row, col

    @staticmethod
    def _legal_action_mask(legal_moves: list[tuple[int, int]]) -> np.ndarray:
        """Convert legal moves into an action mask for MaskablePPO."""
        mask = np.zeros(Board.BOARD_SIZE * Board.BOARD_SIZE, dtype=bool)
        for row, col in legal_moves:
            idx = row * Board.BOARD_SIZE + col
            mask[idx] = True

        # Avoid all-False masks which would break MaskablePPO sampling
        if not np.any(mask):
            mask[:] = True
        return mask

    @staticmethod
    def _resolve_model_path(
        model_path: str | Path | None, search_roots: Iterable[str | Path] | None = None
    ) -> Path:
        """
        Locate the newest available PPO model.

        Search order:
        1. Provided model_path (if a file) or newest .zip inside it (if directory).
        2. Provided search_roots (directories/files) for the newest .zip.
        3. data/models/ppo_pferdeapfel then data/models relative to the project root.
        """
        if model_path:
            candidate = Path(model_path).expanduser()
            if candidate.is_file():
                return candidate
            if candidate.is_dir():
                latest = PPOPlayer._latest_zip_in_dirs([candidate])
                if latest:
                    return latest
            raise FileNotFoundError(f"Provided model_path {candidate} not found or contains no .zip files.")

        roots: list[Path] = []
        if search_roots:
            roots.extend(Path(p).expanduser() for p in search_roots)

        project_root = Path(__file__).resolve().parents[3]
        roots.extend(
            [
                project_root / "data" / "models" / "ppo_pferdeapfel",
                project_root / "data" / "models",
            ]
        )

        latest = PPOPlayer._latest_zip_in_dirs(roots)
        if latest:
            return latest

        raise FileNotFoundError("Could not find any PPO model (.zip). Place a trained model under data/models/.")

    @staticmethod
    def _latest_zip_in_dirs(roots: Iterable[Path]) -> Path | None:
        """Return the newest .zip file across the given roots (searching recursively)."""
        candidates: list[Path] = []
        for root in roots:
            if not root:
                continue
            if root.is_file() and root.suffix == ".zip":
                candidates.append(root)
            elif root.is_dir():
                candidates.extend(root.rglob("*.zip"))

        if not candidates:
            return None

        # Deduplicate while preserving order to avoid repeated disk stats
        unique_candidates = list(dict.fromkeys(candidates))
        return max(unique_candidates, key=lambda p: p.stat().st_mtime)


__all__ = ["PPOPlayer"]
