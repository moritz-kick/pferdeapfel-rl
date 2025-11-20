"""DQN-based RL player wrapper for mode 2."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Optional, Tuple

from src.env.pferdeapfel_env import board_to_observation
from src.game.board import Board
from src.players.base import Player

try:  # pragma: no cover - optional dependency
    from stable_baselines3 import DQN
    from stable_baselines3.common.base_class import BaseAlgorithm
except Exception:  # pragma: no cover - external import guard
    DQN = None
    BaseAlgorithm = Any

logger = logging.getLogger(__name__)


class DQNPlayer(Player):
    """SB3 DQN-backed player. Extra apple placement is ignored for mode 2."""

    DISPLAY_NAME = "dqn"

    def __init__(
        self,
        side: str,
        model: Optional[BaseAlgorithm] = None,
        model_path: Optional[str | Path] = None,
        deterministic: bool = True,
    ) -> None:
        """
        Create a DQN player.

        Args:
            side: "white" or "black" indicating which color this player controls.
            model: Preloaded SB3 model. Useful for testing.
            model_path: Optional filesystem path to load a model from.
            deterministic: Whether to use deterministic actions during inference.
        """
        side_clean = side.lower()
        if side_clean not in ("white", "black"):
            raise ValueError("side must be 'white' or 'black'")

        display_name = f"{side_clean.capitalize()} DQN"
        super().__init__(display_name)
        self.side = side_clean
        self.deterministic = deterministic

        if model is not None:
            self.model = model
        elif model_path is not None:
            if DQN is None:
                msg = "stable-baselines3 is required to load DQN models."
                raise ImportError(msg)
            self.model = DQN.load(str(model_path))
        else:
            if DQN is None:
                msg = "stable-baselines3 not installed; cannot create default DQN model."
                raise ImportError(msg)
            # Create an untrained placeholder model with a dummy env for compatibility
            from stable_baselines3.common.env_util import make_vec_env
            from src.env.pferdeapfel_env import PferdeapfelEnv

            env = make_vec_env(lambda: PferdeapfelEnv(agent_color=self.side), n_envs=1)
            self.model = DQN("MlpPolicy", env, verbose=0)

    def get_move(
        self, board: Board, legal_moves: list[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
        """Predict an action and map it to a legal move."""
        if not legal_moves:
            raise ValueError("No legal moves available for RL player.")

        obs = board_to_observation(board, self.side).reshape(1, -1)
        try:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            move = self._action_to_coord(int(action))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Falling back to random move due to predict error: %s", exc)
            move = random.choice(legal_moves)

        if move not in legal_moves:
            logger.debug("Predicted illegal move %s; choosing random legal move.", move)
            move = random.choice(legal_moves)

        return move, None

    def save(self, path: str | Path) -> None:
        """Persist the underlying model."""
        if hasattr(self.model, "save"):
            self.model.save(str(path))

    def _action_to_coord(self, action: int) -> tuple[int, int]:
        row = action // Board.BOARD_SIZE
        col = action % Board.BOARD_SIZE
        return row, col


__all__ = ["DQNPlayer"]
