"""Base player class for Pferdeäpfel."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from src.game.board import Board


class Player(ABC):
    """Abstract base class for all players."""

    def __init__(self, name: str) -> None:
        """Initialize a player with a name."""
        self.name = name
        self.last_move_metadata: dict = {}

    @abstractmethod
    def get_move(
        self, board: Board, legal_moves: list[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
        """
        Get a move from the player.

        Args:
            board: Current game board state
            legal_moves: List of legal knight move destinations

        Returns:
            Tuple of (move_to, extra_apple_placement)
            - move_to: (row, col) destination for knight move
            - extra_apple_placement: Optional (row, col) for extra apple, or None
        """
        pass

    def __str__(self) -> str:
        """Return player name."""
        return self.name
