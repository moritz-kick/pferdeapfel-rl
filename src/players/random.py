"""Random player implementation."""

import random
from typing import Optional, Tuple

from src.game.board import Board
from src.players.base import Player


class RandomPlayer(Player):
    """Random player that chooses moves randomly."""

    def __init__(self, name: str = "Random") -> None:
        """Initialize random player."""
        super().__init__(name)

    def get_move(
        self, board: Board, legal_moves: list[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
        """
        Get a random legal move.

        Args:
            board: Current game board state
            legal_moves: List of legal knight move destinations

        Returns:
            Tuple of (move_to, extra_apple_placement)
        """
        if not legal_moves:
            raise ValueError("No legal moves available")

        # Choose random move
        move_to = random.choice(legal_moves)

        # Randomly decide whether to place extra apple (30% chance)
        extra_placement = None
        if random.random() < 0.3:
            # Find empty squares
            empty_squares = []
            for row in range(board.BOARD_SIZE):
                for col in range(board.BOARD_SIZE):
                    if board.is_empty(row, col):
                        empty_squares.append((row, col))

            if empty_squares:
                # Try a random empty square
                candidate = random.choice(empty_squares)
                # For simplicity, we'll let the Rules class validate
                # if this would block White (for Black player)
                extra_placement = candidate

        return move_to, extra_placement
