"""Board state management for Pferdeäpfel game."""

from __future__ import annotations

import copy
from typing import Any, Tuple

import numpy as np


class Board:
    """Represents the game board state."""

    # Board constants
    BOARD_SIZE = 8
    EMPTY = 0
    WHITE_HORSE = 1
    BLACK_HORSE = 2
    BROWN_APPLE = 3
    GOLDEN_APPLE = 4

    def __init__(self, mode: int = 3) -> None:
        """Initialize an empty board with horses in starting positions."""
        self.mode = mode
        self.grid = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        # Place horses in opposite corners
        self.grid[0, 0] = self.WHITE_HORSE
        self.grid[7, 7] = self.BLACK_HORSE

        # Track horse positions
        self.white_pos: Tuple[int, int] = (0, 0)
        self.black_pos: Tuple[int, int] = (7, 7)

        # Apple counts
        self.brown_apples_remaining = 28
        self.golden_apples_remaining = 12

        # Track if golden phase has started
        self.golden_phase_started = False

        # Track if golden phase has started
        self.golden_phase_started = False

        # Track if draw condition met (Mode 3)
        self.draw_condition_met = False

        # Move history for undo
        self.move_history: list[dict[str, Any]] = []

    def copy(self) -> Board:
        """Create a deep copy of the board."""
        new_board = Board(mode=self.mode)
        new_board.grid = self.grid.copy()
        new_board.white_pos = self.white_pos
        new_board.black_pos = self.black_pos
        new_board.brown_apples_remaining = self.brown_apples_remaining
        new_board.golden_apples_remaining = self.golden_apples_remaining
        new_board.golden_phase_started = self.golden_phase_started
        if hasattr(self, "draw_condition_met"):
            new_board.draw_condition_met = getattr(self, "draw_condition_met")
        new_board.move_history = copy.deepcopy(self.move_history)
        return new_board

    def get_horse_position(self, player: str) -> Tuple[int, int]:
        """Get the current position of a player's horse."""
        if player == "white":
            return self.white_pos
        return self.black_pos

    def is_valid_square(self, row: int, col: int) -> bool:
        """Check if coordinates are within board bounds."""
        return 0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE

    def is_empty(self, row: int, col: int) -> bool:
        """Check if a square is empty (no horse, no apple)."""
        if not self.is_valid_square(row, col):
            return False
        val = int(self.grid[row, col])
        return val == self.EMPTY

    def get_square(self, row: int, col: int) -> int:
        """Get the value at a square."""
        if not self.is_valid_square(row, col):
            return -1
        return int(self.grid[row, col])

    def place_apple(self, row: int, col: int, apple_type: int) -> bool:
        """Place an apple on the board. Returns True if successful."""
        if not self.is_empty(row, col):
            return False
        self.grid[row, col] = apple_type
        return True

    def has_golden_apple_on_board(self) -> bool:
        """Check if any golden apple exists on the board."""
        return bool(np.any(self.grid == self.GOLDEN_APPLE))
