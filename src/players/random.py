"""Random player implementation."""

import logging
import random
from typing import Optional, Tuple

from src.game.board import Board
from src.players.base import Player

logger = logging.getLogger(__name__)


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
        logger.info(f"{self.name}: get_move called with {len(legal_moves)} legal moves")

        if not legal_moves:
            logger.error(f"{self.name}: No legal moves available!")
            raise ValueError("No legal moves available")

        # Choose random move
        move_to = random.choice(legal_moves)
        logger.info(f"{self.name}: Selected move to {move_to}")

        extra_placement = None

        # --- Classic Mode (Mode 3) Logic ---
        if board.mode == 3:
            logger.info(f"{self.name}: Mode 3, brown apples remaining: {board.brown_apples_remaining}")
            # Only in Brown Phase (if apples remaining)
            if board.brown_apples_remaining > 0:
                # Randomly decide whether to place extra apple (e.g., 50% chance)
                place_apple = random.random() < 0.5
                logger.info(f"{self.name}: Decided to {'place' if place_apple else 'skip'} extra apple")

                if place_apple:
                    # Find all empty squares
                    empty_squares = []
                    for row in range(board.BOARD_SIZE):
                        for col in range(board.BOARD_SIZE):
                            if board.is_empty(row, col):
                                empty_squares.append((row, col))

                    logger.info(f"{self.name}: Found {len(empty_squares)} empty squares")

                    # Filter out squares that would block White's last move (if playing as Black)
                    valid_placements = []
                    from src.game.rules import Rules

                    logger.info(f"{self.name}: Validating placements...")
                    for idx, (r, c) in enumerate(empty_squares):
                        # Temporarily place apple
                        original_val = board.grid[r, c]
                        board.grid[r, c] = Board.BROWN_APPLE

                        # Check if White still has moves
                        white_moves = Rules.get_legal_knight_moves(board, "white")

                        # Restore square
                        board.grid[r, c] = original_val

                        if len(white_moves) > 0:
                            valid_placements.append((r, c))

                        if (idx + 1) % 10 == 0:
                            logger.debug(f"{self.name}: Validated {idx + 1}/{len(empty_squares)} squares")

                    logger.info(f"{self.name}: Found {len(valid_placements)} valid placements")
                    if valid_placements:
                        extra_placement = random.choice(valid_placements)
                        logger.info(f"{self.name}: Selected extra apple placement at {extra_placement}")

        # --- Legacy/Other Modes Logic ---
        elif board.mode == 1:
            # Mode 1: Must place apple.
            # Find empty squares
            empty_squares = []
            for row in range(board.BOARD_SIZE):
                for col in range(board.BOARD_SIZE):
                    if board.is_empty(row, col):
                        empty_squares.append((row, col))

            if empty_squares:
                extra_placement = random.choice(empty_squares)
                logger.info(f"{self.name}: Mode 1, selected extra apple at {extra_placement}")

        logger.info(f"{self.name}: Returning move_to={move_to}, extra_apple={extra_placement}")
        return move_to, extra_placement
