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
                    # SIMULATE THE MOVE to get the correct board state for validation
                    # We need to know where the horse will be and what the board looks like
                    # AFTER the mandatory apple and the move.

                    # 1. Simulate Mandatory Placement
                    sim_grid = board.grid.copy()
                    sim_brown_remaining = board.brown_apples_remaining
                    sim_golden_remaining = board.golden_apples_remaining
                    sim_golden_phase = board.golden_phase_started

                    current_pos = board.get_horse_position(self.name.lower())  # "white" or "black"

                    mandatory_apple = Board.BROWN_APPLE
                    if sim_brown_remaining > 0:
                        sim_brown_remaining -= 1
                    else:
                        mandatory_apple = Board.GOLDEN_APPLE
                        sim_golden_remaining -= 1
                        sim_golden_phase = True

                    sim_grid[current_pos[0], current_pos[1]] = mandatory_apple

                    # 2. Simulate Move
                    # If capturing, the opponent is removed (overwritten)
                    sim_grid[move_to[0], move_to[1]] = (
                        Board.WHITE_HORSE if self.name.lower() == "white" else Board.BLACK_HORSE
                    )

                    # Now find empty squares on this SIMULATED board
                    empty_squares = []
                    for row in range(board.BOARD_SIZE):
                        for col in range(board.BOARD_SIZE):
                            # Check simulated grid for emptiness
                            if sim_grid[row, col] == Board.EMPTY:
                                empty_squares.append((row, col))

                    logger.info(f"{self.name}: Found {len(empty_squares)} empty squares (post-move simulation)")

                    # Filter out squares that would block White's last move
                    valid_placements = []
                    from src.game.rules import Rules

                    # Create a temporary board object for Rules validation
                    # We can't easily clone the whole board object, but we can patch it temporarily
                    # Or better, just manually check the rule on the simulated grid.
                    # The rule is: "Cannot block White's last escape route".
                    # This means we need to check White's legal moves on the simulated board.

                    # Helper to get legal moves on a raw grid
                    def get_simulated_legal_moves(grid, player_color, white_p, black_p, mode):
                        moves = []
                        p_pos = white_p if player_color == "white" else black_p
                        if p_pos is None:
                            return []  # Should not happen unless captured?

                        possible = Rules.get_knight_moves(p_pos[0], p_pos[1])
                        opponent_p = white_p if player_color == "black" else black_p

                        for r, c in possible:
                            if grid[r, c] == Board.EMPTY:
                                moves.append((r, c))
                            elif (r, c) == opponent_p:
                                if mode == 3 and player_color == "black":
                                    moves.append((r, c))
                        return moves

                    # Determine simulated positions
                    sim_white_pos = board.white_pos
                    sim_black_pos = board.black_pos

                    if self.name.lower() == "white":
                        sim_white_pos = move_to
                    else:
                        sim_black_pos = move_to
                        # Check capture
                        if move_to == board.white_pos:
                            sim_white_pos = (
                                None  # Captured? Actually in Mode 3 capture ends game usually, but let's handle it.
                            )

                    logger.info(f"{self.name}: Validating placements...")
                    for idx, (r, c) in enumerate(empty_squares):
                        # Temporarily place apple on simulated grid
                        sim_grid[r, c] = Board.BROWN_APPLE  # Assume brown for validation check

                        # Check if White still has moves (if White is not captured)
                        white_has_moves = True
                        if sim_white_pos is not None:
                            white_moves = get_simulated_legal_moves(
                                sim_grid, "white", sim_white_pos, sim_black_pos, board.mode
                            )
                            if len(white_moves) == 0:
                                # Check if it WAS possible before this apple
                                sim_grid[r, c] = Board.EMPTY
                                white_moves_before = get_simulated_legal_moves(
                                    sim_grid, "white", sim_white_pos, sim_black_pos, board.mode
                                )
                                if len(white_moves_before) > 0:
                                    white_has_moves = False  # We blocked the last move

                        # Restore square
                        sim_grid[r, c] = Board.EMPTY

                        if white_has_moves:
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
                # Ensure we don't place the apple on the square we want to move to
                if move_to in empty_squares:
                    empty_squares.remove(move_to)

                if empty_squares:
                    extra_placement = random.choice(empty_squares)
                    logger.info(f"{self.name}: Mode 1, selected extra apple at {extra_placement}")

        logger.info(f"{self.name}: Returning move_to={move_to}, extra_apple={extra_placement}")
        return move_to, extra_placement
