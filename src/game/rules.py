"""Game rules, move validation, and win conditions for Pferdeäpfel."""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.game.board import Board


class Rules:
    """Handles game rules, legal moves, and win conditions."""

    # Knight move offsets (8 possible knight moves)
    KNIGHT_MOVES = [
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    ]

    @staticmethod
    def get_knight_moves(row: int, col: int) -> List[Tuple[int, int]]:
        """Get all possible knight move destinations from a position."""
        moves = []
        for dr, dc in Rules.KNIGHT_MOVES:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < Board.BOARD_SIZE and 0 <= new_col < Board.BOARD_SIZE:
                moves.append((new_row, new_col))
        return moves

    @staticmethod
    def get_legal_knight_moves(board: Board, player: str) -> List[Tuple[int, int]]:
        """Get legal knight moves for a player (excluding blocked squares)."""
        pos = board.get_horse_position(player)
        row, col = pos
        possible_moves = Rules.get_knight_moves(row, col)
        legal_moves = []

        for new_row, new_col in possible_moves:
            # Check if destination is empty (no horse, no apple)
            if board.is_empty(new_row, new_col):
                legal_moves.append((new_row, new_col))

        return legal_moves

    @staticmethod
    def make_move(
        board: Board,
        player: str,
        move_to: Tuple[int, int],
        extra_apple_placement: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """
        Execute a move for a player.

        Args:
            board: The game board
            player: "white" or "black"
            move_to: Destination square for the knight move
            extra_apple_placement: Optional (row, col) for extra apple placement

        Returns:
            True if move was successful, False otherwise
        """
        # Validate move
        legal_moves = Rules.get_legal_knight_moves(board, player)
        if move_to not in legal_moves:
            return False

        # Save state for undo
        state_snapshot = {
            "white_pos": board.white_pos,
            "black_pos": board.black_pos,
            "grid": board.grid.copy(),
            "white_horse_apples": board.white_horse_apples.copy(),
            "black_horse_apples": board.black_horse_apples.copy(),
            "brown_apples_remaining": board.brown_apples_remaining,
            "golden_apples_remaining": board.golden_apples_remaining,
            "golden_phase_started": board.golden_phase_started,
        }

        # Get current position
        old_pos = board.get_horse_position(player)
        old_row, old_col = old_pos

        # Drop apple from horse onto vacated square
        horse_apples = (
            board.white_horse_apples if player == "white" else board.black_horse_apples
        )
        if horse_apples:
            dropped_apple = horse_apples.pop(0)
            board.grid[old_row, old_col] = dropped_apple
            if dropped_apple == Board.GOLDEN_APPLE:
                board.golden_phase_started = True

        # Move horse
        new_row, new_col = move_to
        board.grid[old_row, old_col] = Board.EMPTY
        board.grid[new_row, new_col] = (
            Board.WHITE_HORSE if player == "white" else Board.BLACK_HORSE
        )

        # Update position tracking
        if player == "white":
            board.white_pos = (new_row, new_col)
        else:
            board.black_pos = (new_row, new_col)

        # Add apple to horse (brown if available, else golden)
        apple_to_add = Board.BROWN_APPLE
        if board.brown_apples_remaining > 0:
            board.brown_apples_remaining -= 1
        else:
            apple_to_add = Board.GOLDEN_APPLE
            board.golden_apples_remaining -= 1
            board.golden_phase_started = True

        horse_apples.append(apple_to_add)

        # Optional extra apple placement
        if extra_apple_placement is not None:
            extra_row, extra_col = extra_apple_placement
            # Validate: square must be empty
            if not board.is_empty(extra_row, extra_col):
                # Rollback move
                board.white_pos = state_snapshot["white_pos"]
                board.black_pos = state_snapshot["black_pos"]
                board.grid = state_snapshot["grid"]
                board.white_horse_apples = state_snapshot["white_horse_apples"]
                board.black_horse_apples = state_snapshot["black_horse_apples"]
                board.brown_apples_remaining = state_snapshot["brown_apples_remaining"]
                board.golden_apples_remaining = state_snapshot["golden_apples_remaining"]
                board.golden_phase_started = state_snapshot["golden_phase_started"]
                return False

            # Check if placement would leave White with no legal moves
            # (only if placing for Black)
            if player == "black":
                # Temporarily place the apple
                board.grid[extra_row, extra_col] = Board.BROWN_APPLE
                white_legal = Rules.get_legal_knight_moves(board, "white")
                board.grid[extra_row, extra_col] = Board.EMPTY

                if len(white_legal) == 0:
                    # Invalid placement - rollback
                    board.white_pos = state_snapshot["white_pos"]
                    board.black_pos = state_snapshot["black_pos"]
                    board.grid = state_snapshot["grid"]
                    board.white_horse_apples = state_snapshot["white_horse_apples"]
                    board.black_horse_apples = state_snapshot["black_horse_apples"]
                    board.brown_apples_remaining = state_snapshot["brown_apples_remaining"]
                    board.golden_apples_remaining = state_snapshot[
                        "golden_apples_remaining"
                    ]
                    board.golden_phase_started = state_snapshot["golden_phase_started"]
                    return False

            # Place the extra apple
            apple_type = Board.BROWN_APPLE
            if board.brown_apples_remaining > 0:
                board.brown_apples_remaining -= 1
            else:
                apple_type = Board.GOLDEN_APPLE
                board.golden_apples_remaining -= 1
                board.golden_phase_started = True

            board.grid[extra_row, extra_col] = apple_type

        # Save move to history
        board.move_history.append(state_snapshot)

        return True

    @staticmethod
    def can_white_move(board: Board) -> bool:
        """Check if White has any legal moves."""
        return len(Rules.get_legal_knight_moves(board, "white")) > 0

    @staticmethod
    def check_win_condition(board: Board) -> Optional[str]:
        """
        Check if the game has ended and return the winner.

        Returns:
            "white" if White wins, "black" if Black wins, None if game continues
        """
        if not Rules.can_white_move(board):
            # White cannot move - game ends
            if board.golden_phase_started or board.has_golden_apple_on_board():
                return "white"
            return "black"
        return None
