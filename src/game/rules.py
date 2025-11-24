"""Game rules, move validation, and win conditions for Pferdeäpfel."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

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

        opponent_pos = board.white_pos if player == "black" else board.black_pos

        for new_row, new_col in possible_moves:
            # Check if destination is empty (no horse, no apple)
            if board.is_empty(new_row, new_col):
                legal_moves.append((new_row, new_col))
            # EXCEPTION: Capture allowed
            # Mode 1 & 2: Both can capture
            # Mode 3: Only Black can capture White
            elif (new_row, new_col) == opponent_pos:
                if board.mode in [1, 2]:
                    legal_moves.append((new_row, new_col))
                elif board.mode == 3 and player == "black":
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
            extra_apple_placement:
                - Mode 1: The REQUIRED placement BEFORE the move.
                - Mode 2: Ignored.
                - Mode 3: The OPTIONAL placement AFTER the move.

        Returns:
            True if move was successful, False otherwise
        """
        # Save state for undo
        state_snapshot = {
            "white_pos": board.white_pos,
            "black_pos": board.black_pos,
            "grid": board.grid.copy(),
            "brown_apples_remaining": board.brown_apples_remaining,
            "golden_apples_remaining": board.golden_apples_remaining,
            "golden_phase_started": board.golden_phase_started,
            "white_match_win_declared": getattr(board, "white_match_win_declared", False),
            "white_won_in_brown_phase": getattr(board, "white_won_in_brown_phase", False),
        }
        if hasattr(board, "draw_condition_met"):
            state_snapshot["draw_condition_met"] = board.draw_condition_met

        # Initialize draw flag for this move (Mode 3)
        board.draw_condition_met = False

        # --- MODE 1: Free Placement ---
        if board.mode == 1:
            # 1. Place Apple (Required)
            if extra_apple_placement is None:
                return False

            apple_row, apple_col = extra_apple_placement
            if not board.is_empty(apple_row, apple_col):
                return False

            board.grid[apple_row, apple_col] = Board.BROWN_APPLE

            # 2. Move
            legal_moves = Rules.get_legal_knight_moves(board, player)
            if move_to not in legal_moves:
                Rules._rollback(board, state_snapshot)
                return False

            new_row, new_col = move_to
            board.grid[new_row, new_col] = Board.WHITE_HORSE if player == "white" else Board.BLACK_HORSE
            if player == "white":
                board.white_pos = (new_row, new_col)
            else:
                board.black_pos = (new_row, new_col)

            # Clear old position (it's empty now)
            old_pos = state_snapshot["white_pos"] if player == "white" else state_snapshot["black_pos"]
            if isinstance(old_pos, tuple):
                board.grid[old_pos[0], old_pos[1]] = Board.EMPTY

        # --- MODE 2: Trail Placement ---
        elif board.mode == 2:
            # 1. Move
            legal_moves = Rules.get_legal_knight_moves(board, player)
            if move_to not in legal_moves:
                return False

            old_pos = board.get_horse_position(player)
            new_row, new_col = move_to

            board.grid[new_row, new_col] = Board.WHITE_HORSE if player == "white" else Board.BLACK_HORSE
            if player == "white":
                board.white_pos = (new_row, new_col)
            else:
                board.black_pos = (new_row, new_col)

            # 2. Leave Trail (Apple on old position)
            board.grid[old_pos[0], old_pos[1]] = Board.BROWN_APPLE

        # --- MODE 3: Classic ---
        elif board.mode == 3:
            # Validate move first (standard check)
            legal_moves = Rules.get_legal_knight_moves(board, player)
            if move_to not in legal_moves:
                return False

            # 1. Mandatory Placement
            old_pos = board.get_horse_position(player)
            old_row, old_col = old_pos

            mandatory_apple = Board.BROWN_APPLE
            brown_exhausted_on_mandatory = False

            if board.brown_apples_remaining > 0:
                board.brown_apples_remaining -= 1
                if board.brown_apples_remaining == 0:
                    brown_exhausted_on_mandatory = True
            else:
                mandatory_apple = Board.GOLDEN_APPLE
                board.golden_apples_remaining -= 1
                board.golden_phase_started = True
                board.white_match_win_declared = True

            board.grid[old_row, old_col] = mandatory_apple

            # 2. Move
            new_row, new_col = move_to
            captured = False
            if player == "black" and (new_row, new_col) == board.white_pos:
                captured = True
                if brown_exhausted_on_mandatory:
                    board.draw_condition_met = True

            board.grid[new_row, new_col] = Board.WHITE_HORSE if player == "white" else Board.BLACK_HORSE
            if player == "white":
                board.white_pos = (new_row, new_col)
            else:
                board.black_pos = (new_row, new_col)

            # 3. Optional Placement
            if extra_apple_placement is not None and not captured:
                extra_row, extra_col = extra_apple_placement
                if not board.is_empty(extra_row, extra_col):
                    Rules._rollback(board, state_snapshot)
                    return False

                # Restriction: Cannot block White's last escape route
                # Logic: If the apple is placed on a square that is a legal move for White,
                # AND it is the ONLY legal move for White, then it's blocking the last escape.

                # 1. Check if the apple placement is on a square White could move to
                white_legal_before = Rules.get_legal_knight_moves(board, "white")

                if (extra_row, extra_col) in white_legal_before:
                    # 2. If it is a legal move, check if it was the ONLY one
                    if len(white_legal_before) == 1:
                        Rules._rollback(board, state_snapshot)
                        return False

                optional_apple = Board.BROWN_APPLE
                if board.brown_apples_remaining > 0:
                    board.brown_apples_remaining -= 1
                else:
                    optional_apple = Board.GOLDEN_APPLE
                    board.golden_apples_remaining -= 1
                    board.golden_phase_started = True
                    board.white_match_win_declared = True

                board.grid[extra_row, extra_col] = optional_apple

        # Save move to history
        board.move_history.append(state_snapshot)
        return True

    @staticmethod
    def _rollback(board: Board, snapshot: dict[str, Any]) -> None:
        """Helper to rollback board state."""
        board.white_pos = snapshot["white_pos"]
        board.black_pos = snapshot["black_pos"]
        board.grid = snapshot["grid"]
        board.brown_apples_remaining = snapshot["brown_apples_remaining"]
        board.golden_apples_remaining = snapshot["golden_apples_remaining"]
        board.golden_phase_started = snapshot["golden_phase_started"]
        board.white_match_win_declared = snapshot.get("white_match_win_declared", False)
        board.white_won_in_brown_phase = snapshot.get("white_won_in_brown_phase", False)
        if "draw_condition_met" in snapshot:
            board.draw_condition_met = snapshot["draw_condition_met"]
        elif hasattr(board, "draw_condition_met"):
            # If not in snapshot but attribute exists, reset it (or keep as is? logic suggests reset)
            # Actually, if it wasn't in snapshot, it might not have been set before.
            # But we added it to __init__, so it should always be there.
            pass

    @staticmethod
    def can_player_move(board: Board, player: str) -> bool:
        """Check if the given player has any legal moves."""
        return len(Rules.get_legal_knight_moves(board, player)) > 0

    @staticmethod
    def can_white_move(board: Board) -> bool:
        """Check if White has any legal moves (legacy helper)."""
        return Rules.can_player_move(board, "white")

    @staticmethod
    def has_white_clinched(board: Board) -> bool:
        """Return True once White has secured the classic-mode match."""
        return bool(getattr(board, "white_match_win_declared", False))

    @staticmethod
    def check_win_condition(board: Board, last_mover: Optional[str] = None) -> Optional[str]:
        """
        Check if the game has ended and return the winner.

        Args:
            board: The game board
            last_mover: The player who made the last move (important for capture win in Mode 1/2)

        Returns:
            "white" if White wins, "black" if Black wins, None if game continues
        """
        # --- MODE 1 & 2: Survival ---
        if board.mode in [1, 2]:
            # 1. Capture (Immediate Win)
            if board.white_pos == board.black_pos:
                return last_mover

            # 2. Check if White is stuck
            if not Rules.can_player_move(board, "white"):
                return "black"

            # 3. Check if Black is stuck
            if not Rules.can_player_move(board, "black"):
                return "white"

            return None

        # --- MODE 3: Classic ---
        if board.golden_phase_started:
            board.white_match_win_declared = True

        # 1. Draw Condition
        if getattr(board, "draw_condition_met", False):
            return "draw"

        # 2. Capture (Black on White)
        if board.black_pos == board.white_pos:
            # If in Golden Phase, White technically won the match but game ends now.
            # If in Brown Phase, Black wins.
            if board.golden_phase_started:
                return "white"
            else:
                return "black"

        # 3. Immobilization
        white_stuck = not Rules.can_player_move(board, "white")
        black_stuck = not Rules.can_player_move(board, "black")

        if white_stuck:
            if board.golden_phase_started:
                return "white"
            else:
                return "black"

        if black_stuck:
            if not board.golden_phase_started:
                board.white_won_in_brown_phase = True
            board.white_match_win_declared = True
            return "white"

        # 4. Golden Apples Exhausted
        # "All 12 Golden Apples are used and White can still move: White gets 24 points."
        if board.golden_apples_remaining == 0:
            return "white"

        return None

    @staticmethod
    def calculate_score(board: Board, winner: str) -> int:
        """
        Calculate the score for the winner.

        Args:
            board: The game board
            winner: "white" or "black"

        Returns:
            The score.
        """
        if board.mode in [1, 2]:
            return 1  # Simple win

        if winner == "black":
            # 1 point for every unused brown apple
            return board.brown_apples_remaining

        if winner == "white":
            if getattr(board, "white_won_in_brown_phase", False):
                return 12
            # Score is based on golden apples that have been placed during the game.
            total_golden = 12
            golden_used = total_golden - board.golden_apples_remaining

            if board.golden_apples_remaining == 0 and Rules.can_player_move(board, "white"):
                return 24

            return max(golden_used, 0)

        return 0
