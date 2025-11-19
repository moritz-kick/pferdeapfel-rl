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
            # EXCEPTION: Black can move to White's square (capture)
            if board.is_empty(new_row, new_col):
                legal_moves.append((new_row, new_col))
            elif player == "black" and (new_row, new_col) == board.white_pos:
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
                temp_apple = Board.BROWN_APPLE
                board.grid[extra_row, extra_col] = temp_apple
                white_legal = Rules.get_legal_knight_moves(board, "white")
                board.grid[extra_row, extra_col] = Board.EMPTY

                if len(white_legal) == 0:
                    Rules._rollback(board, state_snapshot)
                    return False

                optional_apple = Board.BROWN_APPLE
                if board.brown_apples_remaining > 0:
                    board.brown_apples_remaining -= 1
                else:
                    optional_apple = Board.GOLDEN_APPLE
                    board.golden_apples_remaining -= 1
                    board.golden_phase_started = True

                board.grid[extra_row, extra_col] = optional_apple

        # Save move to history
        board.move_history.append(state_snapshot)
        return True

    @staticmethod
    def _rollback(board: Board, snapshot: dict) -> None:
        """Helper to rollback board state."""
        board.white_pos = snapshot["white_pos"]
        board.black_pos = snapshot["black_pos"]
        board.grid = snapshot["grid"]
        board.brown_apples_remaining = snapshot["brown_apples_remaining"]
        board.golden_apples_remaining = snapshot["golden_apples_remaining"]
        board.golden_phase_started = snapshot["golden_phase_started"]
        if "draw_condition_met" in snapshot:
            board.draw_condition_met = snapshot["draw_condition_met"]
        elif hasattr(board, "draw_condition_met"):
            del board.draw_condition_met

    @staticmethod
    def can_player_move(board: Board, player: str) -> bool:
        """Check if the given player has any legal moves."""
        return len(Rules.get_legal_knight_moves(board, player)) > 0

    @staticmethod
    def can_white_move(board: Board) -> bool:
        """Check if White has any legal moves (legacy helper)."""
        return Rules.can_player_move(board, "white")

    @staticmethod
    def check_win_condition(board: Board) -> Optional[str]:
        """
        Check if the game has ended and return the winner.

        Returns:
            "white" if White wins, "black" if Black wins, None if game continues
        """
        # --- MODE 1 & 2: Survival ---
        if board.mode in [1, 2]:
            # Check if White is stuck
            if not Rules.can_player_move(board, "white"):
                return "black"

            # Check if Black is stuck
            if not Rules.can_player_move(board, "black"):
                return "white"

            return None

        # --- MODE 3: Classic ---
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
            # Check for special 24-point conditions

            # 1. Black immobilized
            if not Rules.can_player_move(board, "black"):
                return 24

            # 2. All 12 Golden Apples used
            if board.golden_apples_remaining == 0:
                return 24

            # Otherwise: 1 point for every Golden Apple on board
            count = 0
            for r in range(Board.BOARD_SIZE):
                for c in range(Board.BOARD_SIZE):
                    if board.grid[r, c] == Board.GOLDEN_APPLE:
                        count += 1
            return count

        return 0
