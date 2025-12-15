from __future__ import annotations

from typing import List, Optional, Tuple

from src.game.board import Board
from src.game.game import Game
from src.game.rules import Rules
from src.players.base import Player


class ManualPlayer(Player):
    """A dummy player that does nothing on its own."""

    def get_move(self, board, legal_moves):
        raise NotImplementedError("ManualPlayer should not be asked for moves")


class SimpleDebugEnv:
    """
    A simplified environment for deterministic debugging of game logic.
    Wraps src.game.game.Game.
    """

    def __init__(self, mode: int = 2):
        self.white = ManualPlayer("white")
        self.black = ManualPlayer("black")
        # Disable logging to avoid cluttering disk during exhaustive search
        self.game = Game(self.white, self.black, mode=mode, logging=False)
        self.mode = mode

    def reset(self):
        """Reset the game to initial state."""
        self.game = Game(self.white, self.black, mode=self.mode, logging=False)

    def get_legal_actions(self) -> List[Tuple[Tuple[int, int], Optional[Tuple[int, int]]]]:
        """
        Get all fully specified legal actions (move_to, extra_apple) for the current state.
        Returns a list of tuples.
        """
        if self.game.game_over:
            return []

        legal_moves = Rules.get_legal_knight_moves(self.game.board, self.game.current_player)
        actions = []

        board = self.game.board

        for move in legal_moves:
            if self.mode == 2:
                # Mode 2: No extra apple choice needed
                actions.append((move, None))

            elif self.mode == 1:
                # Mode 1: Move, then place apple
                curr_pos = board.get_horse_position(self.game.current_player)

                # Get all CURRENT empty squares
                empty_sqs = board.get_empty_squares()

                # Logic:
                # Apple can be at `sq` if:
                #   (`sq` is empty NOW AND `sq` != `move`)
                #   OR (`sq` == `curr_pos`)

                valid_apple_spots = []
                # Check all current empty squares
                for r, c in empty_sqs:
                    if (r, c) != move:
                        valid_apple_spots.append((r, c))

                # Add current pos (it becomes empty)
                if curr_pos != move:  # Should always be true for knight move
                    valid_apple_spots.append(curr_pos)

                for apple in valid_apple_spots:
                    actions.append((move, apple))

            elif self.mode == 3:
                # Mode 3: Optional apple.
                # This is complex because of the "cannot block escape" rule.
                # For now, implementing basic logic:
                # Standard: (move, None) is always an option (skip apple)
                actions.append((move, None))

                # Optional apple logic would go here.
                # Given "Start with Mode 2" request, keeping Mode 3 minimal for now.
                pass

        return actions

    def step(self, action: Tuple[Tuple[int, int], Optional[Tuple[int, int]]]) -> bool:
        """
        Apply an action.
        Args:
            action: (move_to, extra_apple)
        Returns:
            success: True if move was valid and applied
        """
        move_to, extra_apple = action
        return self.game.make_move(move_to, extra_apple)

    def undo(self):
        """Undo the last move."""
        self.game.undo_move()

    def get_state_str(self) -> str:
        """Get a string representation of the board for deduplication/logging."""
        # Simple ASCII representation
        chars = [["." for _ in range(8)] for _ in range(8)]
        for r in range(8):
            for c in range(8):
                val = self.game.board.grid[r, c]
                if val == Board.WHITE_HORSE:
                    chars[r][c] = "W"
                elif val == Board.BLACK_HORSE:
                    chars[r][c] = "B"
                elif val == Board.BROWN_APPLE:
                    chars[r][c] = "o"
                elif val == Board.GOLDEN_APPLE:
                    chars[r][c] = "G"

        return "\n".join(["".join(row) for row in chars]) + f"\nTurn: {self.game.current_player}"
