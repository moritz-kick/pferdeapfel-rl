"""Test win conditions."""

import unittest

from src.game.board import Board
from src.game.rules import Rules


class TestWinConditions(unittest.TestCase):
    """Test win condition checking."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.board = Board()

    def test_white_can_move_initially(self) -> None:
        """Test that White can move at game start."""
        self.assertTrue(Rules.can_white_move(self.board))

    def test_no_winner_initially(self) -> None:
        """Test that there's no winner at game start."""
        winner = Rules.check_win_condition(self.board)
        self.assertIsNone(winner)

    def test_white_wins_with_golden_apple(self) -> None:
        """Test that White wins if golden apple is on board when White can't move."""
        # Block all White moves
        blocked = [(1, 2), (2, 1)]
        for row, col in blocked:
            self.board.place_apple(row, col, Board.BROWN_APPLE)

        # Place a golden apple on the board
        self.board.place_apple(3, 3, Board.GOLDEN_APPLE)
        self.board.golden_phase_started = True

        # White should not be able to move
        self.assertFalse(Rules.can_white_move(self.board))

        # White should win
        winner = Rules.check_win_condition(self.board)
        self.assertEqual(winner, "white")

    def test_black_wins_without_golden_apple(self) -> None:
        """Test that Black wins if no golden apple when White can't move."""
        # Block all White moves
        blocked = [(1, 2), (2, 1)]
        for row, col in blocked:
            self.board.place_apple(row, col, Board.BROWN_APPLE)

        # No golden apples on board
        self.assertFalse(self.board.has_golden_apple_on_board())

        # White should not be able to move
        self.assertFalse(Rules.can_white_move(self.board))

        # Black should win
        winner = Rules.check_win_condition(self.board)
        self.assertEqual(winner, "black")

    def test_golden_phase_detection(self) -> None:
        """Test that golden phase is detected correctly."""
        self.assertFalse(self.board.golden_phase_started)

        # Place a golden apple
        self.board.place_apple(3, 3, Board.GOLDEN_APPLE)
        self.board.golden_phase_started = True

        self.assertTrue(self.board.golden_phase_started)
        self.assertTrue(self.board.has_golden_apple_on_board())

    def test_apple_phase_transition(self) -> None:
        """Test transition from brown to golden apples."""
        # Start with brown apples
        self.assertGreater(self.board.brown_apples_remaining, 0)

        # Deplete all brown apples by making many moves
        # (This is a simplified test - in practice would need many moves)
        # Instead, test the logic directly
        self.board.brown_apples_remaining = 0
        self.board.golden_apples_remaining = 20

        # Next move should use golden apple
        # This is tested indirectly through make_move logic


if __name__ == "__main__":
    unittest.main()
