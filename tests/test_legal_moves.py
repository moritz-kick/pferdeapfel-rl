"""Test legal move generation."""

import unittest

from src.game.board import Board
from src.game.rules import Rules


class TestLegalMoves(unittest.TestCase):
    """Test legal move generation."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.board = Board()

    def test_initial_white_moves(self) -> None:
        """Test that White has legal moves from starting position."""
        moves = Rules.get_legal_knight_moves(self.board, "white")
        # From (0,0), knight can move to (1,2) and (2,1)
        self.assertGreater(len(moves), 0)
        self.assertIn((1, 2), moves)
        self.assertIn((2, 1), moves)

    def test_initial_black_moves(self) -> None:
        """Test that Black has legal moves from starting position."""
        moves = Rules.get_legal_knight_moves(self.board, "black")
        # From (7,7), knight can move to (6,5) and (5,6)
        self.assertGreater(len(moves), 0)
        self.assertIn((6, 5), moves)
        self.assertIn((5, 6), moves)

    def test_blocked_square(self) -> None:
        """Test that blocked squares are not legal moves."""
        # Place an apple on (1,2)
        self.board.place_apple(1, 2, Board.BROWN_APPLE)
        moves = Rules.get_legal_knight_moves(self.board, "white")
        # (1,2) should not be in legal moves
        self.assertNotIn((1, 2), moves)

    def test_no_moves_when_blocked(self) -> None:
        """Test that a player with no legal moves returns empty list."""
        # Create a board where White is completely blocked
        # This is a simplified test - in practice, this would require many moves
        board = Board()
        # Block all 8 possible knight moves from (0,0)
        blocked_squares = [(1, 2), (2, 1)]
        for row, col in blocked_squares:
            board.place_apple(row, col, Board.BROWN_APPLE)

        moves = Rules.get_legal_knight_moves(board, "white")
        self.assertEqual(len(moves), 0)

    def test_knight_move_pattern(self) -> None:
        """Test that knight moves follow L-shape pattern."""
        moves = Rules.get_knight_moves(3, 3)
        expected = [
            (1, 2),
            (1, 4),
            (2, 1),
            (2, 5),
            (4, 1),
            (4, 5),
            (5, 2),
            (5, 4),
        ]
        self.assertEqual(set(moves), set(expected))

    def test_move_validation(self) -> None:
        """Test that invalid moves are rejected."""
        # Try to move to an occupied square
        success = Rules.make_move(self.board, "white", (7, 7))  # Black's position
        self.assertFalse(success)

        # Try to move to a square outside the board
        # This should be caught by get_legal_knight_moves, but test make_move too
        legal_moves = Rules.get_legal_knight_moves(self.board, "white")
        self.assertNotIn((10, 10), legal_moves)


if __name__ == "__main__":
    unittest.main()
