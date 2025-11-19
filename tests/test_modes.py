import unittest

from src.game.board import Board
from src.game.game import Game
from src.game.rules import Rules
from src.players.base import Player


class MockPlayer(Player):
    def get_move(self, board):
        return (0, 0), None


class TestGameModes(unittest.TestCase):
    def setUp(self):
        self.white = MockPlayer("White")
        self.black = MockPlayer("Black")

    def test_mode_1_free_placement(self):
        """Test Mode 1: Free Placement (Apple -> Move)."""
        game = Game(self.white, self.black, mode=1)
        board = game.board

        # White turn
        # Try to move without apple (should fail)
        success = Rules.make_move(board, "white", (1, 2), None)
        self.assertFalse(success)

        # Try to place apple on occupied square (should fail)
        success = Rules.make_move(board, "white", (1, 2), (0, 0))  # (0,0) has white horse
        self.assertFalse(success)

        # Valid move: Place apple at (5,5), move to (1,2)
        success = Rules.make_move(board, "white", (1, 2), (5, 5))
        self.assertTrue(success)

        self.assertEqual(board.grid[5, 5], Board.BROWN_APPLE)
        self.assertEqual(board.grid[1, 2], Board.WHITE_HORSE)
        self.assertEqual(board.white_pos, (1, 2))
        # Old position should be empty
        self.assertEqual(board.grid[0, 0], Board.EMPTY)

    def test_mode_2_trail_placement(self):
        """Test Mode 2: Trail Placement (Move -> Leave Trail)."""
        game = Game(self.white, self.black, mode=2)
        board = game.board

        # White turn
        # Move to (1, 2). Extra apple arg should be ignored.
        success = Rules.make_move(board, "white", (1, 2), (5, 5))
        self.assertTrue(success)

        self.assertEqual(board.grid[1, 2], Board.WHITE_HORSE)
        self.assertEqual(board.white_pos, (1, 2))

        # Old position should have an apple
        self.assertEqual(board.grid[0, 0], Board.BROWN_APPLE)

        # (5,5) should be empty (arg ignored)
        self.assertEqual(board.grid[5, 5], Board.EMPTY)

    def test_mode_3_classic(self):
        """Test Mode 3: Classic (Mandatory -> Move -> Optional)."""
        game = Game(self.white, self.black, mode=3)
        board = game.board

        # White turn
        # Move to (1, 2), Optional at (5, 5)
        success = Rules.make_move(board, "white", (1, 2), (5, 5))
        self.assertTrue(success)

        # Mandatory at old pos
        self.assertEqual(board.grid[0, 0], Board.BROWN_APPLE)
        # Horse at new pos
        self.assertEqual(board.grid[1, 2], Board.WHITE_HORSE)
        # Optional at (5, 5)
        self.assertEqual(board.grid[5, 5], Board.BROWN_APPLE)

    def test_mode_1_win_condition(self):
        """Test Mode 1 win condition (cannot move)."""
        game = Game(self.white, self.black, mode=1)
        board = game.board

        # Surround Black with apples/walls so they can't move
        # Black is at (7,7). Moves: (5,6), (6,5)
        board.grid[5, 6] = Board.BROWN_APPLE
        board.grid[6, 5] = Board.BROWN_APPLE

        # Check win condition
        # If Black cannot move, White wins.
        winner = Rules.check_win_condition(board)
        self.assertEqual(winner, "white")


if __name__ == "__main__":
    unittest.main()
