import unittest

from src.game.board import Board
from src.game.game import Game
from src.game.rules import Rules
from src.players.base import Player


class MockPlayer(Player):
    def get_move(
        self, board: Board, legal_moves: list[tuple[int, int]]
    ) -> tuple[tuple[int, int], tuple[int, int] | None]:
        return (0, 0), None


class TestGameModes(unittest.TestCase):
    def setUp(self) -> None:
        self.white = MockPlayer("White")
        self.black = MockPlayer("Black")

    def test_mode_1_free_placement(self) -> None:
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

    def test_mode_2_trail_placement(self) -> None:
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

    def test_mode_3_classic(self) -> None:
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

    def test_mode_1_win_condition(self) -> None:
        """Test Mode 1 win condition (cannot move)."""
        game = Game(self.white, self.black, mode=1)
        board = game.board

        # Surround Black with apples/walls so they can't move
        # Black is at (7,7). Moves: (5,6), (6,5)
        board.grid[5, 6] = Board.BROWN_APPLE
        board.grid[6, 5] = Board.BROWN_APPLE
        # Rebuild cache after direct grid modification
        board.rebuild_empty_cache()

        # Check win condition
        # If Black cannot move, White wins.
        winner = Rules.check_win_condition(board)
        self.assertEqual(winner, "white")

    def test_mode_1_apple_on_only_legal_move(self) -> None:
        """Test Mode 1: Placing apple on the only legal move blocks self and fails.
        
        Edge case: In Mode 1, if a player has only one legal move and places
        the apple on that square, the move should fail because:
        1. Apple is placed first (blocking the square)
        2. Then legal moves are calculated (the square is no longer empty)
        3. The intended move is no longer legal
        4. The entire action is rolled back
        """
        game = Game(self.white, self.black, mode=1)
        board = game.board

        # White is at (0,0). Legal moves from (0,0) are: (1,2) and (2,1)
        # Block one of them so White has only one legal move
        board.grid[1, 2] = Board.BROWN_APPLE
        board._mark_occupied(1, 2)

        # Now White's only legal move is (2, 1)
        legal_moves = Rules.get_legal_knight_moves(board, "white")
        self.assertEqual(len(legal_moves), 1)
        self.assertEqual(legal_moves[0], (2, 1))

        # Try to place apple on (2, 1) - the only legal move - and then move there
        # This should FAIL because after placing the apple, (2,1) is no longer empty
        success = Rules.make_move(board, "white", (2, 1), (2, 1))
        self.assertFalse(success, "Should not be able to place apple on own only legal move")

        # Board should be unchanged (rollback)
        self.assertEqual(board.white_pos, (0, 0))
        self.assertEqual(board.grid[0, 0], Board.WHITE_HORSE)
        # The apple should NOT have been placed (rollback)
        self.assertEqual(board.grid[2, 1], Board.EMPTY)

    def test_mode_1_can_place_apple_on_one_of_multiple_moves(self) -> None:
        """Test Mode 1: Can place apple on one legal move if others remain.
        
        If a player has multiple legal moves, they can place an apple on one
        of them - they just can't move there anymore.
        """
        game = Game(self.white, self.black, mode=1)
        board = game.board

        # White is at (0,0). Legal moves: (1,2) and (2,1)
        legal_moves = Rules.get_legal_knight_moves(board, "white")
        self.assertEqual(len(legal_moves), 2)

        # Place apple on (1, 2), move to (2, 1) - this should succeed
        success = Rules.make_move(board, "white", (2, 1), (1, 2))
        self.assertTrue(success, "Should be able to block one move if another remains")

        # Verify the result
        self.assertEqual(board.white_pos, (2, 1))
        self.assertEqual(board.grid[1, 2], Board.BROWN_APPLE)
        self.assertEqual(board.grid[2, 1], Board.WHITE_HORSE)
        self.assertEqual(board.grid[0, 0], Board.EMPTY)


if __name__ == "__main__":
    unittest.main()
