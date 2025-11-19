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


class TestNewRules(unittest.TestCase):
    def setUp(self) -> None:
        self.white = MockPlayer("White")
        self.black = MockPlayer("Black")
        self.game = Game(self.white, self.black)
        self.board = self.game.board

    def test_initial_setup(self) -> None:
        """Test initial board state."""
        self.assertEqual(self.board.brown_apples_remaining, 28)
        self.assertEqual(self.board.golden_apples_remaining, 12)
        self.assertFalse(self.board.golden_phase_started)
        self.assertEqual(self.board.grid[0, 0], Board.WHITE_HORSE)
        self.assertEqual(self.board.grid[7, 7], Board.BLACK_HORSE)

    def test_turn_sequence_brown_phase(self) -> None:
        """Test a standard turn in Brown Phase."""
        # White moves from (0,0) to (1,2)
        # Mandatory: Apple at (0,0)
        # Move: Horse to (1,2)
        # Optional: Apple at (5,5)

        success = Rules.make_move(self.board, "white", (1, 2), (5, 5))
        self.assertTrue(success)

        # Check Mandatory Placement
        self.assertEqual(self.board.grid[0, 0], Board.BROWN_APPLE)

        # Check Move
        self.assertEqual(self.board.grid[1, 2], Board.WHITE_HORSE)
        self.assertEqual(self.board.white_pos, (1, 2))

        # Check Optional Placement
        self.assertEqual(self.board.grid[5, 5], Board.BROWN_APPLE)

        # Check Supply (2 used)
        self.assertEqual(self.board.brown_apples_remaining, 26)

    def test_capture_win(self) -> None:
        """Test Black capturing White."""
        # Setup: White at (0,0), Black at (1,2)
        self.board.grid[7, 7] = Board.EMPTY
        self.board.grid[1, 2] = Board.BLACK_HORSE
        self.board.black_pos = (1, 2)

        # Black moves to (0,0) (Capture)
        success = Rules.make_move(self.board, "black", (0, 0))
        self.assertTrue(success)

        # Check Win Condition
        winner = Rules.check_win_condition(self.board)
        self.assertEqual(winner, "black")

    def test_draw_condition(self) -> None:
        """Test the specific Draw condition."""
        # Setup: 1 Brown apple remaining.
        # Black at (1,2), White at (0,0).
        self.board.brown_apples_remaining = 1
        self.board.grid[7, 7] = Board.EMPTY
        self.board.grid[1, 2] = Board.BLACK_HORSE
        self.board.black_pos = (1, 2)

        # Black moves to (0,0) (Capture)
        # Mandatory placement uses the LAST brown apple.
        # Capture happens in same turn.

        success = Rules.make_move(self.board, "black", (0, 0))
        self.assertTrue(success)

        # Check Supply
        self.assertEqual(self.board.brown_apples_remaining, 0)

        # Check Win Condition
        winner = Rules.check_win_condition(self.board)
        self.assertEqual(winner, "draw")

    def test_golden_phase_transition(self) -> None:
        """Test transition to Golden Phase."""
        self.board.brown_apples_remaining = 0

        # White moves
        # Mandatory: Golden Apple
        success = Rules.make_move(self.board, "white", (1, 2))
        self.assertTrue(success)

        self.assertTrue(self.board.golden_phase_started)
        self.assertEqual(self.board.grid[0, 0], Board.GOLDEN_APPLE)
        self.assertEqual(self.board.golden_apples_remaining, 11)

    def test_white_wins_on_golden_phase_start(self) -> None:
        """Test White winning match when Golden Phase starts."""
        self.board.brown_apples_remaining = 0

        # White moves, placing Golden Apple
        success = Rules.make_move(self.board, "white", (1, 2))
        self.assertTrue(success)

        winner = Rules.check_win_condition(self.board)
        self.assertIsNone(winner)

        # But board state knows golden phase started
        self.assertTrue(self.board.golden_phase_started)

    def test_white_wins_on_golden_exhaustion(self) -> None:
        """Test White winning when Golden Apples run out."""
        self.board.brown_apples_remaining = 0
        self.board.golden_apples_remaining = 1
        self.board.golden_phase_started = True

        # White moves, using last Golden Apple
        success = Rules.make_move(self.board, "white", (1, 2))
        self.assertTrue(success)

        self.assertEqual(self.board.golden_apples_remaining, 0)

        winner = Rules.check_win_condition(self.board)
        self.assertEqual(winner, "white")

    def test_calculate_score(self) -> None:
        """Test score calculation."""
        # Black wins with 10 brown apples left
        self.board.brown_apples_remaining = 10
        score = Rules.calculate_score(self.board, "black")
        self.assertEqual(score, 10)

        self.board.brown_apples_remaining = 0
        self.board.golden_apples_remaining = 5
        self.board.grid[2, 2] = Board.GOLDEN_APPLE
        self.board.grid[3, 3] = Board.GOLDEN_APPLE

        # White wins (e.g. Golden Phase active)
        score = Rules.calculate_score(self.board, "white")
        self.assertEqual(score, 2)  # 2 golden apples on board


if __name__ == "__main__":
    unittest.main()
