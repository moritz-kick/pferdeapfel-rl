"""Human player implementation (for console input)."""

from typing import Optional, Tuple

from src.game.board import Board
from src.players.base import Player


class HumanPlayer(Player):
    """Human player that gets moves from console input."""

    def __init__(self, name: str = "Human") -> None:
        """Initialize human player."""
        super().__init__(name)

    def get_move(
        self, board: Board, legal_moves: list[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
        """
        Get move from console input.

        Args:
            board: Current game board state
            legal_moves: List of legal knight move destinations

        Returns:
            Tuple of (move_to, extra_apple_placement)
        """
        if not legal_moves:
            raise ValueError("No legal moves available")

        print(f"\n{self.name}'s turn. Legal moves:")
        for i, (row, col) in enumerate(legal_moves):
            print(f"  {i}: ({row}, {col})")

        while True:
            try:
                choice = input("Select move (index): ").strip()
                move_idx = int(choice)
                if 0 <= move_idx < len(legal_moves):
                    move_to = legal_moves[move_idx]
                    break
                print("Invalid index. Try again.")
            except (ValueError, IndexError):
                print("Invalid input. Enter a number.")

        # Ask for optional extra apple placement
        extra_placement = None
        place_extra = input("Place extra apple? (y/n): ").strip().lower()
        if place_extra == "y":
            while True:
                try:
                    row = int(input("Row (0-7): ").strip())
                    col = int(input("Col (0-7): ").strip())
                    if board.is_empty(row, col):
                        extra_placement = (row, col)
                        break
                    print("Square is not empty. Try again.")
                except ValueError:
                    print("Invalid input. Enter numbers.")

        return move_to, extra_placement
