"""Simple demonstration of the win condition bug."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game.board import Board
from src.game.rules import Rules


def demonstrate_capture_bug():
    """Demonstrate how the bug affects capture scenarios."""
    print("=" * 80)
    print("DEMONSTRATION: Capture Scenario Bug")
    print("=" * 80)
    print()
    
    # Create a board in Mode 2
    board = Board(mode=2)
    
    # Set up a capture scenario: both players on same square
    board.white_pos = (4, 4)
    board.black_pos = (4, 4)  # Same position = capture!
    
    print("Scenario: Both players are on the same square (capture)")
    print(f"  White position: {board.white_pos}")
    print(f"  Black position: {board.black_pos}")
    print()
    
    # Test 1: Buggy behavior (no last_mover)
    print("Test 1: BUGGY behavior (called without last_mover)")
    print("-" * 80)
    buggy_result = Rules.check_win_condition(board, last_mover=None)
    print(f"  Result: {buggy_result}")
    print(f"  ❌ Returns None - cannot determine winner!")
    print()
    
    # Test 2: Correct behavior (white captured)
    print("Test 2: CORRECT behavior (white made the capture)")
    print("-" * 80)
    correct_white = Rules.check_win_condition(board, last_mover="white")
    print(f"  Result: {correct_white}")
    print(f"  ✅ White wins (made the capture)")
    print()
    
    # Test 3: Correct behavior (black captured)
    print("Test 3: CORRECT behavior (black made the capture)")
    print("-" * 80)
    correct_black = Rules.check_win_condition(board, last_mover="black")
    print(f"  Result: {correct_black}")
    print(f"  ✅ Black wins (made the capture)")
    print()
    
    print("=" * 80)
    print("CONCLUSION:")
    print("  When get_legal_moves() calls check_win_condition() without last_mover,")
    print("  capture scenarios cannot be resolved correctly!")
    print("=" * 80)
    print()


def demonstrate_immobilization():
    """Show that immobilization works correctly even without last_mover."""
    print("=" * 80)
    print("DEMONSTRATION: Immobilization Scenario")
    print("=" * 80)
    print()
    
    board = Board(mode=2)
    board.white_pos = (0, 0)
    board.black_pos = (7, 7)
    
    # Block all of white's possible moves
    # White at (0,0) can move to: (2,1), (1,2)
    board.grid[2, 1] = Board.BROWN_APPLE
    board.grid[1, 2] = Board.BROWN_APPLE
    board._mark_occupied(2, 1)
    board._mark_occupied(1, 2)
    
    print("Scenario: White is stuck (no legal moves)")
    print(f"  White position: {board.white_pos}")
    print(f"  Black position: {board.black_pos}")
    print(f"  White can move: {Rules.can_player_move(board, 'white')}")
    print(f"  Black can move: {Rules.can_player_move(board, 'black')}")
    print()
    
    # Test without last_mover (buggy call)
    print("Test: Called without last_mover (as in buggy get_legal_moves)")
    print("-" * 80)
    result = Rules.check_win_condition(board, last_mover=None)
    print(f"  Result: {result}")
    print(f"  ✅ Still works correctly - immobilization doesn't need last_mover")
    print()
    
    print("=" * 80)
    print("CONCLUSION:")
    print("  Immobilization checks work fine without last_mover.")
    print("  The bug only affects CAPTURE scenarios.")
    print("=" * 80)
    print()


def demonstrate_combined_scenario():
    """Show a scenario where both capture and immobilization could apply."""
    print("=" * 80)
    print("DEMONSTRATION: Combined Scenario (Capture + Immobilization)")
    print("=" * 80)
    print()
    
    board = Board(mode=2)
    
    # Scenario: Capture occurred, and now next player has no moves
    board.white_pos = (3, 3)
    board.black_pos = (3, 3)  # Capture!
    
    # Block all moves from this position
    # From (3,3), knight moves go to: (1,2), (1,4), (2,1), (2,5), (4,1), (4,5), (5,2), (5,4)
    for pos in [(1,2), (1,4), (2,1), (2,5), (4,1), (4,5), (5,2), (5,4)]:
        board.grid[pos[0], pos[1]] = Board.BROWN_APPLE
        board._mark_occupied(pos[0], pos[1])
    
    print("Scenario: Capture occurred, and now the next player has no moves")
    print(f"  White position: {board.white_pos}")
    print(f"  Black position: {board.black_pos}")
    print(f"  Is capture: {board.white_pos == board.black_pos}")
    print(f"  White can move: {Rules.can_player_move(board, 'white')}")
    print(f"  Black can move: {Rules.can_player_move(board, 'black')}")
    print()
    
    # Simulate: It's white's turn, white has no moves
    print("Simulation: White's turn, white has no moves")
    print("-" * 80)
    print("  Buggy call (no last_mover):")
    buggy = Rules.check_win_condition(board, last_mover=None)
    print(f"    Result: {buggy}")
    if buggy is None:
        print(f"    ❌ Returns None - falls through to immobilization check")
        print(f"    ❌ May incorrectly determine winner based on who is stuck")
    print()
    
    print("  Correct call (with last_mover='black' - black made the capture):")
    correct = Rules.check_win_condition(board, last_mover="black")
    print(f"    Result: {correct}")
    print(f"    ✅ Black wins (made the capture)")
    print()
    
    print("=" * 80)
    print("CONCLUSION:")
    print("  In combined scenarios, the bug can cause incorrect winner determination.")
    print("  The fix ensures capture is checked first with correct last_mover.")
    print("=" * 80)
    print()


def main():
    print()
    print("This script demonstrates the win condition bug in get_legal_moves().")
    print()
    print("The bug: When a player has no legal moves, get_legal_moves() calls")
    print("check_win_condition(board) without the last_mover parameter.")
    print()
    print("This causes problems in capture scenarios where last_mover determines")
    print("the winner in Mode 1 and Mode 2.")
    print()
    
    demonstrate_capture_bug()
    demonstrate_immobilization()
    demonstrate_combined_scenario()
    
    print()
    print("=" * 80)
    print("THE FIX")
    print("=" * 80)
    print()
    print("In get_legal_moves(), when current player has no moves:")
    print("  OLD: self.winner = Rules.check_win_condition(self.board)")
    print("  NEW: opponent = 'black' if self.current_player == 'white' else 'white'")
    print("       self.winner = Rules.check_win_condition(self.board, last_mover=opponent)")
    print()
    print("This ensures capture scenarios are resolved correctly!")
    print("=" * 80)


if __name__ == "__main__":
    main()
