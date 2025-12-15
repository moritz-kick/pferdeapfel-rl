"""Test script to verify the win condition fix works correctly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game.board import Board
from src.game.game import Game
from src.game.rules import Rules
from src.players.heuristic_player import HeuristicPlayer
from src.players.random import RandomPlayer


def test_capture_scenario():
    """Test that capture scenarios are handled correctly."""
    print("Test 1: Capture scenario")
    print("-" * 60)
    
    # Create a scenario where capture happens
    game = Game(RandomPlayer("white"), RandomPlayer("black"), mode=2, logging=False)
    
    # Manually set up a capture scenario
    game.board.white_pos = (3, 3)
    game.board.black_pos = (3, 3)  # Same position = capture
    
    # Test old buggy behavior (no last_mover)
    buggy_winner = Rules.check_win_condition(game.board, last_mover=None)
    print(f"  Buggy (no last_mover): {buggy_winner}")
    
    # Test correct behavior
    correct_winner_white = Rules.check_win_condition(game.board, last_mover="white")
    correct_winner_black = Rules.check_win_condition(game.board, last_mover="black")
    print(f"  Correct (last_mover=white): {correct_winner_white}")
    print(f"  Correct (last_mover=black): {correct_winner_black}")
    
    if buggy_winner != correct_winner_white and buggy_winner != correct_winner_black:
        print("  >>> Bug confirmed: None last_mover returns wrong result! <<<")
    print()


def test_immobilization_scenario():
    """Test that immobilization scenarios work correctly."""
    print("Test 2: Immobilization scenario")
    print("-" * 60)
    
    game = Game(RandomPlayer("white"), RandomPlayer("black"), mode=2, logging=False)
    
    # Set up a scenario where white is stuck
    game.board.white_pos = (0, 0)
    game.board.black_pos = (7, 7)
    
    # Block all white's possible moves
    # White at (0,0) can move to: (2,1), (1,2)
    game.board.grid[2, 1] = Board.BROWN_APPLE
    game.board.grid[1, 2] = Board.BROWN_APPLE
    game.board._mark_occupied(2, 1)
    game.board._mark_occupied(1, 2)
    
    # Test both behaviors
    buggy_winner = Rules.check_win_condition(game.board, last_mover=None)
    correct_winner = Rules.check_win_condition(game.board, last_mover="black")
    
    print(f"  White can move: {Rules.can_player_move(game.board, 'white')}")
    print(f"  Black can move: {Rules.can_player_move(game.board, 'black')}")
    print(f"  Buggy (no last_mover): {buggy_winner}")
    print(f"  Correct (last_mover=black): {correct_winner}")
    
    if buggy_winner == correct_winner == "black":
        print("  >>> Both agree: Black wins (correct) <<<")
    else:
        print("  >>> Disagreement detected! <<<")
    print()


def play_full_game_and_check():
    """Play a full game and check if the fix works."""
    print("Test 3: Full game simulation")
    print("-" * 60)
    
    white = HeuristicPlayer("white")
    black = RandomPlayer("black")
    game = Game(white, black, mode=2, logging=False)
    game.current_player = "white"
    
    moves = 0
    max_moves = 500
    
    print("  Playing game...")
    while not game.game_over and moves < max_moves:
        player = game.get_current_player()
        legal_moves = game.get_legal_moves()
        
        if not legal_moves:
            print(f"  Game ended: {player} has no legal moves")
            break
        
        move_to, extra = player.get_move(game.board, legal_moves)
        game.make_move(move_to, extra)
        moves += 1
    
    print(f"  Total moves: {moves}")
    print(f"  Winner: {game.winner}")
    print(f"  Final positions: White={game.board.white_pos}, Black={game.board.black_pos}")
    print(f"  White can move: {Rules.can_player_move(game.board, 'white')}")
    print(f"  Black can move: {Rules.can_player_move(game.board, 'black')}")
    
    # Verify the winner is correct
    if game.board.white_pos == game.board.black_pos:
        print("  >>> Capture detected - winner should be last_mover <<<")
    elif not Rules.can_player_move(game.board, "white"):
        print("  >>> White stuck - Black should win <<<")
        if game.winner == "black":
            print("  ✓ Correct winner determined")
        else:
            print("  ✗ Wrong winner!")
    elif not Rules.can_player_move(game.board, "black"):
        print("  >>> Black stuck - White should win <<<")
        if game.winner == "white":
            print("  ✓ Correct winner determined")
        else:
            print("  ✗ Wrong winner!")
    print()


def main():
    print("=" * 60)
    print("WIN CONDITION FIX VERIFICATION")
    print("=" * 60)
    print()
    
    test_capture_scenario()
    test_immobilization_scenario()
    play_full_game_and_check()
    
    print("=" * 60)
    print("Note: The fix ensures that when get_legal_moves() detects")
    print("a player has no moves, it passes the opponent as last_mover")
    print("to check_win_condition(). This is critical for capture")
    print("scenarios in Mode 1/2 where last_mover determines the winner.")
    print("=" * 60)


if __name__ == "__main__":
    main()
