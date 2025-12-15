"""Analyze the win condition bug by finding games where outcomes differ."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game.board import Board
from src.game.game import Game
from src.game.rules import Rules
from src.players.heuristic_player import HeuristicPlayer
from src.players.random import RandomPlayer


def play_and_analyze(white_cls, black_cls, white_name, black_name, mode=2, max_games=200):
    """Play games and find ones where buggy vs correct logic differ."""
    print(f"Playing {max_games} games: {white_name} (White) vs {black_name} (Black)")
    print("=" * 80)
    
    divergent_games = []
    
    for game_num in range(1, max_games + 1):
        white = white_cls(white_name)
        black = black_cls(black_name)
        game = Game(white, black, mode=mode, logging=False)
        game.current_player = "white"
        
        moves = 0
        max_moves = 500
        
        # Play the game
        while not game.game_over and moves < max_moves:
            player = game.get_current_player()
            legal_moves = game.get_legal_moves()
            
            if not legal_moves:
                # This is where the bug manifests!
                # get_legal_moves calls check_win_condition without last_mover
                break
            
            move_to, extra = player.get_move(game.board, legal_moves)
            game.make_move(move_to, extra)
            moves += 1
        
        # Now check what the outcome should be
        final_board = game.board
        final_current_player = game.current_player
        
        # BUGGY: What get_legal_moves actually does (calls without last_mover)
        buggy_winner = Rules.check_win_condition(final_board, last_mover=None)
        
        # CORRECT: What it should do (pass the opponent as last_mover)
        # When current_player has no moves, the last_mover is the opponent
        opponent = "black" if final_current_player == "white" else "white"
        correct_winner = Rules.check_win_condition(final_board, last_mover=opponent)
        
        # Also check what the game actually determined
        actual_winner = game.winner
        
        # Check for divergence
        if buggy_winner != correct_winner or (buggy_winner is None and correct_winner is not None):
            print(f"\n>>> DIVERGENCE FOUND in game {game_num} <<<")
            print(f"  Buggy winner (no last_mover): {buggy_winner}")
            print(f"  Correct winner (with last_mover={opponent}): {correct_winner}")
            print(f"  Actual game winner: {actual_winner}")
            print(f"  Final positions: White={final_board.white_pos}, Black={final_board.black_pos}")
            print(f"  White can move: {Rules.can_player_move(final_board, 'white')}")
            print(f"  Black can move: {Rules.can_player_move(final_board, 'black')}")
            print(f"  Current player (stuck): {final_current_player}")
            print(f"  Total moves: {moves}")
            
            # Check if there's a capture
            if final_board.white_pos == final_board.black_pos:
                print(f"  >>> CAPTURE DETECTED! <<<")
            
            divergent_games.append({
                "game_num": game_num,
                "buggy_winner": buggy_winner,
                "correct_winner": correct_winner,
                "actual_winner": actual_winner,
                "final_state": {
                    "white_pos": final_board.white_pos,
                    "black_pos": final_board.black_pos,
                    "white_can_move": Rules.can_player_move(final_board, "white"),
                    "black_can_move": Rules.can_player_move(final_board, "black"),
                    "current_player": final_current_player,
                    "moves": moves,
                }
            })
            
            if len(divergent_games) >= 5:
                break
        
        if game_num % 50 == 0:
            print(f"  Played {game_num} games, found {len(divergent_games)} divergences...")
    
    print(f"\nTotal divergent games found: {len(divergent_games)}")
    return divergent_games


def main():
    print("=" * 80)
    print("WIN CONDITION BUG ANALYSIS")
    print("=" * 80)
    print()
    print("The bug: get_legal_moves() calls check_win_condition(board) without last_mover")
    print("This can cause incorrect winner determination in capture scenarios.")
    print()
    
    # Test different matchups
    matchups = [
        (HeuristicPlayer, RandomPlayer, "Heuristic", "Random"),
        (RandomPlayer, HeuristicPlayer, "Random", "Heuristic"),
        (HeuristicPlayer, HeuristicPlayer, "Heuristic", "Heuristic"),  # Self-play
    ]
    
    all_divergent = []
    
    for white_cls, black_cls, white_name, black_name in matchups:
        print()
        divergent = play_and_analyze(white_cls, black_cls, white_name, black_name, mode=2, max_games=100)
        all_divergent.extend(divergent)
        
        if len(all_divergent) >= 5:
            break
    
    if not all_divergent:
        print("\n" + "=" * 80)
        print("NO DIVERGENCES FOUND")
        print("=" * 80)
        print("\nThis could mean:")
        print("  1. The bug doesn't manifest in these scenarios")
        print("  2. The bug only appears in specific edge cases")
        print("  3. The current implementation handles None last_mover correctly")
        print("\nHowever, the code still has the bug: get_legal_moves should pass last_mover!")
    else:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"\nFound {len(all_divergent)} games with divergent outcomes.")
        print("\nThe fix should be in game.py, get_legal_moves():")
        print("  OLD: self.winner = Rules.check_win_condition(self.board)")
        print("  NEW: opponent = 'black' if self.current_player == 'white' else 'white'")
        print("       self.winner = Rules.check_win_condition(self.board, last_mover=opponent)")


if __name__ == "__main__":
    main()
