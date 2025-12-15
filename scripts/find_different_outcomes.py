"""Find and analyze games where win conditions might be incorrect.

This script plays games and provides detailed analysis to help identify
cases where the win condition logic might not match the game rules.
"""

import sys
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game.game import Game
from src.game.rules import Rules
from src.players.heuristic_player import HeuristicPlayer
from src.players.random import RandomPlayer


def play_detailed_game(white_cls, black_cls, white_name, black_name, mode=2, seed=None):
    """Play a game and return detailed move-by-move information."""
    if seed is not None:
        random.seed(seed)
    
    white = white_cls(white_name)
    black = black_cls(black_name)
    game = Game(white, black, mode=mode, logging=True)
    game.current_player = "white"
    
    moves = []
    move_num = 0
    
    while not game.game_over and move_num < 200:
        current_turn = game.current_player  # Save who's turn it is
        player = game.get_current_player()
        legal_moves = game.get_legal_moves()
        
        if not legal_moves:
            # Game ended because current player has no moves
            # get_legal_moves() already set winner and game_over
            state = {
                "move": move_num,
                "turn": current_turn,  # The player who was stuck
                "action": "no_legal_moves",
                "white_pos": game.board.white_pos,
                "black_pos": game.board.black_pos,
                "white_can_move": Rules.can_player_move(game.board, "white"),
                "black_can_move": Rules.can_player_move(game.board, "black"),
                "capture": game.board.white_pos == game.board.black_pos,
                "winner": game.winner,
                "game_over": game.game_over,
            }
            moves.append(state)
            break
        
        move_to, extra = player.get_move(game.board, legal_moves)
        
        # Record state before move
        before_state = {
            "white_pos": game.board.white_pos,
            "black_pos": game.board.black_pos,
            "white_can_move": Rules.can_player_move(game.board, "white"),
            "black_can_move": Rules.can_player_move(game.board, "black"),
            "capture": game.board.white_pos == game.board.black_pos,
        }
        
        # Make the move (this may change current_player if game doesn't end)
        success = game.make_move(move_to, extra)
        if not success:
            break
        
        # Record state after move
        after_state = {
            "white_pos": game.board.white_pos,
            "black_pos": game.board.black_pos,
            "white_can_move": Rules.can_player_move(game.board, "white"),
            "black_can_move": Rules.can_player_move(game.board, "black"),
            "capture": game.board.white_pos == game.board.black_pos,
            "winner": game.winner,
            "game_over": game.game_over,
        }
        
        moves.append({
            "move": move_num,
            "turn": current_turn,  # The player who made this move
            "move_to": move_to,
            "extra": extra,
            "legal_moves_count": len(legal_moves),
            "before": before_state,
            "after": after_state,
        })
        
        move_num += 1
    
    return {
        "seed": seed,
        "winner": game.winner,
        "total_moves": move_num,
        "moves": moves,
        "final_white_pos": game.board.white_pos,
        "final_black_pos": game.board.black_pos,
    }


def verify_win_condition(result):
    """Verify if the win condition matches the game rules."""
    if not result["moves"]:
        return None, "No moves recorded"
    
    final_move = result["moves"][-1]
    after = final_move["after"]
    
    # Check win condition according to Mode 2 rules:
    # 1. If positions are equal (capture), winner is the player who made the move
    # 2. If white is stuck, black wins
    # 3. If black is stuck, white wins
    
    expected_winner = None
    reason = None
    
    if after["capture"]:
        # Capture: winner should be the player who made the last move
        expected_winner = final_move["turn"]
        reason = f"Capture: {expected_winner} captured opponent"
    elif not after["white_can_move"]:
        expected_winner = "black"
        reason = "White is stuck (no legal moves)"
    elif not after["black_can_move"]:
        expected_winner = "white"
        reason = "Black is stuck (no legal moves)"
    else:
        return None, "Game ended but no clear win condition"
    
    actual_winner = after.get("winner")
    
    if actual_winner != expected_winner:
        return False, f"MISMATCH: Expected {expected_winner} ({reason}), but got {actual_winner}"
    else:
        return True, f"Correct: {expected_winner} wins ({reason})"


def print_game_analysis(result, show_all_moves=False):
    """Print detailed analysis of a game."""
    print(f"\n{'='*80}")
    print(f"GAME ANALYSIS (Seed: {result['seed']})")
    print(f"{'='*80}")
    print(f"\nWinner: {result['winner']}")
    print(f"Total moves: {result['total_moves']}")
    print(f"Final positions: White={result['final_white_pos']}, Black={result['final_black_pos']}")
    
    # Verify win condition
    is_correct, message = verify_win_condition(result)
    print(f"\nWin Condition Verification: {message}")
    if is_correct is False:
        print(f"  ⚠ POTENTIAL BUG DETECTED!")
    
    # Show key moves
    print(f"\n{'='*80}")
    print("KEY MOVES")
    print(f"{'='*80}")
    
    # Show first 3 moves
    print("\nFirst 3 moves:")
    for move_data in result["moves"][:3]:
        print(f"  Move {move_data['move']}: {move_data['turn']} -> {move_data['move_to']}")
        after = move_data["after"]
        if after.get("game_over"):
            print(f"    Game ended! Winner: {after.get('winner')}")
    
    # Show last 5 moves
    if len(result["moves"]) > 3:
        print("\nLast 5 moves:")
        for move_data in result["moves"][-5:]:
            print(f"  Move {move_data['move']}: {move_data['turn']} -> {move_data['move_to']}")
            after = move_data["after"]
            print(f"    After: White={after['white_pos']}, Black={after['black_pos']}")
            print(f"            White can move: {after['white_can_move']}, Black can move: {after['black_can_move']}")
            print(f"            Capture: {after['capture']}")
            if after.get("game_over"):
                print(f"    ⭐ Game ended! Winner: {after.get('winner')}")
    
    # Show all moves if requested
    if show_all_moves:
        print(f"\n{'='*80}")
        print("ALL MOVES")
        print(f"{'='*80}")
        for move_data in result["moves"]:
            print(f"\nMove {move_data['move']}: {move_data['turn']} -> {move_data['move_to']}")
            before = move_data["before"]
            after = move_data["after"]
            print(f"  Before: White={before['white_pos']}, Black={before['black_pos']}")
            print(f"  After:  White={after['white_pos']}, Black={after['black_pos']}")
            print(f"          Capture: {after['capture']}")
            print(f"          White can move: {after['white_can_move']}, Black can move: {after['black_can_move']}")
            if after.get("game_over"):
                print(f"  ⭐ WINNER: {after.get('winner')}")


def find_problematic_games(white_cls, black_cls, white_name, black_name, mode=2, num_games=500):
    """Find games where win condition might be incorrect."""
    print(f"Searching {num_games} games for potential win condition issues...")
    print("="*80)
    
    problematic = []
    
    for seed in range(num_games):
        if (seed + 1) % 100 == 0:
            print(f"  Checked {seed + 1} games...")
        
        result = play_detailed_game(white_cls, black_cls, white_name, black_name, mode=mode, seed=seed)
        is_correct, message = verify_win_condition(result)
        
        if is_correct is False:
            problematic.append((seed, result, message))
            print(f"  ⚠ Seed {seed}: {message}")
    
    return problematic


def main():
    print("="*80)
    print("WIN CONDITION ANALYZER")
    print("="*80)
    print("\nThis script analyzes games to verify win condition logic is correct.")
    print("It checks if winners match the expected winners based on game rules.\n")
    
    mode = 2
    
    # Test 1: HeuristicPlayer vs RandomPlayer
    print("\n" + "="*80)
    print("TEST 1: HeuristicPlayer (White) vs RandomPlayer (Black)")
    print("="*80)
    
    problematic = find_problematic_games(
        HeuristicPlayer, RandomPlayer, "white", "black", mode=mode, num_games=500
    )
    
    if problematic:
        print(f"\n{'='*80}")
        print(f"Found {len(problematic)} problematic games!")
        print(f"{'='*80}")
        
        # Analyze first problematic game in detail
        seed, result, message = problematic[0]
        print(f"\nAnalyzing first problematic game:")
        print_game_analysis(result, show_all_moves=True)
    else:
        print("\n✓ No problematic games found in win condition logic!")
        print("\nShowing a few sample games for manual inspection:")
        for seed in [0, 1, 5, 10, 20]:
            result = play_detailed_game(HeuristicPlayer, RandomPlayer, "white", "black", mode=mode, seed=seed)
            print_game_analysis(result, show_all_moves=False)
    
    # Test 2: RandomPlayer vs HeuristicPlayer
    print("\n" + "="*80)
    print("TEST 2: RandomPlayer (White) vs HeuristicPlayer (Black)")
    print("="*80)
    
    problematic2 = find_problematic_games(
        RandomPlayer, HeuristicPlayer, "white", "black", mode=mode, num_games=500
    )
    
    if problematic2:
        print(f"\nFound {len(problematic2)} problematic games!")
        seed, result, message = problematic2[0]
        print_game_analysis(result, show_all_moves=True)
    
    # Test 3: Self-play
    print("\n" + "="*80)
    print("TEST 3: HeuristicPlayer vs HeuristicPlayer (Self-play)")
    print("="*80)
    
    problematic3 = find_problematic_games(
        HeuristicPlayer, HeuristicPlayer, "white", "black", mode=mode, num_games=200
    )
    
    if problematic3:
        print(f"\nFound {len(problematic3)} problematic games in self-play!")
        seed, result, message = problematic3[0]
        print_game_analysis(result, show_all_moves=True)
    else:
        print("\n✓ No problematic games found in self-play!")
        print("\nShowing a few sample self-play games:")
        for seed in [0, 1, 2, 5, 10]:
            result = play_detailed_game(HeuristicPlayer, HeuristicPlayer, "white", "black", mode=mode, seed=seed)
            print_game_analysis(result, show_all_moves=False)


if __name__ == "__main__":
    main()
