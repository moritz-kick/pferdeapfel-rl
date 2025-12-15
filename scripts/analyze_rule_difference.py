"""Analyze games to find where old and new rule implementations differ.

This script plays games and compares outcomes, helping identify bugs in win condition logic.
"""

import sys
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game.game import Game
from src.game.rules import Rules
from src.players.heuristic_player import HeuristicPlayer
from src.players.random import RandomPlayer


def play_and_log_game(white_cls, black_cls, white_name, black_name, mode=2, seed=None, detailed=False):
    """Play a game and return detailed information about each move."""
    if seed is not None:
        random.seed(seed)
    
    white = white_cls(white_name)
    black = black_cls(black_name)
    game = Game(white, black, mode=mode, logging=True)
    game.current_player = "white"
    
    move_history = []
    move_num = 0
    
    while not game.game_over and move_num < 200:
        player = game.get_current_player()
        legal_moves = game.get_legal_moves()
        
        if not legal_moves:
            # Player is stuck
            state = {
                "move": move_num,
                "turn": game.current_player,
                "legal_moves": [],
                "white_pos": game.board.white_pos,
                "black_pos": game.board.black_pos,
                "white_can_move": Rules.can_player_move(game.board, "white"),
                "black_can_move": Rules.can_player_move(game.board, "black"),
                "capture": game.board.white_pos == game.board.black_pos,
                "winner": game.winner,
                "game_over": game.game_over,
            }
            move_history.append(state)
            break
        
        move_to, extra = player.get_move(game.board, legal_moves)
        
        # State before move
        state_before = {
            "white_pos": game.board.white_pos,
            "black_pos": game.board.black_pos,
            "white_can_move": Rules.can_player_move(game.board, "white"),
            "black_can_move": Rules.can_player_move(game.board, "black"),
            "capture": game.board.white_pos == game.board.black_pos,
        }
        
        # Make move
        success = game.make_move(move_to, extra)
        if not success:
            break
        
        # State after move
        state_after = {
            "white_pos": game.board.white_pos,
            "black_pos": game.board.black_pos,
            "white_can_move": Rules.can_player_move(game.board, "white"),
            "black_can_move": Rules.can_player_move(game.board, "black"),
            "capture": game.board.white_pos == game.board.black_pos,
            "winner": game.winner,
            "game_over": game.game_over,
        }
        
        move_history.append({
            "move": move_num,
            "turn": state_before.get("turn", game.current_player),
            "move_to": move_to,
            "extra": extra,
            "legal_moves_before": legal_moves,
            "before": state_before,
            "after": state_after,
        })
        
        move_num += 1
    
    return {
        "winner": game.winner,
        "moves": move_num,
        "move_history": move_history,
        "final_white_pos": game.board.white_pos,
        "final_black_pos": game.board.black_pos,
    }


def check_win_condition_manually(board, last_mover=None):
    """Manually check win condition to verify correctness."""
    # Mode 1 & 2: Survival
    if board.mode in [1, 2]:
        # 1. Capture
        if board.white_pos == board.black_pos:
            return last_mover
        
        # 2. White stuck
        if not Rules.can_player_move(board, "white"):
            return "black"
        
        # 3. Black stuck
        if not Rules.can_player_move(board, "black"):
            return "white"
        
        return None
    
    # Mode 3: Classic (not used in this analysis)
    return None


def analyze_game_detailed(result, seed):
    """Print detailed analysis of a game."""
    print(f"\n{'='*80}")
    print(f"GAME ANALYSIS (Seed: {seed})")
    print(f"{'='*80}")
    print(f"\nWinner: {result['winner']}")
    print(f"Total moves: {result['moves']}")
    print(f"Final positions: White={result['final_white_pos']}, Black={result['final_black_pos']}")
    
    # Check each move
    print(f"\n{'='*80}")
    print("MOVE-BY-MOVE ANALYSIS")
    print(f"{'='*80}")
    
    for i, move_data in enumerate(result['move_history']):
        print(f"\nMove {move_data['move']}: {move_data['turn']} -> {move_data['move_to']}")
        
        before = move_data['before']
        after = move_data['after']
        
        print(f"  Before: White={before['white_pos']}, Black={before['black_pos']}")
        print(f"          White can move: {before['white_can_move']}, Black can move: {before['black_can_move']}")
        print(f"          Capture: {before['capture']}")
        
        print(f"  After:  White={after['white_pos']}, Black={after['black_pos']}")
        print(f"          White can move: {after['white_can_move']}, Black can move: {after['black_can_move']}")
        print(f"          Capture: {after['capture']}")
        print(f"          Winner: {after.get('winner')}")
        print(f"          Game over: {after.get('game_over')}")
        
        # Manual win condition check
        if after.get('game_over'):
            # Recreate board state to check
            print(f"  Manual win check:")
            if after['capture']:
                print(f"    - Capture detected: winner should be the mover ({move_data['turn']})")
            elif not after['white_can_move']:
                print(f"    - White stuck: winner should be black")
            elif not after['black_can_move']:
                print(f"    - Black stuck: winner should be white")
    
    # Final state analysis
    if result['move_history']:
        final = result['move_history'][-1]['after']
        print(f"\n{'='*80}")
        print("FINAL STATE ANALYSIS")
        print(f"{'='*80}")
        print(f"Winner determined: {final.get('winner')}")
        print(f"White position: {final['white_pos']}")
        print(f"Black position: {final['black_pos']}")
        print(f"Positions equal (capture): {final['capture']}")
        print(f"White can move: {final['white_can_move']}")
        print(f"Black can move: {final['black_can_move']}")
        
        # Verify win condition logic
        if final['capture']:
            last_move = result['move_history'][-1]
            expected_winner = last_move['turn']
            print(f"Expected winner (capture): {expected_winner}")
            if final.get('winner') != expected_winner:
                print(f"  ⚠ MISMATCH! Expected {expected_winner}, got {final.get('winner')}")
        elif not final['white_can_move']:
            print(f"Expected winner (white stuck): black")
            if final.get('winner') != 'black':
                print(f"  ⚠ MISMATCH! Expected black, got {final.get('winner')}")
        elif not final['black_can_move']:
            print(f"Expected winner (black stuck): white")
            if final.get('winner') != 'white':
                print(f"  ⚠ MISMATCH! Expected white, got {final.get('winner')}")


def find_suspicious_games(white_cls, black_cls, white_name, black_name, mode=2, num_games=100):
    """Find games with potentially suspicious outcomes."""
    print(f"Searching {num_games} games for suspicious outcomes...")
    print("="*80)
    
    suspicious = []
    
    for seed in range(num_games):
        result = play_and_log_game(white_cls, black_cls, white_name, black_name, mode=mode, seed=seed)
        
        # Check for suspicious patterns
        if result['move_history']:
            final = result['move_history'][-1]['after']
            
            # Check if win condition seems wrong
            issues = []
            
            if final['capture']:
                last_move = result['move_history'][-1]
                expected_winner = last_move['turn']
                if final.get('winner') != expected_winner:
                    issues.append(f"Capture win: expected {expected_winner}, got {final.get('winner')}")
            
            if not final['white_can_move'] and final.get('winner') != 'black':
                issues.append(f"White stuck but winner is {final.get('winner')}")
            
            if not final['black_can_move'] and final.get('winner') != 'white':
                issues.append(f"Black stuck but winner is {final.get('winner')}")
            
            if issues:
                suspicious.append((seed, result, issues))
                print(f"  Seed {seed}: {'; '.join(issues)}")
    
    return suspicious


def main():
    print("="*80)
    print("RULE DIFFERENCE ANALYZER")
    print("="*80)
    print("\nThis script analyzes games to find potential bugs in win condition logic.")
    print("It checks if the determined winner matches the expected winner based on game state.\n")
    
    mode = 2
    
    # Test 1: HeuristicPlayer vs RandomPlayer
    print("\n" + "="*80)
    print("TEST 1: HeuristicPlayer (White) vs RandomPlayer (Black)")
    print("="*80)
    
    suspicious = find_suspicious_games(HeuristicPlayer, RandomPlayer, "white", "black", mode=mode, num_games=200)
    
    if suspicious:
        print(f"\nFound {len(suspicious)} suspicious games!")
        # Analyze the first one in detail
        seed, result, issues = suspicious[0]
        print(f"\nAnalyzing first suspicious game (seed {seed}):")
        analyze_game_detailed(result, seed)
    else:
        print("\nNo suspicious games found. Checking a few games manually...")
        # Analyze a few random games
        for seed in [0, 1, 2, 10, 50]:
            result = play_and_log_game(HeuristicPlayer, RandomPlayer, "white", "black", mode=mode, seed=seed)
            analyze_game_detailed(result, seed)
    
    # Test 2: Self-play
    print("\n" + "="*80)
    print("TEST 2: HeuristicPlayer vs HeuristicPlayer (Self-play)")
    print("="*80)
    
    suspicious2 = find_suspicious_games(HeuristicPlayer, HeuristicPlayer, "white", "black", mode=mode, num_games=100)
    
    if suspicious2:
        print(f"\nFound {len(suspicious2)} suspicious games in self-play!")
        seed, result, issues = suspicious2[0]
        analyze_game_detailed(result, seed)


if __name__ == "__main__":
    main()
