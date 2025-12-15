"""Find and analyze games where old (buggy) and new (fixed) rules produce different outcomes."""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game.board import Board
from src.game.game import Game
from src.game.rules import Rules
from src.players.heuristic_player import HeuristicPlayer
from src.players.random import RandomPlayer


class GameWithRecording:
    """Wrapper to record game state at each move."""
    
    def __init__(self, white_cls, black_cls, white_name, black_name, mode=2):
        self.white = white_cls(white_name)
        self.black = black_cls(black_name)
        self.game = Game(self.white, self.black, mode=mode, logging=True)
        self.game.current_player = "white"
        self.move_history = []
    
    def play_full_game(self, max_moves=500):
        """Play the game and record all moves."""
        moves = 0
        
        while not self.game.game_over and moves < max_moves:
            player = self.game.get_current_player()
            legal_moves = self.game.get_legal_moves()
            
            if not legal_moves:
                # Record the stuck state
                self._record_state(moves, "stuck", None, None)
                break
            
            move_to, extra = player.get_move(self.game.board, legal_moves)
            
            # Record state before move
            state_before = self._capture_state()
            
            success = self.game.make_move(move_to, extra)
            if not success:
                break
            
            # Record state after move
            state_after = self._capture_state()
            
            self.move_history.append({
                "move_num": moves + 1,
                "player": player.name,
                "move_to": move_to,
                "extra": extra,
                "before": state_before,
                "after": state_after,
            })
            
            moves += 1
        
        return {
            "total_moves": moves,
            "winner": self.game.winner,
            "final_state": self._capture_state(),
            "move_history": self.move_history,
        }
    
    def _capture_state(self):
        """Capture current game state."""
        board = self.game.board
        return {
            "white_pos": board.white_pos,
            "black_pos": board.black_pos,
            "white_can_move": Rules.can_player_move(board, "white"),
            "black_can_move": Rules.can_player_move(board, "black"),
            "current_player": self.game.current_player,
            "game_over": self.game.game_over,
            "winner": self.game.winner,
            "is_capture": board.white_pos == board.black_pos,
        }
    
    def _record_state(self, move_num, reason, move_to, extra):
        """Record a special state (e.g., stuck player)."""
        state = self._capture_state()
        self.move_history.append({
            "move_num": move_num + 1,
            "player": self.game.current_player,
            "reason": reason,
            "move_to": move_to,
            "extra": extra,
            "before": state,
            "after": state,
        })


def check_win_condition_buggy(board: Board, last_mover: Optional[str] = None) -> Optional[str]:
    """Simulate the buggy behavior: called without last_mover."""
    # This is what happens when get_legal_moves calls check_win_condition(board)
    return Rules.check_win_condition(board, last_mover=last_mover)


def check_win_condition_fixed(board: Board, current_player: str) -> Optional[str]:
    """Fixed behavior: correctly determines last_mover."""
    # When current_player has no moves, opponent was the last mover
    opponent = "black" if current_player == "white" else "white"
    return Rules.check_win_condition(board, last_mover=opponent)


def analyze_game_record(record):
    """Analyze a game record to find where buggy vs fixed logic differ."""
    final = record["final_state"]
    moves = record["move_history"]
    
    # Check final state
    current_player = final["current_player"]
    
    # Buggy: called without last_mover (what old code does)
    buggy_winner = check_win_condition_buggy(final, last_mover=None)
    
    # Fixed: correctly passes opponent as last_mover
    fixed_winner = check_win_condition_fixed(final, current_player) if not final["white_can_move"] or not final["black_can_move"] else None
    
    # Also check each move to see where divergence first appears
    divergence_point = None
    for move_data in moves:
        after = move_data["after"]
        if not after["white_can_move"] or not after["black_can_move"]:
            stuck_player = "white" if not after["white_can_move"] else "black"
            buggy = check_win_condition_buggy(after, last_mover=None)
            fixed = check_win_condition_fixed(after, stuck_player)
            
            if buggy != fixed:
                divergence_point = move_data["move_num"]
                break
    
    return {
        "buggy_winner": buggy_winner,
        "fixed_winner": fixed_winner,
        "actual_winner": final["winner"],
        "divergence_point": divergence_point,
        "final_state": final,
    }


def find_divergent_games(num_games=100, mode=2):
    """Find games where buggy and fixed logic produce different outcomes."""
    print("=" * 80)
    print("SEARCHING FOR DIVERGENT GAMES")
    print("=" * 80)
    print()
    
    matchups = [
        (HeuristicPlayer, RandomPlayer, "Heuristic", "Random"),
        (RandomPlayer, HeuristicPlayer, "Random", "Heuristic"),
        (HeuristicPlayer, HeuristicPlayer, "Heuristic", "Heuristic"),  # Self-play
        (RandomPlayer, RandomPlayer, "Random", "Random"),  # Self-play
    ]
    
    all_divergent = []
    
    for white_cls, black_cls, white_name, black_name in matchups:
        print(f"Testing: {white_name} (White) vs {black_name} (Black)")
        print("-" * 80)
        
        for i in range(num_games):
            recorder = GameWithRecording(white_cls, black_cls, white_name, black_name, mode=mode)
            record = recorder.play_full_game()
            
            analysis = analyze_game_record(record)
            
            # Check if buggy and fixed produce different results
            if analysis["buggy_winner"] != analysis["fixed_winner"]:
                print(f"\n>>> DIVERGENCE FOUND in game {i+1} <<<")
                print(f"  Buggy winner: {analysis['buggy_winner']}")
                print(f"  Fixed winner: {analysis['fixed_winner']}")
                print(f"  Actual winner: {analysis['actual_winner']}")
                print(f"  Divergence at move: {analysis['divergence_point']}")
                
                all_divergent.append({
                    "game_num": i + 1,
                    "matchup": f"{white_name} vs {black_name}",
                    "record": record,
                    "analysis": analysis,
                })
                
                if len(all_divergent) >= 3:
                    break
        
        if len(all_divergent) >= 3:
            break
        
        print(f"  Played {num_games} games...")
        print()
    
    return all_divergent


def print_detailed_analysis(divergent_game):
    """Print detailed analysis of a divergent game."""
    game_num = divergent_game["game_num"]
    matchup = divergent_game["matchup"]
    record = divergent_game["record"]
    analysis = divergent_game["analysis"]
    
    print("=" * 80)
    print(f"DETAILED ANALYSIS: Game {game_num}")
    print(f"Matchup: {matchup}")
    print("=" * 80)
    print()
    
    print(f"Total moves: {record['total_moves']}")
    print()
    
    final = analysis["final_state"]
    print("FINAL STATE:")
    print(f"  White position: {final['white_pos']}")
    print(f"  Black position: {final['black_pos']}")
    print(f"  White can move: {final['white_can_move']}")
    print(f"  Black can move: {final['black_can_move']}")
    print(f"  Current player (stuck): {final['current_player']}")
    print(f"  Is capture: {final['is_capture']}")
    print()
    
    print("OUTCOME COMPARISON:")
    print(f"  Buggy logic (no last_mover): {analysis['buggy_winner']}")
    print(f"  Fixed logic (with last_mover): {analysis['fixed_winner']}")
    print(f"  Actual game winner: {analysis['actual_winner']}")
    print()
    
    if analysis['buggy_winner'] != analysis['fixed_winner']:
        print("  >>> BUG CONFIRMED: Outcomes differ! <<<")
        print()
        
        if final['is_capture']:
            print("  This is a CAPTURE scenario.")
            print("  In Mode 1/2, the winner should be the player who made the capture (last_mover).")
            print("  Buggy code returns None when last_mover is missing, causing incorrect winner.")
        else:
            print("  This is an IMMOBILIZATION scenario.")
            print("  The stuck player loses, opponent wins.")
        print()
    
    # Show last few moves
    print("LAST 5 MOVES:")
    print("-" * 80)
    for move_data in record["move_history"][-5:]:
        move_num = move_data["move_num"]
        player = move_data.get("player", "unknown")
        move_to = move_data.get("move_to", "N/A")
        after = move_data["after"]
        
        print(f"\nMove {move_num}: {player} -> {move_to}")
        print(f"  White: {after['white_pos']}, Black: {after['black_pos']}")
        print(f"  White can move: {after['white_can_move']}, Black can move: {after['black_can_move']}")
        
        if after['is_capture']:
            print(f"  >>> CAPTURE! <<<")
        
        if not after['white_can_move'] or not after['black_can_move']:
            stuck = "White" if not after['white_can_move'] else "Black"
            print(f"  >>> {stuck} is stuck! <<<")
    
    print()
    print("=" * 80)


def main():
    print()
    print("This script finds games where the OLD (buggy) and NEW (fixed) win condition")
    print("logic produce different outcomes.")
    print()
    print("The bug: get_legal_moves() calls check_win_condition(board) without last_mover")
    print("The fix: Pass opponent as last_mover when current player is stuck")
    print()
    
    divergent_games = find_divergent_games(num_games=50, mode=2)
    
    if not divergent_games:
        print("\n" + "=" * 80)
        print("NO DIVERGENT GAMES FOUND")
        print("=" * 80)
        print("\nThis could mean:")
        print("  1. The bug is rare and requires specific scenarios")
        print("  2. The current code already has the fix applied")
        print("  3. The bug only manifests in edge cases not covered by these games")
        print("\nHowever, the code should still be fixed to pass last_mover correctly!")
    else:
        print(f"\nFound {len(divergent_games)} divergent games!")
        print()
        
        # Analyze each divergent game in detail
        for game in divergent_games:
            print_detailed_analysis(game)
            print()


if __name__ == "__main__":
    main()
