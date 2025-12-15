"""Find games where old and new win condition logic produce different outcomes."""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game.board import Board
from src.game.game import Game
from src.game.rules import Rules
from src.players.heuristic_player import HeuristicPlayer
from src.players.random import RandomPlayer


def check_win_condition_old(board: Board, last_mover: Optional[str] = None, current_player: Optional[str] = None) -> Optional[str]:
    """
    OLD win condition logic - simulates the bug where get_legal_moves calls
    check_win_condition without last_mover parameter.
    
    The bug: When a player has no legal moves, get_legal_moves() calls
    check_win_condition(board) without last_mover. This can cause issues
    if there's a capture situation that wasn't properly detected.
    """
    # --- MODE 1 & 2: Survival ---
    if board.mode in [1, 2]:
        # 1. Capture (Immediate Win)
        if board.white_pos == board.black_pos:
            # BUG: If last_mover is None (which happens in get_legal_moves),
            # this returns None, and the code falls through to immobilization checks
            # which might give the wrong result
            if last_mover:
                return last_mover
            # When called from get_legal_moves without last_mover, this returns None
            # and falls through - this is the bug!
            return None  # Should have last_mover but doesn't
        
        # 2. Check if White is stuck
        if not Rules.can_player_move(board, "white"):
            return "black"
        
        # 3. Check if Black is stuck
        if not Rules.can_player_move(board, "black"):
            return "white"
        
        return None
    
    # Mode 3 logic (same as new)
    if board.golden_phase_started:
        board.white_match_win_declared = True
    
    if getattr(board, "draw_condition_met", False):
        return "draw"
    
    if board.black_pos == board.white_pos:
        if board.golden_phase_started:
            return "white"
        else:
            return "black"
    
    white_stuck = not Rules.can_player_move(board, "white")
    black_stuck = not Rules.can_player_move(board, "black")
    
    if white_stuck:
        if board.golden_phase_started:
            return "white"
        else:
            return "black"
    
    if black_stuck:
        if not board.golden_phase_started:
            board.white_won_in_brown_phase = True
        board.white_match_win_declared = True
        return "white"
    
    if board.golden_apples_remaining == 0:
        return "white"
    
    return None


def play_game_with_recording(white_cls, black_cls, white_name, black_name, mode=2):
    """Play a game and record all moves and board states."""
    white = white_cls(white_name)
    black = black_cls(black_name)
    game = Game(white, black, mode=mode, logging=True)
    game.current_player = "white"
    
    move_history = []
    moves = 0
    max_moves = 500
    
    while not game.game_over and moves < max_moves:
        player = game.get_current_player()
        legal_moves = game.get_legal_moves()
        
        if not legal_moves:
            break
        
        # Record state before move
        state_before = {
            "turn": game.current_player,
            "white_pos": game.board.white_pos,
            "black_pos": game.board.black_pos,
            "legal_moves": legal_moves.copy(),
            "white_can_move": Rules.can_player_move(game.board, "white"),
            "black_can_move": Rules.can_player_move(game.board, "black"),
        }
        
        move_to, extra = player.get_move(game.board, legal_moves)
        success = game.make_move(move_to, extra)
        
        if not success:
            print(f"  Illegal move attempted at move {moves}!")
            break
        
        # Record state after move
        # OLD: When get_legal_moves finds no moves, it calls check_win_condition without last_mover
        # NEW: Should pass last_mover, but let's simulate the old bug
        state_after = {
            "white_pos": game.board.white_pos,
            "black_pos": game.board.black_pos,
            "winner_old": check_win_condition_old(game.board, last_mover=game.current_player),
            "winner_new": Rules.check_win_condition(game.board, last_mover=game.current_player),
            "game_over": game.game_over,
            "winner": game.winner,
        }
        
        # Check what happens if next player has no moves (simulating get_legal_moves bug)
        if not game.game_over:
            next_player = "black" if game.current_player == "white" else "white"
            next_legal = Rules.get_legal_knight_moves(game.board, next_player)
            if not next_legal:
                # OLD BUG: get_legal_moves calls check_win_condition(board) without last_mover
                # The last_mover should be the current_player (who just moved), not None
                state_after["next_player_stuck"] = next_player
                state_after["winner_old_buggy"] = check_win_condition_old(game.board, last_mover=None)
                # NEW: Should pass last_mover, but get_legal_moves still doesn't - need to fix that!
                # Actually, the fix should be: when get_legal_moves finds no moves, 
                # it should call check_win_condition with last_mover=opponent (the one who just moved)
                state_after["winner_new_correct"] = Rules.check_win_condition(game.board, last_mover=game.current_player)
        
        move_history.append({
            "move_num": moves + 1,
            "before": state_before,
            "move": {"to": move_to, "extra": extra},
            "after": state_after,
        })
        
        moves += 1
        
        if game.game_over:
            break
    
    # Final state - simulate what happens in get_legal_moves when no moves available
    # The current code STILL has the bug: get_legal_moves calls check_win_condition without last_mover
    # But let's check what the correct behavior should be
    final_state = {
        "white_pos": game.board.white_pos,
        "black_pos": game.board.black_pos,
        "white_can_move": Rules.can_player_move(game.board, "white"),
        "black_can_move": Rules.can_player_move(game.board, "black"),
        "winner_old": check_win_condition_old(game.board, last_mover=game.current_player),
        "winner_new": Rules.check_win_condition(game.board, last_mover=game.current_player),
        # Simulate the actual bug: get_legal_moves calls without last_mover
        "winner_buggy_call": Rules.check_win_condition(game.board, last_mover=None),
        "game_over": game.game_over,
        "winner": game.winner,
        "current_player": game.current_player,
    }
    
    return {
        "moves": move_history,
        "final": final_state,
        "total_moves": moves,
        "white_player": white_name,
        "black_player": black_name,
    }


def replay_with_both_rules(game_record):
    """Replay a game record and check outcomes with both old and new rules."""
    # Reconstruct final board state
    if not game_record["moves"]:
        return None, None
    
    final_state = game_record["final"]
    
    # The bug: get_legal_moves calls check_win_condition without last_mover
    # This is what actually happens in the current code (it's still buggy!)
    buggy_winner = final_state.get("winner_buggy_call")
    correct_winner = final_state["winner_new"]
    
    return buggy_winner, correct_winner


def find_divergent_games(num_games=100, mode=2):
    """Find games where old and new rules produce different outcomes."""
    print("=" * 80)
    print(f"Searching for games with divergent outcomes (Mode {mode})")
    print("=" * 80)
    print()
    
    divergent_games = []
    games_played = 0
    
    # Test different matchups
    matchups = [
        (HeuristicPlayer, RandomPlayer, "Heuristic", "Random"),
        (RandomPlayer, HeuristicPlayer, "Random", "Heuristic"),
        (HeuristicPlayer, HeuristicPlayer, "Heuristic", "Heuristic"),
        (RandomPlayer, RandomPlayer, "Random", "Random"),
    ]
    
    for white_cls, black_cls, white_name, black_name in matchups:
        print(f"Testing: {white_name} (White) vs {black_name} (Black)")
        print("-" * 80)
        
        for i in range(num_games):
            games_played += 1
            game_record = play_game_with_recording(white_cls, black_cls, white_name, black_name, mode=mode)
            
            old_winner, new_winner = replay_with_both_rules(game_record)
            
            if old_winner != new_winner:
                print(f"  DIVERGENCE FOUND! Game {games_played}")
                print(f"    Old rules winner: {old_winner}")
                print(f"    New rules winner: {new_winner}")
                print(f"    Actual game winner: {game_record['final']['winner']}")
                print(f"    Total moves: {game_record['total_moves']}")
                
                game_record["matchup"] = f"{white_name} vs {black_name}"
                divergent_games.append((games_played, game_record))
                
                if len(divergent_games) >= 5:  # Find 5 divergent games
                    break
        
        if len(divergent_games) >= 5:
            break
        
        print(f"  Played {num_games} games, found {len([g for g in divergent_games if g[1]['matchup'] == f'{white_name} vs {black_name}'])} divergences")
        print()
    
    return divergent_games


def analyze_divergent_game(game_num, game_record):
    """Provide detailed analysis of a divergent game."""
    print("=" * 80)
    print(f"DETAILED ANALYSIS: Game {game_num}")
    print(f"Matchup: {game_record['matchup']}")
    print("=" * 80)
    print()
    
    moves = game_record["moves"]
    final = game_record["final"]
    
    print(f"Total moves: {len(moves)}")
    print(f"Final positions: White={final['white_pos']}, Black={final['black_pos']}")
    print(f"White can move: {final['white_can_move']}")
    print(f"Black can move: {final['black_can_move']}")
    print()
    
    print("OUTCOME COMPARISON:")
    print(f"  Correct (with last_mover): {final['winner_new']}")
    print(f"  Buggy (no last_mover, as in get_legal_moves): {final.get('winner_buggy_call', 'N/A')}")
    print(f"  Actual game winner: {final['winner']}")
    print(f"  Current player when game ended: {final.get('current_player', 'N/A')}")
    print()
    
    if final.get('winner_buggy_call') != final['winner_new']:
        print("  >>> BUG DETECTED: Outcomes differ! <<<")
        print()
    
    # Find where divergence first appears
    print("MOVE-BY-MOVE ANALYSIS:")
    print("-" * 80)
    
    divergence_found = False
    for move_data in moves[-10:]:  # Show last 10 moves
        move_num = move_data["move_num"]
        before = move_data["before"]
        after = move_data["after"]
        
        if after["winner_old"] != after["winner_new"] and not divergence_found:
            print(f"\n>>> DIVERGENCE FIRST APPEARS AT MOVE {move_num} <<<")
            divergence_found = True
        
        print(f"\nMove {move_num}: {before['turn']} moves to {move_data['move']['to']}")
        print(f"  Before: White={before['white_pos']}, Black={before['black_pos']}")
        print(f"  After:  White={after['white_pos']}, Black={after['black_pos']}")
        print(f"  Old rules winner: {after['winner_old']}")
        print(f"  New rules winner: {after['winner_new']}")
        print(f"  Game over: {after['game_over']}")
        
        if after["white_pos"] == after["black_pos"]:
            print(f"  >>> CAPTURE DETECTED! <<<")
    
    print()
    print("=" * 80)


def main():
    # Find divergent games
    divergent_games = find_divergent_games(num_games=50, mode=2)
    
    if not divergent_games:
        print("\nNo divergent games found. This could mean:")
        print("  1. The fix is correct and there are no bugs")
        print("  2. The old rule reconstruction is incorrect")
        print("  3. Divergences are rare and need more games")
        print("\nTrying self-play scenarios...")
        
        # Try self-play
        print("\n" + "=" * 80)
        print("SELF-PLAY ANALYSIS")
        print("=" * 80)
        print()
        
        for i in range(10):
            game_record = play_game_with_recording(
                HeuristicPlayer, HeuristicPlayer, "Heuristic1", "Heuristic2", mode=2
            )
            old_winner, new_winner = replay_with_both_rules(game_record)
            
            if old_winner != new_winner:
                print(f"  DIVERGENCE in self-play game {i+1}!")
                analyze_divergent_game(i+1, game_record)
                break
            else:
                print(f"  Game {i+1}: Both rules agree ({old_winner})")
    else:
        print(f"\nFound {len(divergent_games)} divergent games!")
        print()
        
        # Analyze first divergent game in detail
        for game_num, game_record in divergent_games[:3]:  # Analyze first 3
            analyze_divergent_game(game_num, game_record)
            print()


if __name__ == "__main__":
    main()
