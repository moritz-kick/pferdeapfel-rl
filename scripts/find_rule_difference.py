"""Find games where old and new rules produce different outcomes."""

import sys
from pathlib import Path
import copy

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game.game import Game
from src.game.rules import Rules
from src.players.heuristic_player import HeuristicPlayer
from src.players.random import RandomPlayer


class OldRulesGame(Game):
    """Game with old win condition logic (buggy version)."""
    
    def get_legal_moves(self):
        """Old version: calls check_win_condition without last_mover when stuck."""
        legal_moves = Rules.get_legal_knight_moves(self.board, self.current_player)
        
        if not legal_moves and not self.game_over:
            # OLD: Called without last_mover parameter
            # This is actually fine for stuck cases, but might have other issues
            self.winner = Rules.check_win_condition(self.board)  # Missing last_mover (but OK for stuck)
            self.game_over = True
        
        return legal_moves
    
    def make_move(self, move_to, extra_apple=None):
        """Override - actually same as new, but let's be explicit."""
        if self.game_over:
            return False
        
        success = Rules.make_move(self.board, self.current_player, move_to, extra_apple)
        
        if success:
            if self.logging:
                self.log_data.append({
                    "turn": self.current_player,
                    "move_to": move_to,
                    "extra_apple": extra_apple,
                    "white_pos": self.board.white_pos,
                    "black_pos": self.board.black_pos,
                    "brown_remaining": self.board.brown_apples_remaining,
                    "golden_remaining": self.board.golden_apples_remaining,
                })
            
            # Check win condition - same as new code actually
            win_result = Rules.check_win_condition(self.board, last_mover=self.current_player)
            
            if self.board.mode == 3 and self.board.golden_phase_started and self.winner is None:
                self.winner = "white"
            
            if win_result:
                self.winner = win_result
                self.game_over = True
            else:
                self.switch_turn()
        
        return success


def play_game_with_both_rules(white_cls, black_cls, white_name, black_name, mode=2, seed=None):
    """Play the same game with both old and new rules, return both results."""
    import random
    if seed is not None:
        random.seed(seed)
    
    # Play with NEW rules
    white_new = white_cls(white_name)
    black_new = black_cls(black_name)
    game_new = Game(white_new, black_new, mode=mode, logging=True)
    game_new.current_player = "white"
    
    moves_new = []
    while not game_new.game_over:
        player = game_new.get_current_player()
        legal_moves = game_new.get_legal_moves()
        if not legal_moves:
            break
        move_to, extra = player.get_move(game_new.board, legal_moves)
        game_new.make_move(move_to, extra)
        moves_new.append({
            "turn": game_new.current_player if not game_new.game_over else ("black" if game_new.current_player == "white" else "white"),
            "move_to": move_to,
            "extra": extra,
            "white_pos": game_new.board.white_pos,
            "black_pos": game_new.board.black_pos,
        })
        if len(moves_new) > 200:  # Safety limit
            break
    
    result_new = {
        "winner": game_new.winner,
        "moves": len(moves_new),
        "white_pos": game_new.board.white_pos,
        "black_pos": game_new.board.black_pos,
        "move_history": moves_new,
    }
    
    # Reset random seed for old rules game
    if seed is not None:
        random.seed(seed)
    
    # Play with OLD rules
    white_old = white_cls(white_name)
    black_old = black_cls(black_name)
    game_old = OldRulesGame(white_old, black_old, mode=mode, logging=True)
    game_old.current_player = "white"
    
    moves_old = []
    while not game_old.game_over:
        player = game_old.get_current_player()
        legal_moves = game_old.get_legal_moves()
        if not legal_moves:
            break
        move_to, extra = player.get_move(game_old.board, legal_moves)
        game_old.make_move(move_to, extra)
        moves_old.append({
            "turn": game_old.current_player if not game_old.game_over else ("black" if game_old.current_player == "white" else "white"),
            "move_to": move_to,
            "extra": extra,
            "white_pos": game_old.board.white_pos,
            "black_pos": game_old.board.black_pos,
        })
        if len(moves_old) > 200:  # Safety limit
            break
    
    result_old = {
        "winner": game_old.winner,
        "moves": len(moves_old),
        "white_pos": game_old.board.white_pos,
        "black_pos": game_old.board.black_pos,
        "move_history": moves_old,
    }
    
    return result_new, result_old


def find_different_outcome_game(white_cls, black_cls, white_name, black_name, mode=2, max_games=1000):
    """Find a game where old and new rules produce different outcomes."""
    print(f"Searching for games with different outcomes (max {max_games} games)...")
    print("=" * 80)
    
    for i in range(max_games):
        if (i + 1) % 100 == 0:
            print(f"  Checked {i + 1} games...")
        
        result_new, result_old = play_game_with_both_rules(
            white_cls, black_cls, white_name, black_name, mode=mode, seed=i
        )
        
        # Check if outcomes differ
        if result_new["winner"] != result_old["winner"]:
            print(f"\n✓ Found different outcome at game {i + 1}!")
            print(f"  NEW rules winner: {result_new['winner']}")
            print(f"  OLD rules winner: {result_old['winner']}")
            print(f"  NEW moves: {result_new['moves']}")
            print(f"  OLD moves: {result_old['moves']}")
            return i, result_new, result_old
    
    print(f"\n✗ No different outcomes found in {max_games} games")
    return None, None, None


def analyze_difference(seed, result_new, result_old, white_cls, black_cls, white_name, black_name, mode=2):
    """Analyze a game where outcomes differ."""
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    
    # Replay both games to get detailed state
    import random
    random.seed(seed)
    
    # NEW rules game
    white_new = white_cls(white_name)
    black_new = black_cls(black_name)
    game_new = Game(white_new, black_new, mode=mode, logging=True)
    game_new.current_player = "white"
    
    states_new = []
    move_num = 0
    while not game_new.game_over and move_num < 200:
        player = game_new.get_current_player()
        legal_moves = game_new.get_legal_moves()
        if not legal_moves:
            states_new.append({
                "move": move_num,
                "turn": game_new.current_player,
                "legal_moves": [],
                "white_pos": game_new.board.white_pos,
                "black_pos": game_new.board.black_pos,
                "white_can_move": Rules.can_player_move(game_new.board, "white"),
                "black_can_move": Rules.can_player_move(game_new.board, "black"),
                "capture": game_new.board.white_pos == game_new.board.black_pos,
                "winner": game_new.winner,
            })
            break
        
        move_to, extra = player.get_move(game_new.board, legal_moves)
        
        states_new.append({
            "move": move_num,
            "turn": game_new.current_player,
            "legal_moves": legal_moves,
            "move_to": move_to,
            "extra": extra,
            "white_pos": game_new.board.white_pos,
            "black_pos": game_new.board.black_pos,
            "white_can_move": Rules.can_player_move(game_new.board, "white"),
            "black_can_move": Rules.can_player_move(game_new.board, "black"),
            "capture": game_new.board.white_pos == game_new.board.black_pos,
        })
        
        game_new.make_move(move_to, extra)
        
        states_new[-1]["after_white_pos"] = game_new.board.white_pos
        states_new[-1]["after_black_pos"] = game_new.board.black_pos
        states_new[-1]["after_white_can_move"] = Rules.can_player_move(game_new.board, "white")
        states_new[-1]["after_black_can_move"] = Rules.can_player_move(game_new.board, "black")
        states_new[-1]["after_capture"] = game_new.board.white_pos == game_new.board.black_pos
        states_new[-1]["winner_after"] = game_new.winner
        states_new[-1]["game_over"] = game_new.game_over
        
        move_num += 1
    
    # OLD rules game
    random.seed(seed)
    
    white_old = white_cls(white_name)
    black_old = black_cls(black_name)
    game_old = OldRulesGame(white_old, black_old, mode=mode, logging=True)
    game_old.current_player = "white"
    
    states_old = []
    move_num = 0
    while not game_old.game_over and move_num < 200:
        player = game_old.get_current_player()
        legal_moves = game_old.get_legal_moves()
        if not legal_moves:
            states_old.append({
                "move": move_num,
                "turn": game_old.current_player,
                "legal_moves": [],
                "white_pos": game_old.board.white_pos,
                "black_pos": game_old.board.black_pos,
                "white_can_move": Rules.can_player_move(game_old.board, "white"),
                "black_can_move": Rules.can_player_move(game_old.board, "black"),
                "capture": game_old.board.white_pos == game_old.board.black_pos,
                "winner": game_old.winner,
            })
            break
        
        move_to, extra = player.get_move(game_old.board, legal_moves)
        
        states_old.append({
            "move": move_num,
            "turn": game_old.current_player,
            "legal_moves": legal_moves,
            "move_to": move_to,
            "extra": extra,
            "white_pos": game_old.board.white_pos,
            "black_pos": game_old.board.black_pos,
            "white_can_move": Rules.can_player_move(game_old.board, "white"),
            "black_can_move": Rules.can_player_move(game_old.board, "black"),
            "capture": game_old.board.white_pos == game_old.board.black_pos,
        })
        
        game_old.make_move(move_to, extra)
        
        states_old[-1]["after_white_pos"] = game_old.board.white_pos
        states_old[-1]["after_black_pos"] = game_old.board.black_pos
        states_old[-1]["after_white_can_move"] = Rules.can_player_move(game_old.board, "white")
        states_old[-1]["after_black_can_move"] = Rules.can_player_move(game_old.board, "black")
        states_old[-1]["after_capture"] = game_old.board.white_pos == game_old.board.black_pos
        states_old[-1]["winner_after"] = game_old.winner
        states_old[-1]["game_over"] = game_old.game_over
        
        move_num += 1
    
    # Find where they diverge
    print(f"\nGame Summary:")
    print(f"  NEW rules: {result_new['winner']} wins in {result_new['moves']} moves")
    print(f"  OLD rules: {result_old['winner']} wins in {result_old['moves']} moves")
    
    # Compare move by move
    min_moves = min(len(states_new), len(states_old))
    divergence_point = None
    
    for i in range(min_moves):
        new_state = states_new[i]
        old_state = states_old[i]
        
        # Check if outcomes differ after this move
        if new_state.get("winner_after") != old_state.get("winner_after"):
            divergence_point = i
            print(f"\n✓ Divergence detected at move {i}!")
            print(f"\nMove {i} Details:")
            print(f"  Turn: {new_state['turn']}")
            print(f"  Move: {new_state['move_to']}")
            print(f"  After move:")
            print(f"    NEW: White={new_state['after_white_pos']}, Black={new_state['after_black_pos']}")
            print(f"    OLD: White={old_state['after_white_pos']}, Black={old_state['after_black_pos']}")
            print(f"    NEW: White can move={new_state['after_white_can_move']}, Black can move={new_state['after_black_can_move']}")
            print(f"    OLD: White can move={old_state['after_white_can_move']}, Black can move={old_state['after_black_can_move']}")
            print(f"    NEW: Capture={new_state['after_capture']}")
            print(f"    OLD: Capture={old_state['after_capture']}")
            print(f"    NEW: Winner={new_state.get('winner_after')}")
            print(f"    OLD: Winner={old_state.get('winner_after')}")
            print(f"    NEW: Game over={new_state.get('game_over')}")
            print(f"    OLD: Game over={old_state.get('game_over')}")
            break
    
    if divergence_point is None:
        print("\n⚠ Games played identically but ended with different winners!")
        print("This suggests the bug is in the final win condition check.")
        if states_new:
            final_new = states_new[-1]
            final_old = states_old[-1] if states_old else {}
            print(f"\nFinal state NEW:")
            print(f"  White pos: {final_new.get('after_white_pos', final_new.get('white_pos'))}")
            print(f"  Black pos: {final_new.get('after_black_pos', final_new.get('black_pos'))}")
            print(f"  White can move: {final_new.get('after_white_can_move', final_new.get('white_can_move'))}")
            print(f"  Black can move: {final_new.get('after_black_can_move', final_new.get('black_can_move'))}")
            print(f"  Capture: {final_new.get('after_capture', final_new.get('capture'))}")
            print(f"  Winner: {final_new.get('winner_after', result_new['winner'])}")
            print(f"\nFinal state OLD:")
            print(f"  White pos: {final_old.get('after_white_pos', final_old.get('white_pos'))}")
            print(f"  Black pos: {final_old.get('after_black_pos', final_old.get('black_pos'))}")
            print(f"  White can move: {final_old.get('after_white_can_move', final_old.get('white_can_move'))}")
            print(f"  Black can move: {final_old.get('after_black_can_move', final_old.get('black_can_move'))}")
            print(f"  Capture: {final_old.get('after_capture', final_old.get('capture'))}")
            print(f"  Winner: {final_old.get('winner_after', result_old['winner'])}")
    
    # Show last few moves
    print(f"\nLast 5 moves (NEW rules):")
    for state in states_new[-5:]:
        print(f"  Move {state['move']}: {state['turn']} -> {state['move_to']}, Winner after: {state.get('winner_after')}")
    
    print(f"\nLast 5 moves (OLD rules):")
    for state in states_old[-5:]:
        print(f"  Move {state['move']}: {state['turn']} -> {state['move_to']}, Winner after: {state.get('winner_after')}")


def test_self_play(white_cls, black_cls, mode=2, num_games=50):
    """Test self-play scenarios with both old and new rules."""
    print("\n" + "=" * 80)
    print("SELF-PLAY ANALYSIS")
    print("=" * 80)
    
    print(f"\nTesting {white_cls.__name__} vs {white_cls.__name__} (self-play)")
    print("-" * 80)
    
    new_wins_white = 0
    new_wins_black = 0
    old_wins_white = 0
    old_wins_black = 0
    
    for i in range(num_games):
        result_new, result_old = play_game_with_both_rules(
            white_cls, white_cls, "white", "black", mode=mode, seed=i
        )
        
        if result_new["winner"] == "white":
            new_wins_white += 1
        elif result_new["winner"] == "black":
            new_wins_black += 1
        
        if result_old["winner"] == "white":
            old_wins_white += 1
        elif result_old["winner"] == "black":
            old_wins_black += 1
    
    print(f"NEW rules: White={new_wins_white}/{num_games} ({new_wins_white*100/num_games:.1f}%), Black={new_wins_black}/{num_games} ({new_wins_black*100/num_games:.1f}%)")
    print(f"OLD rules: White={old_wins_white}/{num_games} ({old_wins_white*100/num_games:.1f}%), Black={old_wins_black}/{num_games} ({old_wins_black*100/num_games:.1f}%)")
    
    # Check for differences
    if new_wins_white != old_wins_white or new_wins_black != old_wins_black:
        print(f"\n⚠ Different outcomes in self-play!")
        print(f"  Difference: White {new_wins_white - old_wins_white:+d}, Black {new_wins_black - old_wins_black:+d}")


def main():
    print("=" * 80)
    print("RULE DIFFERENCE FINDER")
    print("=" * 80)
    print("\nThis script compares OLD (buggy) and NEW (fixed) rule implementations")
    print("to find games where they produce different outcomes.\n")
    
    mode = 2  # Mode 2 for survival
    
    # Test 1: HeuristicPlayer vs RandomPlayer
    print("\n" + "=" * 80)
    print("TEST 1: HeuristicPlayer (White) vs RandomPlayer (Black)")
    print("=" * 80)
    seed, result_new, result_old = find_different_outcome_game(
        HeuristicPlayer, RandomPlayer, "white", "black", mode=mode, max_games=1000
    )
    
    if seed is not None:
        analyze_difference(seed, result_new, result_old, HeuristicPlayer, RandomPlayer, "white", "black", mode)
    
    # Test 2: RandomPlayer vs HeuristicPlayer
    print("\n" + "=" * 80)
    print("TEST 2: RandomPlayer (White) vs HeuristicPlayer (Black)")
    print("=" * 80)
    seed2, result_new2, result_old2 = find_different_outcome_game(
        RandomPlayer, HeuristicPlayer, "white", "black", mode=mode, max_games=1000
    )
    
    if seed2 is not None:
        analyze_difference(seed2, result_new2, result_old2, RandomPlayer, HeuristicPlayer, "white", "black", mode)
    
    # Test 3: Self-play
    test_self_play(HeuristicPlayer, HeuristicPlayer, mode=mode, num_games=50)
    test_self_play(RandomPlayer, RandomPlayer, mode=mode, num_games=50)


if __name__ == "__main__":
    main()
