"""Compare game outcomes between different rule implementations using git checkouts."""

import sys
import subprocess
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_game_with_version(commit_hash, white_cls, black_cls, white_name, black_name, mode=2, seed=0):
    """Run a game using code from a specific git commit."""
    # Save current state
    original_dir = Path.cwd()
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Clone or checkout the specific commit
        result = subprocess.run(
            ["git", "worktree", "add", tmpdir, commit_hash],
            cwd=original_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            # Try alternative approach - just checkout in current repo
            subprocess.run(["git", "checkout", commit_hash], cwd=original_dir, capture_output=True)
            try:
                # Import and run
                import importlib
                import sys
                # Clear module cache
                modules_to_clear = [m for m in sys.modules.keys() if m.startswith('src.')]
                for m in modules_to_clear:
                    del sys.modules[m]
                
                from src.game.game import Game
                from src.players.heuristic_player import HeuristicPlayer
                from src.players.random import RandomPlayer
                import random
                
                random.seed(seed)
                white = white_cls(white_name)
                black = black_cls(black_name)
                game = Game(white, black, mode=mode, logging=False)
                game.current_player = "white"
                
                moves = 0
                while not game.game_over and moves < 200:
                    player = game.get_current_player()
                    legal_moves = game.get_legal_moves()
                    if not legal_moves:
                        break
                    move_to, extra = player.get_move(game.board, legal_moves)
                    game.make_move(move_to, extra)
                    moves += 1
                
                result_data = {
                    "winner": game.winner,
                    "moves": moves,
                    "white_pos": game.board.white_pos,
                    "black_pos": game.board.black_pos,
                }
                
                return result_data
            finally:
                # Restore original
                subprocess.run(["git", "checkout", "-"], cwd=original_dir, capture_output=True)
        else:
            # Use worktree
            try:
                sys.path.insert(0, str(Path(tmpdir)))
                # Similar import and run logic
                # ... (simplified for now)
                return None
            finally:
                subprocess.run(["git", "worktree", "remove", tmpdir], cwd=original_dir, capture_output=True)


# Simpler approach: Just analyze games in detail and let user inspect
def main_simple():
    """Simple approach: analyze current games in detail."""
    import random
    from src.game.game import Game
    from src.game.rules import Rules
    from src.players.heuristic_player import HeuristicPlayer
    from src.players.random import RandomPlayer
    
    print("="*80)
    print("GAME ANALYSIS - Manual Inspection")
    print("="*80)
    print("\nThis will play games and show detailed information.")
    print("You can manually verify if win conditions are correct.\n")
    
    mode = 2
    
    # Play a few games and show details
    for seed in range(10):
        random.seed(seed)
        white = HeuristicPlayer("white")
        black = RandomPlayer("black")
        game = Game(white, black, mode=mode, logging=True)
        game.current_player = "white"
        
        print(f"\n{'='*80}")
        print(f"GAME {seed + 1} (Seed: {seed})")
        print(f"{'='*80}")
        
        move_num = 0
        while not game.game_over and move_num < 200:
            player = game.get_current_player()
            legal_moves = game.get_legal_moves()
            
            if not legal_moves:
                print(f"\nMove {move_num}: {game.current_player} has no legal moves!")
                print(f"  White pos: {game.board.white_pos}")
                print(f"  Black pos: {game.board.black_pos}")
                print(f"  White can move: {Rules.can_player_move(game.board, 'white')}")
                print(f"  Black can move: {Rules.can_player_move(game.board, 'black')}")
                print(f"  Capture: {game.board.white_pos == game.board.black_pos}")
                print(f"  Winner: {game.winner}")
                break
            
            move_to, extra = player.get_move(game.board, legal_moves)
            
            # Show state before move
            if move_num < 3 or move_num > 15:  # Show first few and last few
                print(f"\nMove {move_num}: {game.current_player} -> {move_to}")
                print(f"  Before: White={game.board.white_pos}, Black={game.board.black_pos}")
                print(f"          White can move: {Rules.can_player_move(game.board, 'white')}")
                print(f"          Black can move: {Rules.can_player_move(game.board, 'black')}")
            
            game.make_move(move_to, extra)
            
            if move_num < 3 or move_num > 15:
                print(f"  After:  White={game.board.white_pos}, Black={game.board.black_pos}")
                print(f"          Winner: {game.winner}, Game over: {game.game_over}")
            
            if game.game_over:
                print(f"\nGame ended at move {move_num + 1}")
                print(f"  Final: White={game.board.white_pos}, Black={game.board.black_pos}")
                print(f"         White can move: {Rules.can_player_move(game.board, 'white')}")
                print(f"         Black can move: {Rules.can_player_move(game.board, 'black')}")
                print(f"         Capture: {game.board.white_pos == game.board.black_pos}")
                print(f"         Winner: {game.winner}")
                
                # Verify win condition
                if game.board.white_pos == game.board.black_pos:
                    print(f"  ✓ Capture: Winner should be the mover")
                elif not Rules.can_player_move(game.board, "white"):
                    print(f"  ✓ White stuck: Winner should be black")
                elif not Rules.can_player_move(game.board, "black"):
                    print(f"  ✓ Black stuck: Winner should be white")
                break
            
            move_num += 1
        
        if not game.game_over:
            print(f"\nGame reached move limit (200 moves)")
        
        # Only show first 3 games in detail
        if seed >= 2:
            break
    
    # Self-play test
    print(f"\n{'='*80}")
    print("SELF-PLAY TEST")
    print(f"{'='*80}")
    
    for seed in range(5):
        random.seed(seed)
        white = HeuristicPlayer("white")
        black = HeuristicPlayer("black")
        game = Game(white, black, mode=mode, logging=False)
        game.current_player = "white"
        
        moves = 0
        while not game.game_over and moves < 200:
            player = game.get_current_player()
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break
            move_to, extra = player.get_move(game.board, legal_moves)
            game.make_move(move_to, extra)
            moves += 1
        
        print(f"Game {seed + 1}: {game.winner} wins in {moves} moves")
        print(f"  Final: White={game.board.white_pos}, Black={game.board.black_pos}")
        print(f"         Capture: {game.board.white_pos == game.board.black_pos}")
        print(f"         White stuck: {not Rules.can_player_move(game.board, 'white')}")
        print(f"         Black stuck: {not Rules.can_player_move(game.board, 'black')}")


if __name__ == "__main__":
    main_simple()
