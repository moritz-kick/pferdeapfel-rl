"""Debug script to find games where RandomPlayer wins as black against HeuristicPlayer (white)."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game.game import Game
from src.players.heuristic_player import HeuristicPlayer
from src.players.random import RandomPlayer


def play_until_black_wins(target_wins: int = 3) -> list[Path]:
    """
    Play games until RandomPlayer (black) wins target_wins times against HeuristicPlayer (white).
    
    Returns:
        List of paths to saved game logs
    """
    saved_games = []
    games_played = 0
    black_wins = 0
    
    print(f"Playing games until RandomPlayer (black) wins {target_wins} times...")
    print("=" * 70)
    
    while black_wins < target_wins:
        games_played += 1
        
        # Create players
        white_player = HeuristicPlayer("white")
        black_player = RandomPlayer("black")
        
        # Create game with logging enabled
        game = Game(white_player, black_player, mode=2, logging=True)
        
        # Play game
        max_moves = 200
        move_count = 0
        
        while not game.game_over and move_count < max_moves:
            legal_moves = game.get_legal_moves()
            
            if not legal_moves:
                break
            
            current_player = game.get_current_player()
            move_to, extra = current_player.get_move(game.board, legal_moves)
            
            success = game.make_move(move_to, extra)
            if not success:
                print(f"  Game {games_played}: Illegal move attempted!")
                break
            
            move_count += 1
        
        # Check result
        winner = game.winner if game.winner else "draw"
        
        if winner == "black":
            black_wins += 1
            print(f"  Game {games_played}: BLACK WINS! (Moves: {move_count})")
            
            # Save game log
            log_dir = Path("data/logs/debug_heuristic_white")
            log_path = game.save_log(log_dir)
            
            if log_path:
                saved_games.append(log_path)
                print(f"    Saved to: {log_path}")
        elif winner == "white":
            print(f"  Game {games_played}: White wins (Moves: {move_count})")
        else:
            print(f"  Game {games_played}: Draw (Moves: {move_count})")
        
        # Progress update
        if games_played % 10 == 0:
            print(f"\n  Progress: {black_wins}/{target_wins} black wins after {games_played} games\n")
    
    print("\n" + "=" * 70)
    print(f"Completed! Found {black_wins} black wins in {games_played} games")
    print(f"Saved {len(saved_games)} game logs")
    
    return saved_games


def analyze_game(log_path: Path) -> dict:
    """Analyze a saved game log."""
    import toon_python
    
    with open(log_path, "r") as f:
        data = toon_python.decode(f.read())
    
    moves = data.get("moves", [])
    winner = data.get("winner", "unknown")
    
    # Count moves by player
    white_moves = [m for m in moves if m.get("turn") == "white"]
    black_moves = [m for m in moves if m.get("turn") == "black"]
    
    # Analyze move patterns
    analysis = {
        "file": log_path.name,
        "winner": winner,
        "total_moves": len(moves),
        "white_moves": len(white_moves),
        "black_moves": len(black_moves),
        "white_positions": [m.get("white_pos") for m in moves],
        "black_positions": [m.get("black_pos") for m in moves],
        "moves": moves,
    }
    
    return analysis


def print_analysis(analyses: list[dict]) -> None:
    """Print detailed analysis of the games."""
    print("\n" + "=" * 70)
    print("GAME ANALYSIS")
    print("=" * 70)
    
    for i, analysis in enumerate(analyses, 1):
        print(f"\n### Game {i}: {analysis['file']} ###")
        print(f"Winner: {analysis['winner'].upper()}")
        print(f"Total moves: {analysis['total_moves']}")
        print(f"White moves: {analysis['white_moves']}")
        print(f"Black moves: {analysis['black_moves']}")
        
        # Show first few and last few moves
        moves = analysis['moves']
        print(f"\nFirst 5 moves:")
        for j, move in enumerate(moves[:5], 1):
            print(f"  {j}. {move.get('turn')}: {move.get('move_to')} -> White: {move.get('white_pos')}, Black: {move.get('black_pos')}")
        
        if len(moves) > 5:
            print(f"\nLast 5 moves:")
            for j, move in enumerate(moves[-5:], len(moves) - 4):
                print(f"  {j}. {move.get('turn')}: {move.get('move_to')} -> White: {move.get('white_pos')}, Black: {move.get('black_pos')}")
        
        # Check for patterns
        print(f"\nKey observations:")
        
        # Check if white got stuck
        last_white_move = None
        for move in reversed(moves):
            if move.get('turn') == 'white':
                last_white_move = move
                break
        
        if last_white_move:
            print(f"  - Last white move: {last_white_move.get('move_to')} (position: {last_white_move.get('white_pos')})")
        
        # Check if black captured white
        if analysis['winner'] == 'black':
            # Check if positions match at end
            final_positions = moves[-1] if moves else {}
            if final_positions.get('white_pos') == final_positions.get('black_pos'):
                print(f"  - Black captured white!")
            else:
                print(f"  - White got stuck (no capture)")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Common patterns
    all_white_positions = []
    for analysis in analyses:
        all_white_positions.extend(analysis['white_positions'])
    
    print(f"\nAverage game length: {sum(a['total_moves'] for a in analyses) / len(analyses):.1f} moves")
    print(f"All games ended with black winning")
    
    # Check if white consistently gets stuck in similar positions
    final_white_positions = [a['white_positions'][-1] for a in analyses if a['white_positions']]
    print(f"\nFinal white positions when losing:")
    for pos in final_white_positions:
        print(f"  - {pos}")


def main():
    """Main entry point."""
    # Play games until we get 3 black wins
    saved_games = play_until_black_wins(target_wins=3)
    
    if not saved_games:
        print("No games were saved!")
        return
    
    # Analyze the saved games
    print("\nAnalyzing saved games...")
    analyses = [analyze_game(path) for path in saved_games]
    
    # Print detailed analysis
    print_analysis(analyses)
    
    print(f"\nGame logs saved in: data/logs/debug_heuristic_white/")


if __name__ == "__main__":
    main()

