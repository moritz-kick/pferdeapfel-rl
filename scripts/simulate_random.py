"""Simulate random vs random games."""

import json
import random
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game.game import Game
from src.players.random import RandomPlayer


def simulate_game(log: bool = False, verbose: bool = True) -> dict:
    """
    Simulate a single random vs random game.

    Args:
        log: Whether to log the game
        verbose: Whether to print game progress

    Returns:
        Dictionary with game results
    """
    white = RandomPlayer("White")
    black = RandomPlayer("Black")
    game = Game(white, black, logging=log)

    move_count = 0
    max_moves = 1000  # Safety limit

    if verbose:
        print(f"Starting game. {game.current_player.capitalize()} goes first.")

    while not game.game_over and move_count < max_moves:
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            break

        current_player = game.get_current_player()
        try:
            move_to, extra_apple = current_player.get_move(game.board, legal_moves)
            success = game.make_move(move_to, extra_apple)

            if success:
                move_count += 1
                if verbose and move_count % 10 == 0:
                    print(f"Move {move_count}: {game.current_player.capitalize()}'s turn")
            else:
                if verbose:
                    print(f"Invalid move attempted by {current_player.name}")
                break
        except Exception as e:
            if verbose:
                print(f"Error during move: {e}")
            break

    result = {
        "winner": game.winner,
        "moves": move_count,
        "white_player": white.name,
        "black_player": black.name,
    }

    if game.game_over:
        if verbose:
            winner_name = "White" if game.winner == "white" else "Black"
            print(f"\nGame Over! {winner_name} wins after {move_count} moves.")
    else:
        if verbose:
            print(f"\nGame ended without winner after {move_count} moves.")

    # Save log if requested
    if log and game.logging:
        log_dir = Path("data/logs/game")
        log_path = game.save_log(log_dir)
        if log_path and verbose:
            print(f"Game logged to {log_path}")

    return result


def simulate_multiple(n_games: int = 10, log: bool = False, verbose: bool = True) -> dict:
    """
    Simulate multiple random vs random games.

    Args:
        n_games: Number of games to simulate
        log: Whether to log games
        verbose: Whether to print progress

    Returns:
        Dictionary with statistics
    """
    results = {"white_wins": 0, "black_wins": 0, "total_moves": 0}

    for i in range(n_games):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Game {i+1}/{n_games}")
            print(f"{'='*50}")

        result = simulate_game(log=log, verbose=verbose)
        results["total_moves"] += result["moves"]

        if result["winner"] == "white":
            results["white_wins"] += 1
        elif result["winner"] == "black":
            results["black_wins"] += 1

    if verbose:
        print(f"\n{'='*50}")
        print("Summary:")
        print(f"  White wins: {results['white_wins']}")
        print(f"  Black wins: {results['black_wins']}")
        print(f"  Average moves per game: {results['total_moves'] / n_games:.1f}")
        print(f"{'='*50}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulate random vs random games")
    parser.add_argument(
        "-n", "--num-games", type=int, default=1, help="Number of games to simulate"
    )
    parser.add_argument(
        "--log", action="store_true", help="Log games to data/logs/game/"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress output"
    )
    args = parser.parse_args()

    if args.num_games == 1:
        simulate_game(log=args.log, verbose=not args.quiet)
    else:
        simulate_multiple(n_games=args.num_games, log=args.log, verbose=not args.quiet)
