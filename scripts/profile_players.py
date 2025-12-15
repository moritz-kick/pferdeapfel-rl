"""Standalone script to profile all players (including RandomPlayer)."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.discovery import discover_players
from src.evaluation.profiler import PerformanceProfiler


def main():
    parser = argparse.ArgumentParser(description="Profile all players (including RandomPlayer)")
    parser.add_argument("--mode", type=int, default=2, help="Game mode (default: 2)")
    parser.add_argument("--games", type=int, default=50, help="Number of games per player (default: 50)")
    parser.add_argument("--no-random", action="store_true", help="Exclude RandomPlayer from profiling")
    args = parser.parse_args()

    print("Discovering players...")
    player_classes = discover_players()
    if not player_classes:
        print("No players found!")
        return

    print(f"Found {len(player_classes)} players: {[cls.__name__ for cls in player_classes]}")

    # Filter out HumanPlayer for automated profiling
    player_classes = [cls for cls in player_classes if "Human" not in cls.__name__]
    print(f"Profiling {len(player_classes)} bots: {[cls.__name__ for cls in player_classes]}")

    if not player_classes:
        print("No bot players found!")
        return

    profiler = PerformanceProfiler()

    # Profile all players including RandomPlayer (unless --no-random is set)
    profiler.profile_all_players(
        player_classes,
        mode=args.mode,
        games=args.games,
        include_random=not args.no_random,
    )

    print("Profiling complete!")


if __name__ == "__main__":
    main()

