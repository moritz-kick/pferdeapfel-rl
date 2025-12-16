"""Evaluation script for Mode 2 (Trail Placement)."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.discovery import discover_players
from src.evaluation.profiler import PerformanceProfiler
from src.evaluation.ranking import RankingSystem
from src.evaluation.runner import GameRunner
from src.evaluation.storage import ResultStorage


def main():
    parser = argparse.ArgumentParser(description="Run Mode 2 Evaluation")
    parser.add_argument("--games", type=int, default=50, help="Minimum games per pair per side")
    parser.add_argument("--new", action="store_true", help="Start a new evaluation run, clearing old results")
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Only run performance profiling against RandomPlayer, skip full pairwise evaluation",
    )
    parser.add_argument(
        "--short-eval",
        action="store_true",
        help=(
            "Run a short evaluation against RandomPlayer only "
            "(50 games as white and 50 as black for each bot), "
            "stored separately from the main evaluation."
        ),
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Skip the performance profiling step and run only the full pairwise evaluation",
    )
    args = parser.parse_args()

    print("Discovering players...")
    player_classes = discover_players()
    if not player_classes:
        print("No players found!")
        return

    print(f"Found {len(player_classes)} players: {[cls.__name__ for cls in player_classes]}")

    # Filter out HumanPlayer for automated evaluation
    player_classes = [cls for cls in player_classes if "Human" not in cls.__name__]
    print(f"Running evaluation with {len(player_classes)} bots: {[cls.__name__ for cls in player_classes]}")

    if not player_classes:
        print("No bot players found!")
        return

    storage = ResultStorage()
    ranking = RankingSystem(storage)
    runner = GameRunner()
    profiler = PerformanceProfiler()

    mode = 2

    # Clear old results if --new flag is set
    if args.new:
        storage.clear_results(mode)
        print(f"Cleared old evaluation results for mode {mode}")

    # Optional performance profiling before main evaluation
    if not args.no_profile:
        profiler.profile_all_players(player_classes, mode=mode, games=50, include_random=True)

    # If we only wanted profiling, stop here.
    if args.profile_only:
        return

    # Short evaluation against RandomPlayer only (stored separately, not part of the main eval).
    if args.short_eval:
        names = [cls.__name__ for cls in player_classes]
        random_cls = None
        for cls in player_classes:
            if cls.__name__ == "RandomPlayer":
                random_cls = cls
                break

        if random_cls is None:
            print("RandomPlayer not found among discovered players; cannot run short eval.")
            return

        print("Running short evaluation against RandomPlayer (50 games as white and 50 as black per bot)...")

        for cls in player_classes:
            name = cls.__name__
            if name == "RandomPlayer":
                continue  # Skip self-random; short eval is random vs each other bot

            # Bot as white, Random as black
            for _ in range(50):
                result = runner.run_game(mode, cls, random_cls, name, "RandomPlayer")
                storage.save_short_eval_result(result)

            # Random as white, Bot as black
            for _ in range(50):
                result = runner.run_game(mode, random_cls, cls, "RandomPlayer", name)
                storage.save_short_eval_result(result)

        print("Short evaluation complete. Results stored separately from the main evaluation.")
        return

    names = [cls.__name__ for cls in player_classes]
    # Include self-matchups (e.g., RandomPlayer vs RandomPlayer, GreedyPlayer vs GreedyPlayer)
    # Each pair needs args.games with A as white vs B as black
    # So total games = N players * N players * args.games (includes all pairs and self-play)
    total_games = len(player_classes) * len(player_classes) * args.games

    # Calculate initial progress
    current_results = storage.load_results(mode)
    played_count = 0
    counts = {p1: {p2: 0 for p2 in names} for p1 in names}

    for r in current_results:
        if r.white_player in names and r.black_player in names:
            counts[r.white_player][r.black_player] += 1

    for p1 in names:
        for p2 in names:
            # Include self-matchups (e.g., RandomPlayer vs RandomPlayer, each player vs itself)
            # We cap at args.games because that's our target per direction for the progress bar
            played_count += min(counts.get(p1, {}).get(p2, 0), args.games)

    from tqdm import tqdm

    with tqdm(total=total_games, initial=played_count, desc="Evaluation Progress") as pbar:
        while True:
            # Suggest next match
            suggestion = ranking.suggest_next_match(mode, player_classes, min_games=args.games)

            if not suggestion:
                # print("Evaluation complete for current criteria.")
                break

            p1_cls, p2_cls, p1_name, p2_name = suggestion

            # print(f"Running Match: {p1_name} (White) vs {p2_name} (Black)...")

            result = runner.run_game(mode, p1_cls, p2_cls, p1_name, p2_name)
            storage.save_result(result)

            pbar.update(1)
            pbar.set_postfix(last=f"{result.winner} in {result.moves}")
            # print(f"Result: {result.winner.upper()} won in {result.moves} moves ({result.duration:.2f}s)")

    # Show final rankings
    print("\n--- Final Rankings (Win Points / Games) ---")
    ranks = ranking.get_rankings(mode, [cls.__name__ for cls in player_classes])
    for i, (name, score) in enumerate(ranks, 1):
        print(f"{i}. {name}: {score:.2f}")


if __name__ == "__main__":
    main()
