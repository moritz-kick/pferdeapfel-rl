"""
Benchmark script for PPO player.
Runs 600 games total:
- 100 games PPO vs PPO (Modes 1, 2, 3)
- 100 games PPO vs Random (Modes 1, 2, 3)
Tracks winrate and legal move percentage.
"""

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.game.game import Game
from src.players.random import RandomPlayer
from src.players.rl.ppo_player import PPOPlayer

# Configure logging
logging.basicConfig(level=logging.ERROR, format="%(message)s")
logger = logging.getLogger(__name__)


def run_game(white_player, black_player, mode):
    """Run a single game and return stats."""
    game = Game(white_player, black_player, mode=mode, logging=False)

    total_moves = 0
    ppo_moves_count = 0
    legal_moves_count = 0

    while not game.game_over:
        current_player = game.get_current_player()
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            # Game should handle this, but just in case
            break

        move_to, extra_apple = current_player.get_move(game.board, legal_moves)

        # Track legal moves for PPO players
        if isinstance(current_player, PPOPlayer):
            metadata = getattr(current_player, "last_move_metadata", {})
            if metadata.get("source") == "model":
                legal_moves_count += 1
            ppo_moves_count += 1

        game.make_move(move_to, extra_apple)
        total_moves += 1

    return {
        "winner": game.winner,
        "total_moves": total_moves,
        "ppo_moves": ppo_moves_count,
        "legal_moves": legal_moves_count
    }


def run_benchmark(ppo_model_path, quick=False):
    """Run the benchmark suite."""
    modes = [1, 2, 3]
    games_per_setup = 2 if quick else 100

    # Separate results for PPO vs PPO: track white wins, black wins, draws, and move counts
    ppo_vs_ppo_results = defaultdict(lambda: {
        "white_wins": 0,
        "black_wins": 0,
        "draws": 0,
        "total": 0,
        "total_moves": 0,
        "legal_moves": 0,
        "total_ppo_moves": 0
    })
    
    # Results for PPO vs Random (keep existing structure)
    ppo_vs_random_results = defaultdict(lambda: {"wins": 0, "draws": 0, "total": 0, "legal_moves": 0, "total_ppo_moves": 0})

    print(f"Starting benchmark with model: {ppo_model_path}")
    print(f"Games per setup: {games_per_setup}")
    print("-" * 60)

    start_time = time.time()

    # 1. PPO vs PPO
    for mode in modes:
        print(f"Running PPO vs PPO (Mode {mode})...")
        for i in range(games_per_setup):
            p1 = PPOPlayer("white", ppo_model_path)
            p2 = PPOPlayer("black", ppo_model_path)

            stats = run_game(p1, p2, mode)

            key = f"Mode {mode}"
            ppo_vs_ppo_results[key]["total"] += 1
            if stats["winner"] == "white":
                ppo_vs_ppo_results[key]["white_wins"] += 1
            elif stats["winner"] == "black":
                ppo_vs_ppo_results[key]["black_wins"] += 1
            elif stats["winner"] == "draw":
                ppo_vs_ppo_results[key]["draws"] += 1

            ppo_vs_ppo_results[key]["total_moves"] += stats["total_moves"]
            ppo_vs_ppo_results[key]["legal_moves"] += stats["legal_moves"]
            ppo_vs_ppo_results[key]["total_ppo_moves"] += stats["ppo_moves"]

        print(f"  Completed {games_per_setup} games")

    # 2. PPO vs Random
    for mode in modes:
        print(f"Running PPO vs Random (Mode {mode})...")
        # Split games between PPO as White and PPO as Black
        half_games = games_per_setup // 2

        # PPO as White
        for i in range(half_games):
            ppo = PPOPlayer("white", ppo_model_path)
            rnd = RandomPlayer("black")

            stats = run_game(ppo, rnd, mode)

            key = f"PPO vs Random (Mode {mode})"
            ppo_vs_random_results[key]["total"] += 1
            if stats["winner"] == "white":
                ppo_vs_random_results[key]["wins"] += 1
            elif stats["winner"] == "draw":
                ppo_vs_random_results[key]["draws"] += 1

            ppo_vs_random_results[key]["legal_moves"] += stats["legal_moves"]
            ppo_vs_random_results[key]["total_ppo_moves"] += stats["ppo_moves"]

        # PPO as Black
        for i in range(half_games):
            rnd = RandomPlayer("white")
            ppo = PPOPlayer("black", ppo_model_path)

            stats = run_game(rnd, ppo, mode)

            key = f"PPO vs Random (Mode {mode})"
            ppo_vs_random_results[key]["total"] += 1
            if stats["winner"] == "black":
                ppo_vs_random_results[key]["wins"] += 1
            elif stats["winner"] == "draw":
                ppo_vs_random_results[key]["draws"] += 1

            ppo_vs_random_results[key]["legal_moves"] += stats["legal_moves"]
            ppo_vs_random_results[key]["total_ppo_moves"] += stats["ppo_moves"]

        print(f"  Completed {games_per_setup} games")

    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print(f"BENCHMARK RESULTS (Duration: {duration:.2f}s)")
    print("=" * 80)
    
    # PPO vs PPO Results
    print("\nPPO vs PPO Results:")
    print("-" * 80)
    print(f"{'Mode':<8} | {'White Win %':<12} | {'Black Win %':<12} | {'Draw %':<10} | {'Avg Moves':<12} | {'Total Games':<12}")
    print("-" * 80)
    
    for mode in modes:
        key = f"Mode {mode}"
        data = ppo_vs_ppo_results[key]
        white_win_rate = (data["white_wins"] / data["total"]) * 100 if data["total"] > 0 else 0
        black_win_rate = (data["black_wins"] / data["total"]) * 100 if data["total"] > 0 else 0
        draw_rate = (data["draws"] / data["total"]) * 100 if data["total"] > 0 else 0
        avg_moves = data["total_moves"] / data["total"] if data["total"] > 0 else 0
        
        print(f"{mode:<8} | {white_win_rate:>11.1f}% | {black_win_rate:>11.1f}% | {draw_rate:>9.1f}% | {avg_moves:>11.1f} | {data['total']:>12}")
    
    # Overall PPO vs PPO summary
    overall_white = sum(ppo_vs_ppo_results[f"Mode {m}"]["white_wins"] for m in modes)
    overall_black = sum(ppo_vs_ppo_results[f"Mode {m}"]["black_wins"] for m in modes)
    overall_draws = sum(ppo_vs_ppo_results[f"Mode {m}"]["draws"] for m in modes)
    overall_total = sum(ppo_vs_ppo_results[f"Mode {m}"]["total"] for m in modes)
    overall_moves = sum(ppo_vs_ppo_results[f"Mode {m}"]["total_moves"] for m in modes)
    
    overall_white_rate = (overall_white / overall_total) * 100 if overall_total > 0 else 0
    overall_black_rate = (overall_black / overall_total) * 100 if overall_total > 0 else 0
    overall_draw_rate = (overall_draws / overall_total) * 100 if overall_total > 0 else 0
    overall_avg_moves = overall_moves / overall_total if overall_total > 0 else 0
    
    print("-" * 80)
    print(f"{'Overall':<8} | {overall_white_rate:>11.1f}% | {overall_black_rate:>11.1f}% | {overall_draw_rate:>9.1f}% | {overall_avg_moves:>11.1f} | {overall_total:>12}")
    
    # PPO vs Random Results (keep existing format)
    print("\nPPO vs Random Results:")
    print("-" * 80)
    print(f"{'Setup':<25} | {'Win Rate':<10} | {'Draw Rate':<10} | {'Legal Move %':<15}")
    print("-" * 80)

    for key, data in sorted(ppo_vs_random_results.items()):
        win_rate = (data["wins"] / data["total"]) * 100 if data["total"] > 0 else 0
        draw_rate = (data["draws"] / data["total"]) * 100 if data["total"] > 0 else 0
        legal_pct = (data["legal_moves"] / data["total_ppo_moves"] * 100) if data["total_ppo_moves"] > 0 else 0

        print(f"{key:<25} | {win_rate:>9.1f}% | {draw_rate:>9.1f}% | {legal_pct:>14.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PPO player.")
    parser.add_argument("--model", type=str, help="Path to PPO model zip file")
    parser.add_argument("--quick", action="store_true", help="Run a quick test (2 games per setup)")

    args = parser.parse_args()

    model_path = args.model
    if not model_path:
        # Try to find latest model
        models_dir = project_root / "data" / "models" / "ppo_self_play"
        candidates = list(models_dir.rglob("*.zip"))
        if candidates:
            # Sort by mtime
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            model_path = str(candidates[0])
            print(f"Auto-detected latest model: {model_path}")
        else:
            print("Error: No model provided and none found in data/models")
            sys.exit(1)

    run_benchmark(model_path, args.quick)
