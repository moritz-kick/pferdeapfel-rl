#!/usr/bin/env python
"""
Exhaustive search script for Pferdeäpfel game.
Traverses game tree to a fixed depth to verify rules and stability.
"""

import argparse
import time

from src.debug.debug_env import SimpleDebugEnv


class SearchStats:
    def __init__(self):
        self.nodes_visited = 0
        self.max_depth_reached = 0
        self.start_time = time.time()
        self.games_ended = 0
        self.errors = []


def run_search(mode=2, max_depth=10):
    env = SimpleDebugEnv(mode=mode)
    stats = SearchStats()

    print(f"Starting exhaustive search (DFS) - Mode {mode}, Max Depth {max_depth}")

    try:
        dfs(env, current_depth=0, max_depth=max_depth, stats=stats)
    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")

    duration = time.time() - stats.start_time
    print("\nSearch Complete.")
    print(f"Nodes visited: {stats.nodes_visited}")
    print(f"Max depth reached: {stats.max_depth_reached}")
    print(f"Games ended naturally: {stats.games_ended}")
    print(f"Time taken: {duration:.2f}s")
    print(f"Speed: {stats.nodes_visited / duration:.0f} nodes/sec")

    if stats.errors:
        print(f"\nERRORS FOUND ({len(stats.errors)}):")
        for err in stats.errors[:10]:
            print(err)
        if len(stats.errors) > 10:
            print("...")


def dfs(env: SimpleDebugEnv, current_depth: int, max_depth: int, stats: SearchStats):
    stats.nodes_visited += 1
    stats.max_depth_reached = max(stats.max_depth_reached, current_depth)

    # Progress log
    if stats.nodes_visited % 10000 == 0:
        print(f"Visited {stats.nodes_visited} nodes... (Current Depth: {current_depth})")

    if current_depth >= max_depth:
        return

    if env.game.game_over:
        stats.games_ended += 1
        if current_depth <= 4:
            print(f"\n[INFO] Game ended at depth {current_depth}")
            print(f"Winner: {env.game.winner}")
            print("Final Board State:")
            print(env.get_state_str())
            print("-" * 20)
        return

    actions = env.get_legal_actions()

    if not actions and not env.game.game_over:
        if env.game.game_over:
            stats.games_ended += 1
            return
        else:
            stats.errors.append(f"State has no actions but game_over is False! \n{env.get_state_str()}")
            return

    for action in actions:
        player = env.game.current_player

        # Apply move
        success = env.step(action)

        if not success:
            stats.errors.append(f"Failed to apply legal action {action} for {player}\n{env.get_state_str()}")
            continue

        # Recursive step
        dfs(env, current_depth + 1, max_depth, stats)

        # Backtrack
        env.undo()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=2, help="Game mode (default: 2)")
    parser.add_argument("--depth", type=int, default=10, help="Search depth (default: 10)")
    args = parser.parse_args()

    run_search(mode=args.mode, max_depth=args.depth)
