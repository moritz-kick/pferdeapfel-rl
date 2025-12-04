#!/usr/bin/env python3
"""
Measure average game length (number of moves) for Random vs Random games.

This helps determine appropriate n_steps for PPO training.
"""

import numpy as np
from collections import defaultdict

from src.env.knight_self_play_env import KnightSelfPlayEnv


def measure_game_lengths(n_games_per_mode: int = 100) -> dict:
    """
    Play Random vs Random games and measure their lengths.
    
    Args:
        n_games_per_mode: Number of games to play per mode
        
    Returns:
        Dictionary with statistics per mode and overall
    """
    results = defaultdict(list)
    
    for mode in [1, 2, 3]:
        print(f"\nMode {mode}: Playing {n_games_per_mode} games...")
        
        for game_idx in range(n_games_per_mode):
            env = KnightSelfPlayEnv(
                mode=mode,
                agent_color="white",
                opponent_policy=None,  # Random opponent
            )
            
            obs, info = env.reset()
            done = False
            steps = 0
            
            while not done:
                # Get action mask (length 73: 8 moves + 65 apple options)
                action_mask = env.action_masks()
                move_mask = action_mask[:8]
                apple_mask = action_mask[8:]
                
                # Random valid move
                legal_moves = np.where(move_mask)[0]
                if len(legal_moves) == 0:
                    break  # No legal moves
                move_idx = np.random.choice(legal_moves)
                
                # Random valid apple placement
                legal_apples = np.where(apple_mask)[0]
                if len(legal_apples) == 0:
                    apple_idx = 64  # No apple
                else:
                    apple_idx = np.random.choice(legal_apples)
                
                action = np.array([move_idx, apple_idx])
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                done = terminated or truncated
            
            results[mode].append(steps)
            env.close()
            
            if (game_idx + 1) % 25 == 0:
                print(f"  {game_idx + 1}/{n_games_per_mode} games completed...")
    
    return results


def print_statistics(results: dict) -> None:
    """Print detailed statistics about game lengths."""
    print("\n" + "=" * 60)
    print("Game Length Statistics (Random vs Random)")
    print("=" * 60)
    
    all_games = []
    
    for mode in [1, 2, 3]:
        lengths = results[mode]
        all_games.extend(lengths)
        
        print(f"\nMode {mode}:")
        print(f"  Games played:  {len(lengths)}")
        print(f"  Min steps:     {min(lengths)}")
        print(f"  Max steps:     {max(lengths)}")
        print(f"  Mean steps:    {np.mean(lengths):.1f}")
        print(f"  Median steps:  {np.median(lengths):.1f}")
        print(f"  Std dev:       {np.std(lengths):.1f}")
        print(f"  25th %ile:     {np.percentile(lengths, 25):.1f}")
        print(f"  75th %ile:     {np.percentile(lengths, 75):.1f}")
        print(f"  90th %ile:     {np.percentile(lengths, 90):.1f}")
    
    print("\n" + "-" * 60)
    print("Overall (all modes combined):")
    print(f"  Total games:   {len(all_games)}")
    print(f"  Min steps:     {min(all_games)}")
    print(f"  Max steps:     {max(all_games)}")
    print(f"  Mean steps:    {np.mean(all_games):.1f}")
    print(f"  Median steps:  {np.median(all_games):.1f}")
    print(f"  Std dev:       {np.std(all_games):.1f}")
    print(f"  25th %ile:     {np.percentile(all_games, 25):.1f}")
    print(f"  75th %ile:     {np.percentile(all_games, 75):.1f}")
    print(f"  90th %ile:     {np.percentile(all_games, 90):.1f}")
    
    print("\n" + "=" * 60)
    print("Recommendations for n_steps:")
    print("=" * 60)
    mean_len = np.mean(all_games)
    p90_len = np.percentile(all_games, 90)
    
    print(f"\nWith 64 parallel environments:")
    print(f"  To capture ~1 full game per env:     n_steps = {int(mean_len)}")
    print(f"  To capture 90% of games completely:  n_steps = {int(p90_len)}")
    print(f"  Current setting (n_steps=128):       ~{128/mean_len:.1f} games per env")
    print(f"  Buffer size with n_steps=128:        {128 * 64} samples")


if __name__ == "__main__":
    results = measure_game_lengths(n_games_per_mode=100)
    print_statistics(results)
