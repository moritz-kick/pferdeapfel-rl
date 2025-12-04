"""
Profiling script for PPO self-play training to identify performance bottlenecks.

This script measures time spent in:
1. Environment step (including opponent moves)
2. Action mask computation
3. Observation computation
4. Model inference (predict)
5. PPO learning update
6. Evaluation games (during callbacks)

Usage:
    python scripts/profile_training.py
"""

import cProfile
import io
import pstats
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Timing storage
TIMING_DATA: Dict[str, List[float]] = {}
CALL_COUNTS: Dict[str, int] = {}


@contextmanager
def timer(name: str):
    """Context manager to time a code block."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if name not in TIMING_DATA:
        TIMING_DATA[name] = []
        CALL_COUNTS[name] = 0
    TIMING_DATA[name].append(elapsed)
    CALL_COUNTS[name] += 1


def timed(name: str):
    """Decorator to time a function."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with timer(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def print_timing_report():
    """Print a formatted timing report."""
    print("\n" + "=" * 80)
    print("TIMING REPORT")
    print("=" * 80)
    
    # Calculate totals
    total_times = {}
    for name, times in TIMING_DATA.items():
        total_times[name] = sum(times)
    
    grand_total = sum(total_times.values())
    
    # Sort by total time (descending)
    sorted_names = sorted(total_times.keys(), key=lambda x: total_times[x], reverse=True)
    
    print(f"\n{'Component':<40} {'Total (s)':<12} {'Calls':<10} {'Avg (ms)':<12} {'%':<8}")
    print("-" * 80)
    
    for name in sorted_names:
        times = TIMING_DATA[name]
        total = total_times[name]
        calls = CALL_COUNTS[name]
        avg_ms = (total / calls) * 1000 if calls > 0 else 0
        pct = (total / grand_total * 100) if grand_total > 0 else 0
        print(f"{name:<40} {total:<12.3f} {calls:<10} {avg_ms:<12.4f} {pct:<8.1f}")
    
    print("-" * 80)
    print(f"{'GRAND TOTAL':<40} {grand_total:<12.3f}")
    print("=" * 80)


def profile_env_components():
    """Profile individual environment components."""
    print("\n" + "=" * 80)
    print("PROFILING ENVIRONMENT COMPONENTS")
    print("=" * 80)
    
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor
    
    from src.env.knight_self_play_env import KnightSelfPlayEnv
    from src.game.rules import Rules
    
    # Create environment
    env = KnightSelfPlayEnv(mode="random", agent_color="random", opponent_policy=None)
    
    n_episodes = 50
    n_steps_total = 0
    
    reset_times = []
    step_times = []
    obs_times = []
    mask_times = []
    legal_move_times = []
    opponent_move_times = []
    
    for ep in range(n_episodes):
        # Time reset
        start = time.perf_counter()
        obs, info = env.reset()
        reset_times.append(time.perf_counter() - start)
        
        done = False
        while not done:
            # Time action mask computation
            start = time.perf_counter()
            mask = env.action_masks()
            mask_times.append(time.perf_counter() - start)
            
            # Time legal moves computation (part of action selection)
            start = time.perf_counter()
            legal_moves = Rules.get_legal_knight_moves(env.game.board, env.agent_color)
            legal_move_times.append(time.perf_counter() - start)
            
            # Pick random valid action
            move_mask = mask[:8]
            apple_mask = mask[8:]
            valid_moves = np.where(move_mask)[0]
            valid_apples = np.where(apple_mask)[0]
            
            if len(valid_moves) == 0 or len(valid_apples) == 0:
                break
            
            action = np.array([
                np.random.choice(valid_moves),
                np.random.choice(valid_apples)
            ])
            
            # Time step (includes opponent move)
            start = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            step_times.append(time.perf_counter() - start)
            
            done = terminated or truncated
            n_steps_total += 1
    
    env.close()
    
    def print_stats(name: str, times: List[float]):
        if not times:
            return
        total = sum(times)
        avg = np.mean(times) * 1000
        std = np.std(times) * 1000
        print(f"{name:<30} Total: {total:.3f}s | Avg: {avg:.4f}ms | Std: {std:.4f}ms | Calls: {len(times)}")
    
    print(f"\nRan {n_episodes} episodes, {n_steps_total} total steps\n")
    print_stats("reset()", reset_times)
    print_stats("step() [total]", step_times)
    print_stats("action_masks()", mask_times)
    print_stats("get_legal_knight_moves()", legal_move_times)


def profile_with_model():
    """Profile with actual model inference."""
    print("\n" + "=" * 80)
    print("PROFILING WITH MODEL INFERENCE")
    print("=" * 80)
    
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    
    from src.env.knight_self_play_env import KnightSelfPlayEnv
    
    def mask_fn(env):
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        return base_env.action_masks()
    
    def make_env():
        env = KnightSelfPlayEnv(mode="random", agent_color="random", opponent_policy=None)
        env = ActionMasker(env, mask_fn)
        return Monitor(env)
    
    # Test with different numbers of envs
    for n_envs in [1, 4, 8, 16]:
        print(f"\n--- Testing with {n_envs} parallel environments ---")
        
        env = make_vec_env(make_env, n_envs=n_envs)
        
        # Create a small model
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=0,
            n_steps=64,  # Smaller for faster profiling
            batch_size=32,
            n_epochs=2,
        )
        
        # Measure training time for fixed number of steps
        n_steps = 1024
        n_runs = 2
        run_times = []
        
        for run in range(n_runs):
            start = time.perf_counter()
            model.learn(total_timesteps=n_steps, progress_bar=False, reset_num_timesteps=(run == 0))
            run_times.append(time.perf_counter() - start)
        
        avg_time = np.mean(run_times)
        steps_per_sec = n_steps / avg_time
        
        print(f"  {n_steps} steps in {avg_time:.3f}s ({steps_per_sec:.0f} steps/s)")
        
        env.close()


def profile_opponent_overhead():
    """Profile the overhead of opponent model inference."""
    print("\n" + "=" * 80)
    print("PROFILING OPPONENT MODEL OVERHEAD")
    print("=" * 80)
    
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor
    
    from src.env.knight_self_play_env import KnightSelfPlayEnv
    
    def mask_fn(env):
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        return base_env.action_masks()
    
    # Create a model to use as opponent
    dummy_env = KnightSelfPlayEnv(mode="random", agent_color="random", opponent_policy=None)
    dummy_env = ActionMasker(dummy_env, mask_fn)
    dummy_env = Monitor(dummy_env)
    
    opponent_model = MaskablePPO("MlpPolicy", dummy_env, verbose=0)
    dummy_env.close()
    
    n_episodes = 30
    
    # Test WITHOUT opponent model (random opponent)
    env_random = KnightSelfPlayEnv(mode="random", agent_color="random", opponent_policy=None)
    
    random_times = []
    random_steps = 0
    for _ in range(n_episodes):
        obs, _ = env_random.reset()
        done = False
        start = time.perf_counter()
        while not done:
            mask = env_random.action_masks()
            move_mask = mask[:8]
            apple_mask = mask[8:]
            valid_moves = np.where(move_mask)[0]
            valid_apples = np.where(apple_mask)[0]
            if len(valid_moves) == 0 or len(valid_apples) == 0:
                break
            action = np.array([np.random.choice(valid_moves), np.random.choice(valid_apples)])
            obs, reward, terminated, truncated, info = env_random.step(action)
            done = terminated or truncated
            random_steps += 1
        random_times.append(time.perf_counter() - start)
    env_random.close()
    
    # Test WITH opponent model
    env_model = KnightSelfPlayEnv(mode="random", agent_color="random", opponent_policy=opponent_model)
    
    model_times = []
    model_steps = 0
    for _ in range(n_episodes):
        obs, _ = env_model.reset()
        done = False
        start = time.perf_counter()
        while not done:
            mask = env_model.action_masks()
            move_mask = mask[:8]
            apple_mask = mask[8:]
            valid_moves = np.where(move_mask)[0]
            valid_apples = np.where(apple_mask)[0]
            if len(valid_moves) == 0 or len(valid_apples) == 0:
                break
            action = np.array([np.random.choice(valid_moves), np.random.choice(valid_apples)])
            obs, reward, terminated, truncated, info = env_model.step(action)
            done = terminated or truncated
            model_steps += 1
        model_times.append(time.perf_counter() - start)
    env_model.close()
    
    avg_random = sum(random_times) / n_episodes
    avg_model = sum(model_times) / n_episodes
    
    print(f"\nRandom opponent: {avg_random*1000:.2f}ms/episode ({random_steps} total steps)")
    print(f"Model opponent:  {avg_model*1000:.2f}ms/episode ({model_steps} total steps)")
    print(f"Overhead: {(avg_model/avg_random - 1)*100:.1f}% slower with model opponent")
    print(f"Per-step overhead: ~{(avg_model - avg_random) / (model_steps/n_episodes) * 1000:.3f}ms")


def profile_evaluation_games():
    """Profile the evaluation game loop (most expensive part of callbacks)."""
    print("\n" + "=" * 80)
    print("PROFILING EVALUATION GAMES")
    print("=" * 80)
    
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    from src.env.knight_self_play_env import KnightSelfPlayEnv
    
    def mask_fn(env):
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        return base_env.action_masks()
    
    # Create models
    dummy_env = KnightSelfPlayEnv(mode="random", agent_color="random", opponent_policy=None)
    dummy_env = ActionMasker(dummy_env, mask_fn)
    dummy_env = Monitor(dummy_env)
    
    model = MaskablePPO("MlpPolicy", dummy_env, verbose=0)
    opponent = MaskablePPO("MlpPolicy", dummy_env, verbose=0)
    dummy_env.close()
    
    # === SEQUENTIAL EVALUATION ===
    n_games = 50
    
    print(f"\nSequential evaluation: {n_games} games...")
    
    game_times = []
    step_times = []
    predict_times = []
    mask_times = []
    
    seq_start = time.perf_counter()
    for game_idx in range(n_games):
        current_plays_white = game_idx % 2 == 0
        current_color = "white" if current_plays_white else "black"
        
        env = KnightSelfPlayEnv(
            mode="random",
            agent_color=current_color,
            opponent_policy=opponent,
        )
        
        game_start = time.perf_counter()
        obs, info = env.reset()
        done = False
        
        while not done:
            # Time action mask
            start = time.perf_counter()
            action_masks = env.action_masks()
            mask_times.append(time.perf_counter() - start)
            
            # Time prediction
            start = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            predict_times.append(time.perf_counter() - start)
            
            # Time step
            start = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            step_times.append(time.perf_counter() - start)
            
            done = terminated or truncated
        
        game_times.append(time.perf_counter() - game_start)
        env.close()
    
    seq_total = time.perf_counter() - seq_start
    avg_game = np.mean(game_times) * 1000
    avg_step = np.mean(step_times) * 1000
    avg_predict = np.mean(predict_times) * 1000
    avg_mask = np.mean(mask_times) * 1000
    
    print(f"\nSequential total time for {n_games} games: {seq_total:.2f}s")
    print(f"Average per game: {avg_game:.2f}ms")
    print(f"  - step(): {avg_step:.3f}ms ({avg_step/(avg_step+avg_predict+avg_mask)*100:.1f}% of work)")
    print(f"  - predict(): {avg_predict:.3f}ms ({avg_predict/(avg_step+avg_predict+avg_mask)*100:.1f}% of work)")
    print(f"  - action_masks(): {avg_mask:.3f}ms ({avg_mask/(avg_step+avg_predict+avg_mask)*100:.1f}% of work)")
    
    # === PARALLEL EVALUATION SIMULATION ===
    print(f"\n--- Parallel evaluation would speed up by using multiple processes ---")
    
    # Extrapolate to full evaluation
    n_eval_games_default = 1000
    n_random_games = 300  # 50 per mode×color combination
    total_games = n_eval_games_default + n_random_games
    
    seq_estimated = total_games * avg_game / 1000
    print(f"\nEstimated time for full evaluation ({n_eval_games_default} + {n_random_games} games):")
    print(f"  Sequential: {seq_estimated:.1f}s")
    for workers in [2, 4, 8]:
        # Rough estimate - parallel is not perfectly linear due to overhead
        overhead_factor = 1.2  # Account for process spawning overhead
        parallel_estimated = (seq_estimated / workers) * overhead_factor
        print(f"  {workers} workers: ~{parallel_estimated:.1f}s (estimated {seq_estimated/parallel_estimated:.1f}x speedup)")


def profile_cprofile():
    """Run detailed cProfile analysis."""
    print("\n" + "=" * 80)
    print("DETAILED cProfile ANALYSIS")
    print("=" * 80)
    
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    
    from src.env.knight_self_play_env import KnightSelfPlayEnv
    
    def mask_fn(env):
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        return base_env.action_masks()
    
    def make_env():
        env = KnightSelfPlayEnv(mode="random", agent_color="random", opponent_policy=None)
        env = ActionMasker(env, mask_fn)
        return Monitor(env)
    
    n_envs = 16
    env = make_vec_env(make_env, n_envs=n_envs)
    
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
    )
    
    # Profile a short training run
    profiler = cProfile.Profile()
    profiler.enable()
    
    model.learn(total_timesteps=2048, progress_bar=False)
    
    profiler.disable()
    
    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    print(s.getvalue())
    
    env.close()


def main():
    """Run all profiling tests."""
    print("=" * 80)
    print("PPO SELF-PLAY TRAINING PROFILER")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Run profiling tests
    profile_env_components()
    profile_opponent_overhead()
    profile_with_model()
    profile_evaluation_games()
    
    # Detailed cProfile
    print("\nRunning detailed cProfile analysis (this may take a moment)...")
    profile_cprofile()
    
    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)
    print("""
OPTIMIZATIONS IMPLEMENTED:
1. ✅ Cached empty squares in Board class (O(1) lookups)
2. ✅ SubprocVecEnv option for true parallel env execution (--use-subproc)
3. ✅ Parallel evaluation using ProcessPoolExecutor (--n-eval-workers)
4. ✅ Optimized action mask computation using cached empty squares

USAGE:
  # Default (DummyVecEnv, 4 eval workers)
  python -m src.training.train_ppo_self_play
  
  # With SubprocVecEnv for better parallel env stepping
  python -m src.training.train_ppo_self_play --use-subproc
  
  # With more eval workers for faster evaluation
  python -m src.training.train_ppo_self_play --n-eval-workers 8
  
  # Sequential eval (for debugging)
  python -m src.training.train_ppo_self_play --n-eval-workers 0

ABOUT BATCHED OPPONENT INFERENCE:
  When using a model as opponent, each step requires model.predict() which is slow
  for single observations. "Batched inference" would mean:
  - Collect multiple environment states
  - Run predict() on all of them at once (GPU batching)
  - Distribute results back to environments
  This is complex to implement correctly but could give 2-3x speedup.
  Current approach: Use random opponent for training, model for eval only.
""")


if __name__ == "__main__":
    main()
