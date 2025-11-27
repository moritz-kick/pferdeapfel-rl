"""Benchmark script to find optimal n_envs for PPO training."""

import time
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from src.env.knight_self_play_env import KnightSelfPlayEnv

# Configure logging
logging.basicConfig(level=logging.ERROR)


def benchmark_n_envs(n_envs_list=[4, 8, 16, 32, 64, 128, 256, 512], n_steps=2048):
    print(f"Benchmarking PPO training with {n_steps} steps per rollout...")
    print("-" * 40)
    print(f"{'n_envs':<10} | {'FPS':<10} | {'Time (s)':<10}")
    print("-" * 40)

    results = {}

    for n_envs in n_envs_list:
        try:
            # Create env
            env = make_vec_env(lambda: Monitor(KnightSelfPlayEnv(mode="random")), n_envs=n_envs)

            # Create model
            model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                n_steps=n_steps // n_envs
                if n_steps >= n_envs
                else 1,  # Adjust n_steps to keep total steps roughly constant per update
                batch_size=64,
            )

            # Warmup
            model.learn(total_timesteps=n_envs * 2)

            # Benchmark
            start_time = time.time()
            # Run for a fixed number of total timesteps to get a stable FPS measurement
            # Let's aim for ~10k steps or so
            total_timesteps = 10_000
            model.learn(total_timesteps=total_timesteps)
            end_time = time.time()

            duration = end_time - start_time
            fps = total_timesteps / duration

            print(f"{n_envs:<10} | {int(fps):<10} | {duration:.2f}")
            results[n_envs] = fps

            env.close()

        except Exception as e:
            print(f"{n_envs:<10} | {'ERROR':<10} | {str(e)}")

    print("-" * 40)
    best_n_envs = max(results, key=results.get)
    print(f"Optimal n_envs: {best_n_envs} (FPS: {int(results[best_n_envs])})")


if __name__ == "__main__":
    benchmark_n_envs()
