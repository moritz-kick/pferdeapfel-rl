"""
Benchmark optimal n_envs for PPO training on your machine.
"""
import time
import multiprocessing
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from src.env.pferdeapfel_env import PferdeapfelEnv

# Increase iterations to get stable measurement
TEST_ITERATIONS = 1000
N_ENVS_LIST = [1, 2, 4, 8, 16, 32, 64]

if __name__ == "__main__":
    # Mac M1/M2/M3 usually have 8+ cores.
    cpu_count = multiprocessing.cpu_count()
    print(f"Detected {cpu_count} CPUs.")
    
    print("Benchmarking n_envs with SubprocVecEnv (Multiprocessing)...")
    
    for n_envs in N_ENVS_LIST:
        print(f"\nTesting n_envs={n_envs}")
        env_factory = lambda: PferdeapfelEnv()
        
        # Use SubprocVecEnv for n_envs > 1 to utilize multiple cores
        # This spawns separate processes for each env
        vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
        
        try:
            vec_env = make_vec_env(env_factory, n_envs=n_envs, vec_env_cls=vec_env_cls)
            obs = vec_env.reset()
            
            start = time.time()
            for _ in range(TEST_ITERATIONS):
                # Take random actions
                actions = [vec_env.action_space.sample() for _ in range(n_envs)]
                obs, rewards, dones, infos = vec_env.step(actions)
            elapsed = time.time() - start
            
            total_steps = TEST_ITERATIONS * n_envs
            sps = total_steps / elapsed
            print(f"n_envs={n_envs}: {total_steps} total steps in {elapsed:.2f}s -> {sps:.2f} steps/sec")
            
            vec_env.close()
        except Exception as e:
            print(f"Failed with n_envs={n_envs}: {e}")

    print("\nDone. Use the n_envs with the highest steps/sec!")
