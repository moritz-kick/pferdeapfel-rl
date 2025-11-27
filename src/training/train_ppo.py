"""Training script for PPO agent."""

import argparse
import logging
import glob
import os
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from src.env.knight_self_play_env import KnightSelfPlayEnv

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO agent for Pferdeäpfel.")
    parser.add_argument(
        "--continue",
        dest="continue_training",
        action="store_true",
        help="Continue training from the latest model.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=32,
        help="Number of parallel environments (default: 32, optimal for M1).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20_000_000,
        help="Total timesteps to train (default: 10M).",
    )
    return parser.parse_args()


def get_latest_model(models_dir: Path) -> Path | None:
    """Find the latest model zip file in the directory."""
    files = glob.glob(str(models_dir / "*.zip"))
    if not files:
        return None
    return Path(max(files, key=os.path.getctime))


def main() -> None:
    """Train PPO agent."""
    args = parse_args()

    # Create directories
    models_dir = Path("models/ppo")
    logs_dir = Path("logs/ppo")
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Initialize environment
    # We use a vectorized environment for faster training
    # Note: KnightSelfPlayEnv is a self-play env, so it doesn't need an opponent policy argument
    # We use mode="random" to train on all modes
    logger.info(f"Creating {args.n_envs} parallel environments...")
    env = make_vec_env(lambda: Monitor(KnightSelfPlayEnv(mode="random")), n_envs=args.n_envs)

    model = None
    reset_num_timesteps = True

    if args.continue_training:
        latest_model_path = get_latest_model(models_dir)
        if latest_model_path:
            logger.info(f"Resuming training from {latest_model_path}")
            # Load model
            # Note: We must pass env to load() to continue training with the new env
            model = PPO.load(latest_model_path, env=env)
            reset_num_timesteps = False
        else:
            logger.warning("No existing model found to continue from. Starting new training.")

    if model is None:
        logger.info("Initializing new PPO model...")
        # Initialize PPO model
        # We use MlpPolicy because the board (8x8) is too small for the default NatureCNN used by CnnPolicy.
        # MlpPolicy will automatically flatten the (7, 8, 8) observation.
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(logs_dir),
            learning_rate=3e-4,
            n_steps=2048 // args.n_envs if 2048 >= args.n_envs else 128,  # Adjust n_steps per env
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Entropy coefficient to encourage exploration
        )

    # Callbacks
    # We only save occasionally to avoid disk spam, but we rely on try/finally for the important save.
    checkpoint_callback = CheckpointCallback(
        save_freq=2_000_000,  # Save every 2M steps just in case
        save_path=str(models_dir),
        name_prefix="ppo_knight_selfplay",
    )

    # Eval callback
    eval_env = Monitor(KnightSelfPlayEnv(mode="random"))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir / "best_model"),
        log_path=str(logs_dir),
        eval_freq=100_000,
        deterministic=True,
        render=False,
    )

    # Train
    logger.info(f"Starting training for {args.steps} timesteps...")
    try:
        model.learn(
            total_timesteps=args.steps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving current model...")
    finally:
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = models_dir / f"ppo_knight_selfplay_final_{timestamp}"
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
