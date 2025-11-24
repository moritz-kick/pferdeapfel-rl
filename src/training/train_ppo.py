"""Training script for PPO agent."""

import logging
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from src.env.pferdeapfel_env import PferdeapfelEnv

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def main() -> None:
    """Train PPO agent."""
    # Create directories
    models_dir = Path("models/ppo")
    logs_dir = Path("logs/ppo")
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Initialize environment
    # We use a vectorized environment for faster training
    env = make_vec_env(lambda: Monitor(PferdeapfelEnv(mode=3)), n_envs=4)

    # Initialize PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(logs_dir),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Entropy coefficient to encourage exploration
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(models_dir),
        name_prefix="ppo_pferdeapfel",
    )

    # Eval callback to track win rate against Random
    eval_env = Monitor(PferdeapfelEnv(mode=3))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir / "best_model"),
        log_path=str(logs_dir),
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    # Train
    logger.info("Starting training...")
    try:
        model.learn(
            total_timesteps=3_000_000,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving current model...")
    finally:
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model.save(models_dir / f"ppo_pferdeapfel_final_{timestamp}")
        logger.info("Model saved.")


if __name__ == "__main__":
    main()
