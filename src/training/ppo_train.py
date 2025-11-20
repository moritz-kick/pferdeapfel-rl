"""Basic PPO training script for the Pferdeäpfel mode 2 environment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.env.pferdeapfel_env import PferdeapfelEnv

logger = logging.getLogger(__name__)


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO agent for Pferdeäpfel (mode 2).")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml", help="Config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    trainer_cfg: Dict[str, Any] = config.get("trainer", {})
    env_cfg: Dict[str, Any] = config.get("env", {})

    total_timesteps = int(trainer_cfg.get("total_timesteps", 10000))
    learning_rate = float(trainer_cfg.get("learning_rate", 3e-4))
    n_steps = int(trainer_cfg.get("n_steps", 256))
    batch_size = int(trainer_cfg.get("batch_size", 64))
    gamma = float(trainer_cfg.get("gamma", 0.99))
    tensorboard_log = Path(trainer_cfg.get("tensorboard_log", "data/logs/rl"))
    model_path = Path(trainer_cfg.get("save_path", "data/models/ppo_pferdeapfel"))

    tensorboard_log.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    env_factory = lambda: PferdeapfelEnv(**env_cfg)
    vec_env = make_vec_env(env_factory, n_envs=1)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        tensorboard_log=str(tensorboard_log),
        verbose=1,
    )

    learn_kwargs: Dict[str, Any] = {"total_timesteps": total_timesteps}
    # SB3>=2.1 supports progress_bar; ignore if unavailable.
    try:  # pragma: no cover - depends on installed SB3 version
        learn_kwargs["progress_bar"] = True
        model.learn(**learn_kwargs)
    except TypeError:
        learn_kwargs.pop("progress_bar", None)
        model.learn(**learn_kwargs)

    model.save(str(model_path))
    logger.info("Model saved to %s", model_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
