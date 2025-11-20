"""Basic I/O checks for the RL environment and player wrappers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.env.pferdeapfel_env import PferdeapfelEnv
from src.players.rl.dqn_rl import DQNPlayer


class DummyModel:
    """Minimal SB3-like model for testing."""

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        # Intentionally return an action that may be illegal to trigger fallback logic.
        return [obs.size - 1], None


def test_env_spaces_and_step() -> None:
    """Ensure reset/step produce observations inside the declared space."""
    env = PferdeapfelEnv(agent_color="black", opponent_policy="random")
    obs, _ = env.reset(seed=123)
    assert env.observation_space.contains(obs)

    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)

    assert env.observation_space.contains(obs2)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "legal_moves" in info


def test_dqn_player_fallback_and_logging() -> None:
    """DQNPlayer should map predictions to legal moves and allow simple logging."""
    player = DQNPlayer(side="white", model=DummyModel())

    log_dir = Path("data/logs/rl_player")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "io_test.json"

    env = PferdeapfelEnv(agent_color="white", opponent_policy="none")
    obs, _ = env.reset(seed=42)
    action = env.action_space.sample()
    obs2, reward, terminated, _, _ = env.step(action)

    assert env.board is not None
    move, _ = player.get_move(env.board, [(1, 2), (2, 1)])
    assert move in [(1, 2), (2, 1)]

    with log_file.open("w") as f:
        json.dump(
            {
                "initial_obs_sum": float(np.sum(obs)),
                "next_obs_sum": float(np.sum(obs2)),
                "reward": float(reward),
                "terminated": terminated,
                "chosen_move": move,
            },
            f,
        )

    assert log_file.exists()
