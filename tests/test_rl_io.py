"""Basic I/O checks for the RL environment and player wrappers."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from src.env.pferdeapfel_env import PferdeapfelEnv
from src.game.rules import Rules
from src.players.rl.dqn_rl import DQNPlayer
from src.players.rl.ppo_rl import PPOPlayer


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


class DummyPPOModel:
    """Minimal PPO-like model that respects the provided action mask."""

    def predict(self, obs: np.ndarray, deterministic: bool = True, action_masks=None):
        if action_masks is not None and np.any(action_masks):
            valid_indices = np.flatnonzero(action_masks)
            return [int(valid_indices[-1])], None
        return [0], None


class CountingDummyPPOModel(DummyPPOModel):
    """Tracks predict() calls so tests can assert masking/selection paths."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def predict(self, obs: np.ndarray, deterministic: bool = True, action_masks=None):
        self.calls += 1
        return super().predict(obs, deterministic=deterministic, action_masks=action_masks)


def test_ppo_player_uses_latest_model(tmp_path: Path) -> None:
    """Latest .zip in search roots should be selected."""
    older = tmp_path / "older.zip"
    newest = tmp_path / "newest.zip"
    older.write_text("old")
    newest.write_text("new")

    now = time.time()
    os.utime(older, (now - 100, now - 100))
    os.utime(newest, (now, now))

    resolved = PPOPlayer._resolve_model_path(None, search_roots=[tmp_path])
    assert resolved == newest


def test_ppo_player_respects_action_mask(tmp_path: Path) -> None:
    """PPOPlayer should map predictions to a legal move using the mask."""
    fake_model_path = tmp_path / "fake_model.zip"
    fake_model_path.write_text("weights")

    env = PferdeapfelEnv(agent_color="white", opponent_policy="none")
    _, _ = env.reset(seed=7)
    assert env.board is not None
    legal_moves = Rules.get_legal_knight_moves(env.board, "white")

    player = PPOPlayer("white", model=DummyPPOModel(), model_path=fake_model_path)
    move, extra = player.get_move(env.board, legal_moves)

    assert move in legal_moves
    assert extra is None


def test_self_play_random_opponent_chance_bypasses_model() -> None:
    """When random_opponent_chance=1.0, the opponent should play randomly, not call predict()."""
    model = CountingDummyPPOModel()
    env = PferdeapfelEnv(
        agent_color="white",
        opponent_policy="self_play",
        opponent_model=model,
        random_opponent_chance=1.0,
    )
    env.reset(seed=0)
    assert env.board is not None
    legal_moves = Rules.get_legal_knight_moves(env.board, env.opponent_color)

    move = env._select_opponent_move(legal_moves)

    assert move in legal_moves
    assert model.calls == 0


def test_self_play_best_pool_selected_before_old() -> None:
    """With best_prob=1.0 the env should pick the best snapshot over old/current."""
    best = CountingDummyPPOModel()
    old = CountingDummyPPOModel()
    env = PferdeapfelEnv(
        agent_color="white",
        opponent_policy="self_play",
        opponent_model=old,
        random_opponent_chance=0.0,
    )
    env.set_opponent_pool(best_model=best, old_models=[old], best_prob=1.0, old_prob=0.0, deterministic=True)
    env.reset(seed=1)
    assert env.board is not None
    legal_moves = Rules.get_legal_knight_moves(env.board, env.opponent_color)
    env._select_opponent_move(legal_moves)

    assert best.calls > 0
    assert old.calls == 0
