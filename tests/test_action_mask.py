"""Checks for the action masking integration used by MaskablePPO."""

from __future__ import annotations

import numpy as np

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.env.pferdeapfel_env import PferdeapfelEnv, legal_action_mask
from src.game.board import Board
from src.game.rules import Rules
from src.training.wrappers import ForwardingActionMasker


def test_action_mask_matches_legal_moves() -> None:
    """The mask should mirror the legal moves computed by Rules."""
    env = PferdeapfelEnv(agent_color="black", opponent_policy="none")
    env.reset(seed=123)
    mask = env.get_action_mask()
    assert mask.shape == (Board.BOARD_SIZE * Board.BOARD_SIZE,)
    assert mask.dtype == bool

    assert env.board is not None
    legal_moves = Rules.get_legal_knight_moves(env.board, env.agent_color)
    assert mask.sum() == len(legal_moves)
    for row, col in legal_moves:
        idx = row * Board.BOARD_SIZE + col
        assert mask[idx]


def test_maskable_ppo_respects_action_mask() -> None:
    """MaskablePPO predictions should land on legal squares when masks are supplied."""

    def env_fn():
        base_env = PferdeapfelEnv(agent_color="black", opponent_policy="none")
        return ForwardingActionMasker(base_env, legal_action_mask)

    vec_env = DummyVecEnv([env_fn])
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        n_steps=8,
        batch_size=4,
        learning_rate=1e-3,
        gamma=0.9,
        verbose=0,
    )

    obs = vec_env.reset()
    action_masks = np.array(vec_env.env_method("get_action_mask"))
    selected_action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
    action_scalar = int(np.asarray(selected_action).squeeze())
    move = (action_scalar // Board.BOARD_SIZE, action_scalar % Board.BOARD_SIZE)

    wrapped_env = vec_env.envs[0]
    base_env: PferdeapfelEnv = wrapped_env.env  # type: ignore[attr-defined]
    assert base_env.board is not None
    assert move in Rules.get_legal_knight_moves(base_env.board, base_env.agent_color)

    vec_env.close()
