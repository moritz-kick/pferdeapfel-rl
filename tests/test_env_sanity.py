"""Sanity check for KnightSelfPlayEnv."""

import numpy as np
from src.env.knight_self_play_env import KnightSelfPlayEnv


def test_env_sanity():
    # Test with random mode
    env = KnightSelfPlayEnv(mode="random")
    obs, info = env.reset()

    print(f"Mode selected: {env.game.board.mode}")
    print(f"Agent color: {env.agent_color}")
    print("Observation shape:", obs.shape)
    assert obs.shape == (7, 8, 8)

    # Check channel 0 (My Pos) has exactly one 1.0
    assert np.sum(obs[0]) == 1.0
    # Check channel 1 (Opp Pos) has exactly one 1.0
    assert np.sum(obs[1]) == 1.0

    # Check Mode Channel (3)
    mode_val = obs[3, 0, 0]
    expected_mode_val = {1: 0.0, 2: 0.5, 3: 1.0}[env.game.board.mode]
    assert np.allclose(mode_val, expected_mode_val)
    print(f"Mode channel check passed (val={mode_val})")

    # Check Role Channel (4) - matches agent_color
    role_val = obs[4, 0, 0]
    expected_role = 1.0 if env.agent_color == "white" else 0.0
    assert np.allclose(role_val, expected_role)
    print(f"Role channel check passed (val={role_val})")

    print("Initial observation check passed.")

    # Make a random move
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    print("Step result:", reward, done, info)
    print("New observation shape:", obs.shape)

    if not done:
        # Check that observation is still from agent's perspective
        my_pos = env.game.board.get_horse_position(env.agent_color)
        assert obs[0, my_pos[0], my_pos[1]] == 1.0

        # Role channel should remain the same (agent doesn't switch sides)
        new_role_val = obs[4, 0, 0]
        assert np.allclose(new_role_val, role_val), "Agent perspective should stay consistent"
        print("Agent perspective consistency check passed.")

    print("Sanity check complete.")


if __name__ == "__main__":
    test_env_sanity()
