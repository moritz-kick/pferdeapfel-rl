"""Sanity check for KnightSelfPlayEnv."""

import numpy as np
from src.env.knight_self_play_env import KnightSelfPlayEnv
from src.game.board import Board
from src.game.rules import Rules


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


def test_mode_1_action_mask_apple_on_legal_move():
    """Test that Mode 1 action mask correctly handles apple placement on legal moves.
    
    The mask should:
    - Allow placing apple on a legal move destination IF other moves exist
    - Forbid placing apple on the ONLY legal move destination (would block self)
    """
    env = KnightSelfPlayEnv(mode=1, agent_color="white")
    obs, info = env.reset()
    
    board = env.game.board
    legal_moves = Rules.get_legal_knight_moves(board, "white")
    
    # White starts at (0,0), legal moves are (1,2) and (2,1)
    assert len(legal_moves) == 2
    assert (1, 2) in legal_moves
    assert (2, 1) in legal_moves
    
    # Get action mask
    mask = env.action_masks()
    apple_mask = mask[8:]  # First 8 are move masks, rest are apple masks
    
    # Move destinations are NEVER valid for apple placement in Mode 1
    # (because after moving there, the player occupies that square)
    idx_1_2 = 1 * 8 + 2  # Square (1,2) -> index 10
    idx_2_1 = 2 * 8 + 1  # Square (2,1) -> index 17
    
    assert apple_mask[idx_1_2] == False, "Should NOT allow apple on move destination (1,2)"
    assert apple_mask[idx_2_1] == False, "Should NOT allow apple on move destination (2,1)"
    
    # But the player's current position IS valid (will be empty after move)
    idx_0_0 = 0 * 8 + 0
    assert apple_mask[idx_0_0] == True, "Should allow apple on current position (0,0)"
    
    # Other empty non-move squares are valid
    idx_5_5 = 5 * 8 + 5
    assert apple_mask[idx_5_5] == True, "Should allow apple on (5,5)"
    
    # Now block one move so only one remains
    board.grid[1, 2] = Board.BROWN_APPLE
    board._mark_occupied(1, 2)
    
    legal_moves = Rules.get_legal_knight_moves(board, "white")
    assert len(legal_moves) == 1
    assert legal_moves[0] == (2, 1)
    
    # Get updated action mask
    mask = env.action_masks()
    apple_mask = mask[8:]
    
    # With new Mode 1 rules (move first, then place apple):
    # - We must exclude ALL legal move destinations from apple mask
    # - (2,1) is a legal move destination, so it's NOT valid for apple
    # - Player's current position (0,0) WILL be empty after they move
    idx_0_0 = 0 * 8 + 0  # Square (0,0) -> index 0
    assert apple_mask[idx_0_0] == True, "Should allow apple on old position (0,0)"
    
    # (2,1) is a legal move destination - NOT valid for apple placement
    assert apple_mask[idx_2_1] == False, "Should NOT allow apple on move destination (2,1)"
    
    # Other empty squares should still be allowed
    idx_5_5 = 5 * 8 + 5  # Square (5,5) -> index 45
    assert apple_mask[idx_5_5] == True, "Should allow apple on non-move squares"
    
    print("Mode 1 action mask test passed.")


if __name__ == "__main__":
    test_env_sanity()
    test_mode_1_action_mask_apple_on_legal_move()
