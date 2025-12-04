"""
Self-play Gymnasium environment for Pferdeäpfel with Action Masking.

This environment is designed for self-play where a single policy
controls the agent, while the opponent is handled separately.

The key insight: In the original design, the same policy controlled both players
and the reward was given to whoever won. This caused the policy to learn to
"throw the game" quickly because winning fast = maximum discounted reward.

This version fixes the issue by:
1. Assigning the agent to ONE color per episode (randomly chosen)
2. Having a separate opponent make moves (random by default, or a provided policy)
3. Giving rewards only from the agent's perspective
4. Using ACTION MASKING to prevent illegal moves entirely

This is the standard approach for self-play in board games (AlphaGo, etc).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.game.board import Board
from src.game.game import Game
from src.game.rules import Rules
from src.players.random import RandomPlayer

logger = logging.getLogger(__name__)


class KnightSelfPlayEnv(gym.Env):
    """
    Gymnasium environment for Pferdeäpfel with proper self-play handling.

    Key Design: The agent is assigned to ONE color per episode, and the opponent
    is controlled by a separate policy (default: random). This prevents the 
    degenerate case where a single policy controlling both sides learns to 
    "throw the game" for quick rewards.

    Observation Space:
        Box(low=0, high=1, shape=(7, 8, 8), dtype=np.float32)
        - Channel 0: Current Player Position (1.0 at pos, 0.0 elsewhere)
        - Channel 1: Opponent Player Position (1.0 at pos, 0.0 elsewhere)
        - Channel 2: Blocked Squares (1.0 if blocked/apple/visited, 0.0 elsewhere)
        - Channel 3: Mode ID (Normalized: 1->0.0, 2->0.5, 3->1.0)
        - Channel 4: Current Role (1.0 if White, 0.0 if Black)
        - Channel 5: Brown Apples Remaining (Normalized: count / 28.0)
        - Channel 6: Golden Apples Remaining (Normalized: count / 12.0)

    Action Space:
        MultiDiscrete([8, 65])
        - 0: Knight Move Index (0-7) corresponding to Rules.KNIGHT_MOVES
        - 1: Apple Placement Index (0-63 for board squares, 64 for "No Apple")
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        mode: Union[int, str] = "random",
        agent_color: str = "random",
        opponent_policy: Optional[Any] = None,
    ) -> None:
        """
        Initialize the environment.

        Args:
            mode: Game mode (1, 2, 3) or "random"
            agent_color: Which color the RL agent plays ("white", "black", or "random")
            opponent_policy: Optional policy for opponent moves. If None, uses random.
                           Can be a stable-baselines3 model or any callable(obs) -> action
        """
        super().__init__()
        self.mode_config = mode
        self.agent_color_config = agent_color
        self.opponent_policy = opponent_policy

        # Define action and observation space
        self.action_space = spaces.MultiDiscrete([8, 65])
        # (7, 8, 8) tensor for CNN/MLP
        self.observation_space = spaces.Box(low=0, high=1, shape=(7, 8, 8), dtype=np.float32)

        # Initialize game components
        self.white_player = RandomPlayer("White")
        self.black_player = RandomPlayer("Black")
        self.game: Optional[Game] = None

        # The color the RL agent is playing this episode (fixed per episode)
        self.agent_color: str = "white"
        
        # Move counter for shaping rewards
        self.move_count = 0

    def set_opponent_policy(self, policy: Any) -> None:
        """Update the opponent policy (useful for curriculum learning)."""
        self.opponent_policy = policy

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to a new game state."""
        super().reset(seed=seed)

        # Determine mode for this episode
        if self.mode_config == "random":
            mode = int(self.np_random.integers(1, 4))  # 1, 2, or 3
        else:
            mode = int(self.mode_config)

        # Determine which color the agent plays this episode
        if self.agent_color_config == "random":
            self.agent_color = "white" if self.np_random.random() < 0.5 else "black"
        else:
            self.agent_color = self.agent_color_config

        # Create new game
        self.game = Game(self.white_player, self.black_player, mode=mode)
        self.move_count = 0

        # If opponent goes first (agent is black and white starts), let opponent move
        if self.game.current_player != self.agent_color:
            self._opponent_move()

        return self._get_observation(), self._get_info()

    def _opponent_move(self) -> bool:
        """
        Let the opponent make a move.
        
        Returns:
            True if game continues, False if game ended.
        """
        if self.game is None or self.game.game_over:
            return False

        opp_color = self.game.current_player
        legal_moves = Rules.get_legal_knight_moves(self.game.board, opp_color)

        if not legal_moves:
            # Opponent is stuck - game should end
            self.game.get_legal_moves()  # This triggers game over check
            return False

        if self.opponent_policy is not None:
            # Use provided opponent policy
            opp_obs = self._get_observation_for_player(opp_color)
            action = None
            if hasattr(self.opponent_policy, 'predict'):
                # If opponent is a MaskablePPO model, provide action masks for its color
                try:
                    from sb3_contrib import MaskablePPO  # type: ignore
                    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy  # type: ignore
                    is_maskable = isinstance(self.opponent_policy, MaskablePPO) and isinstance(getattr(self.opponent_policy, 'policy', None), MaskableActorCriticPolicy)
                except Exception:
                    is_maskable = False
                if is_maskable:
                    opp_masks = self._get_action_masks_for_color(opp_color)
                    action, _ = self.opponent_policy.predict(opp_obs, deterministic=False, action_masks=opp_masks)
                else:
                    action, _ = self.opponent_policy.predict(opp_obs, deterministic=False)
            else:
                # Callable
                action = self.opponent_policy(opp_obs)
            move_idx, apple_idx = action
            
            # Decode move
            current_pos = self.game.board.get_horse_position(opp_color)
            dr, dc = Rules.KNIGHT_MOVES[move_idx]
            move_to = (current_pos[0] + dr, current_pos[1] + dc)

            extra_apple = None
            if apple_idx < 64:
                extra_apple = (apple_idx // 8, apple_idx % 8)

            # If move is illegal, fall back to random legal move
            if move_to not in legal_moves:
                move_to = legal_moves[self.np_random.integers(0, len(legal_moves))]
                extra_apple = self._get_random_valid_apple()
        else:
            # Random opponent - just pick from legal moves
            move_to = legal_moves[self.np_random.integers(0, len(legal_moves))]
            extra_apple = self._get_random_valid_apple()

        success = self.game.make_move(move_to, extra_apple)
        
        # If move failed (bad apple), try again without apple
        if not success:
            self.game.make_move(move_to, None)

        return not self.game.game_over

    def _get_action_masks_for_color(self, color: str) -> np.ndarray:
        """Compute flattened action mask (length 73) for a specific player color.

        This mirrors ``action_masks`` but allows specifying the actor (agent vs opponent).
        """
        if self.game is None:
            return np.ones(73, dtype=bool)

        board = self.game.board
        current_pos = board.get_horse_position(color)

        move_mask = np.zeros(8, dtype=bool)
        legal_moves = Rules.get_legal_knight_moves(board, color)
        legal_move_set = set(legal_moves)
        for idx, (dr, dc) in enumerate(Rules.KNIGHT_MOVES):
            target = (current_pos[0] + dr, current_pos[1] + dc)
            if target in legal_move_set:
                move_mask[idx] = True

        apple_mask = np.zeros(65, dtype=bool)
        
        # OPTIMIZATION: Use cached empty squares
        empty_squares = board.get_empty_squares()
        
        if board.mode == 1:
            # Mode 1: Apple placement is REQUIRED BEFORE moving
            # Can place on any empty square, BUT if placing on a legal move destination,
            # we must have at least one OTHER legal move remaining
            has_multiple_moves = len(legal_move_set) > 1
            for r, c in empty_squares:
                if (r, c) in legal_move_set:
                    # Can only place here if we have other moves
                    if has_multiple_moves:
                        apple_mask[r * 8 + c] = True
                else:
                    # Not a legal move destination - always allowed
                    apple_mask[r * 8 + c] = True
        elif board.mode == 2:
            apple_mask[64] = True
        elif board.mode == 3:
            # Mode 3: Cannot place on move targets or block White's only escape
            white_legal_moves = Rules.get_legal_knight_moves(board, "white")
            white_legal_set = set(white_legal_moves)
            white_only_one_move = len(white_legal_moves) == 1
            
            for r, c in empty_squares:
                if (r, c) in legal_move_set:
                    continue
                if white_only_one_move and (r, c) in white_legal_set:
                    continue
                apple_mask[r * 8 + c] = True
            apple_mask[64] = True

        if not move_mask.any():
            move_mask[:] = True
        if not apple_mask.any():
            apple_mask[64] = True

        return np.concatenate([move_mask, apple_mask])

    def _get_random_valid_apple(self) -> Optional[Tuple[int, int]]:
        """Get a random valid apple placement position."""
        if self.game is None:
            return None
        
        # For mode 1, apple is required
        # For mode 2, apple is automatic (trail)
        # For mode 3, apple is optional
        if self.game.board.mode == 2:
            return None  # Mode 2 handles apple automatically
        
        empty_squares = []
        for r in range(8):
            for c in range(8):
                if self.game.board.is_empty(r, c):
                    empty_squares.append((r, c))
        
        if empty_squares:
            # For mode 1, always place (required)
            # For mode 3, 50% chance to place optional apple
            if self.game.board.mode == 1:
                return empty_squares[self.np_random.integers(0, len(empty_squares))]
            elif self.game.board.mode == 3:
                if self.np_random.random() < 0.5:
                    return empty_squares[self.np_random.integers(0, len(empty_squares))]
        return None

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        The agent makes a move, then the opponent responds (if game continues).
        Rewards are from the agent's perspective only.

        Args:
            action: [move_index, apple_index]
        """
        if self.game is None:
            raise RuntimeError("Call reset() before step()")

        # 1. Decode Action
        move_idx, apple_idx = action

        # Get current position (agent's position)
        current_pos = self.game.board.get_horse_position(self.agent_color)

        # Calculate target move
        dr, dc = Rules.KNIGHT_MOVES[move_idx]
        move_to = (current_pos[0] + dr, current_pos[1] + dc)

        # Calculate apple placement
        extra_apple = None
        if apple_idx < 64:
            extra_apple = (apple_idx // 8, apple_idx % 8)

        # 2. Validate and Execute Move
        legal_moves = Rules.get_legal_knight_moves(self.game.board, self.agent_color)

        if move_to not in legal_moves:
            # Invalid move - heavy penalty and end episode
            return self._get_observation(), -1.0, True, False, {"error": "Invalid move"}

        # Try to make the move
        success = self.game.make_move(move_to, extra_apple)

        if not success:
            # Invalid apple placement or other rule violation
            return self._get_observation(), -1.0, True, False, {"error": "Invalid action (apple/rules)"}

        self.move_count += 1

        # 3. Check if game ended after agent's move
        if self.game.game_over:
            reward = self._calculate_reward()
            return self._get_observation(), reward, True, False, {"winner": self.game.winner}

        # 4. Opponent's turn
        game_continues = self._opponent_move()

        # 5. Check if game ended after opponent's move
        if self.game.game_over or not game_continues:
            reward = self._calculate_reward()
            return self._get_observation(), reward, True, False, {"winner": self.game.winner}

        # 6. Game continues - apply per-move reward shaping
        # Mode 3 White benefits from longer games (more golden apples = more points)
        # All other situations should encourage faster, decisive play
        step_reward = self._get_step_reward()

        return self._get_observation(), step_reward, False, False, {}

    def _get_step_reward(self) -> float:
        """
        Calculate per-step reward based on mode and color.
        
        Mode 3 White: POSITIVE per-move reward (+0.005)
            - White benefits from longer games in golden phase
            - More golden apples placed = higher score
            - Encourages survival and prolonging the game
        
        All other situations: NEGATIVE per-move penalty (-0.005)
            - Encourages faster, decisive play
            - Captures and quick wins are better
            - Prevents aimless wandering
        """
        if self.game is None:
            return 0.0
        
        is_mode_3 = self.game.board.mode == 3
        is_white = self.agent_color == "white"
        
        if is_mode_3 and is_white:
            # White in Mode 3 wants longer games for golden phase scoring
            return 0.005
        else:
            # All other cases: encourage faster play
            return -0.005

    def _calculate_reward(self) -> float:
        """
        Calculate final reward based on game outcome.
        
        Mode-aware terminal rewards:
        - Mode 3 White: Bonus for longer games (golden phase scoring)
        - Mode 3 Black: Bonus for FASTER wins (capture before golden phase)
        - Mode 1 & 2: Standard win/loss with slight length considerations
        """
        if self.game is None:
            return 0.0

        is_mode_3 = self.game.board.mode == 3
        is_white = self.agent_color == "white"
        
        if self.game.winner == self.agent_color:
            # WIN!
            if is_mode_3:
                if is_white:
                    # White wins - longer games with golden phase = better
                    # Bonus based on golden apples used (12 - remaining)
                    golden_used = 12 - self.game.board.golden_apples_remaining
                    golden_bonus = golden_used * 0.02  # Up to 0.24 bonus
                    game_length_bonus = min(self.move_count * 0.005, 0.25)
                    return 1.0 + golden_bonus + game_length_bonus
                else:
                    # Black wins - faster is better (capture before golden phase)
                    # Bonus for quick wins, penalty for slow ones
                    speed_bonus = max(0.5 - self.move_count * 0.01, 0.0)
                    return 1.0 + speed_bonus
            else:
                # Mode 1 & 2 - slight preference for faster wins
                speed_bonus = max(0.3 - self.move_count * 0.005, 0.0)
                return 1.0 + speed_bonus
                
        elif self.game.winner == "draw":
            return 0.0
        else:
            # LOSS
            if is_mode_3 and is_white:
                # White losing - less penalty if survived long (golden phase scoring)
                survival_reduction = min(self.move_count * 0.005, 0.2)
                return -1.0 + survival_reduction
            else:
                # Black or Mode 1/2 - standard loss penalty
                return -1.0

    def _get_observation(self) -> np.ndarray:
        """Get observation from the agent's perspective."""
        return self._get_observation_for_player(self.agent_color)

    def _get_observation_for_player(self, player: str) -> np.ndarray:
        """Get observation from a specific player's perspective."""
        if self.game is None:
            return np.zeros((7, 8, 8), dtype=np.float32)

        obs = np.zeros((7, 8, 8), dtype=np.float32)

        my_pos = self.game.board.get_horse_position(player)
        opp_color = "black" if player == "white" else "white"
        opp_pos = self.game.board.get_horse_position(opp_color)

        # Channel 0: My Position
        obs[0, my_pos[0], my_pos[1]] = 1.0

        # Channel 1: Opponent Position
        obs[1, opp_pos[0], opp_pos[1]] = 1.0

        # Channel 2: Blocked Squares - OPTIMIZED using cached empty squares
        # Instead of checking all 64 squares, mark only non-empty ones
        empty_squares = self.game.board.get_empty_squares()
        obs[2, :, :] = 1.0  # Start with all blocked
        for r, c in empty_squares:
            obs[2, r, c] = 0.0  # Mark empty squares as unblocked

        # Channel 3: Mode ID (Normalized)
        mode_val = 0.0
        if self.game.board.mode == 2:
            mode_val = 0.5
        elif self.game.board.mode == 3:
            mode_val = 1.0
        obs[3, :, :] = mode_val

        # Channel 4: Current Role
        role_val = 1.0 if player == "white" else 0.0
        obs[4, :, :] = role_val

        # Channel 5: Brown Apples Remaining (Normalized)
        brown_val = self.game.board.brown_apples_remaining / 28.0
        obs[5, :, :] = brown_val

        # Channel 6: Golden Apples Remaining (Normalized)
        golden_val = self.game.board.golden_apples_remaining / 12.0
        obs[6, :, :] = golden_val

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get auxiliary info."""
        if self.game is None:
            return {}
        return {
            "agent_color": self.agent_color,
            "turn": self.game.current_player,
            "white_pos": self.game.board.white_pos,
            "black_pos": self.game.board.black_pos,
            "move_count": self.move_count,
        }

    def render(self) -> Optional[List[str]]:
        """Render the environment."""
        if self.game is None:
            return None

        # Simple ANSI render
        grid = self.game.board.grid
        output = []
        output.append(f"Agent: {self.agent_color} | Mode: {self.game.board.mode}")
        output.append("  0 1 2 3 4 5 6 7")
        for r in range(8):
            row_str = f"{r} "
            for c in range(8):
                val = grid[r, c]
                char = "."
                if val == Board.WHITE_HORSE:
                    char = "W"
                elif val == Board.BLACK_HORSE:
                    char = "B"
                elif val == Board.BROWN_APPLE:
                    char = "o"
                elif val == Board.GOLDEN_APPLE:
                    char = "G"
                row_str += f"{char} "
            output.append(row_str)

        if self.render_mode == "human":
            for line in output:
                print(line)

        return output

    def action_masks(self) -> np.ndarray:
        """Return flattened boolean action mask for MaskablePPO.

        MaskablePPO (sb3-contrib) expects a single 1D boolean array whose length
        equals the sum of the discrete action dimensions when using a
        ``MultiDiscrete`` action space. Our action space is ``MultiDiscrete([8, 65])``
        (8 knight move indices, 64 board squares + 1 "no apple" option).

        The returned array therefore has length 73 and is ordered as:
        [move_0 .. move_7, apple_0 .. apple_63, apple_no_choice]

        True = action is valid, False = action is invalid.
        """
        if self.game is None:
            # If no game, allow everything (will be reset anyway)
            return np.ones(73, dtype=bool)
        
        board = self.game.board
        current_pos = board.get_horse_position(self.agent_color)
        
        # --- Mask for knight moves (8 possible moves) ---
        move_mask = np.zeros(8, dtype=bool)
        legal_moves = Rules.get_legal_knight_moves(board, self.agent_color)
        legal_move_set = set(legal_moves)
        
        for idx, (dr, dc) in enumerate(Rules.KNIGHT_MOVES):
            target = (current_pos[0] + dr, current_pos[1] + dc)
            if target in legal_move_set:
                move_mask[idx] = True
        
        # --- Mask for apple placement (64 squares + 1 for "no apple") ---
        apple_mask = np.zeros(65, dtype=bool)
        
        # OPTIMIZATION: Use cached empty squares instead of iterating over all 64 squares
        empty_squares = board.get_empty_squares()
        
        if board.mode == 1:
            # Mode 1: Apple placement is REQUIRED BEFORE moving
            # Can place on any empty square, BUT if placing on a legal move destination,
            # we must have at least one OTHER legal move remaining (don't block ourselves)
            has_multiple_moves = len(legal_move_set) > 1
            for r, c in empty_squares:
                if (r, c) in legal_move_set:
                    # Can only place here if we have other moves
                    if has_multiple_moves:
                        apple_mask[r * 8 + c] = True
                else:
                    # Not a legal move destination - always allowed
                    apple_mask[r * 8 + c] = True
            # "No apple" (index 64) is NOT valid in mode 1
            
        elif board.mode == 2:
            # Mode 2: Apple is automatic (trail), no choice needed
            # Only "no apple" action is valid
            apple_mask[64] = True
            
        elif board.mode == 3:
            # Mode 3: Apple placement is OPTIONAL after moving
            # Restriction 1: Cannot block White's last remaining escape route
            # Restriction 2: Cannot place on move target (player will be there after move!)
            white_legal_moves = Rules.get_legal_knight_moves(board, "white")
            white_legal_set = set(white_legal_moves)
            white_only_one_move = len(white_legal_moves) == 1
            
            for r, c in empty_squares:
                # Cannot place apple on squares we're moving to
                if (r, c) in legal_move_set:
                    continue
                # Check if this placement would block White's only escape
                if white_only_one_move and (r, c) in white_legal_set:
                    continue
                apple_mask[r * 8 + c] = True
            # "No apple" is always valid in mode 3
            apple_mask[64] = True
        
        # Safety: ensure at least one action is valid per dimension
        if not move_mask.any():
            move_mask[:] = True
        if not apple_mask.any():
            apple_mask[64] = True
        
        # Concatenate into single flattened mask (length 8 + 65 = 73)
        return np.concatenate([move_mask, apple_mask])
