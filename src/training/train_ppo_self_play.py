"""
AlphaZero-style self-play training for PPO agent with Action Masking.

This implements the core AlphaZero training loop:
1. Train the current model by playing against the best model
2. Periodically evaluate current vs best
3. If current wins >= threshold, current becomes the new best
4. Repeat

Uses MaskablePPO from sb3-contrib to enforce legal moves via action masking.
This ensures the agent continuously improves against its strongest version,
rather than wasting capacity learning what moves are illegal.

Performance Optimizations:
- SubprocVecEnv for true parallel environment execution
- Parallelized evaluation using multiprocessing
- Cached empty squares in board for O(1) lookups
"""

import argparse
import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from src.env.knight_self_play_env import KnightSelfPlayEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Parallel Evaluation Helper Functions (must be at module level for pickling)
# ============================================================================

def _play_single_eval_game(
    game_idx: int,
    current_model_path: str,
    opponent_model_path: Optional[str],
    mode: str = "random",
) -> dict:
    """
    Play a single evaluation game. Used for parallel evaluation.
    
    This function is designed to be called in a separate process.
    Models are loaded from paths to avoid pickling issues.
    
    Args:
        game_idx: Index of the game (used to determine colors)
        current_model_path: Path to the current model file
        opponent_model_path: Path to the opponent model file (None for random)
        mode: Game mode ("random", 1, 2, or 3)
    
    Returns:
        dict with game result info
    """
    # Determine colors - alternate
    current_plays_white = game_idx % 2 == 0
    current_color = "white" if current_plays_white else "black"
    
    # Load models
    try:
        current_model = MaskablePPO.load(current_model_path)
    except Exception as e:
        return {"error": f"Failed to load current model: {e}"}
    
    opponent_policy = None
    if opponent_model_path is not None:
        try:
            opponent_policy = MaskablePPO.load(opponent_model_path)
        except Exception:
            # Fall back to random
            pass
    
    # Create environment
    env = KnightSelfPlayEnv(
        mode=mode,
        agent_color=current_color,
        opponent_policy=opponent_policy,
    )
    
    obs, info = env.reset()
    done = False
    game_mode = env.game.board.mode if env.game else 1
    
    while not done:
        action_masks = env.action_masks()
        action, _ = current_model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    env.close()
    
    winner = info.get("winner")
    error = info.get("error")
    
    return {
        "game_idx": game_idx,
        "current_color": current_color,
        "winner": winner,
        "error": error,
        "mode": game_mode,
    }


def _play_eval_game_vs_random(
    game_idx: int,
    model_path: str,
    mode: int,
    color: str,
) -> dict:
    """
    Play a single game against random opponent. Used for parallel evaluation.
    
    Args:
        game_idx: Index of the game
        model_path: Path to the model file
        mode: Game mode (1, 2, or 3)
        color: Color to play as ("white" or "black")
    
    Returns:
        dict with game result info
    """
    try:
        model = MaskablePPO.load(model_path)
    except Exception as e:
        return {"error": f"Failed to load model: {e}"}
    
    env = KnightSelfPlayEnv(
        mode=mode,
        agent_color=color,
        opponent_policy=None,  # Random opponent
    )
    
    obs, info = env.reset()
    done = False
    
    while not done:
        action_masks = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    env.close()
    
    return {
        "game_idx": game_idx,
        "mode": mode,
        "color": color,
        "winner": info.get("winner"),
        "won": info.get("winner") == color,
    }


class SelfPlayCallback(BaseCallback):
    """
    Callback for AlphaZero-style self-play training.
    
    Periodically evaluates the current model against the best model.
    If the current model wins >= win_rate_threshold, it becomes the new best.
    
    Enhanced evaluation criteria:
    1. Must beat the best model overall (standard self-play)
    2. Must beat Random from BOTH colors (detects color bias)
    3. Minimum per-color win rate threshold (prevents weak color exploitation)
    4. Color bias tracker: Stops if consistently weak on one color
    
    Performance optimizations:
    - Parallel evaluation using ProcessPoolExecutor
    - Configurable number of worker processes
    """

    def __init__(
        self,
        eval_freq: int = 50_000,
        n_eval_games: int = 100,
        win_rate_threshold: float = 0.55,
        min_per_color_wr: float = 0.50,
        best_model_path: Path = Path("data/models/ppo_self_play/best_model"),
        n_eval_workers: int = 16,
        verbose: int = 1,
    ):
        """
        Args:
            eval_freq: Evaluate every N timesteps
            n_eval_games: Number of games to play for evaluation
            win_rate_threshold: Win rate needed to become new best (0.55 = 55%)
            min_per_color_wr: Minimum win rate from each color vs Random (0.50 = 50%)
            best_model_path: Path to save the best model
            n_eval_workers: Number of parallel workers for evaluation (0=sequential)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_games = n_eval_games
        self.win_rate_threshold = win_rate_threshold
        self.min_per_color_wr = min_per_color_wr
        self.best_model_path = Path(best_model_path)
        self.n_eval_workers = n_eval_workers
        self.last_eval_timestep = 0
        
        # Temp path for saving current model during evaluation
        self._temp_model_path = Path(best_model_path).parent / "_temp_eval_model"
        
        # Statistics
        self.generation = 0
        self.eval_results: list[dict] = []
        
        # Color bias tracking
        # Counts consecutive evaluations where one color performs poorly
        self.consecutive_white_weak = 0
        self.consecutive_black_weak = 0
        self.color_bias_threshold = 10  # Stop if weak on one color 10 times in a row
        self.color_bias_detected = False

    def _on_step(self) -> bool:
        """Called after each step."""
        # Check if we've passed the next evaluation threshold
        if self.num_timesteps - self.last_eval_timestep >= self.eval_freq:
            self.last_eval_timestep = self.num_timesteps
            self._evaluate_and_maybe_update()
        
        # Check for color bias issue
        if self.color_bias_detected:
            logger.warning("\n" + "!"*60)
            logger.warning("COLOR BIAS DETECTED - Training may be stuck!")
            logger.warning(f"Weak on white {self.consecutive_white_weak}x, black {self.consecutive_black_weak}x in a row")
            logger.warning("Consider debugging or adjusting training parameters.")
            logger.warning("!"*60 + "\n")
            # Don't stop training, but log the issue
            self.color_bias_detected = False  # Reset flag after logging
        
        return True

    def _evaluate_and_maybe_update(self) -> None:
        """Evaluate current model vs best and update if better.
        
        IMPORTANT: The model must perform well from BOTH white and black perspectives
        to become the new best. This prevents learning color-biased strategies.
        
        We use TWO evaluation criteria:
        1. Must beat the best model overall (standard self-play)
        2. Must beat Random from BOTH colors (detects color bias that self-play hides)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Generation {self.generation}: Evaluating current vs best model...")
        logger.info(f"{'='*60}")
        
        # Load best model for comparison (if exists)
        best_model_file = self.best_model_path / "best_model.zip"
        
        if not best_model_file.exists():
            # No best model yet - current becomes best
            logger.info("No best model exists yet. Current model becomes the first best!")
            self._save_as_best()
            self.generation += 1
            return
        
        # Try to load best model - MaskablePPO first, then fallback to PPO
        best_model = None
        try:
            best_model = MaskablePPO.load(str(best_model_file))
            logger.info("Loaded MaskablePPO best model for evaluation")
        except Exception as e_mask:
            # Fallback to regular PPO (for backwards compatibility with old models)
            try:
                from stable_baselines3 import PPO
                best_model = PPO.load(str(best_model_file))
                logger.info("Loaded non-maskable PPO best model for evaluation (legacy)")
            except Exception as e_ppo:
                logger.warning(f"Failed to load best model (maskable: {e_mask}; ppo: {e_ppo}). Current becomes best.")
                self._save_as_best()
                self.generation += 1
                return
        
        # EVALUATION 1: Current vs Best (self-play style)
        wins, losses, draws, details = self._play_evaluation_games(
            current_model=self.model,
            opponent_model=best_model,
            n_games=self.n_eval_games,
        )
        
        total_games = wins + losses + draws
        win_rate = wins / total_games if total_games > 0 else 0.0
        
        logger.info(f"Results vs Best: Wins={wins}, Losses={losses}, Draws={draws}")
        logger.info(f"Overall Win Rate: {win_rate:.1%} (threshold: {self.win_rate_threshold:.1%})")
        
        # Log per-mode-per-color stats vs Best
        logger.info("Per-color breakdown vs Best:")
        for color in ["white", "black"]:
            games = details.get(f"{color}_games", 0)
            color_wins = details.get(f"{color}_wins", 0)
            if games > 0:
                logger.info(f"  As {color.capitalize()}: {color_wins}/{games} ({color_wins/games:.1%})")
        
        # EVALUATION 2: Current vs Random (color bias detection)
        # This is critical because self-play hides color bias!
        # Enhanced: Test each mode × color combination separately (6 combinations)
        logger.info("\n" + "-"*50)
        logger.info("Evaluating vs Random (per-mode, per-color)...")
        random_results = self._evaluate_vs_random_detailed(
            model=self.model,
            n_games_per_combo=50,  # 50 games per mode × color = 300 total
        )
        
        # Aggregate results
        random_white_total = sum(random_results["per_mode"][m]["white_wins"] for m in [1, 2, 3])
        random_black_total = sum(random_results["per_mode"][m]["black_wins"] for m in [1, 2, 3])
        random_white_games = sum(random_results["per_mode"][m]["white_games"] for m in [1, 2, 3])
        random_black_games = sum(random_results["per_mode"][m]["black_games"] for m in [1, 2, 3])
        
        random_white_wr = random_white_total / random_white_games if random_white_games > 0 else 0.0
        random_black_wr = random_black_total / random_black_games if random_black_games > 0 else 0.0
        random_overall_wr = (random_white_total + random_black_total) / (random_white_games + random_black_games) if (random_white_games + random_black_games) > 0 else 0.0
        
        logger.info(f"\nVs Random Summary:")
        logger.info(f"  Overall: {random_overall_wr:.1%} (need ≥{self.win_rate_threshold:.1%})")
        logger.info(f"  As White: {random_white_wr:.1%} ({random_white_total}/{random_white_games})")
        logger.info(f"  As Black: {random_black_wr:.1%} ({random_black_total}/{random_black_games})")
        
        # Log per-mode breakdown
        logger.info("\nPer-mode breakdown vs Random:")
        mode_color_wr = {}  # Store for later analysis
        for mode in [1, 2, 3]:
            mode_data = random_results["per_mode"][mode]
            for color in ["white", "black"]:
                wins = mode_data[f"{color}_wins"]
                games = mode_data[f"{color}_games"]
                wr = wins / games if games > 0 else 0.0
                mode_color_wr[(mode, color)] = wr
                logger.info(f"  Mode {mode}, {color.capitalize()}: {wr:.1%} ({wins}/{games})")
        
        # Calculate per-color win rates vs Best
        white_vs_best_games = details.get("white_games", 0)
        white_vs_best_wins = details.get("white_wins", 0)
        black_vs_best_games = details.get("black_games", 0)
        black_vs_best_wins = details.get("black_wins", 0)
        
        white_vs_best_wr = white_vs_best_wins / white_vs_best_games if white_vs_best_games > 0 else 0.0
        black_vs_best_wr = black_vs_best_wins / black_vs_best_games if black_vs_best_games > 0 else 0.0
        
        # Store detailed results
        eval_result = {
            "generation": self.generation,
            "timesteps": self.num_timesteps,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "white_vs_best_wr": white_vs_best_wr,
            "black_vs_best_wr": black_vs_best_wr,
            "white_win_rate": random_white_wr,
            "black_win_rate": random_black_wr,
            "random_overall_wr": random_overall_wr,
            "became_best": False,
            "mode_color_stats": mode_color_wr,
        }
        self.eval_results.append(eval_result)
        
        # ============================================================
        # STRICT PROMOTION CRITERIA (prevents color bias)
        # ============================================================
        # A model must be good at BOTH colors to be promoted!
        # 
        # VS BEST MODEL:
        #   1. Overall win rate >= threshold (55%)
        #   2. White win rate >= threshold (55%) - NEW!
        #   3. Black win rate >= threshold (55%) - NEW!
        #
        # VS RANDOM:
        #   4. Overall win rate >= threshold (55%)
        #   5. White win rate >= per-color threshold (50%) - RAISED from 30%!
        #   6. Black win rate >= per-color threshold (50%) - RAISED from 30%!
        #
        # This prevents the "great at black, terrible at white" collapse.
        # ============================================================
        
        # Vs Best - require EACH color to meet threshold
        is_good_vs_best_overall = win_rate >= self.win_rate_threshold
        is_good_vs_best_white = white_vs_best_wr >= self.win_rate_threshold
        is_good_vs_best_black = black_vs_best_wr >= self.win_rate_threshold
        is_good_vs_best = is_good_vs_best_overall and is_good_vs_best_white and is_good_vs_best_black
        
        # Vs Random - require EACH color to meet per-color threshold
        is_good_vs_random_overall = random_overall_wr >= self.win_rate_threshold
        white_ok = random_white_wr >= self.min_per_color_wr
        black_ok = random_black_wr >= self.min_per_color_wr
        is_good_vs_random = is_good_vs_random_overall and white_ok and black_ok
        
        # Track color bias
        self._update_color_bias_tracking(random_white_wr, random_black_wr)
        
        # Log decision breakdown
        logger.info(f"\nPromotion Criteria Check:")
        logger.info(f"  vs Best Overall: {win_rate:.1%} >= {self.win_rate_threshold:.1%}? {'✓' if is_good_vs_best_overall else '✗'}")
        logger.info(f"  vs Best White:   {white_vs_best_wr:.1%} >= {self.win_rate_threshold:.1%}? {'✓' if is_good_vs_best_white else '✗'}")
        logger.info(f"  vs Best Black:   {black_vs_best_wr:.1%} >= {self.win_rate_threshold:.1%}? {'✓' if is_good_vs_best_black else '✗'}")
        logger.info(f"  vs Random Overall: {random_overall_wr:.1%} >= {self.win_rate_threshold:.1%}? {'✓' if is_good_vs_random_overall else '✗'}")
        logger.info(f"  vs Random White:   {random_white_wr:.1%} >= {self.min_per_color_wr:.1%}? {'✓' if white_ok else '✗'}")
        logger.info(f"  vs Random Black:   {random_black_wr:.1%} >= {self.min_per_color_wr:.1%}? {'✓' if black_ok else '✗'}")
        
        if is_good_vs_best and is_good_vs_random:
            logger.info(f"\n🎉 New best model! Generation {self.generation} -> {self.generation + 1}")
            logger.info(f"   vs Best: {win_rate:.1%} (W:{white_vs_best_wr:.1%}, B:{black_vs_best_wr:.1%})")
            logger.info(f"   vs Random: {random_overall_wr:.1%} (W:{random_white_wr:.1%}, B:{random_black_wr:.1%})")
            self._save_as_best()
            eval_result["became_best"] = True
            self.generation += 1
            
            # Reset bias counters on success
            self.consecutive_white_weak = 0
            self.consecutive_black_weak = 0
            
            # Update the opponent policy in all environments
            self._update_opponent_in_envs()
        else:
            reasons = []
            if not is_good_vs_best_overall:
                reasons.append(f"vs Best Overall: {win_rate:.1%} < {self.win_rate_threshold:.1%}")
            if not is_good_vs_best_white:
                reasons.append(f"vs Best White: {white_vs_best_wr:.1%} < {self.win_rate_threshold:.1%}")
            if not is_good_vs_best_black:
                reasons.append(f"vs Best Black: {black_vs_best_wr:.1%} < {self.win_rate_threshold:.1%}")
            if not is_good_vs_random_overall:
                reasons.append(f"vs Random Overall: {random_overall_wr:.1%} < {self.win_rate_threshold:.1%}")
            if not white_ok:
                reasons.append(f"vs Random White: {random_white_wr:.1%} < {self.min_per_color_wr:.1%}")
            if not black_ok:
                reasons.append(f"vs Random Black: {random_black_wr:.1%} < {self.min_per_color_wr:.1%}")
            
            logger.info(f"\n❌ Model not promoted. Failed criteria:")
            for reason in reasons:
                logger.info(f"   - {reason}")
    
    def _update_color_bias_tracking(self, white_wr: float, black_wr: float) -> None:
        """Track consecutive weak performance on each color."""
        weak_threshold = self.min_per_color_wr
        
        if white_wr < weak_threshold:
            self.consecutive_white_weak += 1
        else:
            self.consecutive_white_weak = 0
        
        if black_wr < weak_threshold:
            self.consecutive_black_weak += 1
        else:
            self.consecutive_black_weak = 0
        
        # Check if we've hit the bias threshold
        if self.consecutive_white_weak >= self.color_bias_threshold:
            self.color_bias_detected = True
            logger.warning(f"⚠️ WHITE COLOR BIAS: Weak performance {self.consecutive_white_weak} times in a row!")
        if self.consecutive_black_weak >= self.color_bias_threshold:
            self.color_bias_detected = True
            logger.warning(f"⚠️ BLACK COLOR BIAS: Weak performance {self.consecutive_black_weak} times in a row!")
    
    def _evaluate_vs_random_detailed(
        self,
        model: MaskablePPO,
        n_games_per_combo: int,
    ) -> dict:
        """
        Evaluate model against Random opponent with detailed per-mode-per-color breakdown.
        
        Tests all 6 combinations: 3 modes × 2 colors.
        Uses parallel evaluation when n_eval_workers > 0.
        
        Args:
            model: The model to evaluate
            n_games_per_combo: Games per (mode, color) combination
            
        Returns:
            Dictionary with detailed statistics
        """
        results = {
            "per_mode": {
                1: {"white_wins": 0, "white_games": 0, "black_wins": 0, "black_games": 0},
                2: {"white_wins": 0, "white_games": 0, "black_wins": 0, "black_games": 0},
                3: {"white_wins": 0, "white_games": 0, "black_wins": 0, "black_games": 0},
            },
        }
        
        if self.n_eval_workers > 0:
            # PARALLEL EVALUATION
            # Save model to temp file for worker processes
            model.save(str(self._temp_model_path))
            temp_model_path = str(self._temp_model_path) + ".zip"
            
            # Create all game tasks
            tasks = []
            game_idx = 0
            for mode in [1, 2, 3]:
                for color in ["white", "black"]:
                    for _ in range(n_games_per_combo):
                        tasks.append((game_idx, temp_model_path, mode, color))
                        game_idx += 1
            
            # Run in parallel
            with ProcessPoolExecutor(max_workers=self.n_eval_workers) as executor:
                futures = [
                    executor.submit(_play_eval_game_vs_random, *task)
                    for task in tasks
                ]
                
                for future in as_completed(futures):
                    result = future.result()
                    if "error" not in result:
                        mode = result["mode"]
                        color = result["color"]
                        results["per_mode"][mode][f"{color}_games"] += 1
                        if result["won"]:
                            results["per_mode"][mode][f"{color}_wins"] += 1
        else:
            # SEQUENTIAL EVALUATION (fallback)
            for mode in [1, 2, 3]:
                for color in ["white", "black"]:
                    wins = 0
                    for _ in range(n_games_per_combo):
                        env = KnightSelfPlayEnv(
                            mode=mode,
                            agent_color=color,
                            opponent_policy=None,  # Random opponent
                        )
                        
                        obs, info = env.reset()
                        done = False
                        
                        while not done:
                            action_masks = env.action_masks()
                            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
                            obs, reward, terminated, truncated, info = env.step(action)
                            done = terminated or truncated
                        
                        if info.get("winner") == color:
                            wins += 1
                        
                        env.close()
                    
                    results["per_mode"][mode][f"{color}_wins"] = wins
                    results["per_mode"][mode][f"{color}_games"] = n_games_per_combo
        
        return results
    
    def _evaluate_vs_random(
        self,
        model: MaskablePPO,
        n_games_per_color: int,
    ) -> Tuple[int, int, dict]:
        """
        Evaluate model against Random opponent.
        
        This is critical for detecting color bias that self-play hides.
        When a model plays itself, the win rate per color depends on the strategy,
        but against Random, a truly good model should win from both colors.
        
        Args:
            model: The model to evaluate
            n_games_per_color: Games to play as each color
            
        Returns:
            (wins_as_white, wins_as_black, details)
        """
        wins_white = 0
        wins_black = 0
        
        details = {
            "mode_stats": {1: {"wins": 0, "total": 0}, 2: {"wins": 0, "total": 0}, 3: {"wins": 0, "total": 0}},
        }
        
        # Test as White
        for _ in range(n_games_per_color):
            env = KnightSelfPlayEnv(
                mode="random",
                agent_color="white",
                opponent_policy=None,  # None = Random opponent
            )
            
            obs, info = env.reset()
            done = False
            game_mode = env.game.board.mode if env.game else 1
            
            while not done:
                action_masks = env.action_masks()
                action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            details["mode_stats"][game_mode]["total"] += 1
            if info.get("winner") == "white":
                wins_white += 1
                details["mode_stats"][game_mode]["wins"] += 1
            
            env.close()
        
        # Test as Black
        for _ in range(n_games_per_color):
            env = KnightSelfPlayEnv(
                mode="random",
                agent_color="black",
                opponent_policy=None,  # None = Random opponent
            )
            
            obs, info = env.reset()
            done = False
            game_mode = env.game.board.mode if env.game else 1
            
            while not done:
                action_masks = env.action_masks()
                action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            details["mode_stats"][game_mode]["total"] += 1
            if info.get("winner") == "black":
                wins_black += 1
                details["mode_stats"][game_mode]["wins"] += 1
            
            env.close()
        
        return wins_white, wins_black, details
    
    def _play_evaluation_games(
        self,
        current_model: MaskablePPO,
        opponent_model: Any,  # Can be MaskablePPO or PPO
        n_games: int,
    ) -> Tuple[int, int, int, dict]:
        """
        Play n_games between current and opponent model.
        
        Games are split evenly: half with current as white, half as black.
        Uses parallel evaluation when n_eval_workers > 0.
        
        Args:
            current_model: The current MaskablePPO model being trained
            opponent_model: The opponent model (can be MaskablePPO or regular PPO)
            n_games: Number of games to play (should be even)
        
        Returns: (wins, losses, draws, details) from current model's perspective
                 details contains per-color and per-mode statistics
        """
        wins = 0
        losses = 0
        draws = 0
        
        # Track detailed stats for analysis
        details = {
            "white_wins": 0, "white_losses": 0, "white_draws": 0, "white_games": 0,
            "black_wins": 0, "black_losses": 0, "black_draws": 0, "black_games": 0,
            "mode_stats": {1: {"wins": 0, "total": 0}, 2: {"wins": 0, "total": 0}, 3: {"wins": 0, "total": 0}},
        }
        
        if self.n_eval_workers > 0:
            # PARALLEL EVALUATION
            # Save models to temp files for worker processes
            current_model.save(str(self._temp_model_path))
            current_model_path = str(self._temp_model_path) + ".zip"
            
            best_model_file = self.best_model_path / "best_model.zip"
            opponent_model_path = str(best_model_file) if best_model_file.exists() else None
            
            # Run games in parallel
            with ProcessPoolExecutor(max_workers=self.n_eval_workers) as executor:
                futures = [
                    executor.submit(
                        _play_single_eval_game,
                        game_idx,
                        current_model_path,
                        opponent_model_path,
                        "random",
                    )
                    for game_idx in range(n_games)
                ]
                
                for future in as_completed(futures):
                    result = future.result()
                    if "error" in result and result.get("winner") is None:
                        losses += 1
                        continue
                    
                    current_color = result["current_color"]
                    winner = result["winner"]
                    game_mode = result.get("mode", 1)
                    error = result.get("error")
                    
                    # Track per-color stats
                    if current_color == "white":
                        details["white_games"] += 1
                    else:
                        details["black_games"] += 1
                    
                    details["mode_stats"][game_mode]["total"] += 1
                    
                    if error:
                        losses += 1
                        if current_color == "white":
                            details["white_losses"] += 1
                        else:
                            details["black_losses"] += 1
                    elif winner == current_color:
                        wins += 1
                        details["mode_stats"][game_mode]["wins"] += 1
                        if current_color == "white":
                            details["white_wins"] += 1
                        else:
                            details["black_wins"] += 1
                    elif winner == "draw":
                        draws += 1
                        if current_color == "white":
                            details["white_draws"] += 1
                        else:
                            details["black_draws"] += 1
                    else:
                        losses += 1
                        if current_color == "white":
                            details["white_losses"] += 1
                        else:
                            details["black_losses"] += 1
        else:
            # SEQUENTIAL EVALUATION (fallback)
            for game_idx in range(n_games):
                # Alternate who plays white/black - ensures equal exposure
                current_plays_white = game_idx % 2 == 0
                current_color = "white" if current_plays_white else "black"
                
                # Create evaluation environment with random mode
                env = KnightSelfPlayEnv(
                    mode="random",
                    agent_color=current_color,
                    opponent_policy=opponent_model,
                )
                
                obs, info = env.reset()
                done = False
                game_mode = env.game.board.mode if env.game else 1
                
                while not done:
                    # Current model makes a move with action masking
                    action_masks = env.action_masks()
                    action, _ = current_model.predict(obs, deterministic=True, action_masks=action_masks)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                
                # Determine outcome
                winner = info.get("winner")
                
                # Track per-color stats
                if current_plays_white:
                    details["white_games"] += 1
                else:
                    details["black_games"] += 1
                
                # Track mode stats
                details["mode_stats"][game_mode]["total"] += 1
                
                # Check if there was an error (invalid move) - this counts as a loss
                if "error" in info:
                    losses += 1
                    if current_plays_white:
                        details["white_losses"] += 1
                    else:
                        details["black_losses"] += 1
                elif winner == current_color:
                    wins += 1
                    details["mode_stats"][game_mode]["wins"] += 1
                    if current_plays_white:
                        details["white_wins"] += 1
                    else:
                        details["black_wins"] += 1
                elif winner == "draw":
                    draws += 1
                    if current_plays_white:
                        details["white_draws"] += 1
                    else:
                        details["black_draws"] += 1
                elif winner is None:
                    # No winner and no error - shouldn't happen, but treat as draw
                    draws += 1
                    if current_plays_white:
                        details["white_draws"] += 1
                    else:
                        details["black_draws"] += 1
                else:
                    losses += 1
                    if current_plays_white:
                        details["white_losses"] += 1
                    else:
                        details["black_losses"] += 1
                
                env.close()
        
        return wins, losses, draws, details
    
    def _save_as_best(self) -> None:
        """Save current model as the new best."""
        self.best_model_path.mkdir(parents=True, exist_ok=True)
        save_path = self.best_model_path / "best_model"
        self.model.save(str(save_path))
        
        # Also save generation info
        info_path = self.best_model_path / "info.txt"
        with open(info_path, "w") as f:
            f.write(f"Generation: {self.generation}\n")
            f.write(f"Timesteps: {self.num_timesteps}\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
        
        logger.info(f"Saved best model to {save_path}")
    
    def _update_opponent_in_envs(self) -> None:
        """Update the opponent policy in all training environments."""
        if not hasattr(self.training_env, 'envs'):
            return
        
        # Load the new best model for opponent
        best_model_file = self.best_model_path / "best_model.zip"
        if best_model_file.exists():
            try:
                new_opponent = MaskablePPO.load(str(best_model_file))
                # Update each environment's opponent
                for env in self.training_env.envs:
                    # Unwrap to get the actual KnightSelfPlayEnv
                    base_env = env
                    while hasattr(base_env, 'env'):
                        base_env = base_env.env
                    if hasattr(base_env, 'set_opponent_policy'):
                        base_env.set_opponent_policy(new_opponent)
                
                logger.info("Updated opponent policy in all training environments")
            except Exception as e:
                logger.warning(f"Failed to update opponent policy: {e}")


class IncrementalCSVLogger(BaseCallback):
    """
    Callback to save training history incrementally after each evaluation.
    
    This ensures training data is preserved even if training is interrupted,
    and enables real-time plotting of progress.
    """
    
    def __init__(
        self,
        self_play_callback: "SelfPlayCallback",
        log_path: Path,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.self_play_callback = self_play_callback
        self.log_path = Path(log_path)
        self.last_saved_count = 0
    
    def _on_step(self) -> bool:
        """Check if there are new evaluation results to save."""
        results = self.self_play_callback.eval_results
        if len(results) > self.last_saved_count:
            self._save_history()
            self.last_saved_count = len(results)
        return True
    
    def _save_history(self) -> None:
        """Save all evaluation results to CSV."""
        results = self.self_play_callback.eval_results
        if not results:
            return
        
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            f.write("generation,timesteps,wins,losses,draws,win_rate,white_vs_best,black_vs_best,white_vs_random,black_vs_random,random_overall_wr,became_best\n")
            for r in results:
                white_vs_best = r.get('white_vs_best_wr', 0.0)
                black_vs_best = r.get('black_vs_best_wr', 0.0)
                white_vs_random = r.get('white_win_rate', 0.0)
                black_vs_random = r.get('black_win_rate', 0.0)
                random_overall = r.get('random_overall_wr', 0.0)
                f.write(f"{r['generation']},{r['timesteps']},{r['wins']},{r['losses']},{r['draws']},{r['win_rate']:.4f},{white_vs_best:.4f},{black_vs_best:.4f},{white_vs_random:.4f},{black_vs_random:.4f},{random_overall:.4f},{r['became_best']}\n")
        
        if self.verbose:
            logger.debug(f"Saved {len(results)} evaluation records to {self.log_path}")


class OpponentUpdateCallback(BaseCallback):
    """
    Periodically update the opponent from the saved best model.
    
    This is useful when using multiple parallel environments,
    as the SelfPlayCallback only updates after evaluation.
    """
    
    def __init__(
        self,
        update_freq: int = 100_000,
        best_model_path: Path = Path("data/models/ppo_self_play/best_model"),
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.update_freq = update_freq
        self.best_model_path = Path(best_model_path)
        self.last_update_timestep = 0
        self.current_opponent_gen = -1
    
    def _on_step(self) -> bool:
        # Check if we've passed the next update threshold
        if self.num_timesteps - self.last_update_timestep >= self.update_freq:
            self.last_update_timestep = self.num_timesteps
            self._maybe_update_opponent()
        
        return True
    
    def _maybe_update_opponent(self) -> None:
        """Check if there's a newer best model and update opponent if so."""
        info_path = self.best_model_path / "info.txt"
        if not info_path.exists():
            return
        
        # Read current generation
        try:
            with open(info_path, "r") as f:
                for line in f:
                    if line.startswith("Generation:"):
                        gen = int(line.split(":")[1].strip())
                        if gen > self.current_opponent_gen:
                            self._update_opponent()
                            self.current_opponent_gen = gen
                        break
        except Exception as e:
            logger.warning(f"Failed to check for opponent update: {e}")
    
    def _update_opponent(self) -> None:
        """Load and set the new opponent model."""
        best_model_file = self.best_model_path / "best_model.zip"
        if not best_model_file.exists():
            return
        
        try:
            # Attempt MaskablePPO first, fallback to PPO
            new_opponent = None
            try:
                new_opponent = MaskablePPO.load(str(best_model_file))
                if self.verbose:
                    logger.info("Opponent updated with MaskablePPO model")
            except Exception as e_mask:
                try:
                    from stable_baselines3 import PPO  # type: ignore
                    new_opponent = PPO.load(str(best_model_file))
                    if self.verbose:
                        logger.info("Opponent updated with non-maskable PPO model")
                except Exception as e_ppo:
                    raise RuntimeError(f"Failed to load opponent model (maskable: {e_mask}; ppo: {e_ppo})") from e_ppo
            
            if hasattr(self.training_env, 'envs'):
                for env in self.training_env.envs:
                    base_env = env
                    while hasattr(base_env, 'env'):
                        base_env = base_env.env
                    if hasattr(base_env, 'set_opponent_policy'):
                        base_env.set_opponent_policy(new_opponent)
                
                if self.verbose:
                    logger.info(f"Synced opponent to generation {self.current_opponent_gen + 1}")
        except Exception as e:
            logger.warning(f"Failed to update opponent: {e}")


def mask_fn(env: gym.Env) -> np.ndarray:
    """Get action masks from the environment for MaskablePPO."""
    # Unwrap to get the actual KnightSelfPlayEnv
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    return base_env.action_masks()


def create_self_play_env(
    best_model_path: Optional[Path] = None,
    mode: str = "random",
) -> gym.Env:
    """Create a self-play environment with the best model as opponent and action masking."""
    opponent_policy = None
    
    if best_model_path is not None:
        model_file = best_model_path / "best_model.zip"
        if model_file.exists():
            try:
                # Try loading as MaskablePPO first
                opponent_policy = MaskablePPO.load(str(model_file))
                logger.info(f"Loaded maskable opponent from {model_file}")
            except Exception as e_mask:
                # Fallback to regular PPO
                try:
                    from stable_baselines3 import PPO  # type: ignore
                    opponent_policy = PPO.load(str(model_file))
                    logger.info(f"Loaded non-maskable PPO opponent from {model_file}")
                except Exception as e_ppo:
                    logger.warning(f"Failed to load opponent model (maskable: {e_mask}; ppo: {e_ppo}). Using random.")
    
    env = KnightSelfPlayEnv(
        mode=mode,
        agent_color="random",
        opponent_policy=opponent_policy,
    )
    # Wrap with ActionMasker for MaskablePPO compatibility
    env = ActionMasker(env, mask_fn)
    return Monitor(env)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AlphaZero-style self-play training for PPO agent."
    )
    parser.add_argument(
        "--continue",
        dest="continue_training",
        action="store_true",
        help="Continue training from the latest model.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh training, ignoring any existing models.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=16,
        help="Number of parallel environments (default: 16). Profiling shows 16 is optimal; 64+ causes slowdown due to opponent model inference overhead.",
    )
    parser.add_argument(
        "--use-subproc",
        action="store_true",
        help="Use SubprocVecEnv for true parallel env execution. WARNING: With model opponents, DummyVecEnv is often faster due to model loading overhead per subprocess.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50_000_000,
        help="Total timesteps to train (default: 5M).",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=100_000,
        help="Evaluate current vs best every N timesteps (default: 100K).",
    )
    parser.add_argument(
        "--n-eval-games",
        type=int,
        default=1000,
        help="Number of games to play for evaluation (default: 1000).",
    )
    parser.add_argument(
        "--n-eval-workers",
        type=int,
        default=16,
        help="Number of parallel workers for evaluation (default: 16, 0=sequential). Higher values speed up the 1300-game evaluation phase.",
    )
    parser.add_argument(
        "--win-threshold",
        type=float,
        default=0.55,
        help="Win rate threshold to become new best (default: 0.55).",
    )
    parser.add_argument(
        "--min-per-color",
        type=float,
        default=0.50,
        help="Minimum win rate vs Random required from each color (default: 0.50 = 50%%).",
    )
    parser.add_argument(
        "--from-pretrained",
        type=str,
        default=None,
        help="Path to a pretrained model to start from (e.g., from random training).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Train against random opponent for N steps before starting self-play. "
             "Recommended: 15_000_000 for AlphaZero-style curriculum. 0 = no warmup.",
    )
    parser.add_argument(
        "--warmup-n-envs",
        type=int,
        default=64,
        help="Number of envs during warmup phase (default: 64). "
             "Warmup can use more envs since there's no model inference overhead.",
    )
    return parser.parse_args()


def main() -> None:
    """Main training loop."""
    args = parse_args()
    
    # Directories
    models_dir = Path("data/models/ppo_self_play")
    best_model_dir = models_dir / "best_model"
    logs_dir = Path("data/logs/ppo_self_play")
    
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # If --fresh, clear the best model directory to start completely fresh
    if args.fresh:
        best_model_file = best_model_dir / "best_model.zip"
        if best_model_file.exists():
            logger.info(f"--fresh: Removing old best model {best_model_file}")
            best_model_file.unlink()
        info_file = best_model_dir / "info.txt"
        if info_file.exists():
            info_file.unlink()
    
    # Create environments
    vec_env_cls = SubprocVecEnv if args.use_subproc else DummyVecEnv
    logger.info(f"Creating {args.n_envs} parallel self-play environments using {vec_env_cls.__name__}...")
    
    def make_env():
        return create_self_play_env(best_model_path=best_model_dir, mode="random")
    
    # Note: make_vec_env uses DummyVecEnv by default
    # For SubprocVecEnv, we need to create it directly
    if args.use_subproc:
        env = SubprocVecEnv([make_env for _ in range(args.n_envs)])
    else:
        env = make_vec_env(make_env, n_envs=args.n_envs)
    
    # Initialize or load model
    model = None
    reset_num_timesteps = True
    
    if args.from_pretrained and not args.fresh:
        # Start from a pretrained model (e.g., one trained against random)
        pretrained_path = Path(args.from_pretrained)
        if pretrained_path.exists():
            logger.info(f"Loading pretrained model from {pretrained_path}")
            model = MaskablePPO.load(str(pretrained_path), env=env)
            
            # Also set this as the initial "best" if no best exists
            if not (best_model_dir / "best_model.zip").exists():
                model.save(str(best_model_dir / "best_model"))
                with open(best_model_dir / "info.txt", "w") as f:
                    f.write("Generation: 0\n")
                    f.write("Timesteps: 0\n")
                    f.write(f"Date: {datetime.now().isoformat()}\n")
                    f.write("Note: Initialized from pretrained model\n")
                logger.info("Set pretrained model as initial best")
        else:
            logger.warning(f"Pretrained model not found at {pretrained_path}")
    
    # Auto-resume: Try to load the most recent model if available (unless --fresh)
    if model is None and not args.fresh:
        # Look for models in order of preference:
        # 1. Most recent final model (ppo_selfplay_final_*)
        # 2. Most recent checkpoint (ppo_selfplay_*)
        # 3. Best model (if user wants to continue from best)
        
        final_models = list(models_dir.glob("ppo_selfplay_final_*.zip"))
        checkpoint_models = list(models_dir.glob("ppo_selfplay_[0-9]*.zip"))  # Numeric checkpoints only
        
        candidate_models = final_models + checkpoint_models
        
        if candidate_models:
            latest = max(candidate_models, key=os.path.getctime)
            logger.info(f"Auto-resuming from most recent model: {latest}")
            model = MaskablePPO.load(str(latest), env=env)
            reset_num_timesteps = False if args.continue_training else True
        elif (best_model_dir / "best_model.zip").exists() and args.continue_training:
            # If no training checkpoints but best model exists and user wants to continue
            logger.info(f"Loading from best model: {best_model_dir / 'best_model.zip'}")
            model = MaskablePPO.load(str(best_model_dir / "best_model.zip"), env=env)
            reset_num_timesteps = False
    
    if model is None:
        logger.info("Initializing new MaskablePPO model with action masking...")
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(logs_dir),
            learning_rate=3e-4,
            n_steps=128,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )
        
        # Save initial model as first best (if no best exists)
        if not (best_model_dir / "best_model.zip").exists():
            model.save(str(best_model_dir / "best_model"))
            with open(best_model_dir / "info.txt", "w") as f:
                f.write("Generation: 0\n")
                f.write("Timesteps: 0\n")
                f.write(f"Date: {datetime.now().isoformat()}\n")
            logger.info("Saved initial model as generation 0 best")
    
    # ========================================================================
    # WARMUP PHASE: Train against random before self-play (AlphaZero-style)
    # ========================================================================
    if args.warmup_steps > 0 and not args.continue_training:
        warmup_envs = args.warmup_n_envs
        logger.info(f"""
{'='*60}
WARMUP PHASE: Training vs Random Opponent
{'='*60}
Warmup Steps: {args.warmup_steps:,}
Warmup Envs: {warmup_envs} (more envs OK - no model inference overhead)
This builds a strong foundation before self-play begins.
{'='*60}
        """)
        
        # Create warmup environments (no opponent model = random)
        def make_warmup_env():
            warmup_env = KnightSelfPlayEnv(
                mode="random",
                agent_color="random",
                opponent_policy=None,  # Random opponent
            )
            warmup_env = ActionMasker(warmup_env, mask_fn)
            return Monitor(warmup_env)
        
        if args.use_subproc:
            warmup_vec_env = SubprocVecEnv([make_warmup_env for _ in range(warmup_envs)])
        else:
            warmup_vec_env = make_vec_env(make_warmup_env, n_envs=warmup_envs)
        
        # Set the warmup environment
        model.set_env(warmup_vec_env)
        
        # Train against random
        model.learn(
            total_timesteps=args.warmup_steps,
            progress_bar=True,
            reset_num_timesteps=True,
        )
        
        # Save warmup model as best for self-play phase
        model.save(str(best_model_dir / "best_model"))
        with open(best_model_dir / "info.txt", "w") as f:
            f.write("Generation: 0\n")
            f.write(f"Timesteps: {args.warmup_steps}\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write("Note: Warmup model trained vs Random\n")
        logger.info(f"Warmup complete! Saved warmup model as initial best.")
        
        # Clean up warmup env
        warmup_vec_env.close()
        
        # Create new self-play environments that use the warmup model as opponent
        logger.info("Creating self-play environments with trained opponent...")
        
        def make_selfplay_env():
            return create_self_play_env(best_model_path=best_model_dir, mode="random")
        
        if args.use_subproc:
            env = SubprocVecEnv([make_selfplay_env for _ in range(args.n_envs)])
        else:
            env = make_vec_env(make_selfplay_env, n_envs=args.n_envs)
        
        # Switch to new environment for self-play phase
        model.set_env(env)
        
        logger.info(f"""
{'='*60}
WARMUP COMPLETE - Starting Self-Play Phase
{'='*60}
Remaining Steps: {args.steps:,}
Now training against the warmup model (and future best models).
{'='*60}
        """)
    
    # Callbacks
    self_play_callback = SelfPlayCallback(
        eval_freq=args.eval_freq,
        n_eval_games=args.n_eval_games,
        win_rate_threshold=args.win_threshold,
        min_per_color_wr=args.min_per_color,
        best_model_path=best_model_dir,
        n_eval_workers=args.n_eval_workers,
        verbose=1,
    )
    
    opponent_update_callback = OpponentUpdateCallback(
        update_freq=args.eval_freq // 2,  # Sync more frequently than eval
        best_model_path=best_model_dir,
        verbose=1,
    )
    
    # Incremental CSV logger for real-time plotting
    csv_logger = IncrementalCSVLogger(
        self_play_callback=self_play_callback,
        log_path=logs_dir / "self_play_history.csv",
        verbose=1,
    )
    
    # Train!
    vec_env_type = "SubprocVecEnv" if args.use_subproc else "DummyVecEnv"
    eval_type = f"{args.n_eval_workers} workers" if args.n_eval_workers > 0 else "sequential"
    
    # Performance note
    perf_note = ""
    if args.n_envs > 32:
        perf_note = "\n⚠️  WARNING: n_envs may be suboptimal. Profiling shows 16 is fastest."
    
    # Warmup note
    warmup_note = ""
    if args.warmup_steps > 0:
        warmup_note = f"\nWarmup Phase: {args.warmup_steps:,} steps vs Random (already completed)"
    
    logger.info(f"""
{'='*60}
Starting AlphaZero-style Self-Play Training
{'='*60}
Total Timesteps: {args.steps:,}{warmup_note}
Parallel Environments: {args.n_envs} ({vec_env_type}){perf_note}
Evaluation: {eval_type}
Evaluation Frequency: {args.eval_freq:,} steps
Eval Games per Check: {args.n_eval_games}

STRICT Promotion Criteria (prevents color bias):
  vs Best PPO (ALL must pass):
    - Overall: ≥{args.win_threshold:.0%}
    - As White: ≥{args.win_threshold:.0%}
    - As Black: ≥{args.win_threshold:.0%}
  vs Random (ALL must pass):
    - Overall: ≥{args.win_threshold:.0%}
    - As White: ≥{args.min_per_color:.0%}
    - As Black: ≥{args.min_per_color:.0%}

Evaluation Structure:
  - vs Best PPO: {args.n_eval_games} games (alternating colors)
  - vs Random: 300 games (50 per mode × color combination)
  - Color bias tracker: Warns after 10 consecutive weak evals

💡 Tip: Run 'python scripts/plot_training.py' in another terminal
   to monitor training progress in real-time!
{'='*60}
    """)
    
    try:
        model.learn(
            total_timesteps=args.steps,
            callback=[self_play_callback, opponent_update_callback, csv_logger],
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps,
        )
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user.")
    finally:
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = models_dir / f"ppo_selfplay_final_{timestamp}"
        model.save(str(final_path))
        logger.info(f"Saved final model to {final_path}")
        
        # Print summary
        logger.info(f"""
{'='*60}
Training Summary
{'='*60}
Final Generation: {self_play_callback.generation}
Total Evaluations: {len(self_play_callback.eval_results)}
Color Bias Status:
  Consecutive White Weak: {self_play_callback.consecutive_white_weak}
  Consecutive Black Weak: {self_play_callback.consecutive_black_weak}
        """)
        
        if self_play_callback.eval_results:
            # Save evaluation history with full per-color details
            history_path = logs_dir / "self_play_history.csv"
            with open(history_path, "w") as f:
                f.write("generation,timesteps,wins,losses,draws,win_rate,white_vs_best,black_vs_best,white_vs_random,black_vs_random,random_overall_wr,became_best\n")
                for r in self_play_callback.eval_results:
                    white_vs_best = r.get('white_vs_best_wr', 0.0)
                    black_vs_best = r.get('black_vs_best_wr', 0.0)
                    white_vs_random = r.get('white_win_rate', 0.0)
                    black_vs_random = r.get('black_win_rate', 0.0)
                    random_overall = r.get('random_overall_wr', 0.0)
                    f.write(f"{r['generation']},{r['timesteps']},{r['wins']},{r['losses']},{r['draws']},{r['win_rate']:.4f},{white_vs_best:.4f},{black_vs_best:.4f},{white_vs_random:.4f},{black_vs_random:.4f},{random_overall:.4f},{r['became_best']}\n")
            logger.info(f"Saved evaluation history to {history_path}")


if __name__ == "__main__":
    main()
