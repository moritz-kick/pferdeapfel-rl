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
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from src.env.knight_self_play_env import KnightSelfPlayEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SelfPlayCallback(BaseCallback):
    """
    Callback for AlphaZero-style self-play training.
    
    Periodically evaluates the current model against the best model.
    If the current model wins >= win_rate_threshold, it becomes the new best.
    """

    def __init__(
        self,
        eval_freq: int = 50_000,
        n_eval_games: int = 100,
        win_rate_threshold: float = 0.55,
        best_model_path: Path = Path("data/models/ppo_self_play/best_model"),
        verbose: int = 1,
    ):
        """
        Args:
            eval_freq: Evaluate every N timesteps
            n_eval_games: Number of games to play for evaluation
            win_rate_threshold: Win rate needed to become new best (0.55 = 55%)
            best_model_path: Path to save the best model
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_games = n_eval_games
        self.win_rate_threshold = win_rate_threshold
        self.best_model_path = Path(best_model_path)
        self.last_eval_timestep = 0
        
        # Statistics
        self.generation = 0
        self.eval_results: list[dict] = []

    def _on_step(self) -> bool:
        """Called after each step."""
        # Check if we've passed the next evaluation threshold
        if self.num_timesteps - self.last_eval_timestep >= self.eval_freq:
            self.last_eval_timestep = self.num_timesteps
            self._evaluate_and_maybe_update()
        
        return True

    def _evaluate_and_maybe_update(self) -> None:
        """Evaluate current model vs best and update if better.
        
        IMPORTANT: The model must perform well from BOTH white and black perspectives
        to become the new best. This prevents learning color-biased strategies.
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
        
        # Evaluate: current model plays against best model
        wins, losses, draws, details = self._play_evaluation_games(
            current_model=self.model,
            opponent_model=best_model,
            n_games=self.n_eval_games,
        )
        
        total_games = wins + losses + draws
        win_rate = wins / total_games if total_games > 0 else 0.0
        
        # Calculate per-color win rates
        white_wr = details["white_wins"] / details["white_games"] if details["white_games"] > 0 else 0.0
        black_wr = details["black_wins"] / details["black_games"] if details["black_games"] > 0 else 0.0
        
        logger.info(f"Results: Wins={wins}, Losses={losses}, Draws={draws}")
        logger.info(f"Overall Win Rate: {win_rate:.1%} (threshold: {self.win_rate_threshold:.1%})")
        logger.info(f"  As White: {white_wr:.1%} ({details['white_wins']}/{details['white_games']})")
        logger.info(f"  As Black: {black_wr:.1%} ({details['black_wins']}/{details['black_games']})")
        
        # Log per-mode performance
        for mode, stats in details["mode_stats"].items():
            if stats["total"] > 0:
                mode_wr = stats["wins"] / stats["total"]
                logger.info(f"  Mode {mode}: {mode_wr:.1%} ({stats['wins']}/{stats['total']})")
        
        self.eval_results.append({
            "generation": self.generation,
            "timesteps": self.num_timesteps,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "white_win_rate": white_wr,
            "black_win_rate": black_wr,
            "became_best": False,  # Will update below if becomes best
        })
        
        # CRITICAL: Require model to be good from BOTH perspectives
        # This prevents the "color collapse" problem where model only wins as one color
        min_per_color_wr = 0.40  # Must win at least 40% from each color
        
        is_good_overall = win_rate >= self.win_rate_threshold
        is_balanced = white_wr >= min_per_color_wr and black_wr >= min_per_color_wr
        
        if is_good_overall and is_balanced:
            logger.info(f"🎉 New best model! Generation {self.generation} -> {self.generation + 1}")
            logger.info(f"   (Balanced: White {white_wr:.1%}, Black {black_wr:.1%})")
            self._save_as_best()
            self.eval_results[-1]["became_best"] = True
            self.generation += 1
            
            # Update the opponent policy in all environments
            self._update_opponent_in_envs()
        else:
            if not is_good_overall:
                logger.info(f"Current model not good enough (need {self.win_rate_threshold:.1%}, got {win_rate:.1%})")
            elif not is_balanced:
                logger.info(f"Current model is COLOR-BIASED:")
                logger.info(f"   White: {white_wr:.1%}, Black: {black_wr:.1%} (need ≥{min_per_color_wr:.1%} each)")
                logger.info("   Model must learn to win from BOTH perspectives!")
    
    def _play_evaluation_games(
        self,
        current_model: MaskablePPO,
        opponent_model: Any,  # Can be MaskablePPO or PPO
        n_games: int,
    ) -> Tuple[int, int, int, dict]:
        """
        Play n_games between current and opponent model.
        
        Games are split evenly: half with current as white, half as black.
        This ensures we measure performance from BOTH perspectives.
        
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
        default=32,
        help="Number of parallel environments (default: 32).",
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
        "--win-threshold",
        type=float,
        default=0.55,
        help="Win rate threshold to become new best (default: 0.55).",
    )
    parser.add_argument(
        "--from-pretrained",
        type=str,
        default=None,
        help="Path to a pretrained model to start from (e.g., from random training).",
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
    
    # Create environments
    logger.info(f"Creating {args.n_envs} parallel self-play environments...")
    
    def make_env():
        return create_self_play_env(best_model_path=best_model_dir, mode="random")
    
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
            n_steps=2048 // args.n_envs if 2048 >= args.n_envs else 128,
            batch_size=64,
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
    
    # Callbacks
    self_play_callback = SelfPlayCallback(
        eval_freq=args.eval_freq,
        n_eval_games=args.n_eval_games,
        win_rate_threshold=args.win_threshold,
        best_model_path=best_model_dir,
        verbose=1,
    )
    
    opponent_update_callback = OpponentUpdateCallback(
        update_freq=args.eval_freq // 2,  # Sync more frequently than eval
        best_model_path=best_model_dir,
        verbose=1,
    )
    
    # Train!
    logger.info(f"""
{'='*60}
Starting AlphaZero-style Self-Play Training
{'='*60}
Total Timesteps: {args.steps:,}
Parallel Environments: {args.n_envs}
Evaluation Frequency: {args.eval_freq:,} steps
Eval Games per Check: {args.n_eval_games}
Win Rate Threshold: {args.win_threshold:.0%}
{'='*60}
    """)
    
    try:
        model.learn(
            total_timesteps=args.steps,
            callback=[self_play_callback, opponent_update_callback],
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
        """)
        
        if self_play_callback.eval_results:
            # Save evaluation history
            history_path = logs_dir / "self_play_history.csv"
            with open(history_path, "w") as f:
                f.write("generation,timesteps,wins,losses,draws,win_rate,became_best\n")
                for r in self_play_callback.eval_results:
                    f.write(f"{r['generation']},{r['timesteps']},{r['wins']},{r['losses']},{r['draws']},{r['win_rate']:.4f},{r['became_best']}\n")
            logger.info(f"Saved evaluation history to {history_path}")


if __name__ == "__main__":
    main()
