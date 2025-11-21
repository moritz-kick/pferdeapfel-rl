"""Basic PPO training script for the Pferdeäpfel mode 2 environment."""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path
from collections import deque
from typing import Any, Dict, List, Sequence, Tuple

from datetime import datetime
import numpy as np
import yaml
from sb3_contrib import MaskablePPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.env.pferdeapfel_env import PferdeapfelEnv, legal_action_mask
from src.training.wrappers import ForwardingActionMasker

logger = logging.getLogger(__name__)


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


class StopAfterEpisodesCallback(BaseCallback):
    """Stop training after a target number of completed episodes (games)."""

    def __init__(self, target_episodes: int, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.target_episodes = target_episodes
        self.episodes = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None:
            self.episodes += int(np.sum(dones))

        if self.episodes >= self.target_episodes:
            logger.info("Reached target of %d episodes; stopping current learn().", self.target_episodes)
            return False
        return True


class SelfPlaySyncCallback(BaseCallback):
    """
    Periodically run a round-robin tournament between the current policy and a pool
    of recent snapshots. The best performer becomes the preferred opponent, while
    older snapshots remain in a pool for diversity during training episodes.
    """

    def __init__(
        self,
        env_factory,
        sync_interval: int,
        eval_episodes: int,
        opponent_deterministic: bool,
        best_prob: float,
        old_prob: float,
        max_old_models: int = 4,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.env_factory = env_factory
        self.sync_interval = sync_interval
        self.eval_episodes = eval_episodes
        self.opponent_deterministic = opponent_deterministic
        self._next_eval = sync_interval
        self._opponent_model = None
        self._best_model = None
        self._old_models: deque = deque(maxlen=max_old_models)
        self._best_prob = best_prob
        self._old_prob = old_prob

    def _on_training_start(self) -> None:
        # Freeze initial opponent and broadcast to all envs
        self._opponent_model = self._snapshot_model()
        self._best_model = self._opponent_model
        filtered_old = [m for m in self._old_models if m is not self._best_model]
        self.training_env.env_method(
            "set_opponent_model", self.model, deterministic=self.opponent_deterministic
        )
        self.training_env.env_method(
            "set_opponent_pool",
            self._best_model,
            filtered_old,
            self._best_prob,
            self._old_prob,
            deterministic=self.opponent_deterministic,
        )

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval:
            return True

        best_model, best_label, scores = self._run_tournament()
        self._best_model = best_model
        filtered_old = [m for m in self._old_models if m is not self._best_model]
        self.training_env.env_method(
            "set_opponent_pool",
            self._best_model,
            filtered_old,
            self._best_prob,
            self._old_prob,
            deterministic=self.opponent_deterministic,
        )
        logger.info(
            "Self-play tournament at %d steps: scores=%s (best=%s)",
            self.num_timesteps,
            scores,
            best_label,
        )

        self._next_eval += self.sync_interval
        return True

    def _snapshot_model(self):
        """Create a frozen copy of the current policy."""
        algo_cls = self.model.__class__
        with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
            self.model.save(tmp.name)
            tmp.flush()
            return algo_cls.load(tmp.name)

    def _evaluate_pair(self, model_a: BaseAlgorithm, model_b: BaseAlgorithm) -> Tuple[int, int]:
        """Play model_a (agent) vs model_b (opponent) and return (wins_a, wins_b)."""
        wins_a = 0
        wins_b = 0
        for _ in range(self.eval_episodes):
            env = self.env_factory()
            # Eliminate random opponents during eval to keep matches deterministic-ish
            if hasattr(env, "random_opponent_chance"):
                env.random_opponent_chance = 0.0
            env.set_opponent_model(model_b, deterministic=self.opponent_deterministic)
            env.set_opponent_pool(model_b, [], 0.0, 0.0, deterministic=self.opponent_deterministic)
            obs, _ = env.reset()
            done = False
            while not done:
                mask_fn = getattr(env, "get_action_mask", None)
                action_masks = mask_fn() if callable(mask_fn) else None
                action, _ = model_a.predict(obs, deterministic=True, action_masks=action_masks)
                obs, _, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
                if done:
                    winner = info.get("winner")
                    if winner == env.agent_color:
                        wins_a += 1
                    elif winner is not None:
                        wins_b += 1
            env.close()
        return wins_a, wins_b

    def _run_tournament(self) -> Tuple[BaseAlgorithm, str, List[Tuple[str, int]]]:
        """
        Evaluate the current model plus the last snapshots (max_old_models) round-robin.
        Returns (best_model, best_label, scores) where scores is [(label, wins), ...].
        """
        current_snapshot = self._snapshot_model()
        models: Sequence[BaseAlgorithm] = (current_snapshot,) + tuple(self._old_models)
        labels = [f"current_{self.num_timesteps}"] + [f"old_{idx}" for idx in range(len(self._old_models))]

        scores = [0 for _ in models]
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                win_i, win_j = self._evaluate_pair(models[i], models[j])
                scores[i] += win_i
                scores[j] += win_j

        best_idx = int(np.argmax(scores))
        best_model = models[best_idx]
        best_label = labels[best_idx]

        # Update history: keep current snapshot at the front
        self._old_models.appendleft(current_snapshot)

        score_pairs: List[Tuple[str, int]] = list(zip(labels, scores))
        return best_model, best_label, score_pairs


def make_vec_env_from_config(
    env_cfg: Dict[str, Any], n_envs: int, require_same_process: bool = False
):
    """Create a vectorized environment with sensible defaults for self-play."""
    def env_factory():
        base_env = PferdeapfelEnv(**env_cfg)
        return ForwardingActionMasker(base_env, legal_action_mask)

    requires_same_process = require_same_process or env_cfg.get("opponent_policy") in ("self_play", "self")
    vec_env_cls = SubprocVecEnv if n_envs > 1 and not requires_same_process else DummyVecEnv

    if requires_same_process and vec_env_cls is SubprocVecEnv:
        logger.info(
            "Self-play opponent sharing requires same-process envs; falling back to DummyVecEnv (n_envs=%d).", n_envs
        )
        vec_env_cls = DummyVecEnv

    vec_env = make_vec_env(env_factory, n_envs=n_envs, vec_env_cls=vec_env_cls)
    return vec_env, env_factory, vec_env_cls


def maybe_load_initial_weights(model: BaseAlgorithm, initial_model_path: Path | None) -> None:
    """Load pretrained weights into the current policy if a path is provided."""
    if not initial_model_path:
        return
    if not initial_model_path.exists():
        logger.warning("Initial model path %s not found; training from scratch.", initial_model_path)
        return

    try:
        algo_cls = model.__class__
        pretrained = algo_cls.load(str(initial_model_path))
        model.policy.load_state_dict(pretrained.policy.state_dict())
        logger.info("Initialized policy weights from %s", initial_model_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load initial weights from %s (%s); using fresh weights.", initial_model_path, exc)


def broadcast_self_play_opponent(vec_env, env_cfg: Dict[str, Any], model: BaseAlgorithm) -> None:
    """Share the current model with environments that use self-play opponents."""
    if env_cfg.get("opponent_policy") not in ("self_play", "self"):
        return

    opponent_det = bool(env_cfg.get("opponent_deterministic", False))
    if isinstance(vec_env, DummyVecEnv):
        vec_env.env_method("set_opponent_model", model, deterministic=opponent_det)
    else:
        logger.warning(
            "Skipping initial self-play opponent broadcast because SubprocVecEnv cannot share the model directly. "
            "Switch to DummyVecEnv if you need live opponent updates."
        )


def resolve_run_name(config_value: str | None) -> str:
    """Return a non-empty run name, falling back to a timestamped default."""
    name = (config_value or "").strip()
    if name:
        return name
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def resolve_model_root(raw_model_path: Path) -> Path:
    """Treat save_path as a directory root; fall back to data/models if needed."""
    if raw_model_path.suffix:
        base = raw_model_path.parent
    else:
        base = raw_model_path
    if str(base):
        return base
    return Path("data/models")


def ensure_unique_run_name(run_name: str, tensorboard_log: Path, model_root: Path) -> str:
    """
    Avoid overwriting logs/models when a run name is reused by appending a counter suffix.
    """
    candidate = run_name
    idx = 1
    while (tensorboard_log / candidate).exists() or (model_root / candidate).exists():
        candidate = f"{run_name}_{idx:02d}"
        idx += 1
    if candidate != run_name:
        logger.info("Adjusted run name to avoid overwrite: %s -> %s", run_name, candidate)
    return candidate


def find_latest_model(model_root: Path) -> Path | None:
    """Return the most recently modified .zip model under model_root (recursively)."""
    if not model_root.exists():
        return None
    candidates = [p for p in model_root.rglob("*.zip") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO agent for Pferdeäpfel (mode 2).")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml", help="Config file.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name to tag logs and saved model.")
    args = parser.parse_args()

    config = load_config(args.config)
    trainer_cfg: Dict[str, Any] = config.get("trainer", {})
    env_cfg: Dict[str, Any] = config.get("env", {})

    total_timesteps = int(trainer_cfg.get("total_timesteps", 10000))
    learning_rate = float(trainer_cfg.get("learning_rate", 3e-4))
    n_steps = int(trainer_cfg.get("n_steps", 256))
    batch_size = int(trainer_cfg.get("batch_size", 64))
    n_envs = int(trainer_cfg.get("n_envs", 1))
    gamma = float(trainer_cfg.get("gamma", 0.99))
    ent_coef = float(trainer_cfg.get("ent_coef", 0.01))
    tensorboard_log = Path(trainer_cfg.get("tensorboard_log", "data/logs/rl"))
    raw_model_path = Path(trainer_cfg.get("save_path", "data/models/ppo_pferdeapfel"))
    cfg_run_name = str(trainer_cfg.get("run_name", "")).strip()
    run_name = resolve_run_name(args.run_name or cfg_run_name)
    initial_model_raw = str(trainer_cfg.get("initial_model_path", "") or "").strip()
    initial_model_path = Path(initial_model_raw) if initial_model_raw else None
    warmup_games = int(trainer_cfg.get("warmup_games", 0))
    warmup_timesteps = int(trainer_cfg.get("warmup_total_timesteps", warmup_games * 256)) if warmup_games else 0
    warmup_n_envs = int(trainer_cfg.get("warmup_n_envs", n_envs))
    warmup_env_cfg: Dict[str, Any] = {**env_cfg, **trainer_cfg.get("warmup_env", {})}
    if warmup_games:
        warmup_env_cfg.setdefault("opponent_policy", "random")
        warmup_env_cfg.setdefault("agent_color", "random")

    tensorboard_log.mkdir(parents=True, exist_ok=True)
    model_root = resolve_model_root(raw_model_path)
    run_name = ensure_unique_run_name(run_name, tensorboard_log, model_root)
    model_run_dir = model_root / run_name
    model_run_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_run_dir / f"{run_name}.zip"

    if initial_model_path is None:
        latest_model = find_latest_model(model_root)
        if latest_model:
            initial_model_path = latest_model
            logger.info("Using latest model as initial weights: %s", latest_model)

    sp_cfg: Dict[str, Any] = trainer_cfg.get("self_play", {})
    enable_self_play = bool(sp_cfg.get("enabled", False))

    # Stage 1: optional warmup vs. random legal opponent to learn valid moves.
    if warmup_games:
        vec_env, env_factory, _ = make_vec_env_from_config(
            warmup_env_cfg, n_envs=warmup_n_envs, require_same_process=False
        )
    else:
        vec_env, env_factory, _ = make_vec_env_from_config(
            env_cfg, n_envs=n_envs, require_same_process=enable_self_play
        )

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        ent_coef=ent_coef,
        tensorboard_log=str(tensorboard_log),
        verbose=1,
    )
    maybe_load_initial_weights(model, initial_model_path)

    # Optional warmup phase against a random legal opponent.
    if warmup_games:
        warmup_cb = StopAfterEpisodesCallback(target_episodes=warmup_games)
        learn_kwargs: Dict[str, Any] = {
            "total_timesteps": max(warmup_timesteps, warmup_games),
            "callback": warmup_cb,
            "tb_log_name": run_name,
        }
        try:  # pragma: no cover - depends on installed SB3 version
            learn_kwargs["progress_bar"] = True
            model.learn(**learn_kwargs)
        except TypeError:
            learn_kwargs.pop("progress_bar", None)
            model.learn(**learn_kwargs)
        logger.info("Warmup finished after %d episodes (timesteps=%d).", warmup_cb.episodes, model.num_timesteps)
        vec_env.close()
        # Switch to the main self-play training stage.
        vec_env, env_factory, _ = make_vec_env_from_config(
            env_cfg, n_envs=n_envs, require_same_process=enable_self_play
        )
        model.set_env(vec_env)

    callbacks = []
    if enable_self_play:
        callbacks.append(
            SelfPlaySyncCallback(
                env_factory=env_factory,
                sync_interval=int(sp_cfg.get("sync_interval", 50_000)),
                eval_episodes=int(sp_cfg.get("eval_episodes", 8)),
                opponent_deterministic=bool(sp_cfg.get("opponent_deterministic", True)),
                best_prob=float(sp_cfg.get("best_opponent_prob", 0.4)),
                old_prob=float(sp_cfg.get("old_opponent_prob", 0.3)),
                max_old_models=4,
            )
        )

    # Enable self-play by letting the opponent use the current policy weights.
    broadcast_self_play_opponent(vec_env, env_cfg, model)

    learn_kwargs = {
        "total_timesteps": total_timesteps,
        "reset_num_timesteps": not warmup_games,
        "callback": callbacks or None,
        "tb_log_name": run_name,
    }
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
