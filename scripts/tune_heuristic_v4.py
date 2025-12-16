"""Bayesian parameter tuning for `HeuristicPlayerV4` via self-play tournaments.

This script now:
- Uses Optuna's Bayesian/TPE sampler to propose promising configurations.
- Evaluates the Pareto/front-runner configs in a round-robin tournament.
- Records win-rate and average decision time per move.
- Saves two plots:
  * win-rate vs. average move time (scatter)
  * average move time per config (bar chart) to visualize compute cost.
"""
# ruff: noqa: I001

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import optuna  # type: ignore[import-not-found]
from optuna.samplers import TPESampler  # type: ignore[import-not-found]
from optuna.trial import Trial, TrialState  # type: ignore[import-not-found]

# Make `src` importable when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.game.game import Game  # noqa: E402
from src.players.heuristic_player import HeuristicPlayerV4  # noqa: E402


Color = str
Move = Tuple[int, int]


@dataclass
class V4Config:
    """Configuration for a `HeuristicPlayerV4` instance."""

    depth: int = 3
    beam_width: int = 6
    space_weight: float = 1.0
    mobility_weight: float = 0.45
    opp_mobility_weight: float = 0.6
    center_bonus: float = 0.4
    danger_penalty: float = 2500.0

    def to_kwargs(self) -> Dict[str, float]:
        return {
            "depth": self.depth,
            "beam_width": self.beam_width,
            "space_weight": self.space_weight,
            "mobility_weight": self.mobility_weight,
            "opp_mobility_weight": self.opp_mobility_weight,
            "center_bonus": self.center_bonus,
            "danger_penalty": self.danger_penalty,
        }

    def label(self) -> str:
        """Short human-readable identifier."""
        return (
            f"d{self.depth}_bw{self.beam_width}_sw{self.space_weight:.2f}"
            f"_mw{self.mobility_weight:.2f}_ow{self.opp_mobility_weight:.2f}"
            f"_cb{self.center_bonus:.2f}_dp{int(self.danger_penalty)}"
        )


@dataclass
class ConfigStats:
    """Aggregated stats for a configuration across all games."""

    label: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_move_time: float = 0.0  # seconds
    total_moves: int = 0

    def win_rate(self) -> float:
        games = self.wins + self.losses + self.draws
        if games == 0:
            return 0.0
        # Count a draw as half-win
        return (self.wins + 0.5 * self.draws) / games

    def avg_move_time_ms(self) -> float:
        if self.total_moves == 0:
            return 0.0
        return (self.total_move_time / self.total_moves) * 1000.0


@dataclass
class BOTrialResult:
    """Result of a single Optuna trial against the baseline config."""

    config: V4Config
    win_rate: float
    avg_move_time_ms: float


def sample_configs(n: int, seed: int | None = None) -> List[V4Config]:
    """Randomly sample `n` configurations within a sane search space."""
    if seed is not None:
        random.seed(seed)

    configs: List[V4Config] = []
    for _ in range(n):
        depth = random.choice([2, 8])
        beam_width = random.choice([4, 6, 8, 12])
        space_weight = random.uniform(0.1, 2)
        mobility_weight = random.uniform(0.1, 0.9)
        opp_mobility_weight = random.uniform(0.1, 0.9)
        center_bonus = random.uniform(0.1, 0.9)
        danger_penalty = random.uniform(100.0, 3500.0)

        configs.append(
            V4Config(
                depth=depth,
                beam_width=beam_width,
                space_weight=space_weight,
                mobility_weight=mobility_weight,
                opp_mobility_weight=opp_mobility_weight,
                center_bonus=center_bonus,
                danger_penalty=danger_penalty,
            )
        )

    # Always include the current default as a baseline (first entry)
    baseline = V4Config()
    return [baseline] + configs


def suggest_config(trial: Trial) -> V4Config:
    """Suggest a V4 configuration using Optuna's search space."""
    depth = trial.suggest_int("depth", 2, 6, step=1)
    beam_width = trial.suggest_categorical("beam_width", [4, 6, 8, 10, 12])
    space_weight = trial.suggest_float("space_weight", 0.2, 1.6, step=0.05)
    mobility_weight = trial.suggest_float("mobility_weight", 0.2, 0.9, step=0.05)
    opp_mobility_weight = trial.suggest_float("opp_mobility_weight", 0.2, 0.9, step=0.05)
    center_bonus = trial.suggest_float("center_bonus", 0.1, 0.9, step=0.05)
    danger_penalty = trial.suggest_float("danger_penalty", 200.0, 3500.0, step=50.0)

    return V4Config(
        depth=depth,
        beam_width=int(beam_width),
        space_weight=space_weight,
        mobility_weight=mobility_weight,
        opp_mobility_weight=opp_mobility_weight,
        center_bonus=center_bonus,
        danger_penalty=danger_penalty,
    )


def config_from_trial(trial: optuna.trial.FrozenTrial) -> V4Config:
    """Rebuild a V4Config from an Optuna trial's parameters."""
    return V4Config(
        depth=int(trial.params["depth"]),
        beam_width=int(trial.params["beam_width"]),
        space_weight=float(trial.params["space_weight"]),
        mobility_weight=float(trial.params["mobility_weight"]),
        opp_mobility_weight=float(trial.params["opp_mobility_weight"]),
        center_bonus=float(trial.params["center_bonus"]),
        danger_penalty=float(trial.params["danger_penalty"]),
    )


def evaluate_against_baseline(
    cfg: V4Config,
    baseline_cfg: V4Config,
    games_per_direction: int,
    max_moves: int,
    seed: int | None = None,
) -> Tuple[float, float]:
    """Play cfg vs baseline both colors; return (win_rate, avg_move_time_ms for cfg)."""
    wins = losses = draws = 0
    total_time = 0.0
    total_moves = 0
    rng = random.Random(seed)

    for _ in range(games_per_direction):
        # cfg as white
        winner, _, _, per_stats = play_single_game(
            cfg, baseline_cfg, max_moves=max_moves, seed=rng.randint(0, 10_000_000)
        )
        if winner == "white":
            wins += 1
        elif winner == "black":
            losses += 1
        else:
            draws += 1
        w_time, w_moves = per_stats["white"]
        total_time += w_time
        total_moves += w_moves

        # cfg as black
        winner, _, _, per_stats = play_single_game(
            baseline_cfg, cfg, max_moves=max_moves, seed=rng.randint(0, 10_000_000)
        )
        if winner == "black":
            wins += 1
        elif winner == "white":
            losses += 1
        else:
            draws += 1
        b_time, b_moves = per_stats["black"]
        total_time += b_time
        total_moves += b_moves

    games_played = 2 * games_per_direction
    win_rate = (wins + 0.5 * draws) / games_played
    avg_move_time_ms = (total_time / total_moves) * 1000.0 if total_moves else 0.0
    return win_rate, avg_move_time_ms


def bayesian_optimize_configs(
    baseline_cfg: V4Config,
    n_trials: int,
    games_per_direction: int,
    max_moves: int,
    seed: int | None = None,
) -> Tuple[optuna.study.Study, List[BOTrialResult]]:
    """Run Optuna TPE to propose configs; score vs baseline."""
    sampler = TPESampler(seed=seed, multivariate=True, group=True)
    study = optuna.create_study(
        directions=["maximize", "minimize"],  # maximize win-rate, minimize avg move time
        sampler=sampler,
    )

    rng = random.Random(seed)
    trial_results: List[BOTrialResult] = []

    def objective(trial: Trial) -> Tuple[float, float]:
        cfg = suggest_config(trial)
        wr, avg_ms = evaluate_against_baseline(
            cfg,
            baseline_cfg,
            games_per_direction=games_per_direction,
            max_moves=max_moves,
            seed=rng.randint(0, 10_000_000),
        )
        trial.set_user_attr("config_label", cfg.label())
        trial.set_user_attr("config_dict", asdict(cfg))
        trial_results.append(BOTrialResult(config=cfg, win_rate=wr, avg_move_time_ms=avg_ms))
        return wr, avg_ms

    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    return study, trial_results


def select_candidates_from_study(
    study: optuna.study.Study,
    baseline_cfg: V4Config,
    top_k: int,
) -> List[V4Config]:
    """Select configs (baseline + Pareto) for the round-robin tournament."""
    configs: List[V4Config] = []
    seen: set[str] = set()

    def add_cfg(cfg: V4Config) -> None:
        lab = cfg.label()
        if lab not in seen:
            seen.add(lab)
            configs.append(cfg)

    add_cfg(baseline_cfg)

    # Start with Pareto front (best_trials for multi-objective)
    for trial in study.best_trials:
        add_cfg(config_from_trial(trial))
        if len(configs) >= top_k + 1:  # +1 for baseline
            return configs

    # Fill remaining slots by win-rate then speed
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    completed.sort(key=lambda t: (-t.values[0], t.values[1]))
    for trial in completed:
        add_cfg(config_from_trial(trial))
        if len(configs) >= top_k + 1:
            break

    return configs


def play_single_game(
    cfg_white: V4Config,
    cfg_black: V4Config,
    max_moves: int = 200,
    seed: int | None = None,
) -> Tuple[str, int, float, Dict[Color, Tuple[float, int]]]:
    """Play one Mode-2 game between two V4 configs.

    Returns:
        winner: "white", "black", or "draw"
        moves: total moves in the game
        duration: wall-clock game duration (s)
        per_player_stats: {color: (total_move_time_s, move_count)}
    """
    if seed is not None:
        random.seed(seed)

    white = HeuristicPlayerV4("white", **cfg_white.to_kwargs())
    black = HeuristicPlayerV4("black", **cfg_black.to_kwargs())

    game = Game(white, black, mode=2, logging=False)

    start_time = time.time()
    per_player_time = {"white": 0.0, "black": 0.0}
    per_player_moves = {"white": 0, "black": 0}

    try:
        while not game.game_over:
            if len(game.board.move_history) > max_moves:
                game.game_over = True
                game.winner = "draw"
                break

            current_player = game.get_current_player()
            color: Color = game.current_player
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                # Game logic in `get_legal_moves` should already end the game + set winner.
                break

            t0 = time.time()
            move_to, extra = current_player.get_move(game.board, legal_moves)
            t1 = time.time()

            per_player_time[color] += t1 - t0
            per_player_moves[color] += 1

            success = game.make_move(move_to, extra)
            if not success:
                # Illegal move: treat as immediate loss for the side that moved.
                game.game_over = True
                if color == "white":
                    game.winner = "black"
                else:
                    game.winner = "white"
                break

    except Exception:
        # Any crash -> loss for side to move
        if game.current_player == "white":
            game.winner = "black"
        else:
            game.winner = "white"
        game.game_over = True

    duration = time.time() - start_time
    winner = game.winner or "draw"
    total_moves = len(game.board.move_history)

    stats: Dict[Color, Tuple[float, int]] = {
        "white": (per_player_time["white"], per_player_moves["white"]),
        "black": (per_player_time["black"], per_player_moves["black"]),
    }
    return winner, total_moves, duration, stats


def run_tournament(
    configs: List[V4Config],
    games_per_pair: int,
    seed: int | None = None,
    max_moves: int = 200,
) -> Dict[str, ConfigStats]:
    """Round-robin tournament between parameter configs."""
    if seed is not None:
        random.seed(seed)

    labels = [cfg.label() for cfg in configs]
    stats: Dict[str, ConfigStats] = {lab: ConfigStats(label=lab) for lab in labels}

    num = len(configs)
    total_matchups = num * (num - 1) // 2
    print(f"Running round-robin between {num} configs ({total_matchups} unique pairs).")
    print(f"{games_per_pair} games per direction per pair (total games ~= {2 * games_per_pair * total_matchups}).")

    match_index = 0
    for i in range(num):
        for j in range(i + 1, num):
            cfg_i = configs[i]
            cfg_j = configs[j]
            label_i = cfg_i.label()
            label_j = cfg_j.label()
            match_index += 1

            print(f"\nPair {match_index}/{total_matchups}: {label_i} vs {label_j}")

            for g in range(games_per_pair):
                # A as white, B as black
                winner, _, _, per_stats = play_single_game(
                    cfg_i, cfg_j, max_moves=max_moves, seed=random.randint(0, 10_000_000)
                )
                if winner == "white":
                    stats[label_i].wins += 1
                    stats[label_j].losses += 1
                elif winner == "black":
                    stats[label_j].wins += 1
                    stats[label_i].losses += 1
                else:
                    stats[label_i].draws += 1
                    stats[label_j].draws += 1

                w_time, w_moves = per_stats["white"]
                b_time, b_moves = per_stats["black"]
                stats[label_i].total_move_time += w_time
                stats[label_i].total_moves += w_moves
                stats[label_j].total_move_time += b_time
                stats[label_j].total_moves += b_moves

                # B as white, A as black
                winner, _, _, per_stats = play_single_game(
                    cfg_j, cfg_i, max_moves=max_moves, seed=random.randint(0, 10_000_000)
                )
                if winner == "white":
                    stats[label_j].wins += 1
                    stats[label_i].losses += 1
                elif winner == "black":
                    stats[label_i].wins += 1
                    stats[label_j].losses += 1
                else:
                    stats[label_i].draws += 1
                    stats[label_j].draws += 1

                w_time, w_moves = per_stats["white"]
                b_time, b_moves = per_stats["black"]
                stats[label_j].total_move_time += w_time
                stats[label_j].total_moves += w_moves
                stats[label_i].total_move_time += b_time
                stats[label_i].total_moves += b_moves

    return stats


def choose_best_config(
    configs: List[V4Config],
    stats: Dict[str, ConfigStats],
) -> Tuple[V4Config, ConfigStats]:
    """Pick the best configuration by win-rate, breaking ties by speed."""
    best_cfg: V4Config | None = None
    best_stat: ConfigStats | None = None

    for cfg in configs:
        lab = cfg.label()
        s = stats[lab]
        if best_cfg is None:
            best_cfg, best_stat = cfg, s
            continue

        wr = s.win_rate()
        best_wr = best_stat.win_rate()

        if wr > best_wr + 1e-6:
            best_cfg, best_stat = cfg, s
        elif math.isclose(wr, best_wr, rel_tol=1e-6, abs_tol=1e-6):
            # Same win-rate → prefer faster average move time
            if s.avg_move_time_ms() < best_stat.avg_move_time_ms():
                best_cfg, best_stat = cfg, s

    assert best_cfg is not None and best_stat is not None
    return best_cfg, best_stat


def plot_results(
    stats: Dict[str, ConfigStats],
    best_label: str,
    output_path: Path,
) -> None:
    """Create a scatter plot: win-rate vs avg move time."""
    x = []
    y = []
    colors = []

    for label, s in stats.items():
        x.append(s.avg_move_time_ms())
        y.append(s.win_rate())
        colors.append("red" if label == best_label else "blue")

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, c=colors, alpha=0.8)
    plt.xlabel("Average move time [ms]")
    plt.ylabel("Win rate (vs. other configs)")
    plt.title("HeuristicPlayerV4 parameter tuning: strength vs speed")

    # Annotate the best configuration
    best_stat = stats[best_label]
    plt.annotate(
        "BEST",
        xy=(best_stat.avg_move_time_ms(), best_stat.win_rate()),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
        color="red",
        weight="bold",
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_time_by_config(stats: Dict[str, ConfigStats], output_path: Path) -> None:
    """Create a bar chart of avg move time per configuration (slower at top)."""
    ordered = sorted(stats.items(), key=lambda kv: kv[1].avg_move_time_ms(), reverse=True)
    labels = [lab for lab, _ in ordered]
    times = [s.avg_move_time_ms() for _, s in ordered]

    plt.figure(figsize=(10, max(4, len(labels) * 0.35)))
    plt.barh(labels, times, color="skyblue", alpha=0.85)
    plt.xlabel("Average move time [ms]")
    plt.title("HeuristicPlayerV4: per-config compute cost")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian tune HeuristicPlayerV4 parameters via self-play."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of Optuna/Bayesian trials to run (overrides --configs).",
    )
    parser.add_argument(
        "--configs",
        type=int,
        default=None,
        help="Deprecated alias for --trials to keep older commands working.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="How many Optuna-suggested configs to keep (baseline added automatically).",
    )
    parser.add_argument(
        "--games-per-eval",
        type=int,
        default=2,
        help="Games per direction vs baseline during Bayesian search (per trial).",
    )
    parser.add_argument(
        "--games-per-pair",
        type=int,
        default=2,
        help="Games per direction for each pair of configs (A-white/B-black and B-white/A-black).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help="Abort a game after this many half-moves to avoid marathon draws.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="v4_tuning_results.png",
        help="Path for the win-rate vs time plot (relative to project root).",
    )
    parser.add_argument(
        "--time-output",
        type=str,
        default="v4_tuning_time.png",
        help="Path for the per-config time bar plot (relative to project root).",
    )
    args = parser.parse_args()

    bo_trials = args.trials if args.trials is not None else args.configs
    if bo_trials is None:
        bo_trials = 15

    baseline_cfg = V4Config()
    print(f"Running Bayesian optimization (Optuna TPE) for {bo_trials} trials...")
    study, bo_results = bayesian_optimize_configs(
        baseline_cfg,
        n_trials=bo_trials,
        games_per_direction=args.games_per_eval,
        max_moves=args.max_moves,
        seed=args.seed,
    )
    print(f"Completed {len(bo_results)} trials. Pareto-best candidates:")
    for trial in study.best_trials:
        wr, avg_ms = trial.values
        cfg = config_from_trial(trial)
        print(f"- {cfg.label()}: win_rate={wr:.3f}, avg_move_time_ms={avg_ms:.2f}")

    configs = select_candidates_from_study(study, baseline_cfg, top_k=args.top_k)
    print(f"\nSelected {len(configs)} configs (including baseline) for round-robin:")
    for cfg in configs:
        print(f"- {cfg.label()}")

    stats = run_tournament(
        configs,
        games_per_pair=args.games_per_pair,
        seed=args.seed,
        max_moves=args.max_moves,
    )

    best_cfg, best_stat = choose_best_config(configs, stats)
    best_label = best_cfg.label()

    print("\n=== BEST CONFIGURATION ===")
    print(f"Label: {best_label}")
    for k, v in asdict(best_cfg).items():
        print(f"  {k}: {v}")
    print(f"Win-rate: {best_stat.win_rate():.3f}  (W={best_stat.wins}, L={best_stat.losses}, D={best_stat.draws})")
    print(f"Average move time: {best_stat.avg_move_time_ms():.2f} ms over {best_stat.total_moves} moves")

    scatter_path = ROOT / args.output
    time_path = ROOT / args.time_output
    print(f"\nSaving win-rate vs avg-move-time plot to: {scatter_path}")
    plot_results(stats, best_label, scatter_path)
    print(f"Saving per-config move-time bar chart to: {time_path}")
    plot_time_by_config(stats, time_path)


if __name__ == "__main__":
    main()
