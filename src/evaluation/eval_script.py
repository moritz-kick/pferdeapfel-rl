"""Pairwise evaluation of available players in mode 2."""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

from src.game.game import Game
from src.players.base import Player
from src.players.mcts import MCTSPlayer
from src.players.minimax import AlphaBetaPlayer
from src.players.random import RandomPlayer

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from src.players.rl.ppo_rl import PPOPlayer
except Exception as exc:  # pragma: no cover - guard import
    PPOPlayer = None
    _PPO_IMPORT_ERR = exc
else:  # pragma: no cover - guard import
    _PPO_IMPORT_ERR = None

try:  # pragma: no cover - optional dependency
    from src.players.rl.dqn_rl import DQNPlayer
except Exception as exc:  # pragma: no cover - guard import
    DQNPlayer = None
    _DQN_IMPORT_ERR = exc
else:  # pragma: no cover - guard import
    _DQN_IMPORT_ERR = None

# Default search roots for auto-discovery
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIRS = [
    PROJECT_ROOT / "data" / "models" / "ppo_pferdeapfel",
    PROJECT_ROOT / "data" / "models",
]


@dataclass(frozen=True)
class PlayerSpec:
    """Descriptor for a player that can be instantiated for either side."""

    name: str
    builder: Callable[[str], Player]


def discover_models(paths: Sequence[Path] | None = None) -> List[Path]:
    """
    Find .zip models under the provided paths (files or directories), sorted newest first.

    Args:
        paths: Optional list of files/directories to search. If omitted, defaults to DEFAULT_MODEL_DIRS.
    """
    roots = list(paths) if paths else DEFAULT_MODEL_DIRS
    candidates: list[Path] = []

    for root in roots:
        root_expanded = root.expanduser()
        if not root_expanded.exists():
            continue
        if root_expanded.is_file() and root_expanded.suffix == ".zip":
            candidates.append(root_expanded.resolve())
        elif root_expanded.is_dir():
            candidates.extend(p.resolve() for p in root_expanded.rglob("*.zip"))

    # Deduplicate while preserving order
    unique: list[Path] = list(dict.fromkeys(candidates))

    def _mtime_or_zero(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    unique.sort(key=_mtime_or_zero, reverse=True)
    return unique


def _build_rl_player(model_path: Path, side: str = "black") -> Player:
    """Instantiate an RL player for the given model, preferring PPO, falling back to DQN."""
    errors: list[str] = []

    if PPOPlayer is not None:
        try:
            return PPOPlayer(side, model_path=model_path)
        except Exception as exc:  # pragma: no cover - dependency/runtime guard
            errors.append(f"PPO load failed: {exc}")
    else:
        errors.append(f"PPO unavailable ({_PPO_IMPORT_ERR})")

    if DQNPlayer is not None:
        try:
            return DQNPlayer(side, model_path=model_path)
        except Exception as exc:  # pragma: no cover - dependency/runtime guard
            errors.append(f"DQN load failed: {exc}")
    else:
        errors.append(f"DQN unavailable ({_DQN_IMPORT_ERR})")

    raise RuntimeError(f"Could not load model {model_path}: {'; '.join(errors)}")


def play_single_game(white: Player, black: Player, mode: int = 2) -> str:
    """Play one game between two players; White always starts."""
    game = Game(white, black, mode=mode, logging=False)
    game.current_player = "white"

    while not game.game_over:
        player = game.get_current_player()
        legal_moves = game.get_legal_moves()
        move_to, extra = player.get_move(game.board, legal_moves)
        game.make_move(move_to, extra)
    return game.winner or "draw"


def evaluate_model(model_path: Path, games: int) -> Dict[str, int]:
    """Run a batch of games for a single RL model vs Random (legacy helper)."""
    return play_matchup(
        player_a_builder=lambda side: _build_rl_player(model_path, side),
        player_b_builder=lambda _side: RandomPlayer("RandomPlayer"),
        games_per_side=games,
    )


def _write_results(output: Path, rows: Iterable[Dict[str, object]]) -> None:
    """Append results to CSV, writing the header if needed."""
    fieldnames = ["agent", "opponent", "games", "agent_wins", "opponent_wins", "draws", "win_rate"]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _pair_key(agent: str, opponent: str) -> tuple[str, str]:
    """Generate an order-insensitive key for a matchup."""
    return tuple(sorted((agent, opponent)))


def _load_existing_pairs(output: Path) -> set[tuple[str, str]]:
    """Read existing CSV rows to avoid replaying matchups."""
    if not output.exists():
        return set()

    pairs: set[tuple[str, str]] = set()
    try:
        with output.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                agent = row.get("agent", "").strip()
                opponent = row.get("opponent", "").strip()
                if agent and opponent:
                    pairs.add(_pair_key(agent, opponent))
    except OSError:
        return set()

    return pairs


def _build_player_specs(model_paths: Sequence[Path]) -> list[PlayerSpec]:
    """Collect all playable agents for evaluation."""
    specs: list[PlayerSpec] = [
        PlayerSpec(name="RandomPlayer", builder=lambda _side: RandomPlayer("RandomPlayer")),
        PlayerSpec(name="MCTSPlayer", builder=lambda side: MCTSPlayer(side)),
        PlayerSpec(name="AlphaBetaPlayer", builder=lambda side: AlphaBetaPlayer(side)),
    ]

    for path in model_paths:
        specs.append(
            PlayerSpec(
                name=path.name,
                builder=lambda side, model_path=path: _build_rl_player(model_path, side),
            )
        )

    return specs


def play_matchup(
    player_a_builder: Callable[[str], Player],
    player_b_builder: Callable[[str], Player],
    games_per_side: int,
    mode: int = 2,
) -> Dict[str, int]:
    """
    Run color-balanced games between the provided builders and aggregate results.

    Each matchup plays `games_per_side` with player A as White/player B as Black,
    and the same number with colors swapped.
    """
    total_games = max(0, games_per_side) * 2
    results = {"agent_wins": 0, "opponent_wins": 0, "draws": 0, "games": total_games}

    for _ in range(games_per_side):
        winner = play_single_game(player_a_builder("white"), player_b_builder("black"), mode=mode)
        if winner == "white":
            results["agent_wins"] += 1
        elif winner == "black":
            results["opponent_wins"] += 1
        else:
            results["draws"] += 1

    for _ in range(games_per_side):
        winner = play_single_game(player_b_builder("white"), player_a_builder("black"), mode=mode)
        if winner == "white":
            results["opponent_wins"] += 1
        elif winner == "black":
            results["agent_wins"] += 1
        else:
            results["draws"] += 1

    return results


def evaluate(models: Sequence[Path], games: int, output: Path) -> None:
    """Evaluate all available players against each other and record results with balanced colors."""
    player_specs = _build_player_specs(models)
    if len(player_specs) < 2:
        raise SystemExit("Need at least two players to evaluate.")

    existing_pairs = _load_existing_pairs(output)
    rows: list[dict[str, object]] = []
    for spec_a, spec_b in combinations(player_specs, 2):
        matchup_key = _pair_key(spec_a.name, spec_b.name)
        if matchup_key in existing_pairs:
            logger.info("Skipping existing matchup: %s vs %s", spec_a.name, spec_b.name)
            continue

        total_games = games * 2
        logger.info(
            "Evaluating matchup: %s vs %s (%d per color, %d total)",
            spec_a.name,
            spec_b.name,
            games,
            total_games,
        )
        res = play_matchup(spec_a.builder, spec_b.builder, games, mode=2)
        games_played = res.get("games", total_games)
        win_rate = res["agent_wins"] / games_played if games_played else 0.0
        row = {
            "agent": spec_a.name,
            "opponent": spec_b.name,
            "games": games_played,
            "agent_wins": res["agent_wins"],
            "opponent_wins": res["opponent_wins"],
            "draws": res["draws"],
            "win_rate": f"{win_rate:.3f}",
        }
        rows.append(row)
        logger.info("Results for %s vs %s: %s", spec_a.name, spec_b.name, row)

    if rows:
        _write_results(output, rows)
        logger.info("Wrote results for %d matchup(s) to %s", len(rows), output)
    else:
        logger.warning("No results to write (no new matchups to evaluate?).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all available players against each other (mode 2).")
    parser.add_argument(
        "--models",
        type=Path,
        nargs="*",
        help="Model files or directories (recursively searched for .zip). Defaults to data/models/ppo_pferdeapfel.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Deprecated: single model path (file or directory). Use --models instead.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Number of games per color for each matchup (total games doubled).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("src/evaluation/results.csv"), help="CSV path for aggregated results."
    )
    args = parser.parse_args()

    input_paths: list[Path] = []
    if args.models:
        input_paths.extend(args.models)
    if args.model:
        input_paths.append(args.model)

    model_paths = discover_models(input_paths or None)

    evaluate(model_paths, args.games, args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    main()
