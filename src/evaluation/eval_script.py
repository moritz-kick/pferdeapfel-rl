"""Simple evaluation script: RL agent vs Random player in mode 2."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict

from src.game.game import Game
from src.players.random import RandomPlayer

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from src.players.rl.dqn_rl import DQNPlayer
except Exception as exc:  # pragma: no cover - guard import
    DQNPlayer = None
    logger.warning("DQNPlayer unavailable: %s", exc)


def play_single_game(model_path: Path) -> str:
    """Play one game of Random (White) vs RL (Black)."""
    if DQNPlayer is None:
        raise ImportError("DQNPlayer not available. Install stable-baselines3 and its dependencies.")

    white = RandomPlayer("White")
    black = DQNPlayer("Black", model_path=model_path)

    game = Game(white, black, mode=2, logging=False)
    # Force White to start to align with rules
    game.current_player = "white"

    while not game.game_over:
        player = game.get_current_player()
        legal_moves = game.get_legal_moves()
        move_to, extra = player.get_move(game.board, legal_moves)
        game.make_move(move_to, extra)
    return game.winner or "draw"


def evaluate(model_path: Path, games: int, output: Path) -> Dict[str, int]:
    """Run evaluation games and persist aggregated results."""
    results = {"agent_wins": 0, "opponent_wins": 0, "draws": 0}
    for _ in range(games):
        winner = play_single_game(model_path)
        if winner == "black":
            results["agent_wins"] += 1
        elif winner == "white":
            results["opponent_wins"] += 1
        else:
            results["draws"] += 1

    win_rate = results["agent_wins"] / games if games else 0.0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["agent", "opponent", "games", "agent_wins", "opponent_wins", "draws", "win_rate"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "agent": model_path.name,
                "opponent": "RandomPlayer",
                "games": games,
                "agent_wins": results["agent_wins"],
                "opponent_wins": results["opponent_wins"],
                "draws": results["draws"],
                "win_rate": f"{win_rate:.3f}",
            }
        )
    logger.info("Evaluation complete: %s", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RL agent vs Random.")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained RL model (.zip).")
    parser.add_argument("--games", type=int, default=10, help="Number of games to run.")
    parser.add_argument(
        "--output", type=Path, default=Path("src/evaluation/results.csv"), help="CSV path for aggregated results."
    )
    args = parser.parse_args()

    evaluate(args.model, args.games, args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
