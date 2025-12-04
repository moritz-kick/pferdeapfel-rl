#!/usr/bin/env python3
"""Test PPO vs itself to verify color symmetry hypothesis."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.game.board import Board
from src.game.rules import Rules
from src.game.game import Game
from src.players.random import RandomPlayer
from src.players.rl.ppo_player import PPOPlayer


def main():
    model_path = "data/models/ppo_self_play/best_model/best_model.zip"

    print("Testing PPO vs itself (same model as both white and black)")
    print("=" * 60)

    for mode in [1, 2, 3]:
        results = {"white_wins": 0, "black_wins": 0, "draws": 0}

        for _ in range(20):
            p1 = PPOPlayer("white", model_path)
            p2 = PPOPlayer("black", model_path)
            game = Game(p1, p2, mode=mode, logging=False)

            while not game.game_over:
                current = game.get_current_player()
                legal = game.get_legal_moves()
                if not legal:
                    break
                move, apple = current.get_move(game.board, legal)
                if move in legal:
                    game.make_move(move, apple)
                else:
                    break

            if game.winner == "white":
                results["white_wins"] += 1
            elif game.winner == "black":
                results["black_wins"] += 1
            else:
                results["draws"] += 1

        total = results["white_wins"] + results["black_wins"] + results["draws"]
        print(f"\nMode {mode} (20 games):")
        print(f"  White wins: {results['white_wins']}/20 = {results['white_wins'] / 20:.1%}")
        print(f"  Black wins: {results['black_wins']}/20 = {results['black_wins'] / 20:.1%}")

        # Show the asymmetry
        if results["white_wins"] > results["black_wins"] + 4:
            print("  -> White-favored strategy!")
        elif results["black_wins"] > results["white_wins"] + 4:
            print("  -> Black-favored strategy!")
        else:
            print("  -> Relatively balanced")


if __name__ == "__main__":
    main()
