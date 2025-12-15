#!/usr/bin/env python
"""
Play a game with the Smart Minimax Player.
"""

import sys
import time
import argparse
import random
from typing import Optional

from src.game.game import Game
from src.game.board import Board
from src.players.minimax import MinimaxPlayer
from src.players.random import RandomPlayer
from src.debug.debug_env import SimpleDebugEnv  # Use for ASCII printing


def print_board(board: Board):
    chars = [["." for _ in range(8)] for _ in range(8)]
    for r in range(8):
        for c in range(8):
            val = board.grid[r, c]
            if val == Board.WHITE_HORSE:
                chars[r][c] = "W"
            elif val == Board.BLACK_HORSE:
                chars[r][c] = "B"
            elif val == Board.BROWN_APPLE:
                chars[r][c] = "o"
            elif val == Board.GOLDEN_APPLE:
                chars[r][c] = "G"

    print("  0 1 2 3 4 5 6 7")
    for r in range(8):
        print(f"{r} {' '.join(chars[r])}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Run Smart Player Game")
    parser.add_argument("--depth", type=int, default=8, help="Opening depth for Smart Player")
    parser.add_argument("--limit", type=int, default=10, help="Number of deep moves")
    parser.add_argument("--role", type=str, default="white", choices=["white", "black"], help="Role of Smart Player")
    args = parser.parse_args()

    print(f"Initializing Game (Mode 2)... Smart Player as {args.role.upper()}")

    if args.role == "white":
        p1 = MinimaxPlayer("white", opening_depth=args.depth, opening_limit=args.limit)
        p2 = RandomPlayer("black")
    else:
        p1 = RandomPlayer("white")
        p2 = MinimaxPlayer("black", opening_depth=args.depth, opening_limit=args.limit)

    game = Game(p1, p2, mode=2, logging=False)

    turn = 0
    while not game.game_over:
        turn += 1
        current_player = game.get_current_player()
        print(f"\n--- Turn {turn}: {current_player.name} ---")

        legal_moves = game.get_legal_moves()
        if not legal_moves:
            print(f"{current_player.name} has no moves!")
            break

        move, extra_apple = current_player.get_move(game.board, legal_moves)

        print(f"Move: {move}")
        success = game.make_move(move, extra_apple)

        if not success:
            print("CRITICAL: Move failed!")
            break

        print_board(game.board)
        # Sleep slightly for readability
        time.sleep(0.5)

    print(f"\nGAME OVER! Winner: {game.winner}")

    # Check if Smart player won
    smart_name = args.role
    if game.winner == smart_name:
        print("✅ Smart Player Won!")
    else:
        print("❌ Smart Player Lost (or Draw)")


if __name__ == "__main__":
    main()
