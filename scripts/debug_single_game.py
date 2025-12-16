"""Find and visualize a single debug game between two specific players.

This script:
- Lets you pick white and black player classes.
- Repeatedly plays games until a desired winner (player + color) is achieved.
- Logs ONLY the matching game to a TOON file.
- Visualizes ONLY that matching game move by move on the terminal.
- Prints two special final-board visualizations:
  1) Global move numbers where one player has all odd and the other all even.
  2) Per-player move numbers starting from 1 for each player, using colors.

Spacing is chosen so 1- and 2-digit numbers keep the board aligned.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.discovery import discover_players
from src.game.game import Game
from src.game.board import Board
from src.players.base import Player


Coord = Tuple[int, int]
LogEntry = Dict[str, Any]


def _discover_player_map() -> Dict[str, Type[Player]]:
    """Return a mapping from class name -> Player subclass."""
    classes = discover_players()
    return {cls.__name__: cls for cls in classes}


def _make_player(player_map: Dict[str, Type[Player]], name: str, color: str) -> Player:
    if name not in player_map:
        raise ValueError(f"Unknown player class '{name}'. Available: {sorted(player_map.keys())}")
    cls = player_map[name]
    # Pass the color string as the player name (consistent with other scripts)
    return cls(color)


def print_simple_board(board: Board) -> None:
    """Simple ASCII board (single game, per-move)."""
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

    print("   " + " ".join(str(c) for c in range(8)))
    for r in range(8):
        print(f"{r:2} " + " ".join(chars[r]))
    print()


def _build_move_maps(log_data: List[LogEntry]) -> Tuple[List[List[Optional[int]]], Dict[str, List[List[Optional[int]]]]]:
    """
    Build:
    - odd_even_map[row][col] = global move index (1,2,3,...) for the player who moved there
    - per_player_map[color][row][col] = per-player move index (1,2,3,...) for that color
    """
    size = 8
    odd_even_map: List[List[Optional[int]]] = [[None for _ in range(size)] for _ in range(size)]
    per_player_map: Dict[str, List[List[Optional[int]]]] = {
        "white": [[None for _ in range(size)] for _ in range(size)],
        "black": [[None for _ in range(size)] for _ in range(size)],
    }

    move_index = 1
    per_color_counter = {"white": 1, "black": 1}

    for entry in log_data:
        turn = entry.get("turn")
        move_to: Coord = entry.get("move_to")
        if turn not in ("white", "black") or move_to is None:
            continue
        r, c = move_to
        if not (0 <= r < size and 0 <= c < size):
            continue

        # Global numbering (odd/even naturally alternate by move_index)
        odd_even_map[r][c] = move_index

        # Per-player numbering
        idx = per_color_counter[turn]
        per_player_map[turn][r][c] = idx
        per_color_counter[turn] = idx + 1

        move_index += 1

    return odd_even_map, per_player_map


def _print_numbered_board_odd_even(odd_even_map: List[List[Optional[int]]]) -> None:
    """Final board: global move numbers, one player gets odd and the other even implicitly."""
    size = 8
    print("\nFinal board with global move numbers (odd/even by player):")
    # Column header
    print("    " + "  ".join(f"{c:2d}" for c in range(size)))
    for r in range(size):
        row_cells: List[str] = []
        for c in range(size):
            n = odd_even_map[r][c]
            cell = "." if n is None else str(n)
            # width 2 to keep alignment for 1-2 digit numbers
            row_cells.append(f"{cell:>2}")
        print(f"{r:2d}  " + "  ".join(row_cells))


def _print_numbered_board_colored(per_player_map: Dict[str, List[List[Optional[int]]]]) -> None:
    """
    Final board: per-player move numbers starting from 1 for each.
    Uses ANSI colors to distinguish players while keeping spacing stable.
    """
    size = 8
    WHITE_COLOR = "\033[97m"  # bright white
    BLACK_COLOR = "\033[94m"  # blue
    RESET = "\033[0m"

    print("\nFinal board with per-player move numbers (both start at 1, colored):")
    # Column header (no color)
    print("    " + "  ".join(f"{c:2d}" for c in range(size)))

    for r in range(size):
        row_str_parts: List[str] = []
        for c in range(size):
            n_white = per_player_map["white"][r][c]
            n_black = per_player_map["black"][r][c]

            # A square should belong to at most one player in valid games,
            # but if both ended up there, prefer showing both markers compactly.
            if n_white is not None and n_black is None:
                raw = f"{n_white:>2}"
                colored = f"{WHITE_COLOR}{raw}{RESET}"
            elif n_black is not None and n_white is None:
                raw = f"{n_black:>2}"
                colored = f"{BLACK_COLOR}{raw}{RESET}"
            elif n_white is not None and n_black is not None:
                # Degenerate case; show 'X ' to keep width 2
                raw = "X "
                colored = f"{WHITE_COLOR}{raw}{RESET}"
            else:
                raw = " ."
                colored = raw

            row_str_parts.append(colored)

        print(f"{r:2d}  " + "  ".join(row_str_parts))


def run_single_game(
    white_class: str,
    black_class: str,
    mode: int,
    target_color: Optional[str],
    target_class: Optional[str],
    max_moves: int = 1000,
    per_move_sleep: float = 0.3,
) -> Tuple[Game, int]:
    """
    Play a single game between the given player classes.

    Returns:
        (game, move_count)
    """
    player_map = _discover_player_map()
    white_player = _make_player(player_map, white_class, "white")
    black_player = _make_player(player_map, black_class, "black")

    game = Game(white_player, black_player, mode=mode, logging=True)

    move_count = 0
    print(f"\nStarting game: {white_class} (white) vs {black_class} (black).")
    print(f"{game.current_player.capitalize()} to move first.\n")
    print_simple_board(game.board)

    while not game.game_over and move_count < max_moves:
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            break

        current_player = game.get_current_player()
        try:
            move_to, extra_apple = current_player.get_move(game.board, legal_moves)
        except Exception as e:
            print(f"Error during move by {current_player.name}: {e}")
            break

        success = game.make_move(move_to, extra_apple)
        if not success:
            print(f"Invalid move attempted by {current_player.name}: {move_to}, extra={extra_apple}")
            break

        move_count += 1
        print(f"Move {move_count}: {current_player.name} -> {move_to}, extra={extra_apple}")
        print_simple_board(game.board)
        if per_move_sleep > 0:
            time.sleep(per_move_sleep)

    # Determine whether this game matches the desired win condition.
    # If no target is specified, any finished game is acceptable.
    matched = False
    if game.game_over:
        winner_color = game.winner
        winner_class: Optional[str] = None
        if winner_color == "white":
            winner_class = white_class
        elif winner_color == "black":
            winner_class = black_class

        if target_color is None and target_class is None:
            matched = True
        else:
            color_ok = (target_color is None) or (winner_color == target_color)
            class_ok = (target_class is None) or (winner_class == target_class)
            matched = bool(color_ok and class_ok)

    if not matched:
        print("\nGame finished but did not match desired win condition.")

    return game, move_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find a single debug game between two specific players, "
            "log it, visualize it move-by-move, and print special final-board views."
        )
    )
    parser.add_argument("--white", type=str, required=True, help="White player class name (e.g. HeuristicPlayerV2)")
    parser.add_argument("--black", type=str, required=True, help="Black player class name (e.g. RandomPlayer)")
    parser.add_argument(
        "--mode",
        type=int,
        default=2,
        help="Game mode (default: 2)",
    )
    parser.add_argument(
        "--target-color",
        type=str,
        choices=["white", "black"],
        default=None,
        help="Required winner color (white/black). If omitted, any winner is accepted.",
    )
    parser.add_argument(
        "--target-class",
        type=str,
        default=None,
        help="Required winner player class name (e.g. HeuristicPlayerV2). If omitted, any class is accepted.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=100,
        help="Maximum number of games to try before giving up (default: 100).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.3,
        help="Seconds to sleep between moves for readability (default: 0.3).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="data/logs/debug_games",
        help="Directory to store the matching game's TOON log (default: data/logs/debug_games).",
    )

    args = parser.parse_args()

    attempts = 0
    matched_game: Optional[Game] = None
    matched_moves = 0

    while attempts < args.max_games:
        attempts += 1
        print(f"\n=== Attempt {attempts}/{args.max_games} ===")
        game, move_count = run_single_game(
            white_class=args.white,
            black_class=args.black,
            mode=args.mode,
            target_color=args.target_color,
            target_class=args.target_class,
            max_moves=1000,
            per_move_sleep=args.sleep,
        )

        if game.game_over:
            winner_color = game.winner
            print(f"\nGame over after {move_count} moves. Winner: {winner_color}")

        # Check whether this attempt matched the desired condition
        winner_color = game.winner
        winner_class: Optional[str] = None
        if winner_color == "white":
            winner_class = args.white
        elif winner_color == "black":
            winner_class = args.black

        color_ok = (args.target_color is None) or (winner_color == args.target_color)
        class_ok = (args.target_class is None) or (winner_class == args.target_class)
        if game.game_over and color_ok and class_ok:
            matched_game = game
            matched_moves = move_count
            break
        else:
            print("Desired winner not achieved in this game; trying another (without visualizing previous ones).")

    if matched_game is None:
        print("\nNo matching game found within the attempt limit.")
        sys.exit(1)

    # Save log for the matched game
    log_dir = Path(args.log_dir)
    log_path = matched_game.save_log(log_dir)
    if log_path:
        print(f"\nMatching game logged to {log_path}")

    # Final position + winner
    print(f"\nFINAL RESULT: Winner is {matched_game.winner}")
    print_simple_board(matched_game.board)

    # Build and print the two requested final-board visualizations
    odd_even_map, per_player_map = _build_move_maps(matched_game.log_data)
    _print_numbered_board_odd_even(odd_even_map)
    _print_numbered_board_colored(per_player_map)


if __name__ == "__main__":
    main()


