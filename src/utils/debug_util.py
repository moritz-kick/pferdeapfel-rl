"""Debug utilities for logging incomplete games."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import toon_python

if TYPE_CHECKING:
    from src.game.game import Game


def write_debug_log(game: Game, log_dir: Path) -> Path:
    """
    Write a debug log for an incomplete game.

    Args:
        game: The game instance to log
        log_dir: Directory to save debug logs

    Returns:
        Path to the created debug log file
    """
    # Create debug log directory
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"debug_{timestamp}.toon"

    # Collect comprehensive game state
    debug_data = {
        "timestamp": timestamp,
        "game_state": {
            "current_player": game.current_player,
            "starting_player": game.starting_player,
            "game_over": game.game_over,
            "winner": game.winner,
            "mode": game.board.mode,
        },
        "board_state": {
            "white_pos": game.board.white_pos,
            "black_pos": game.board.black_pos,
            "brown_apples_remaining": game.board.brown_apples_remaining,
            "golden_apples_remaining": game.board.golden_apples_remaining,
            "golden_phase_started": game.board.golden_phase_started,
            "grid": game.board.grid.tolist(),  # Convert numpy array to list
        },
        "players": {
            "white": type(game.white_player).__name__,
            "black": type(game.black_player).__name__,
        },
        "moves": game.log_data,
        "move_count": len(game.log_data),
    }

    # Add legal moves for debugging
    from src.game.rules import Rules

    debug_data["legal_moves"] = {
        "white": Rules.get_legal_knight_moves(game.board, "white"),
        "black": Rules.get_legal_knight_moves(game.board, "black"),
    }

    # Write to file
    with open(log_path, "w") as f:
        f.write(toon_python.encode(debug_data))

    return log_path
