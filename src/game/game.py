"""Game controller that manages game state and player turns."""

from __future__ import annotations

import logging
import random
import toon_python
from pathlib import Path
from typing import Any, Optional

from src.game.board import Board
from src.game.rules import Rules
from src.players.base import Player

logger = logging.getLogger(__name__)


class Game:
    """Manages a game of Pferdeäpfel."""

    def __init__(self, white_player: Player, black_player: Player, mode: int = 3, logging: bool = False) -> None:
        """Initialize a new game."""
        self.board = Board(mode=mode)
        self.white_player = white_player
        self.black_player = black_player
        self.logging = logging
        self.log_data: list[dict[str, Any]] = []
        self.current_player: str = random.choice(["white", "black"])
        self.starting_player = self.current_player
        self.winner: Optional[str] = None
        self.game_over = False

    def get_current_player(self) -> Player:
        """Get the Player object for the current player."""
        if self.current_player == "white":
            return self.white_player
        return self.black_player

    def switch_turn(self) -> None:
        """Switch to the other player's turn."""
        self.current_player = "black" if self.current_player == "white" else "white"

    def make_move(self, move_to: tuple[int, int], extra_apple: Optional[tuple[int, int]] = None) -> bool:
        """
        Make a move for the current player.

        Returns:
            True if move was successful, False otherwise
        """
        logger.info(f"make_move called: player={self.current_player}, move_to={move_to}, extra_apple={extra_apple}")

        if self.game_over:
            logger.warning("Attempted move on game that's already over")
            return False

        success = Rules.make_move(self.board, self.current_player, move_to, extra_apple)
        logger.info(f"Rules.make_move returned: {success}")

        if success:
            # Log move if logging is enabled
            if self.logging:
                self.log_data.append(
                    {
                        "turn": self.current_player,
                        "move_to": move_to,
                        "extra_apple": extra_apple,
                        "white_pos": self.board.white_pos,
                        "black_pos": self.board.black_pos,
                        "brown_remaining": self.board.brown_apples_remaining,
                        "golden_remaining": self.board.golden_apples_remaining,
                    }
                )

            # Check win condition
            self.winner = Rules.check_win_condition(self.board, last_mover=self.current_player)
            if self.winner:
                logger.info(f"Game over! Winner: {self.winner}")
                self.game_over = True
            else:
                logger.info(f"Switching turn from {self.current_player}")
                self.switch_turn()
                logger.info(f"Now it's {self.current_player}'s turn")

        return success

    def undo_move(self) -> bool:
        """Undo the last move. Returns True if successful."""
        if not self.board.move_history:
            return False

        state = self.board.move_history.pop()
        self.board.white_pos = state["white_pos"]
        self.board.black_pos = state["black_pos"]
        self.board.grid = state["grid"]
        self.board.brown_apples_remaining = state["brown_apples_remaining"]
        self.board.golden_apples_remaining = state["golden_apples_remaining"]
        self.board.golden_phase_started = state["golden_phase_started"]

        # Remove last log entry if logging
        if self.logging and self.log_data:
            self.log_data.pop()

        # Switch turn back
        self.switch_turn()

        # Reset game over state
        self.winner = None
        self.game_over = False

        return True

    def get_legal_moves(self) -> list[tuple[int, int]]:
        """Get legal moves for the current player."""
        logger.debug(f"get_legal_moves for {self.current_player}")
        legal_moves = Rules.get_legal_knight_moves(self.board, self.current_player)
        logger.debug(f"Found {len(legal_moves)} legal moves for {self.current_player}")

        if not legal_moves and not self.game_over:
            # Current player is stuck – determine the winner immediately
            logger.info(f"{self.current_player} has no legal moves - game ending")
            self.winner = Rules.check_win_condition(self.board)
            self.game_over = True
            logger.info(f"Winner: {self.winner}")

        return legal_moves

    def save_log(self, log_dir: Path) -> Optional[Path]:
        """Save game log to a TOON file. Returns the path if successful."""
        if not self.logging or not self.log_data:
            return None

        # Find next available log file number
        log_dir.mkdir(parents=True, exist_ok=True)
        log_num = 1
        while (log_dir / f"game_{log_num:03d}.toon").exists():
            log_num += 1

        log_path = log_dir / f"game_{log_num:03d}.toon"
        log_content = {
            "white_player": self.white_player.name,
            "black_player": self.black_player.name,
            "starting_player": self.starting_player,
            "winner": self.winner,
            "moves": self.log_data,
        }

        with open(log_path, "w") as f:
            f.write(toon_python.encode(log_content))

        return log_path
