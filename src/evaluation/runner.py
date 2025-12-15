"""Game runner module for evaluations."""

import time
from typing import Type

from src.evaluation.storage import GameResult
from src.game.game import Game
from src.players.base import Player


class GameRunner:
    """Runs games between players for evaluation."""

    def run_game(
        self, mode: int, white_cls: Type[Player], black_cls: Type[Player], white_name: str, black_name: str
    ) -> GameResult:
        """
        Run a single game.

        Args:
            mode: Game mode (1, 2, 3)
            white_cls: Class for white player
            black_cls: Class for black player
            white_name: Name instance for white
            black_name: Name instance for black

        Returns:
            GameResult object
        """
        # Instantiate players
        # Fix: Players expect 'white' or 'black' as name/color, but we want to track agent names.
        # We instantiate with color, then update name if needed (though players often use name for logic).
        # Actually, Minimax uses self.name for logic.
        # So we MUST pass "white" and "black".
        p1 = white_cls("white")
        p2 = black_cls("black")

        # Start game
        game = Game(p1, p2, mode=mode, logging=False)
        # Note: We disable internal game logging to separate log files,
        # or we could keep it if we want replays.
        # For evaluation, we mainly care about the result metrics.

        start_time = time.time()

        # Run game loop
        # We can use game.play() if it exists or manual loop
        # The existing Game class doesn't seem to have a blocking "play full game" method,
        # it usually relies on external loop or GUI.
        # Let's write a simple loop here.

        metadata = {"white_class": white_cls.__name__, "black_class": black_cls.__name__}

        # Track player-specific metrics before game starts
        if hasattr(p1, "nodes_evaluated"):
            p1.nodes_evaluated = 0
        if hasattr(p2, "nodes_evaluated"):
            p2.nodes_evaluated = 0

        try:
            while not game.game_over:
                # We can enforce a max moves limit to prevent infinite loops in buggy bots
                if len(game.board.move_history) > 200:
                    game.game_over = True
                    game.winner = "draw"
                    metadata["termination"] = "max_moves_exceeded"
                    break

                current_player = game.get_current_player()
                legal_moves = game.get_legal_moves()

                if not legal_moves:
                    # Should be handled by game logic (checkmate/stalemate), but just in case
                    break

                move_to, extra = current_player.get_move(game.board, legal_moves)

                success = game.make_move(move_to, extra)
                if not success:
                    # Illegal move logic
                    # Game usually handles this by ignoring, but for eval we might want to DQ?
                    # Current Game implementation prints error and returns False.
                    # We should probably count this as a loss or retry limit.
                    # For now, let's treat it as a loss for the current player?
                    # Or just break loop.
                    game.game_over = True
                    # If P1 made illegal move, P2 wins
                    game.winner = "black" if current_player == p1 else "white"
                    metadata["error"] = f"Illegal move by {current_player.name}: {move_to}, {extra}"
                    break

            winner = game.winner if game.winner else "draw"

            # Determine termination reason
            if winner != "draw":
                if game.board.white_pos == game.board.black_pos:
                    metadata["termination"] = "capture"
                else:
                    metadata["termination"] = "stuck"
            else:
                metadata["termination"] = "draw"

        except Exception as e:
            # Crash = Loss
            import traceback

            traceback.print_exc()
            winner = "draw"  # default
            if game.current_player == "white":
                winner = "black"
                metadata["white_error"] = str(e)
            else:
                winner = "white"
                metadata["black_error"] = str(e)

        duration = time.time() - start_time

        # Collect player-specific metadata after game
        if hasattr(p1, "nodes_evaluated"):
            metadata["white_nodes_evaluated"] = p1.nodes_evaluated
        if hasattr(p1, "last_search_depth"):
            metadata["white_search_depth"] = p1.last_search_depth
        if hasattr(p2, "nodes_evaluated"):
            metadata["black_nodes_evaluated"] = p2.nodes_evaluated
        if hasattr(p2, "last_search_depth"):
            metadata["black_search_depth"] = p2.last_search_depth

        return GameResult(
            timestamp=start_time,
            mode=mode,
            white_player=white_name,
            black_player=black_name,
            winner=winner,
            moves=len(game.board.move_history),
            duration=duration,
            white_error=metadata.get("white_error"),
            black_error=metadata.get("black_error"),
            metadata=metadata,
        )
