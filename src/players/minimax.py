"""Classical minimax/alpha-beta search player."""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

from src.game.board import Board
from src.game.rules import Rules
from src.players.base import Player

Move = Tuple[int, int]
Action = Tuple[Move, Optional[Move]]


class AlphaBetaPlayer(Player):
    """Player that selects moves via minimax search with alpha-beta pruning."""

    DISPLAY_NAME = "alphabeta"

    def __init__(self, side: str, depth: int = 3, time_limit: Optional[float] = None) -> None:
        """
        Initialize a search player.

        Args:
            side: "white" or "black".
            depth: Maximum search depth (plies).
            time_limit: Optional wall-clock limit per move in seconds.
        """
        side_clean = side.lower()
        if side_clean not in ("white", "black"):
            raise ValueError("side must be 'white' or 'black'")

        super().__init__(f"{side_clean.capitalize()} AlphaBeta")
        self.side = side_clean
        self.depth = max(1, depth)
        self.time_limit = time_limit

    def get_move(self, board: Board, legal_moves: list[Move]) -> tuple[Move, Optional[Move]]:
        """Choose the best move using alpha-beta search."""
        if not legal_moves:
            raise ValueError("No legal moves available for AlphaBeta player.")

        actions = self._generate_actions(board, self.side, legal_moves)
        if not actions:
            raise ValueError("No valid actions available for AlphaBeta search.")

        start_time = time.time()
        best_action = actions[0]
        best_value = -math.inf

        for action in actions:
            board_after, next_player, winner = self._apply_action(board, self.side, action)
            if board_after is None:
                continue

            if winner:
                value = self._terminal_value(winner)
            else:
                value = self._alphabeta(
                    board_after,
                    player=next_player,
                    depth=self.depth - 1,
                    alpha=-math.inf,
                    beta=math.inf,
                    last_mover=self.side,
                    start_time=start_time,
                )

            if value > best_value:
                best_value = value
                best_action = action

            if self._time_exceeded(start_time):
                break

        return best_action

    def _alphabeta(
        self,
        board: Board,
        player: str,
        depth: int,
        alpha: float,
        beta: float,
        last_mover: Optional[str],
        start_time: float,
    ) -> float:
        """Recursive alpha-beta search."""
        winner = Rules.check_win_condition(board, last_mover=last_mover)
        if winner:
            return self._terminal_value(winner)

        if depth == 0 or self._time_exceeded(start_time):
            return self._evaluate_board(board)

        actions = self._generate_actions(board, player)
        if not actions:
            terminal = Rules.check_win_condition(board, last_mover=last_mover)
            if terminal:
                return self._terminal_value(terminal)
            return self._evaluate_board(board)

        maximizing = player == self.side
        if maximizing:
            value = -math.inf
            for action in actions:
                child_board, next_player, winner = self._apply_action(board, player, action)
                if child_board is None:
                    continue
                score = self._terminal_value(winner) if winner else self._alphabeta(
                    child_board,
                    player=next_player,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    last_mover=player,
                    start_time=start_time,
                )
                value = max(value, score)
                alpha = max(alpha, value)
                if beta <= alpha or self._time_exceeded(start_time):
                    break
            return value if value != -math.inf else self._evaluate_board(board)

        value = math.inf
        for action in actions:
            child_board, next_player, winner = self._apply_action(board, player, action)
            if child_board is None:
                continue
            score = self._terminal_value(winner) if winner else self._alphabeta(
                child_board,
                player=next_player,
                depth=depth - 1,
                alpha=alpha,
                beta=beta,
                last_mover=player,
                start_time=start_time,
            )
            value = min(value, score)
            beta = min(beta, value)
            if beta <= alpha or self._time_exceeded(start_time):
                break
        return value if value != math.inf else self._evaluate_board(board)

    def _evaluate_board(self, board: Board) -> float:
        """Heuristic evaluation of a non-terminal state from self.side perspective."""
        my_moves = len(Rules.get_legal_knight_moves(board, self.side))
        opp_moves = len(Rules.get_legal_knight_moves(board, self._opponent(self.side)))

        mobility_score = (my_moves - opp_moves) * 10.0

        my_pos = board.get_horse_position(self.side)
        opp_pos = board.get_horse_position(self._opponent(self.side))
        centrality_score = self._centrality(my_pos) - self._centrality(opp_pos)

        return mobility_score + centrality_score

    def _generate_actions(
        self, board: Board, player: str, legal_moves: Optional[List[Move]] = None
    ) -> List[Action]:
        """Enumerate legal (move, extra_apple) actions for a player."""
        moves = legal_moves if legal_moves is not None else Rules.get_legal_knight_moves(board, player)
        if not moves:
            return []

        if board.mode == 2:
            return [(move, None) for move in moves]

        empties = [(r, c) for r in range(Board.BOARD_SIZE) for c in range(Board.BOARD_SIZE) if board.is_empty(r, c)]
        actions: list[Action] = []

        for move in moves:
            extras: list[Optional[Move]] = []
            if board.mode == 3:
                extras.append(None)

            if board.mode in (1, 3):
                for empty in empties:
                    if board.mode == 1 and empty == move:
                        continue
                    extras.append(empty)

            for extra in extras:
                board_copy = board.copy()
                if Rules.make_move(board_copy, player, move, extra):
                    actions.append((move, extra))

        return actions

    def _apply_action(self, board: Board, player: str, action: Action) -> tuple[Optional[Board], str, Optional[str]]:
        """Apply an action to a copied board and return successor info."""
        board_copy = board.copy()
        move_to, extra = action
        if not Rules.make_move(board_copy, player, move_to, extra):
            return None, player, None
        winner = Rules.check_win_condition(board_copy, last_mover=player)
        return board_copy, self._opponent(player), winner

    def _centrality(self, pos: Tuple[int, int]) -> float:
        """Score that prefers central squares for better mobility."""
        row, col = pos
        center = (Board.BOARD_SIZE - 1) / 2
        return -(abs(row - center) + abs(col - center))

    def _terminal_value(self, winner: str) -> float:
        if winner == "draw":
            return 0.0
        if winner == self.side:
            return 1_000_000.0
        return -1_000_000.0

    def _time_exceeded(self, start_time: float) -> bool:
        return self.time_limit is not None and (time.time() - start_time) >= self.time_limit

    @staticmethod
    def _opponent(player: str) -> str:
        return "black" if player == "white" else "white"


__all__ = ["AlphaBetaPlayer"]
