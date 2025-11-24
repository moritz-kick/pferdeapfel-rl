"""Mobility-aware greedy player covering all Pferdeäpfel modes."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

from src.game.board import Board
from src.game.rules import Rules
from src.players.base import Player
from src.players.random import RandomPlayer


class GreedyPlayer(Player):
    """
    Greedy heuristic player that evaluates move + apple combinations.

    - Always captures when a legal capture is available.
    - Chooses apple placements (mandatory or optional) that reduce the opponent's
      future mobility whenever possible.
    - Falls back to the random player only if no evaluated action is feasible.
    """

    DISPLAY_NAME = "greedy"

    def __init__(self, color: str) -> None:
        super().__init__(color.capitalize())
        normalized = color.lower()
        self.color = normalized if normalized in {"white", "black"} else "white"
        self._fallback_random = RandomPlayer(self.name)

    # --------------------------------------------------------------------- API
    def get_move(
        self, board: Board, legal_moves: list[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
        if not legal_moves:
            raise ValueError("GreedyPlayer received no legal moves")

        if board.mode == 1:
            return self._play_mode1(board, legal_moves)
        if board.mode == 2:
            return self._play_mode2(board, legal_moves)
        if board.mode == 3:
            return self._play_mode3(board, legal_moves)
        return self._fallback_random.get_move(board, legal_moves)

    # ----------------------------------------------------------------- Helpers
    def _opponent(self) -> str:
        return "black" if self.color == "white" else "white"

    def _move_is_capture(self, board: Board, move: Tuple[int, int]) -> bool:
        opponent_pos = board.black_pos if self.color == "white" else board.white_pos
        return move == opponent_pos

    def _empty_squares(self, board: Board) -> Iterable[Tuple[int, int]]:
        for row in range(Board.BOARD_SIZE):
            for col in range(Board.BOARD_SIZE):
                if board.is_empty(row, col):
                    yield (row, col)

    def _simulate_turn(
        self, board: Board, move: Tuple[int, int], extra_apple: Optional[Tuple[int, int]]
    ) -> Optional[Board]:
        board_copy = board.copy()
        success = Rules.make_move(board_copy, self.color, move, extra_apple)
        if not success:
            return None
        return board_copy

    def _score_board(self, board_after: Board, opp_baseline: Optional[int]) -> float:
        winner = Rules.check_win_condition(board_after, last_mover=self.color)
        if winner == self.color:
            return math.inf
        if winner and winner != "draw":
            return -math.inf

        opponent = self._opponent()
        my_mobility = len(Rules.get_legal_knight_moves(board_after, self.color))
        opp_mobility = len(Rules.get_legal_knight_moves(board_after, opponent))

        score = (my_mobility - opp_mobility) + 0.1 * my_mobility
        if opp_baseline is not None:
            score += 0.05 * (opp_baseline - opp_mobility)

        if board_after.mode == 3:
            if board_after.golden_phase_started:
                score += 5.0 if self.color == "white" else -5.0
            if opponent == "white" and not Rules.can_player_move(board_after, opponent):
                score += 3.0  # Black immobilized White

        return score

    def _score_action(
        self,
        board: Board,
        move: Tuple[int, int],
        extra_apple: Optional[Tuple[int, int]],
        opp_baseline: Optional[int],
    ) -> Optional[float]:
        simulated = self._simulate_turn(board, move, extra_apple)
        if simulated is None:
            return None
        return self._score_board(simulated, opp_baseline)

    # --------------------------------------------------------------- Mode logic
    def _play_mode2(
        self, board: Board, legal_moves: list[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
        for move in legal_moves:
            if self._move_is_capture(board, move):
                return move, None

        opp_baseline = len(Rules.get_legal_knight_moves(board, self._opponent()))
        best_score = -math.inf
        best_move: Optional[Tuple[int, int]] = None

        for move in legal_moves:
            score = self._score_action(board, move, None, opp_baseline)
            if score is None:
                continue
            if score > best_score:
                best_score = score
                best_move = move

        if best_move is not None:
            return best_move, None
        return self._fallback_random.get_move(board, legal_moves)

    def _play_mode1(
        self, board: Board, legal_moves: list[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
        opponent_moves = set(Rules.get_legal_knight_moves(board, self._opponent()))
        placements = list(self._empty_squares(board))
        if not placements:
            return self._fallback_random.get_move(board, legal_moves)

        prioritized = [sq for sq in placements if sq in opponent_moves]
        others = [sq for sq in placements if sq not in opponent_moves]
        candidate_placements = prioritized + others

        opp_baseline = len(opponent_moves)
        best_score = -math.inf
        best_action: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None

        for placement in candidate_placements:
            for move in legal_moves:
                score = self._score_action(board, move, placement, opp_baseline)
                if score is None:
                    continue
                if math.isinf(score) and score > 0:
                    return move, placement
                if score > best_score:
                    best_score = score
                    best_action = (move, placement)

        if best_action:
            return best_action
        return self._fallback_random.get_move(board, legal_moves)

    def _play_mode3(
        self, board: Board, legal_moves: list[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
        if self.color == "black":
            for move in legal_moves:
                if self._move_is_capture(board, move):
                    return move, None

        opponent = self._opponent()
        opp_baseline = len(Rules.get_legal_knight_moves(board, opponent))
        best_score = -math.inf
        best_action: Optional[Tuple[Tuple[int, int], Optional[Tuple[int, int]]]] = None

        for move in legal_moves:
            base_board = self._simulate_turn(board, move, None)
            if base_board is None:
                continue
            base_score = self._score_board(base_board, opp_baseline)
            if base_score > best_score:
                best_score = base_score
                best_action = (move, None)

            opponent_moves_after = Rules.get_legal_knight_moves(base_board, opponent)
            for target in opponent_moves_after:
                if base_board.get_square(*target) != Board.EMPTY:
                    continue
                score = self._score_action(board, move, target, opp_baseline)
                if score is None:
                    continue
                if math.isinf(score) and score > 0:
                    return move, target
                if score > best_score:
                    best_score = score
                    best_action = (move, target)

        if best_action:
            return best_action
        return self._fallback_random.get_move(board, legal_moves)

