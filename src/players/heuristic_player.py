"""Heuristic player for Mode 2, focusing on speed and survival."""

from __future__ import annotations

import logging
import random
from typing import List, Optional, Tuple

from src.game.board import Board
from src.game.rules import Rules
from src.players.base import Player

logger = logging.getLogger(__name__)


class HeuristicPlayer(Player):
    """
    A fast, heuristic-based player for Mode 2.

    Strategy:
    1. Immediate win check (Capture).
    2. Avoid immediate loss (Getting stuck).
    3. Maximize Mobility (Number of legal moves).
    4. Minimize Opponent Mobility (Aggression).
    """

    def __init__(self, name: str, aggression: float = 0.5) -> None:
        super().__init__(name)
        self.aggression = aggression  # Weight for minimizing opponent moves [0.0, 1.0]

    def get_move(
        self, board: Board, legal_moves: List[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
        """Select the best move based on heuristics."""
        if not legal_moves:
            return (0, 0), None

        opponent = "black" if self.name == "white" else "white"
        opp_pos = board.get_horse_position(opponent)

        # Randomize order to break ties unpredictably
        random.shuffle(legal_moves)

        # 1. Immediate Win: Capture
        for move in legal_moves:
            if move == opp_pos:
                return move, None

        # First pass: Filter out moves that allow immediate capture
        # These are the worst moves and should be avoided even if all moves are bad
        safe_moves = []
        capture_allowing_moves = []
        
        for move in legal_moves:
            # Quick check: simulate move and see if opponent can capture
            sim_board = board.fast_copy()
            current_pos = board.get_horse_position(self.name)
            sim_board.grid[move[0], move[1]] = Board.WHITE_HORSE if self.name == "white" else Board.BLACK_HORSE
            sim_board._mark_occupied(move[0], move[1])
            sim_board.grid[current_pos[0], current_pos[1]] = Board.BROWN_APPLE
            sim_board._mark_occupied(current_pos[0], current_pos[1])
            if self.name == "white":
                sim_board.white_pos = move
            else:
                sim_board.black_pos = move
            
            opp_moves = Rules.get_legal_knight_moves(sim_board, opponent)
            if move in opp_moves:
                capture_allowing_moves.append(move)
            else:
                safe_moves.append(move)
        
        # Prefer safe moves even if they have low scores
        moves_to_evaluate = safe_moves if safe_moves else capture_allowing_moves
        
        best_move = moves_to_evaluate[0]
        best_score = float("-inf")

        for move in moves_to_evaluate:
            # Simulate move to evaluate future state
            # leveraging the fact that we can just check immediate next state
            move_score = self._evaluate_move(board, move, opponent)

            if move_score > best_score:
                best_score = move_score
                best_move = move

        return best_move, None

    def _evaluate_move(self, board: Board, move: Tuple[int, int], opponent: str) -> float:
        """
        Evaluate a candidate move.
        Higher score is better.
        """

        current_pos = board.get_horse_position(self.name)
        sim_board = board.fast_copy()
        # 1. Update grid
        sim_board.grid[move[0], move[1]] = Board.WHITE_HORSE if self.name == "white" else Board.BLACK_HORSE
        sim_board._mark_occupied(move[0], move[1])

        # 2. Leave trail
        sim_board.grid[current_pos[0], current_pos[1]] = Board.BROWN_APPLE
        sim_board._mark_occupied(current_pos[0], current_pos[1])

        if self.name == "white":
            sim_board.white_pos = move
        else:
            sim_board.black_pos = move

        # Calculate Heuristics

        # A. unique legal moves from new position (My Mobility)
        my_moves = Rules.get_legal_knight_moves(sim_board, self.name)
        num_my_moves = len(my_moves)

        # If I have 0 moves next turn, this is a suicidal move (unless I already won, checked above)
        if num_my_moves == 0:
            return float("-inf")

        # B. Opponent moves (Their Mobility)
        opp_moves = Rules.get_legal_knight_moves(sim_board, opponent)
        num_opp_moves = len(opp_moves)

        # 2. Avoid Immediate Loss (Being Captured)
        # If the opponent can move to our new position, they capture us next turn.
        if move in opp_moves:
            return float("-inf")

        # Score = (My Mobility) - (k * Opponent Mobility)
        # We want to maximize our options and limit theirs.
        score = float(num_my_moves) - (self.aggression * float(num_opp_moves))

        return score
