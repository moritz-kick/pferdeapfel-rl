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

    # Explicit variant tag for profiling/GUI
    VERSION = "v1"

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


class HeuristicPlayerV2(Player):
    """
    Space-aware variant of the heuristic player for Mode 2.

    Goals:
    - Keep the original speed target by using only cheap computations.
    - Prioritize safe moves that preserve our future space while trimming the opponent's.
    - Light center preference to avoid hugging the edges early.
    """

    VERSION = "v2"

    def __init__(
        self,
        name: str,
        aggression: float = 0.4,
        space_weight: float = 0.35,
        center_bonus: float = 0.25,
    ) -> None:
        super().__init__(name)
        self.aggression = aggression
        self.space_weight = space_weight
        self.center_bonus = center_bonus

    def get_move(
        self, board: Board, legal_moves: List[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
        if not legal_moves:
            return (0, 0), None

        opponent = "black" if self.name == "white" else "white"
        opponent_pos = board.get_horse_position(opponent)

        # Immediate win
        for move in legal_moves:
            if move == opponent_pos:
                return move, None

        # Shuffle to avoid deterministic bad ties
        random.shuffle(legal_moves)

        best_move = legal_moves[0]
        best_score = float("-inf")
        best_dist = float("inf")

        for move in legal_moves:
            score = self._evaluate_move(board, move, opponent, opponent_pos)
            dist = abs(move[0] - 3.5) + abs(move[1] - 3.5)
            if score > best_score or (score == best_score and dist < best_dist):
                best_score = score
                best_move = move
                best_dist = dist

        return best_move, None

    def _evaluate_move(
        self, board: Board, move: Tuple[int, int], opponent: str, opponent_pos: Tuple[int, int]
    ) -> float:
        current_pos = board.get_horse_position(self.name)
        sim_board = board.fast_copy()

        # Apply move + trail
        sim_board.grid[move[0], move[1]] = Board.WHITE_HORSE if self.name == "white" else Board.BLACK_HORSE
        sim_board._mark_occupied(move[0], move[1])
        sim_board.grid[current_pos[0], current_pos[1]] = Board.BROWN_APPLE
        sim_board._mark_occupied(current_pos[0], current_pos[1])

        if self.name == "white":
            sim_board.white_pos = move
        else:
            sim_board.black_pos = move

        # Mobility checks
        my_moves = Rules.get_legal_knight_moves(sim_board, self.name)
        if not my_moves:
            return float("-inf")

        opp_moves = Rules.get_legal_knight_moves(sim_board, opponent)
        # Avoid stepping into an immediate capture
        if move in opp_moves:
            return float("-inf")

        # Minimax-style: evaluate the worst advantage we retain after any opponent reply.
        reply_advantage = self._worst_reply_advantage(sim_board, opponent, opp_moves)
        if reply_advantage == float("-inf"):
            return float("-inf")

        # Territory (reachable empty squares) difference
        my_area = self._reachable_area(sim_board, move, sim_board.get_horse_position(opponent))
        opp_area = self._reachable_area(sim_board, sim_board.get_horse_position(opponent), move)

        mobility_score = float(len(my_moves)) - self.aggression * float(len(opp_moves))
        area_score = (my_area - opp_area) * self.space_weight
        center_score = self._center_bonus(move)

        return mobility_score + area_score + center_score + (0.8 * reply_advantage)

    def _worst_reply_advantage(
        self, board_after_self: Board, opponent: str, opp_moves: List[Tuple[int, int]]
    ) -> float:
        """
        Return the minimum (my_moves - opp_moves) we can keep after any opponent reply.
        """
        if not opp_moves:
            return float("inf")  # opponent already stuck

        opp_pos = board_after_self.get_horse_position(opponent)
        worst_advantage = float("inf")

        for opp_move in opp_moves:
            opp_board = board_after_self.fast_copy()
            opp_board.grid[opp_move[0], opp_move[1]] = Board.WHITE_HORSE if opponent == "white" else Board.BLACK_HORSE
            opp_board._mark_occupied(opp_move[0], opp_move[1])
            opp_board.grid[opp_pos[0], opp_pos[1]] = Board.BROWN_APPLE
            opp_board._mark_occupied(opp_pos[0], opp_pos[1])

            if opponent == "white":
                opp_board.white_pos = opp_move
            else:
                opp_board.black_pos = opp_move

            my_future = Rules.get_legal_knight_moves(opp_board, self.name)
            if not my_future:
                return float("-inf")  # opponent can trap us immediately

            opp_future = Rules.get_legal_knight_moves(opp_board, opponent)
            advantage = float(len(my_future) - len(opp_future))
            if advantage < worst_advantage:
                worst_advantage = advantage

        return worst_advantage

    def _reachable_area(self, board: Board, start: Tuple[int, int], blocked: Tuple[int, int]) -> int:
        """
        Count how many empty squares are reachable for a knight without stepping on `blocked`.
        Uses only the cached empty set for speed.
        """
        empties = board.get_empty_squares()
        visited = set()
        stack = [start]

        while stack:
            row, col = stack.pop()
            for nr, nc in Rules.get_knight_moves(row, col):
                pos = (nr, nc)
                if pos == blocked or pos in visited:
                    continue
                if pos not in empties:
                    continue
                visited.add(pos)
                stack.append(pos)

        return len(visited)

    def _center_bonus(self, pos: Tuple[int, int]) -> float:
        row, col = pos
        return self.center_bonus if 2 <= row <= 5 and 2 <= col <= 5 else 0.0


class HeuristicPlayerV3(Player):
    """
    Simpler, more tactical heuristic for Mode 2.

    Goals:
    - Prefer moves that let us later "sit on" squares the opponent would like to move to.
      (We move there first, leave an apple, and remove that square from their future options.)
    - Still avoid immediate suicide and keep a reasonable amount of our own mobility.
    - Keep the logic cheap: only look one move ahead for each side.
    """

    VERSION = "v3"

    def __init__(
        self,
        name: str,
        block_weight: float = 1.0,
        my_mobility_weight: float = 0.2,
        opp_mobility_weight: float = 0.3,
    ) -> None:
        super().__init__(name)
        # How much we value "blocking" the opponent's potential landing squares
        self.block_weight = block_weight
        # Small bias to keep our own future options
        self.my_mobility_weight = my_mobility_weight
        # Small penalty for letting the opponent keep many options
        self.opp_mobility_weight = opp_mobility_weight

    def get_move(
        self, board: Board, legal_moves: List[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
        if not legal_moves:
            return (0, 0), None

        opponent = "black" if self.name == "white" else "white"
        opponent_pos = board.get_horse_position(opponent)

        # 1) Immediate capture if possible
        for move in legal_moves:
            if move == opponent_pos:
                return move, None

        # Shuffle to avoid deterministic tie-breaking
        random.shuffle(legal_moves)

        best_move = legal_moves[0]
        best_score = float("-inf")

        for move in legal_moves:
            score = self._evaluate_move(board, move, opponent)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move, None

    def _evaluate_move(
        self, board: Board, move: Tuple[int, int], opponent: str
    ) -> float:
        """
        Evaluate a candidate move using a simple "blocking future squares" idea.

        For a hypothetical move:
        - Look at all our legal moves from that new position (our next-turn options).
        - Look at all the opponent's legal moves from their position after our move.
        - Count how many squares are in the intersection of these sets.

        Intuition:
        - Any square in the intersection is a square *both* could reach in one move.
          If we go there first on our next turn, we leave an apple and the opponent
          can never use it – we effectively remove one of their potential landing fields.
        - Even if we only "steal" 1 out of 8 of their options, it's still decent.
          If we steal 1/1 or 2/2, that's extremely strong.
        """
        current_pos = board.get_horse_position(self.name)
        sim_board = board.fast_copy()

        # Apply our move
        sim_board.grid[move[0], move[1]] = Board.WHITE_HORSE if self.name == "white" else Board.BLACK_HORSE
        sim_board._mark_occupied(move[0], move[1])
        sim_board.grid[current_pos[0], current_pos[1]] = Board.BROWN_APPLE
        sim_board._mark_occupied(current_pos[0], current_pos[1])

        if self.name == "white":
            sim_board.white_pos = move
        else:
            sim_board.black_pos = move

        # Our next-move options from the new square
        my_next_moves = Rules.get_legal_knight_moves(sim_board, self.name)
        if not my_next_moves:
            # We would immediately trap ourselves next turn (unless we already captured,
            # which is handled earlier), so this is a terrible move.
            return float("-inf")

        # Opponent options after our move
        opp_next_moves = Rules.get_legal_knight_moves(sim_board, opponent)

        # Avoid stepping onto a square that can be captured immediately next turn.
        if move in opp_next_moves:
            return float("-inf")

        # How many of our future landing squares are also potential landing squares for the opponent?
        my_set = set(my_next_moves)
        opp_set = set(opp_next_moves)
        blocked_squares = my_set & opp_set
        num_blocked = len(blocked_squares)

        # Ratio of opponent moves we could potentially "steal" by going there first next turn.
        opp_count = max(1, len(opp_next_moves))
        blocked_ratio = num_blocked / opp_count

        # Heuristic score:
        # - Reward blocking opponent squares.
        # - Slightly reward our own mobility.
        # - Slightly punish letting the opponent keep many options.
        score = (
            self.block_weight * (float(num_blocked) + blocked_ratio)
            + self.my_mobility_weight * float(len(my_next_moves))
            - self.opp_mobility_weight * float(len(opp_next_moves))
        )

        return score


class HeuristicPlayerV4(Player):
    """
    Beam-search lookahead variant for Mode 2.

    Goals:
    - Keep moves fast (tiny branching factor) while looking a few plies ahead.
    - Use space-aware evaluation (bitboard flood fill) to stay competitive with Minimax.
    - Prefer safe moves first to avoid immediate recaptures.
    """

    VERSION = "v4"
    WIN_SCORE = 100000.0

    def __init__(
        self,
        name: str,
        depth: int = 3,
        beam_width: int = 6,
        root_beam: Optional[int] = None,
        space_weight: float = 0.45,
        mobility_weight: float = 0.4,
        opp_mobility_weight: float = 0.55,
        center_bonus: float = 0.45,
        danger_penalty: float = 1150.0,
    ) -> None:
        super().__init__(name)
        self.depth = depth
        self.beam_width = beam_width
        self.root_beam = root_beam or max(beam_width + 2, beam_width)
        self.space_weight = space_weight
        self.mobility_weight = mobility_weight
        self.opp_mobility_weight = opp_mobility_weight
        self.center_bonus = center_bonus
        self.danger_penalty = danger_penalty
        self.nodes_evaluated = 0
        self.last_search_depth = depth

    def get_move(
        self, board: Board, legal_moves: List[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
        if not legal_moves:
            return (0, 0), None

        opponent = "black" if self.name == "white" else "white"
        opp_pos = board.get_horse_position(opponent)

        # Immediate capture if available
        for move in legal_moves:
            if move == opp_pos:
                return move, None

        self.nodes_evaluated = 0
        self.last_search_depth = self.depth

        # Pre-score moves and reuse simulated boards at root to cut copies
        scored_moves: List[Tuple[float, Tuple[int, int], Board]] = []
        for move in legal_moves:
            sim_board = board.fast_copy()
            self._apply_mode2_move(sim_board, self.name, move)
            base_score = self._static_eval(sim_board, opponent, move, self.name)
            scored_moves.append((base_score, move, sim_board))

        # Order by static score, evaluate a small beam
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        best_move = scored_moves[0][1]
        best_score = float("-inf")

        alpha = float("-inf")
        beta = float("inf")

        for base_score, move, sim_board in scored_moves[: self.root_beam]:
            score = self._search(sim_board, self.depth - 1, alpha, beta, False, opponent)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            if beta <= alpha:
                break

        return best_move, None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def _search(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        is_self_turn: bool,
        opponent: str,
    ) -> float:
        self.nodes_evaluated += 1

        # Terminal: capture already happened
        if board.white_pos == board.black_pos:
            return self.WIN_SCORE if not is_self_turn else -self.WIN_SCORE

        if depth == 0:
            return self._fast_eval(board, opponent)

        current = self.name if is_self_turn else opponent
        legal_moves = Rules.get_legal_knight_moves(board, current)

        if not legal_moves:
            return -self.WIN_SCORE if is_self_turn else self.WIN_SCORE

        ordered_moves = self._order_moves(board, legal_moves, current, opponent, is_self_turn)

        if is_self_turn:
            value = float("-inf")
            for move in ordered_moves:
                sim_board = board.fast_copy()
                self._apply_mode2_move(sim_board, current, move)
                value = max(value, self._search(sim_board, depth - 1, alpha, beta, False, opponent))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value

        value = float("inf")
        for move in ordered_moves:
            sim_board = board.fast_copy()
            self._apply_mode2_move(sim_board, current, move)
            value = min(value, self._search(sim_board, depth - 1, alpha, beta, True, opponent))
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

    def _order_moves(
        self,
        board: Board,
        moves: List[Tuple[int, int]],
        mover: str,
        opponent: str,
        maximizing: bool,
    ) -> List[Tuple[int, int]]:
        scored: List[Tuple[float, Tuple[int, int]]] = []
        for move in moves:
            sim_board = board.fast_copy()
            self._apply_mode2_move(sim_board, mover, move)
            score = self._static_eval(sim_board, opponent, move, mover)
            scored.append((score, move))

        scored.sort(key=lambda x: x[0], reverse=maximizing)
        return [m for _, m in scored[: self.beam_width]]

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def _static_eval(
        self, board: Board, opponent: str, move_just_played: Tuple[int, int], mover: str
    ) -> float:
        """
        Cheap single-state evaluation for ordering and root scoring.
        Positive favors self.
        """
        my_moves = len(Rules.get_legal_knight_moves(board, self.name))
        opp_moves = len(Rules.get_legal_knight_moves(board, opponent))

        # Suicide or immediate recapture avoidance
        if mover == self.name:
            if my_moves == 0:
                return -self.WIN_SCORE
            if move_just_played in Rules.get_legal_knight_moves(board, opponent):
                return -self.danger_penalty
        else:
            # Opponent just moved; if they walked into capture range, we like it
            if board.get_horse_position(opponent) in Rules.get_legal_knight_moves(board, self.name):
                return self.danger_penalty

        # Space estimate
        my_area = self._flood_fill(board, board.get_horse_position(self.name))
        opp_area = self._flood_fill(board, board.get_horse_position(opponent))

        mobility_score = self.mobility_weight * float(my_moves) - self.opp_mobility_weight * float(opp_moves)
        space_score = self.space_weight * float(my_area - opp_area)
        center_score = self._center_bonus(board.get_horse_position(self.name))

        # Bonus if opponent has no moves after this state
        if opp_moves == 0:
            return self.WIN_SCORE

        return space_score + mobility_score + center_score

    def _fast_eval(self, board: Board, opponent: str) -> float:
        my_moves = len(Rules.get_legal_knight_moves(board, self.name))
        opp_moves = len(Rules.get_legal_knight_moves(board, opponent))

        if my_moves == 0:
            return -self.WIN_SCORE
        if opp_moves == 0:
            return self.WIN_SCORE

        my_area = self._flood_fill(board, board.get_horse_position(self.name))
        opp_area = self._flood_fill(board, board.get_horse_position(opponent))

        mobility_score = self.mobility_weight * float(my_moves) - self.opp_mobility_weight * float(opp_moves)
        space_score = self.space_weight * float(my_area - opp_area)
        center_score = self._center_bonus(board.get_horse_position(self.name))

        return space_score + mobility_score + center_score

    def _center_bonus(self, pos: Tuple[int, int]) -> float:
        row, col = pos
        return self.center_bonus if 2 <= row <= 5 and 2 <= col <= 5 else 0.0

    def _apply_mode2_move(self, board: Board, player: str, move_to: Tuple[int, int]) -> None:
        old_pos = board.get_horse_position(player)
        board.grid[move_to[0], move_to[1]] = Board.WHITE_HORSE if player == "white" else Board.BLACK_HORSE
        board._mark_occupied(move_to[0], move_to[1])
        if player == "white":
            board.white_pos = move_to
        else:
            board.black_pos = move_to
        board.grid[old_pos[0], old_pos[1]] = Board.BROWN_APPLE
        board._mark_occupied(old_pos[0], old_pos[1])

    # ------------------------------------------------------------------
    # Bitboard flood fill utilities (copied from Minimax for speed)
    # ------------------------------------------------------------------
    FILE_A = 0x0101010101010101
    FILE_B = 0x0202020202020202
    FILE_G = 0x4040404040404040
    FILE_H = 0x8080808080808080

    NOT_A = ~FILE_A
    NOT_B = ~FILE_B
    NOT_G = ~FILE_G
    NOT_H = ~FILE_H
    NOT_AB = ~(FILE_A | FILE_B)
    NOT_GH = ~(FILE_G | FILE_H)

    @classmethod
    def _expand_knight_moves(cls, mask: int) -> int:
        res = 0
        res |= (mask & cls.NOT_H) << 17
        res |= (mask & cls.NOT_GH) << 10
        res |= (mask & cls.NOT_GH) >> 6
        res |= (mask & cls.NOT_H) >> 15
        res |= (mask & cls.NOT_A) << 15
        res |= (mask & cls.NOT_AB) << 6
        res |= (mask & cls.NOT_AB) >> 10
        res |= (mask & cls.NOT_A) >> 17
        return res & 0xFFFFFFFFFFFFFFFF

    def _flood_fill(self, board: Board, start_pos: Tuple[int, int], max_depth: int = 64) -> int:
        r, c = start_pos
        if not board.is_valid_square(r, c):
            return 0

        start_mask = 1 << (r * 8 + c)
        allowed_mask = ~board.occupied_mask & 0xFFFFFFFFFFFFFFFF

        visited = start_mask
        front = start_mask

        for _ in range(max_depth):
            new_front = self._expand_knight_moves(front)
            new_front &= allowed_mask
            new_front &= ~visited
            if new_front == 0:
                break
            visited |= new_front
            front = new_front

        return visited.bit_count()
