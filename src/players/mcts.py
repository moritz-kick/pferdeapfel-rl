"""Monte Carlo Tree Search player implementation."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.game.board import Board
from src.game.rules import Rules
from src.players.base import Player

Move = Tuple[int, int]
Action = Tuple[Move, Optional[Move]]


@dataclass
class _Node:
    """Internal tree node used by the MCTSPlayer."""

    board: Board
    player: str
    parent: Optional["_Node"] = None
    action: Optional[Action] = None
    terminal_winner: Optional[str] = None
    children: Dict[Action, "_Node"] = field(default_factory=dict)
    visits: int = 0
    value: float = 0.0
    untried_actions: Optional[List[Action]] = None

    def is_terminal(self) -> bool:
        """Return True if this node already has a winner."""
        return self.terminal_winner is not None

    def is_fully_expanded(self) -> bool:
        """Return True if all actions from this node have been explored."""
        return self.untried_actions is not None and len(self.untried_actions) == 0


class MCTSPlayer(Player):
    """Monte Carlo Tree Search based player."""

    DISPLAY_NAME = "mcts"

    def __init__(
        self,
        side: str,
        simulations: int = 1024,
        exploration: float = math.sqrt(2.0),
        rollout_depth: int = 100,
        time_limit: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Create an MCTS player.

        Args:
            side: "white" or "black".
            simulations: Number of simulations to run per move.
            exploration: UCT exploration constant.
            rollout_depth: Maximum playout depth during rollouts.
            time_limit: Optional wall-clock limit in seconds.
            seed: Optional RNG seed for reproducibility.
        """
        side_clean = side.lower()
        if side_clean not in ("white", "black"):
            raise ValueError("side must be 'white' or 'black'")

        super().__init__(f"{side_clean.capitalize()} MCTS")
        self.side = side_clean
        self.simulations = max(1, simulations)
        self.exploration = exploration
        self.rollout_depth = max(1, rollout_depth)
        self.time_limit = time_limit
        self.rng = random.Random(seed)

    def get_move(self, board: Board, legal_moves: list[Move]) -> tuple[Move, Optional[Move]]:
        """Return the move selected by running MCTS from the current state."""
        if not legal_moves:
            raise ValueError("No legal moves available for MCTS player.")

        root_board = board.copy()
        root = _Node(board=root_board, player=self.side)
        root.untried_actions = self._generate_actions(root_board, self.side, legal_moves)

        if not root.untried_actions:
            # Fallback: try to recover any valid action before giving up.
            fallback_actions = self._generate_actions(root_board, self.side)
            if not fallback_actions:
                raise ValueError("No valid actions available for MCTS search.")
            chosen_action = fallback_actions[0]
            return chosen_action[0], chosen_action[1]

        start_time = time.time()
        iterations = 0
        while iterations < self.simulations:
            if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
                break

            node = self._select_node(root)
            if not node.is_terminal():
                expanded = self._expand(node)
                if expanded is not None:
                    node = expanded
                elif node.children:
                    node = self.rng.choice(list(node.children.values()))

            reward = self._rollout(node)
            self._backpropagate(node, reward)
            iterations += 1

        if not root.children:
            # If expansion failed for some reason, choose a safe legal move.
            safe_action = root.untried_actions[0] if root.untried_actions else (legal_moves[0], None)
            return safe_action[0], safe_action[1]

        best_child = max(root.children.values(), key=lambda c: self._value_for_player(c, self.side))
        assert best_child.action is not None
        return best_child.action

    def _select_node(self, node: _Node) -> _Node:
        """Traverse the tree using UCT until an expandable node is found."""
        current = node
        while current.children and current.is_fully_expanded() and not current.is_terminal():
            current = self._uct_select(current)
        return current

    def _expand(self, node: _Node) -> Optional[_Node]:
        """Expand one untried action from the node and return the child."""
        if node.untried_actions is None:
            node.untried_actions = self._generate_actions(node.board, node.player)

        if not node.untried_actions:
            return None

        action = self._pop_random(node.untried_actions)
        board_after, next_player, winner = self._apply_action(node.board, node.player, action)
        if board_after is None:
            return None

        child = _Node(
            board=board_after,
            player=next_player,
            parent=node,
            action=action,
            terminal_winner=winner,
        )
        node.children[action] = child
        return child

    def _rollout(self, node: _Node) -> float:
        """Simulate a random playout from the given node."""
        if node.terminal_winner is not None:
            return self._result_value(node.terminal_winner)

        board_sim = node.board.copy()
        player = node.player
        last_mover = self._opponent(player)

        for _ in range(self.rollout_depth):
            winner = Rules.check_win_condition(board_sim, last_mover=last_mover)
            if winner:
                return self._result_value(winner)

            actions = self._generate_actions(board_sim, player)
            if not actions:
                winner = Rules.check_win_condition(board_sim, last_mover=last_mover)
                return self._result_value(winner) if winner else 0.5

            action = self.rng.choice(actions)
            success = Rules.make_move(board_sim, player, action[0], action[1])
            if not success:
                # Should not happen because _generate_actions validates moves,
                # but fall back to neutral reward to keep search robust.
                return 0.5

            last_mover = player
            player = self._opponent(player)

        return 0.5

    def _backpropagate(self, node: _Node, reward_root_pov: float) -> None:
        """Propagate rollout result up the tree."""
        current: Optional[_Node] = node
        while current is not None:
            current.visits += 1
            # Store value from the perspective of the player to move at this node.
            pov_value = self._pov_value(current.player, reward_root_pov)
            current.value += pov_value
            current = current.parent

    def _uct_select(self, node: _Node) -> _Node:
        """Select a child node using the UCT formula."""
        log_parent_visits = math.log(max(1, node.visits))

        def score(child: _Node) -> float:
            if child.visits == 0:
                return float("inf")
            # Translate the child's average value into the current node player's POV.
            child_avg = child.value / child.visits
            exploitation = child_avg if child.player == node.player else 1.0 - child_avg
            exploration = self.exploration * math.sqrt(log_parent_visits / child.visits)
            return exploitation + exploration

        return max(node.children.values(), key=score)

    def _generate_actions(
        self, board: Board, player: str, legal_moves: Optional[List[Move]] = None
    ) -> List[Action]:
        """Enumerate valid (move, extra_apple) action pairs."""
        moves = legal_moves if legal_moves is not None else Rules.get_legal_knight_moves(board, player)
        if not moves:
            return []

        if board.mode == 2:
            return [(move, None) for move in moves]

        empties = [(r, c) for r in range(Board.BOARD_SIZE) for c in range(Board.BOARD_SIZE) if board.is_empty(r, c)]
        actions: List[Action] = []

        for move in moves:
            extras: List[Optional[Move]] = []
            if board.mode == 3:
                extras.append(None)  # Optional apple placement

            if board.mode in (1, 3):
                for empty in empties:
                    if board.mode == 1 and empty == move:
                        continue  # Can't place on the destination before moving
                    extras.append(empty)

            for extra in extras:
                board_copy = board.copy()
                if Rules.make_move(board_copy, player, move, extra):
                    actions.append((move, extra))

        return actions

    def _apply_action(self, board: Board, player: str, action: Action) -> tuple[Optional[Board], str, Optional[str]]:
        """Apply an action to a copy of the board."""
        board_copy = board.copy()
        move_to, extra = action
        if not Rules.make_move(board_copy, player, move_to, extra):
            return None, player, None
        winner = Rules.check_win_condition(board_copy, last_mover=player)
        return board_copy, self._opponent(player), winner

    def _result_value(self, winner: str) -> float:
        """Return rollout reward from this player's perspective."""
        if winner == self.side:
            return 1.0
        if winner == "draw":
            return 0.5
        return 0.0

    def _pov_value(self, player: str, reward_root_pov: float) -> float:
        """Convert a reward from root POV to the given player's POV."""
        if reward_root_pov == 0.5:
            return 0.5
        if player == self.side:
            return reward_root_pov
        return 1.0 - reward_root_pov

    def _value_for_player(self, node: _Node, player: str) -> float:
        """Average value stored on node translated into the given player's POV."""
        if node.visits == 0:
            return 0.0
        avg = node.value / node.visits
        return avg if node.player == player else 1.0 - avg

    def _pop_random(self, actions: List[Action]) -> Action:
        """Remove and return a random action from the list."""
        idx = self.rng.randrange(len(actions))
        return actions.pop(idx)

    @staticmethod
    def _opponent(player: str) -> str:
        return "black" if player == "white" else "white"


__all__ = ["MCTSPlayer"]
