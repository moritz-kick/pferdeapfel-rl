"""Basic checks for the Monte Carlo Tree Search player."""

from __future__ import annotations

from src.game.board import Board
from src.game.rules import Rules
from src.players import mcts
from src.players.mcts import MCTSPlayer


def test_mcts_returns_legal_move_mode2() -> None:
    """MCTS should always return a legal move in trail mode."""
    board = Board(mode=2)
    legal_moves = Rules.get_legal_knight_moves(board, "white")
    player = MCTSPlayer("white", simulations=8, rollout_depth=6, seed=123)

    move, extra = player.get_move(board, legal_moves)

    assert move in legal_moves
    assert extra is None


def test_mcts_seed_reproducible() -> None:
    """Using the same seed should produce the same decision on the same state."""
    board_a = Board(mode=2)
    board_b = Board(mode=2)
    legal_a = Rules.get_legal_knight_moves(board_a, "white")
    legal_b = Rules.get_legal_knight_moves(board_b, "white")

    player_a = MCTSPlayer("white", simulations=16, rollout_depth=8, seed=99)
    player_b = MCTSPlayer("white", simulations=16, rollout_depth=8, seed=99)

    move_a, extra_a = player_a.get_move(board_a, legal_a)
    move_b, extra_b = player_b.get_move(board_b, legal_b)

    assert move_a == move_b
    assert extra_a == extra_b


def test_mcts_generates_required_extra_in_mode1() -> None:
    """In mode 1 an extra apple placement is required."""
    board = Board(mode=1)
    legal_moves = Rules.get_legal_knight_moves(board, "white")
    player = MCTSPlayer("white", simulations=4, rollout_depth=4, seed=7)

    move, extra = player.get_move(board, legal_moves)
    assert move in legal_moves
    assert extra is not None

    board_copy = board.copy()
    assert Rules.make_move(board_copy, "white", move, extra) is True


def test_mcts_uct_penalizes_opponent_success() -> None:
    """Tree policy should treat opponent value as a penalty, not a bonus."""
    player = MCTSPlayer("black", simulations=1, seed=0)
    root = mcts._Node(board=Board(mode=2), player="black")

    good_for_black = mcts._Node(
        board=Board(mode=2),
        player="white",
        parent=root,
        action=((0, 0), None),
        visits=10,
        value=0.0,  # Opponent loses from their POV -> great for black
    )
    bad_for_black = mcts._Node(
        board=Board(mode=2),
        player="white",
        parent=root,
        action=((0, 1), None),
        visits=10,
        value=10.0,  # Opponent always wins from their POV -> terrible for black
    )

    root.children = {good_for_black.action: good_for_black, bad_for_black.action: bad_for_black}
    root.visits = good_for_black.visits + bad_for_black.visits

    chosen = player._uct_select(root)
    assert chosen is good_for_black
