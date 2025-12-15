import cProfile
import pstats
import io
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.players.minimax import MinimaxPlayer
from src.game.board import Board
from src.game.rules import Rules


def profile():
    board = Board(mode=2)
    # Use valid parameters but maybe slightly reduced depth if 8 is insanely slow.
    # User default is 8. Let's try 6 to be safe but representative, or 8 if we are brave.
    # Let's start with 6.
    player = MinimaxPlayer("white", opening_depth=6, normal_depth=4)

    print("Starting profiling with depth 6...")
    legal_moves = Rules.get_legal_knight_moves(board, "white")

    pr = cProfile.Profile()
    pr.enable()

    player.get_move(board, legal_moves)

    pr.disable()
    s = io.StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(40)
    print(s.getvalue())


if __name__ == "__main__":
    profile()
