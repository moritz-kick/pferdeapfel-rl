"""More detailed debug script to find the Mode 1 issue."""

import logging
from pathlib import Path
import sys

from src.game.game import Game
from src.game.board import Board
from src.game.rules import Rules
from src.players.random import RandomPlayer
from src.players.rl.ppo_player import PPOPlayer

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

def test_single_game():
    """Test a single Mode 1 game with detailed logging."""
    print("\n" + "="*70)
    print("DETAILED MODE 1 TEST - RANDOM VS RANDOM")
    print("="*70 + "\n")
    
    white = RandomPlayer("White")
    black = RandomPlayer("Black")
    game = Game(white, black, mode=1, logging=True)
    
    move_count = 0
    max_moves = 20
    
    while not game.game_over and move_count < max_moves:
        move_count += 1
        current_player = game.get_current_player()
        player_color = game.current_player
        
        # Get legal moves
        legal_moves = Rules.get_legal_knight_moves(game.board, player_color)
        print(f"\n--- Move #{move_count} ({player_color.upper()}) ---")
        print(f"Legal moves available: {legal_moves}")
        print(f"Board state BEFORE move:")
        print(f"  White pos: {game.board.white_pos}")
        print(f"  Black pos: {game.board.black_pos}")
        print(f"  Apples on board: {sum(1 for r in range(8) for c in range(8) if game.board.grid[r,c] == Board.BROWN_APPLE)}")
        print(f"  Brown apples remaining: {game.board.brown_apples_remaining}")
        
        if not legal_moves:
            print(f"❌ NO LEGAL MOVES! Game should end.")
            break
        
        # Get move from player
        try:
            move_to, extra_apple = current_player.get_move(game.board, legal_moves)
            print(f"Player returned: move={move_to}, apple={extra_apple}")
        except Exception as e:
            print(f"❌ Error getting move: {e}")
            break
        
        # Make the move
        success = game.make_move(move_to, extra_apple)
        print(f"Move execution: {'✓ SUCCESS' if success else '✗ FAILED'}")
        
        if not success:
            print(f"❌ Move failed for {player_color}")
            print(f"Board state AFTER failed move:")
            print(f"  White pos: {game.board.white_pos}")
            print(f"  Black pos: {game.board.black_pos}")
            print(f"  Apples on board: {sum(1 for r in range(8) for c in range(8) if game.board.grid[r,c] == Board.BROWN_APPLE)}")
            break
        
        print(f"Board state AFTER move:")
        print(f"  White pos: {game.board.white_pos}")
        print(f"  Black pos: {game.board.black_pos}")
        print(f"  Apples on board: {sum(1 for r in range(8) for c in range(8) if game.board.grid[r,c] == Board.BROWN_APPLE)}")
        print(f"  Brown apples remaining: {game.board.brown_apples_remaining}")
    
    if game.game_over:
        print(f"\n✓ Game finished! Winner: {game.winner}")
    else:
        print(f"\n✗ Game did not finish (moved {move_count} times)")

if __name__ == "__main__":
    test_single_game()
