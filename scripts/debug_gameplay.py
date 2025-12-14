"""Debug script to visualize gameplay and detect unfinished games."""

import logging
from pathlib import Path
from typing import Optional
import argparse
import sys

from src.game.game import Game
from src.game.board import Board
from src.game.rules import Rules
from src.players.random import RandomPlayer
from src.players.rl.ppo_player import PPOPlayer

# Suppress logging during gameplay
logging.basicConfig(level=logging.WARNING)


class GameVisualizer:
    """Visualizes game state and moves."""

    BOARD_SYMBOLS = {
        Board.EMPTY: ".",
        Board.WHITE_HORSE: "♘",  # White knight
        Board.BLACK_HORSE: "♞",  # Black knight
        Board.BROWN_APPLE: "🍎",  # Brown apple
        Board.GOLDEN_APPLE: "✨",  # Golden apple
    }

    @staticmethod
    def visualize_board(board: Board) -> str:
        """Generate a visual representation of the board."""
        lines = []
        lines.append("\n     0   1   2   3   4   5   6   7")
        lines.append("   ┌───┬───┬───┬───┬───┬───┬───┬───┐")

        for row in range(8):
            row_str = f" {row} │"
            for col in range(8):
                cell_value = int(board.grid[row, col])
                symbol = GameVisualizer.BOARD_SYMBOLS.get(cell_value, "?")
                row_str += f" {symbol} │"
            lines.append(row_str)
            lines.append("   ├───┼───┼───┼───┼───┼───┼───┼───┤")

        lines[-1] = lines[-1].replace("├", "└").replace("┼", "┴").replace("┤", "┘")
        return "\n".join(lines)

    @staticmethod
    def visualize_move(
        board: Board,
        player: str,
        move_to: tuple,
        extra_apple: Optional[tuple],
        move_number: int,
        turn_number: int,
    ) -> str:
        """Generate a description of a move."""
        pos = board.white_pos if player == "white" else board.black_pos
        player_display = "White ♘" if player == "white" else "Black ♞"
        
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"Turn #{turn_number} - {player_display}'s Move #{move_number}")
        lines.append(f"{'='*60}")
        lines.append(f"Move: {pos} → {move_to}")
        if extra_apple:
            lines.append(f"Extra Apple Placement: {extra_apple}")
        lines.append(f"Brown apples remaining: {board.brown_apples_remaining}")
        lines.append(f"Golden apples remaining: {board.golden_apples_remaining}")
        if board.golden_phase_started:
            lines.append(f"⭐ GOLDEN PHASE ACTIVE")
        
        return "\n".join(lines)


def simulate_and_display_first_moves(mode: int) -> None:
    """Simulate and display the first moves for each color in a given mode."""
    print(f"\n{'='*70}")
    print(f"MODE {mode} - FIRST MOVES VISUALIZATION")
    print(f"{'='*70}")

    for player_type in ["Random vs Random", "Random vs PPO", "PPO vs PPO"]:
        print(f"\n{'─'*70}")
        print(f"{player_type}")
        print(f"{'─'*70}")

        # Create players based on type
        if player_type == "Random vs Random":
            white = RandomPlayer("White-Random")
            black = RandomPlayer("Black-Random")
        elif player_type == "Random vs PPO":
            white = RandomPlayer("White-Random")
            try:
                black = PPOPlayer("black", "data/models/ppo_self_play/best_model/best_model.zip")
            except Exception as e:
                print(f"⚠️  Could not load PPO model: {e}")
                continue
        else:  # PPO vs PPO
            try:
                white = PPOPlayer("white", "data/models/ppo_self_play/best_model/best_model.zip")
                black = PPOPlayer("black", "data/models/ppo_self_play/best_model/best_model.zip")
            except Exception as e:
                print(f"⚠️  Could not load PPO model: {e}")
                continue

        game = Game(white, black, mode=mode, logging=False)
        turn_num = 0

        # Play first 2 moves (one for each player)
        for move_count in range(2):
            turn_num += 1
            current_player = game.get_current_player()
            player_color = game.current_player

            # Get legal moves
            legal_moves = Rules.get_legal_knight_moves(game.board, player_color)

            if not legal_moves:
                print(f"\n⚠️  NO LEGAL MOVES for {player_color.upper()}! Game should end.")
                break

            # Get move from player
            try:
                move_to, extra_apple = current_player.get_move(game.board, legal_moves)
            except Exception as e:
                print(f"\n❌ Error getting move: {e}")
                break

            # Display move info
            print(GameVisualizer.visualize_move(
                game.board, player_color, move_to, extra_apple, move_count + 1, turn_num
            ))

            # Make the move
            success = game.make_move(move_to, extra_apple)
            if not success:
                print(f"❌ Move failed!")
                break

            # Display board after move
            print(GameVisualizer.visualize_board(game.board))

        print()


def run_simulation(
    white_player_type: str,
    black_player_type: str,
    mode: int,
    num_games: int = 50,
) -> dict:
    """
    Run simulation and track game completion.

    Returns: Dictionary with stats and list of unfinished games
    """
    unfinished_games = []
    finished_games = 0
    max_moves = 0
    min_moves = float('inf')
    total_moves = 0

    for game_num in range(num_games):
        # Create players
        white = _create_player(white_player_type, "white")
        black = _create_player(black_player_type, "black")

        if white is None or black is None:
            print(f"⚠️  Skipping game {game_num + 1}: Could not create players")
            continue

        game = Game(white, black, mode=mode, logging=False)
        move_count = 0
        max_allowed_moves = 1000  # Prevent infinite loops

        # Play the game
        while not game.game_over and move_count < max_allowed_moves:
            current_player = game.get_current_player()
            player_color = game.current_player

            # Get legal moves
            legal_moves = Rules.get_legal_knight_moves(game.board, player_color)

            if not legal_moves:
                # Game should end due to no legal moves, but let's mark it
                unfinished_games.append({
                    "game_num": game_num + 1,
                    "reason": f"No legal moves for {player_color}",
                    "move_count": move_count,
                    "game": game,
                })
                break

            try:
                move_to, extra_apple = current_player.get_move(game.board, legal_moves)
                success = game.make_move(move_to, extra_apple)
                if not success:
                    unfinished_games.append({
                        "game_num": game_num + 1,
                        "reason": "Move execution failed",
                        "move_count": move_count,
                        "game": game,
                    })
                    break
                move_count += 1
            except Exception as e:
                unfinished_games.append({
                    "game_num": game_num + 1,
                    "reason": f"Exception during move: {str(e)[:100]}",
                    "move_count": move_count,
                    "game": game,
                })
                break

        if game.game_over:
            finished_games += 1
            total_moves += move_count
            max_moves = max(max_moves, move_count)
            min_moves = min(min_moves, move_count)
        elif move_count >= max_allowed_moves:
            unfinished_games.append({
                "game_num": game_num + 1,
                "reason": f"Exceeded max moves ({max_allowed_moves})",
                "move_count": move_count,
                "game": game,
            })

    return {
        "finished": finished_games,
        "total": num_games,
        "unfinished_count": len(unfinished_games),
        "avg_moves": total_moves / finished_games if finished_games > 0 else 0,
        "max_moves": max_moves if finished_games > 0 else 0,
        "min_moves": min_moves if finished_games > 0 else float('inf'),
        "unfinished_games": unfinished_games,
    }


def _create_player(player_type: str, color: str):
    """Create a player of the specified type."""
    if player_type == "Random":
        return RandomPlayer(f"{color.capitalize()}-Random")
    elif player_type == "PPO":
        try:
            return PPOPlayer(color, "data/models/ppo_self_play/best_model/best_model.zip")
        except Exception as e:
            print(f"❌ Failed to load PPO model: {e}")
            return None
    return None


def print_game_replay(game_data: dict) -> None:
    """Print a complete game replay."""
    game = game_data["game"]
    
    print(f"\n{'='*70}")
    print(f"UNFINISHED GAME #{game_data['game_num']} - COMPLETE REPLAY")
    print(f"{'='*70}")
    print(f"Reason: {game_data['reason']}")
    print(f"Total moves played: {game_data['move_count']}")
    print(f"\nFinal game state:")
    print(f"  White position: {game.board.white_pos}")
    print(f"  Black position: {game.board.black_pos}")
    print(f"  Brown apples remaining: {game.board.brown_apples_remaining}")
    print(f"  Golden apples remaining: {game.board.golden_apples_remaining}")
    print(f"  Golden phase started: {game.board.golden_phase_started}")
    print(f"  Game over: {game.game_over}")
    print(f"  Winner: {game.winner}")
    
    print("\nFinal board state:")
    print(GameVisualizer.visualize_board(game.board))
    
    # Print move history if available
    if game.log_data:
        print(f"\nMove history ({len(game.log_data)} moves):")
        for i, move in enumerate(game.log_data, 1):
            player = "White ♘" if move["turn"] == "white" else "Black ♞"
            extra = f" + Apple @{move['extra_apple']}" if move["extra_apple"] else ""
            print(f"  {i}. {player}: {move['white_pos'] if move['turn'] == 'white' else move['black_pos']} "
                  f"→ {move['move_to']}{extra}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Debug gameplay and detect unfinished games")
    parser.add_argument("--games", type=int, default=50, help="Number of games per simulation")
    parser.add_argument("--first-moves", action="store_true", help="Show first moves visualization only")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("PFERDEÄPFEL GAMEPLAY DEBUG & VISUALIZATION")
    print("="*70)

    if args.first_moves:
        # Only show first moves for each mode
        for mode in [1, 2, 3]:
            simulate_and_display_first_moves(mode)
        return

    # Run full simulations for each mode
    modes_to_test = [1, 2, 3]
    matchups = [
        ("Random", "Random", "Random vs Random"),
        ("Random", "PPO", "Random vs PPO"),
        ("PPO", "PPO", "PPO vs PPO"),
    ]

    for mode in modes_to_test:
        print(f"\n\n{'='*70}")
        print(f"MODE {mode} SIMULATION ({args.games} GAMES PER MATCHUP)")
        print(f"{'='*70}\n")

        for white_type, black_type, display_name in matchups:
            print(f"\n{display_name}:")
            print("─" * 50)

            results = run_simulation(white_type, black_type, mode, args.games)

            # Print summary
            print(f"✓ Finished: {results['finished']}/{results['total']}")
            print(f"✗ Unfinished: {results['unfinished_count']}")

            if results['finished'] > 0:
                print(f"Average moves: {results['avg_moves']:.1f}")
                print(f"Move range: {results['min_moves']} - {results['max_moves']}")

            # Print unfinished games
            if results['unfinished_games']:
                print(f"\n⚠️  UNFINISHED GAMES DETECTED ({len(results['unfinished_games'])}):")
                for uf_game in results['unfinished_games'][:2]:  # Show first 2 in summary
                    print(f"  • Game #{uf_game['game_num']}: {uf_game['reason']} (move {uf_game['move_count']})")

                # Print detailed replay for each unfinished game
                for uf_game in results['unfinished_games']:
                    print_game_replay(uf_game)


if __name__ == "__main__":
    main()
