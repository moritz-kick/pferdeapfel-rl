import time
import sys
from src.game.game import Game
from src.players.random import RandomPlayer


def run_stress_test(num_games=100):
    print(f"Starting stress test: {num_games} games (Random vs Random)...")

    total_moves = 0
    total_time = 0
    games_completed = 0
    errors = 0

    white_player = RandomPlayer("White")
    black_player = RandomPlayer("Black")

    for i in range(num_games):
        start_time = time.time()
        game = Game(white_player, black_player, mode=3, logging=False)

        moves_in_game = 0
        try:
            while not game.game_over:
                # Safety break for infinite loops
                if moves_in_game > 1000:
                    raise TimeoutError("Game exceeded 1000 moves")

                current_player = game.get_current_player()
                legal_moves = game.get_legal_moves()

                if not legal_moves:
                    # Should be handled by game.make_move or game logic, but just in case
                    break

                move_to, extra_apple = current_player.get_move(game.board, legal_moves)
                success = game.make_move(move_to, extra_apple)

                if not success:
                    raise RuntimeError(f"Invalid move attempted by {current_player.name}")

                moves_in_game += 1

            end_time = time.time()
            duration = end_time - start_time

            total_moves += moves_in_game
            total_time += duration
            games_completed += 1

            # Optional: Print progress every 10 games
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_games} games...")

        except Exception as e:
            errors += 1
            print(f"Error in game {i + 1}: {e}")
            # Continue to next game

    print("\n--- Stress Test Results ---")
    print(f"Games Completed: {games_completed}/{num_games}")
    print(f"Errors: {errors}")

    if games_completed > 0:
        avg_moves = total_moves / games_completed
        avg_time = total_time / games_completed
        print(f"Total Time: {total_time:.4f}s")
        print(f"Average Time per Game: {avg_time:.4f}s")
        print(f"Average Moves per Game: {avg_moves:.2f}")
    else:
        print("No games completed successfully.")


if __name__ == "__main__":
    run_stress_test()
