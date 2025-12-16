"""Main GUI for Pferdeäpfel game using PySide6."""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.game.game import Game
from src.game.rules import Rules
from src.players.base import Player
from src.players.greedy import GreedyPlayer
from src.players.heuristic_player import HeuristicPlayer, HeuristicPlayerV2, HeuristicPlayerV3
from src.players.human import HumanPlayer
from src.players.minimax import MinimaxPlayer
from src.players.random import RandomPlayer
from src.utils.debug_util import write_debug_log
from src.utils.toon_parser import parse_toon


class BoardWidget(QWidget):
    """Widget that displays the game board."""

    SQUARE_SIZE = 60
    BOARD_MARGIN = 20

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the board widget."""
        super().__init__(parent)
        self.game: Optional[Game] = None
        self.selected_square: Optional[tuple[int, int]] = None
        # Mode 1 & 3: Move waiting for extra/required apple placement
        self.pending_move: Optional[tuple[int, int]] = None
        self.pending_apple: Optional[tuple[int, int]] = None
        self.legal_moves: list[tuple[int, int]] = []
        self.setMinimumSize(
            self.BOARD_MARGIN * 2 + self.SQUARE_SIZE * 8,
            self.BOARD_MARGIN * 2 + self.SQUARE_SIZE * 8,
        )

    def set_game(self, game: Optional[Game]) -> None:
        """Set the game instance to display."""
        self.game = game
        self.update_legal_moves()
        self.update()

    def update_legal_moves(self) -> None:
        """Update the list of legal moves for the current player."""
        if self.game and not self.game.game_over:
            self.legal_moves = self.game.get_legal_moves()
        else:
            self.legal_moves = []

    def get_square_at(self, x: int, y: int) -> Optional[tuple[int, int]]:
        """Get the board square at pixel coordinates."""
        if not self.game:
            return None

        col = (x - self.BOARD_MARGIN) // self.SQUARE_SIZE
        row = (y - self.BOARD_MARGIN) // self.SQUARE_SIZE

        if 0 <= row < 8 and 0 <= col < 8:
            return (row, col)
        return None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse clicks on the board."""
        if not self.game or self.game.game_over:
            return

        pos = event.position()
        square = self.get_square_at(int(pos.x()), int(pos.y()))
        if square is None:
            return

        current_player = self.game.get_current_player()
        if not isinstance(current_player, HumanPlayer):
            return

        mode = self.game.board.mode

        # --- MODE 1: Free Placement (Move, then Apple) ---
        if mode == 1:
            # If we have a pending move, this click is for the apple placement
            if self.pending_move is not None:
                # Execute move with the apple placement (Game validates apple placement)
                success = self.game.make_move(self.pending_move, square)
                if success:
                    self.pending_move = None
                    self.selected_square = None
                    self.update_legal_moves()
                    self.update()
                    parent = self.parent()
                    if isinstance(parent, GameWindow):
                        parent.update_ui()
                else:
                    QMessageBox.warning(self, "Invalid Move", "Move invalid (apple must be on empty square).")
                    self.pending_move = None
                    self.selected_square = None
                    self.update()

            # No pending move, this click is to select the move destination
            else:
                if square in self.legal_moves:
                    self.pending_move = square
                    self.selected_square = square
                    self.update()

        # --- MODE 2: Trail Placement (Just Move) ---
        elif mode == 2:
            if square in self.legal_moves:
                success = self.game.make_move(square)
                if success:
                    self.update_legal_moves()
                    self.update()
                    parent = self.parent()
                    if isinstance(parent, GameWindow):
                        parent.update_ui()

        # --- MODE 3: Classic (Move, then Optional Apple) ---
        elif mode == 3:
            # Right-click to confirm move without extra apple
            if event.button() == Qt.MouseButton.RightButton and self.pending_move is not None:
                success = self.game.make_move(self.pending_move, None)
                if success:
                    self.pending_move = None
                    self.selected_square = None
                    self.update_legal_moves()
                    self.update()
                    parent = self.parent()
                    if isinstance(parent, GameWindow):
                        parent.update_ui()

            # If we have a pending move, this click is for extra apple placement
            elif self.pending_move is not None:
                # Place extra apple and complete the move (Game validates apple placement)
                success = self.game.make_move(self.pending_move, square)
                if success:
                    self.pending_move = None
                    self.selected_square = None
                    self.update_legal_moves()
                    self.update()
                    parent = self.parent()
                    if isinstance(parent, GameWindow):
                        parent.update_ui()
                else:
                    # Invalid placement (e.g., would block White, or occupied)
                    QMessageBox.warning(self, "Invalid Move", "Invalid move or apple placement.")
                    self.pending_move = None
                    self.selected_square = None
                    self.update()

            elif square in self.legal_moves:
                # Select a move - wait for extra apple placement
                # or right-click to confirm without extra
                self.pending_move = square
                self.selected_square = square
                self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint the board."""
        if not self.game:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        board = self.game.board

        # Draw board squares
        for row in range(8):
            for col in range(8):
                x = self.BOARD_MARGIN + col * self.SQUARE_SIZE
                y = self.BOARD_MARGIN + row * self.SQUARE_SIZE

                # Alternate square colors
                is_light = (row + col) % 2 == 0
                color = QColor(240, 217, 181) if is_light else QColor(181, 136, 99)
                painter.fillRect(x, y, self.SQUARE_SIZE, self.SQUARE_SIZE, color)

                # Highlight legal moves
                if (row, col) in self.legal_moves:
                    highlight = QColor(144, 238, 144, 150)
                    painter.fillRect(x, y, self.SQUARE_SIZE, self.SQUARE_SIZE, highlight)

                # Highlight selected square (pending move or pending apple)
                if self.selected_square == (row, col):
                    color = QColor(255, 0, 0)
                    if self.pending_apple == (row, col):
                        color = QColor(139, 69, 19)  # Brown for apple

                    painter.setPen(QPen(color, 3))
                    painter.drawRect(x, y, self.SQUARE_SIZE, self.SQUARE_SIZE)

                # Draw contents
                square_val = board.get_square(row, col)
                if square_val == board.WHITE_HORSE:
                    painter.setPen(QPen(QColor(255, 255, 255), 2))
                    painter.setBrush(QColor(255, 255, 255))
                    painter.drawEllipse(x + 10, y + 10, self.SQUARE_SIZE - 20, self.SQUARE_SIZE - 20)
                    painter.setPen(QPen(QColor(0, 0, 0), 2))
                    painter.drawText(
                        x + self.SQUARE_SIZE // 2 - 5,
                        y + self.SQUARE_SIZE // 2 + 5,
                        "W",
                    )
                elif square_val == board.BLACK_HORSE:
                    painter.setPen(QPen(QColor(0, 0, 0), 2))
                    painter.setBrush(QColor(0, 0, 0))
                    painter.drawEllipse(x + 10, y + 10, self.SQUARE_SIZE - 20, self.SQUARE_SIZE - 20)
                    painter.setPen(QPen(QColor(255, 255, 255), 1))
                    painter.drawText(
                        x + self.SQUARE_SIZE // 2 - 5,
                        y + self.SQUARE_SIZE // 2 + 5,
                        "B",
                    )
                elif square_val == board.BROWN_APPLE:
                    painter.setPen(QPen(QColor(139, 69, 19), 2))
                    painter.setBrush(QColor(139, 69, 19))
                    painter.drawEllipse(x + 15, y + 15, self.SQUARE_SIZE - 30, self.SQUARE_SIZE - 30)
                elif square_val == board.GOLDEN_APPLE:
                    painter.setPen(QPen(QColor(255, 215, 0), 2))
                    painter.setBrush(QColor(255, 215, 0))
                    painter.drawEllipse(x + 15, y + 15, self.SQUARE_SIZE - 30, self.SQUARE_SIZE - 30)


class GameWindow(QWidget):
    """Main game window."""

    def __init__(self) -> None:
        """Initialize the game window."""
        super().__init__()
        self.project_root = Path(__file__).resolve().parents[2]
        self.game: Optional[Game] = None
        self.player_factories = self._build_player_factories()
        self.log_dir = Path("data/logs/game")
        self.debug_log_dir = Path("data/logs/debug")
        self.init_ui()
        self.load_config()
        self.new_game()

    def closeEvent(self, event: Any) -> None:
        """Handle window close event - write debug log if game is incomplete."""
        if self.game and self.logging_button.isChecked() and not self.game.game_over:
            if isinstance(self.white_player, RandomPlayer) and isinstance(self.black_player, RandomPlayer):
                try:
                    log_path = write_debug_log(self.game, self.debug_log_dir)
                    logging.info(f"Debug log written to: {log_path}")
                except Exception as e:
                    logging.error(f"Failed to write debug log: {e}")

        event.accept()

    def _build_player_factories(self) -> dict[str, Callable[[str], Player]]:
        """Return mapping of selector key to player factory."""
        factories: dict[str, Callable[[str], Player]] = {
            "human": lambda color: HumanPlayer(color.capitalize()),
            "random": lambda color: RandomPlayer(color.capitalize()),
            "greedy": lambda color: GreedyPlayer(color),
            # Heuristic variants v1–v3
            "heuristic": lambda color: HeuristicPlayerV3(color.capitalize()),  # default to latest
            "heuristic_v1": lambda color: HeuristicPlayer(color.capitalize()),
            "heuristic_v2": lambda color: HeuristicPlayerV2(color.capitalize()),
            "heuristic_v3": lambda color: HeuristicPlayerV3(color.capitalize()),
            "minimax": lambda color: MinimaxPlayer(color.capitalize()),
        }

        return factories

    def init_ui(self) -> None:
        """Initialize the UI components."""
        self.setWindowTitle("Pferdeäpfel")
        self.setMinimumSize(700, 600)

        layout = QVBoxLayout()

        # Player selection
        player_layout = QHBoxLayout()
        player_layout.addWidget(QLabel("White:"))
        self.white_combo = QComboBox()
        self.white_combo.addItems(list(self.player_factories.keys()))
        self.white_combo.currentTextChanged.connect(self.on_player_changed)
        player_layout.addWidget(self.white_combo)

        player_layout.addWidget(QLabel("Black:"))
        self.black_combo = QComboBox()
        self.black_combo.addItems(list(self.player_factories.keys()))
        self.black_combo.currentTextChanged.connect(self.on_player_changed)
        player_layout.addWidget(self.black_combo)

        player_layout.addStretch()
        layout.addLayout(player_layout)

        # Minimax Settings
        settings_layout = QHBoxLayout()
        self.optimal_opening_checkbox = QCheckBox("Use Optimal Opening Moves (Minimax)")
        self.optimal_opening_checkbox.setChecked(True)
        self.optimal_opening_checkbox.toggled.connect(self.on_player_changed)  # Recreate players on toggle
        settings_layout.addWidget(self.optimal_opening_checkbox)
        settings_layout.addStretch()
        layout.addLayout(settings_layout)

        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["1: Free Placement", "2: Trail Placement", "3: Classic"])
        self.mode_combo.setCurrentIndex(2)  # Default to Classic
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # Board
        self.board_widget = BoardWidget()
        self.board_widget.set_game(None)
        layout.addWidget(self.board_widget, alignment=Qt.AlignmentFlag.AlignCenter)

        # Status bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Starting game...")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()

        self.turn_label = QLabel("")
        status_layout.addWidget(self.turn_label)

        layout.addLayout(status_layout)

        # Info labels
        info_layout = QHBoxLayout()
        self.apple_label = QLabel("")
        info_layout.addWidget(self.apple_label)
        info_layout.addStretch()
        self.depth_label = QLabel("")  # New depth label
        info_layout.addWidget(self.depth_label)
        # Add some space
        info_layout.addSpacing(20)
        self.ai_status_label = QLabel("")
        info_layout.addWidget(self.ai_status_label)
        layout.addLayout(info_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.logging_button = QPushButton("Logging: ON")
        self.logging_button.setCheckable(True)
        self.logging_button.setChecked(True)
        self.logging_button.toggled.connect(self.on_logging_toggled)
        button_layout.addWidget(self.logging_button)

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.on_undo)
        button_layout.addWidget(self.undo_button)

        self.restart_button = QPushButton("Restart")
        self.restart_button.clicked.connect(self.new_game)
        button_layout.addWidget(self.restart_button)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def load_config(self) -> None:
        """Load GUI configuration from config.json."""
        config_path = Path(__file__).parent / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    white = config.get("white", "human")
                    black = config.get("black", "random")
                    logs = config.get("logs", True)
                    mode = config.get("mode", 3)  # Default to 3 (Classic)

                    idx = self.white_combo.findText(white)
                    if idx >= 0:
                        self.white_combo.setCurrentIndex(idx)
                    idx = self.black_combo.findText(black)
                    if idx >= 0:
                        self.black_combo.setCurrentIndex(idx)

                    self.logging_button.setChecked(logs)

                    # Set mode (1 -> index 0, 2 -> index 1, 3 -> index 2)
                    if 1 <= mode <= 3:
                        self.mode_combo.setCurrentIndex(mode - 1)
            except Exception:
                pass

    def create_players(self) -> tuple[Player, Player]:
        """Create player instances based on combo box selections."""
        white_type = self.white_combo.currentText()
        black_type = self.black_combo.currentText()

        # Helper to create player with current settings
        def create(type_name: str, color: str) -> Player:
            if type_name == "minimax":
                return MinimaxPlayer(
                    color.capitalize(),
                    use_optimal_opening=self.optimal_opening_checkbox.isChecked(),
                )

            factory = self.player_factories.get(type_name, self.player_factories["human"])
            # Minimax factory in the dict is still valid but we bypass it here for the custom setting
            return factory(color.capitalize() if type_name != "greedy" else color)

        white_player = create(white_type, "white")
        black_player = create(black_type, "black")

        return white_player, black_player

    def new_game(self) -> None:
        """Start a new game."""
        try:
            white_player, black_player = self.create_players()
        except Exception as exc:
            logging.error("Failed to create players: %s", exc)
            QMessageBox.critical(self, "Player Error", f"Could not start a new game: {exc}")
            return
        self.white_player = white_player
        self.black_player = black_player
        logging_enabled = self.logging_button.isChecked()

        mode_text = self.mode_combo.currentText()
        mode = int(mode_text.split(":")[0])

        self.game = Game(white_player, black_player, mode=mode, logging=logging_enabled)
        self.board_widget.pending_move = None
        self.board_widget.pending_apple = None
        self.board_widget.selected_square = None
        self.board_widget.set_game(self.game)
        self.update_ui()

    def on_player_changed(self) -> None:
        """Handle player selection change."""
        if self.game:
            self.new_game()

    def on_mode_changed(self) -> None:
        """Handle mode selection change."""
        self.new_game()

    def on_logging_toggled(self, checked: bool) -> None:
        """Handle logging toggle."""
        if self.game:
            self.game.logging = checked
        self.logging_button.setText(f"Logging: {'ON' if checked else 'OFF'}")

    def on_undo(self) -> None:
        """Handle undo button click."""
        if self.game and self.game.undo_move():
            self.board_widget.update_legal_moves()
            self.update_ui()

    def update_ui(self) -> None:
        """Update UI elements to reflect current game state."""
        if not self.game:
            return

        self.board_widget.update()

        # Update status
        if self.game.game_over:
            if self.game.winner == "draw":
                self.status_label.setText("Game Over! It's a Draw!")
                self.turn_label.setText("")
            else:
                winner_name = "White" if self.game.winner == "white" else "Black"
                # Ensure winner is not None before calling calculate_score
                winner_arg = self.game.winner if self.game.winner else "white"
                score = Rules.calculate_score(self.game.board, winner_arg)
                self.status_label.setText(f"Game Over! {winner_name} wins! Score: {score}")
                self.turn_label.setText("")

            # Save log if logging
            if self.game.logging:
                log_path = self.game.save_log(self.log_dir)
                if log_path:
                    QMessageBox.information(self, "Game Logged", f"Game saved to {log_path}")
        else:
            current = self.game.get_current_player()
            status_text = f"Current player: {current.name}"
            if isinstance(current, HumanPlayer):
                mode = self.game.board.mode
                if mode == 1:
                    if self.board_widget.pending_move:
                        status_text += " - Select empty square to place apple"
                    else:
                        status_text += " - Select move destination"
                elif mode == 3 and self.board_widget.pending_move:
                    status_text += " - Click empty square for extra apple, or right-click to skip"
            self.status_label.setText(status_text)
            self.turn_label.setText(f"Turn: {self.game.current_player.capitalize()}")

        # Update apple counts
        board = self.game.board
        apple_text = f"Brown: {board.brown_apples_remaining}, Golden: {board.golden_apples_remaining}"
        if board.golden_phase_started:
            apple_text += " (Golden phase active)"
        self.apple_label.setText(apple_text)

        # Handle AI player moves
        if not self.game.game_over:
            current_player = self.game.get_current_player()
            if not isinstance(current_player, HumanPlayer):
                # Use QTimer to allow UI to update before AI move
                QTimer.singleShot(100, self.make_ai_move)

    def make_ai_move(self) -> None:
        """Make a move for an AI player."""
        if not self.game or self.game.game_over:
            return

        current_player = self.game.get_current_player()
        if isinstance(current_player, HumanPlayer):
            return

        try:
            logging.debug(f"[GUI] make_ai_move: {current_player.name}'s turn")
            legal_moves = self.game.get_legal_moves()
            logging.debug(f"[GUI] Legal moves: {len(legal_moves)}")

            if not legal_moves:
                logging.debug(f"[GUI] No legal moves for {current_player.name}")
                return

            logging.debug(f"[GUI] Getting move from {current_player.name}...")
            move_to, extra_apple = current_player.get_move(self.game.board, legal_moves)
            logging.debug(f"[GUI] {current_player.name} chose: move_to={move_to}, extra_apple={extra_apple}")

            success = self.game.make_move(move_to, extra_apple)
            logging.debug(f"[GUI] Move success: {success}")

            if success:
                self.board_widget.update_legal_moves()
                self.update_ui()

                # Update AI status label
                metadata = getattr(current_player, "last_move_metadata", None)
                if metadata:
                    source = metadata.get("source", "unknown")
                    reason = metadata.get("reason", "")
                    status_text = f"AI Status: {source.upper()}"
                    if reason:
                        status_text += f" ({reason})"
                    self.ai_status_label.setText(status_text)

                    # Log if enabled
                    if self.game.logging:
                        logging.info(f"AI Move Metadata: {metadata}")
                else:
                    self.ai_status_label.setText("")

                # Update Minimax Depth stats
                if isinstance(current_player, MinimaxPlayer):
                    depth = current_player.last_search_depth
                    nodes = current_player.nodes_evaluated
                    self.depth_label.setText(f"Depth: {depth}, Nodes: {nodes}")
                else:
                    self.depth_label.setText("")

        except Exception as e:
            logging.error(f"[GUI ERROR] AI move failed: {e}")
            import traceback

            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"AI move failed: {e}")

    def keyPressEvent(self, event: Any) -> None:
        """Handle key press events."""
        if event.key() == Qt.Key.Key_H:
            QMessageBox.information(
                self,
                "Help",
                "Controls:\n"
                "- Left Click: Select piece / Move / Place Apple\n"
                "- Right Click: Confirm move without optional placement (Mode 3)\n"
                "- 1, 2, 3: Restart in Mode 1, 2, or 3\n"
                "- H: Show this help\n\n"
                "Modes:\n"
                "1. Free Placement: Move -> Place apple\n"
                "2. Trail Placement: Move -> Leave trail\n"
                "3. Classic: Mandatory Apple -> Move -> Optional Apple",
            )
        elif event.key() == Qt.Key.Key_1:
            self.mode_combo.setCurrentIndex(0)
        elif event.key() == Qt.Key.Key_2:
            self.mode_combo.setCurrentIndex(1)
        elif event.key() == Qt.Key.Key_3:
            self.mode_combo.setCurrentIndex(2)

    def load_game_log(self, log_path: Path) -> None:
        """Load and replay a game log."""
        try:
            with open(log_path) as f:
                if log_path.suffix == ".toon":
                    data = parse_toon(f.read())
                else:
                    data = json.load(f)

            # Reset game to Classic Mode (Mode 3)
            self.mode_combo.setCurrentIndex(2)
            self.new_game()
            assert self.game is not None

            moves = data.get("moves", [])
            if not moves:
                QMessageBox.information(self, "Log Loaded", "Log contains no moves.")
                return

            # Determine starting player from the first move
            first_move_turn = moves[0]["turn"]
            if self.game.current_player != first_move_turn:
                self.game.switch_turn()

            # Replay moves
            replayed_count = 0
            for move in moves:
                player_name = move["turn"]
                move_to = tuple(move["move_to"])
                extra_apple = tuple(move["extra_apple"]) if move["extra_apple"] else None

                # Ensure correct player
                if self.game.current_player != player_name:
                    # Force turn switch if out of sync
                    logging.warning(f"Warning: Turn mismatch. Log: {player_name}, Game: {self.game.current_player}")
                    self.game.switch_turn()

                success = self.game.make_move(move_to, extra_apple)
                if not success:
                    logging.error(f"Failed to replay move {replayed_count + 1}: {move}")
                    QMessageBox.warning(self, "Replay Error", f"Failed to replay move {replayed_count + 1}")
                    break
                replayed_count += 1

            self.update_ui()
            QMessageBox.information(self, "Log Loaded", f"Successfully replayed {replayed_count} moves.")

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load log: {e}")


def main() -> None:
    """Main entry point for the GUI."""
    import argparse

    # Configure logging
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    )

    parser = argparse.ArgumentParser(description="Pferdeäpfel Game GUI")
    parser.add_argument("--load-log", type=Path, help="Path to game log to load")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args, unknown = parser.parse_known_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    app = QApplication(sys.argv[:1] + unknown)
    window = GameWindow()

    if args.load_log:
        # Use QTimer to load after window is shown, to ensure dialogs appear correctly
        QTimer.singleShot(100, lambda: window.load_game_log(args.load_log))

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
