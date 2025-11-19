"""Main GUI for Pferdeäpfel game using PySide6."""

import json
import sys
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QApplication,
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
from src.players.human import HumanPlayer
from src.players.random import RandomPlayer


class BoardWidget(QWidget):
    """Widget that displays the game board."""

    SQUARE_SIZE = 60
    BOARD_MARGIN = 20

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the board widget."""
        super().__init__(parent)
        self.game: Optional[Game] = None
        self.selected_square: Optional[tuple[int, int]] = None
        # Mode 3: Move waiting for extra apple or confirmation
        self.pending_move: Optional[tuple[int, int]] = None
        # Mode 1: Apple waiting for move
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

        # --- MODE 1: Free Placement (Apple FIRST, then Move) ---
        if mode == 1:
            # If we have a pending apple, this click is for the move
            if self.pending_apple is not None:
                if square in self.legal_moves:
                    # Execute move with the pending apple
                    success = self.game.make_move(square, self.pending_apple)
                    if success:
                        self.pending_apple = None
                        self.selected_square = None
                        self.update_legal_moves()
                        self.update()
                        parent = self.parent()
                        if isinstance(parent, GameWindow):
                            parent.update_ui()
                    else:
                        # Move failed (shouldn't happen if in legal_moves, but maybe blocking check)
                        QMessageBox.warning(self, "Invalid Move", "Move invalid.")
                        self.pending_apple = None
                        self.update()
                else:
                    # Clicked somewhere else, cancel pending apple
                    self.pending_apple = None
                    self.selected_square = None
                    self.update()

            # No pending apple, this click is to place the apple
            else:
                if self.game.board.is_empty(square[0], square[1]):
                    self.pending_apple = square
                    self.selected_square = square
                    self.update()
                else:
                    # Clicked occupied square
                    pass

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

        # --- MODE 3: Classic (Move FIRST, then Optional Apple) ---
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
                if self.game.board.is_empty(square[0], square[1]):
                    # Place extra apple and complete the move
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
                        # Invalid placement (e.g., would block White)
                        QMessageBox.warning(
                            self, "Invalid Placement", "Cannot place apple here (would block White's moves)."
                        )
                        self.pending_move = None
                        self.update()
                else:
                    # Square not empty, cancel pending move
                    self.pending_move = None
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
        self.game: Optional[Game] = None
        self.log_dir = Path("data/logs/game")
        self.init_ui()
        self.load_config()
        self.new_game()

    def init_ui(self) -> None:
        """Initialize the UI components."""
        self.setWindowTitle("Pferdeäpfel")
        self.setMinimumSize(700, 600)

        layout = QVBoxLayout()

        # Player selection
        player_layout = QHBoxLayout()
        player_layout.addWidget(QLabel("White:"))
        self.white_combo = QComboBox()
        self.white_combo.addItems(["human", "random"])
        self.white_combo.currentTextChanged.connect(self.on_player_changed)
        player_layout.addWidget(self.white_combo)

        player_layout.addWidget(QLabel("Black:"))
        self.black_combo = QComboBox()
        self.black_combo.addItems(["human", "random"])
        self.black_combo.currentTextChanged.connect(self.on_player_changed)
        player_layout.addWidget(self.black_combo)

        player_layout.addStretch()
        layout.addLayout(player_layout)

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

                    idx = self.white_combo.findText(white)
                    if idx >= 0:
                        self.white_combo.setCurrentIndex(idx)
                    idx = self.black_combo.findText(black)
                    if idx >= 0:
                        self.black_combo.setCurrentIndex(idx)
                    self.logging_button.setChecked(logs)
            except Exception:
                pass  # Use defaults

    def create_players(
        self,
    ) -> tuple[HumanPlayer | RandomPlayer, HumanPlayer | RandomPlayer]:
        """Create player instances based on combo box selections."""
        white_type = self.white_combo.currentText()
        black_type = self.black_combo.currentText()

        white_player: HumanPlayer | RandomPlayer = (
            HumanPlayer("White") if white_type == "human" else RandomPlayer("White")
        )
        black_player: HumanPlayer | RandomPlayer = (
            HumanPlayer("Black") if black_type == "human" else RandomPlayer("Black")
        )

        return white_player, black_player

    def new_game(self) -> None:
        """Start a new game."""
        white_player, black_player = self.create_players()
        logging = self.logging_button.isChecked()

        mode_text = self.mode_combo.currentText()
        mode = int(mode_text.split(":")[0])

        self.game = Game(white_player, black_player, mode=mode, logging=logging)
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
                # If game_over is True but winner is None (shouldn't happen):
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
                    if self.board_widget.pending_apple:
                        status_text += " - Select move destination"
                    else:
                        status_text += " - Select empty square to place apple"
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
            if isinstance(current_player, RandomPlayer):
                # Use QTimer to allow UI to update before AI move
                QTimer.singleShot(100, self.make_ai_move)

    def make_ai_move(self) -> None:
        """Make a move for an AI player."""
        if not self.game or self.game.game_over:
            return

        current_player = self.game.get_current_player()
        if not isinstance(current_player, RandomPlayer):
            return

        try:
            legal_moves = self.game.get_legal_moves()
            if not legal_moves:
                return

            move_to, extra_apple = current_player.get_move(self.game.board, legal_moves)
            success = self.game.make_move(move_to, extra_apple)

            if success:
                self.board_widget.update_legal_moves()
                self.update_ui()
        except Exception as e:
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
                "1. Free Placement: Place apple -> Move\n"
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
                data = json.load(f)

            # Reset game to Classic Mode (Mode 3)
            self.mode_combo.setCurrentIndex(2)
            self.new_game()

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
                    print(f"Warning: Turn mismatch. Log: {player_name}, Game: {self.game.current_player}")
                    self.game.switch_turn()

                success = self.game.make_move(move_to, extra_apple)
                if not success:
                    print(f"Failed to replay move {replayed_count + 1}: {move}")
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

    parser = argparse.ArgumentParser(description="Pferdeäpfel Game GUI")
    parser.add_argument("--load-log", type=Path, help="Path to game log to load")
    # Use parse_known_args because QApplication also handles some args
    args, unknown = parser.parse_known_args()

    app = QApplication(sys.argv[:1] + unknown)
    window = GameWindow()

    if args.load_log:
        # Use QTimer to load after window is shown, to ensure dialogs appear correctly
        QTimer.singleShot(100, lambda: window.load_game_log(args.load_log))

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
