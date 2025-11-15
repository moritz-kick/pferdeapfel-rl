"""Main GUI for Pferdeäpfel game using PySide6."""

import json
import random
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
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
        self.pending_move: Optional[tuple[int, int]] = None  # Move waiting for extra apple or confirmation
        self.legal_moves: list[tuple[int, int]] = []
        self.setMinimumSize(
            self.BOARD_MARGIN * 2 + self.SQUARE_SIZE * 8,
            self.BOARD_MARGIN * 2 + self.SQUARE_SIZE * 8,
        )

    def set_game(self, game: Game) -> None:
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

    def mousePressEvent(self, event) -> None:
        """Handle mouse clicks on the board."""
        if not self.game or self.game.game_over:
            return

        square = self.get_square_at(event.position().x(), event.position().y())
        if square is None:
            return

        current_player = self.game.get_current_player()
        if isinstance(current_player, HumanPlayer):
            # Right-click to confirm move without extra apple
            if event.button() == Qt.RightButton and self.pending_move is not None:
                success = self.game.make_move(self.pending_move, None)
                if success:
                    self.pending_move = None
                    self.selected_square = None
                    self.update_legal_moves()
                    self.update()
                    self.parent().update_ui()  # type: ignore
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
                        self.parent().update_ui()  # type: ignore
                    else:
                        # Invalid placement (e.g., would block White)
                        from PySide6.QtWidgets import QMessageBox
                        QMessageBox.warning(
                            self.parent(),  # type: ignore
                            "Invalid Placement",
                            "Cannot place apple here (would block White's moves)."
                        )
                        self.pending_move = None
                        self.update()
                else:
                    # Square not empty, cancel pending move
                    self.pending_move = None
                    self.update()
            elif square in self.legal_moves:
                # Select a move - wait for extra apple placement or right-click to confirm without extra
                self.pending_move = square
                self.selected_square = square
                self.update()

    def paintEvent(self, event) -> None:
        """Paint the board."""
        if not self.game:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

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

                # Highlight selected square (pending move)
                if self.selected_square == (row, col) or self.pending_move == (row, col):
                    painter.setPen(QPen(QColor(255, 0, 0), 3))
                    painter.drawRect(x, y, self.SQUARE_SIZE, self.SQUARE_SIZE)

                # Draw contents
                square_val = board.get_square(row, col)
                if square_val == board.WHITE_HORSE:
                    painter.setPen(QPen(QColor(255, 255, 255), 2))
                    painter.setBrush(QColor(255, 255, 255))
                    painter.drawEllipse(
                        x + 10, y + 10, self.SQUARE_SIZE - 20, self.SQUARE_SIZE - 20
                    )
                    painter.drawText(
                        x + self.SQUARE_SIZE // 2 - 5,
                        y + self.SQUARE_SIZE // 2 + 5,
                        "W",
                    )
                elif square_val == board.BLACK_HORSE:
                    painter.setPen(QPen(QColor(0, 0, 0), 2))
                    painter.setBrush(QColor(0, 0, 0))
                    painter.drawEllipse(
                        x + 10, y + 10, self.SQUARE_SIZE - 20, self.SQUARE_SIZE - 20
                    )
                    painter.setPen(QPen(QColor(255, 255, 255), 1))
                    painter.drawText(
                        x + self.SQUARE_SIZE // 2 - 5,
                        y + self.SQUARE_SIZE // 2 + 5,
                        "B",
                    )
                elif square_val == board.BROWN_APPLE:
                    painter.setPen(QPen(QColor(139, 69, 19), 2))
                    painter.setBrush(QColor(139, 69, 19))
                    painter.drawEllipse(
                        x + 15, y + 15, self.SQUARE_SIZE - 30, self.SQUARE_SIZE - 30
                    )
                elif square_val == board.GOLDEN_APPLE:
                    painter.setPen(QPen(QColor(255, 215, 0), 2))
                    painter.setBrush(QColor(255, 215, 0))
                    painter.drawEllipse(
                        x + 15, y + 15, self.SQUARE_SIZE - 30, self.SQUARE_SIZE - 30
                    )


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

        # Board
        self.board_widget = BoardWidget()
        self.board_widget.set_game(None)
        layout.addWidget(self.board_widget, alignment=Qt.AlignCenter)

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

    def create_players(self) -> tuple[HumanPlayer | RandomPlayer, HumanPlayer | RandomPlayer]:
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
        self.game = Game(white_player, black_player, logging=logging)
        self.board_widget.pending_move = None
        self.board_widget.selected_square = None
        self.board_widget.set_game(self.game)
        self.update_ui()

    def on_player_changed(self) -> None:
        """Handle player selection change."""
        if self.game:
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
            winner_name = "White" if self.game.winner == "white" else "Black"
            self.status_label.setText(f"Game Over! {winner_name} wins!")
            self.turn_label.setText("")
            # Save log if logging
            if self.game.logging:
                log_path = self.game.save_log(self.log_dir)
                if log_path:
                    QMessageBox.information(
                        self, "Game Logged", f"Game saved to {log_path}"
                    )
        else:
            current = self.game.get_current_player()
            status_text = f"Current player: {current.name}"
            if isinstance(current, HumanPlayer) and self.board_widget.pending_move:
                status_text += " - Click empty square for extra apple, or right-click to skip"
            self.status_label.setText(status_text)
            self.turn_label.setText(
                f"Turn: {self.game.current_player.capitalize()}"
            )

        # Update apple counts
        board = self.game.board
        apple_text = (
            f"Brown: {board.brown_apples_remaining}, "
            f"Golden: {board.golden_apples_remaining}"
        )
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


def main() -> None:
    """Main entry point for the GUI."""
    app = QApplication(sys.argv)
    window = GameWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
