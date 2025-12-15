"""Evaluation result storage module."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GameResult:
    """Represents the result of a single game evaluation."""

    timestamp: float
    mode: int
    white_player: str
    black_player: str
    winner: str  # "white", "black", "draw"
    moves: int
    duration: float
    white_error: Optional[str] = None
    black_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameResult":
        return cls(**data)


class ResultStorage:
    """Handles storage and retrieval of evaluation results."""

    def __init__(self, data_dir: str = "data/evaluation"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, mode: int) -> Path:
        return self.data_dir / f"eval_{mode}.jsonl"

    def save_result(self, result: GameResult) -> None:
        """Append a result to the storage file."""
        file_path = self._get_file_path(result.mode)
        with open(file_path, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    def load_results(self, mode: int) -> List[GameResult]:
        """Load all results for a given mode."""
        file_path = self._get_file_path(mode)
        if not file_path.exists():
            return []

        results = []
        with open(file_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    results.append(GameResult.from_dict(data))
                except json.JSONDecodeError:
                    continue

        return results

    def get_match_count(self, mode: int, white: str, black: str) -> int:
        """Count how many games have been played between two players in a mode."""
        results = self.load_results(mode)
        count = 0
        for r in results:
            if r.white_player == white and r.black_player == black:
                count += 1
        return count

    def clear_results(self, mode: int) -> None:
        """Delete all results for a given mode."""
        file_path = self._get_file_path(mode)
        if file_path.exists():
            file_path.unlink()
