"""Ranking and matchmaking system for evaluation."""

from typing import List, Optional, Tuple, Type

from src.evaluation.storage import ResultStorage
from src.players.base import Player


class RankingSystem:
    """Manages rankings and suggests matches."""

    def __init__(self, storage: ResultStorage) -> None:
        """Initialize with result storage."""
        self.storage = storage

    def suggest_next_match(
        self, mode: int, player_classes: List[Type[Player]], min_games: int = 50
    ) -> Optional[Tuple[Type[Player], Type[Player], str, str]]:
        """
        Suggest the next match to run to ensure all pairs play min_games per side.
        Includes self-play matchups (e.g., RandomPlayer vs RandomPlayer) and ensures
        color balance for fairness.

        Returns:
            (WhiteClass, BlackClass, WhiteName, BlackName) or None if done.
        """
        results = self.storage.load_results(mode)

        # Count games played between pairs (white_name -> black_name)
        # This ensures color balance: we need min_games with A as white vs B as black
        # AND min_games with B as white vs A as black
        counts = {cls.__name__: {cls2.__name__: 0 for cls2 in player_classes} for cls in player_classes}

        for r in results:
            if r.white_player in counts and r.black_player in counts[r.white_player]:
                counts[r.white_player][r.black_player] += 1

        # Find pair with fewest games (including self-matchups)
        best_pair = None
        min_count = float("inf")

        for p1 in player_classes:
            name1 = p1.__name__
            for p2 in player_classes:
                name2 = p2.__name__

                # Include self-matchups (e.g., RandomPlayer vs RandomPlayer, GreedyPlayer vs GreedyPlayer)
                # For self-play, we still need to balance white/black games
                count = counts[name1][name2]
                if count < min_games:
                    if count < min_count:
                        min_count = count
                        best_pair = (p1, p2, name1, name2)

        return best_pair

    def get_rankings(self, mode: int, player_names: List[str]) -> List[Tuple[str, float]]:
        """
        Calculate simple win rate rankings.

        Returns:
            List of (player_name, score) sorted by score desc.
        """
        results = self.storage.load_results(mode)

        stats = {name: {"wins": 0, "games": 0} for name in player_names}

        for r in results:
            if r.winner == "white" and r.white_player in stats:
                stats[r.white_player]["wins"] += 1
            elif r.winner == "black" and r.black_player in stats:
                stats[r.black_player]["wins"] += 1

            if r.white_player in stats:
                stats[r.white_player]["games"] += 1
            if r.black_player in stats:
                stats[r.black_player]["games"] += 1

        rankings = []
        for name, data in stats.items():
            if data["games"] > 0:
                score = data["wins"] / data["games"]
            else:
                score = 0.0
            rankings.append((name, score))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
