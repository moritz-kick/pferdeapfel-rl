"""Performance profiling module for players against RandomPlayer."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Type

from src.evaluation.runner import GameRunner
from src.players.base import Player
from src.players.greedy import GreedyPlayer
from src.players.random import RandomPlayer


@dataclass
class PlayerProfile:
    """Performance profile for a player."""

    player_name: str
    player_version: str  # e.g., "v1", "v2"
    timestamp: float
    mode: int
    games_played: int
    avg_duration: float  # Average game duration in seconds
    total_duration: float  # Total time for all games
    avg_moves: float  # Average moves per game
    min_duration: float
    max_duration: float
    metadata: Dict  # Additional metadata (e.g., nodes_evaluated for Minimax)


class PerformanceProfiler:
    """Profiles player performance against RandomPlayer."""

    def __init__(self, data_dir: str = "data/evaluation"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.profile_file = self.data_dir / "player_profiles.jsonl"

    def _load_profiles(self) -> Dict[str, List[PlayerProfile]]:
        """Load all existing profiles, grouped by player name."""
        profiles_by_player: Dict[str, List[PlayerProfile]] = {}

        if not self.profile_file.exists():
            return profiles_by_player

        with open(self.profile_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    profile = PlayerProfile(**data)
                    if profile.player_name not in profiles_by_player:
                        profiles_by_player[profile.player_name] = []
                    profiles_by_player[profile.player_name].append(profile)
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"Warning: Failed to parse profile line: {e}")
                    continue

        # Sort profiles by timestamp (newest first)
        for player_name in profiles_by_player:
            profiles_by_player[player_name].sort(key=lambda p: p.timestamp, reverse=True)

        return profiles_by_player

    def load_profiles(self, mode: Optional[int] = None) -> Dict[str, List[PlayerProfile]]:
        """
        Load all existing profiles, optionally filtered by mode.
        
        Args:
            mode: Optional mode filter. If provided, only return profiles for this mode.
            
        Returns:
            Dictionary mapping player names to their profiles (newest first)
        """
        profiles_by_player = self._load_profiles()
        
        if mode is not None:
            filtered = {}
            for player_name, profiles in profiles_by_player.items():
                filtered_profiles = [p for p in profiles if p.mode == mode]
                if filtered_profiles:
                    filtered[player_name] = filtered_profiles
            return filtered
        
        return profiles_by_player

    def _save_profile(self, profile: PlayerProfile) -> None:
        """Save a profile to the JSONL file."""
        with open(self.profile_file, "a") as f:
            f.write(json.dumps(asdict(profile)) + "\n")

    def _get_latest_profile(self, player_name: str) -> Optional[PlayerProfile]:
        """Get the latest profile for a player."""
        profiles_by_player = self._load_profiles()
        profiles = profiles_by_player.get(player_name, [])
        return profiles[0] if profiles else None

    def _determine_version(self, player_name: str, new_avg_duration: float) -> str:
        """
        Determine version based on performance delta.
        If significant change (>10% delta), increment version.
        Otherwise, use same version as latest.
        """
        latest = self._get_latest_profile(player_name)

        if latest is None:
            return "v1"  # First version

        # Calculate delta percentage
        if latest.avg_duration == 0:
            delta_pct = float("inf") if new_avg_duration > 0 else 0.0
        else:
            delta_pct = abs((new_avg_duration - latest.avg_duration) / latest.avg_duration) * 100

        # If delta > 10%, create new version
        if delta_pct > 10.0:
            # Extract version number and increment
            if latest.player_version.startswith("v"):
                try:
                    version_num = int(latest.player_version[1:])
                    return f"v{version_num + 1}"
                except ValueError:
                    return "v2"
            return "v2"
        else:
            # No significant change, use same version
            return latest.player_version

    def profile_player(
        self,
        player_cls: Type[Player],
        mode: int = 2,
        games: int = 10,
        random_as_white: bool = True,
    ) -> PlayerProfile:
        """
        Profile a player against RandomPlayer.

        Args:
            player_cls: Player class to profile
            mode: Game mode (default: 2)
            games: Number of games to play (default: 10)
            random_as_white: If True, RandomPlayer plays white; if False, plays black

        Returns:
            PlayerProfile with performance metrics
        """
        player_name = player_cls.__name__
        runner = GameRunner()

        durations = []
        move_counts = []
        total_start = time.time()

        # Collect metadata (e.g., nodes_evaluated for Minimax)
        metadata = {}

        print(f"  Profiling {player_name} against RandomPlayer ({games} games)...")

        for i in range(games):
            if random_as_white:
                white_cls = RandomPlayer
                black_cls = player_cls
                white_name = "RandomPlayer"
                black_name = player_name
            else:
                white_cls = player_cls
                black_cls = RandomPlayer
                white_name = player_name
                black_name = "RandomPlayer"

            result = runner.run_game(mode, white_cls, black_cls, white_name, black_name)
            durations.append(result.duration)
            move_counts.append(result.moves)

            # Collect metadata from result
            # Track the player's metadata (white or black depending on setup)
            if result.metadata:
                if random_as_white:
                    # Player is black
                    if "black_nodes_evaluated" in result.metadata:
                        if "nodes_evaluated" not in metadata:
                            metadata["nodes_evaluated"] = []
                        metadata["nodes_evaluated"].append(result.metadata["black_nodes_evaluated"])
                    if "black_search_depth" in result.metadata:
                        if "search_depth" not in metadata:
                            metadata["search_depth"] = []
                        metadata["search_depth"].append(result.metadata["black_search_depth"])
                else:
                    # Player is white
                    if "white_nodes_evaluated" in result.metadata:
                        if "nodes_evaluated" not in metadata:
                            metadata["nodes_evaluated"] = []
                        metadata["nodes_evaluated"].append(result.metadata["white_nodes_evaluated"])
                    if "white_search_depth" in result.metadata:
                        if "search_depth" not in metadata:
                            metadata["search_depth"] = []
                        metadata["search_depth"].append(result.metadata["white_search_depth"])

        total_duration = time.time() - total_start

        # Aggregate metadata (convert lists to stats)
        aggregated_metadata = {}
        for key, value_list in metadata.items():
            if isinstance(value_list, list) and len(value_list) > 0:
                if all(isinstance(v, (int, float)) for v in value_list):
                    aggregated_metadata[f"avg_{key}"] = sum(value_list) / len(value_list)
                    aggregated_metadata[f"total_{key}"] = sum(value_list)
                    aggregated_metadata[f"min_{key}"] = min(value_list)
                    aggregated_metadata[f"max_{key}"] = max(value_list)
                aggregated_metadata[key] = value_list  # Keep raw list for analysis
            else:
                aggregated_metadata[key] = value_list

        metadata = aggregated_metadata

        avg_duration = sum(durations) / len(durations) if durations else 0.0
        avg_moves = sum(move_counts) / len(move_counts) if move_counts else 0.0

        # Determine version:
        # - Prefer static VERSION attribute on the class (for hand-labelled variants like v1/v2/v3)
        # - Fallback to automatic performance-based versioning for everything else.
        version = getattr(player_cls, "VERSION", None)
        if not isinstance(version, str):
            version = self._determine_version(player_name, avg_duration)

        profile = PlayerProfile(
            player_name=player_name,
            player_version=version,
            timestamp=time.time(),
            mode=mode,
            games_played=games,
            avg_duration=avg_duration,
            total_duration=total_duration,
            avg_moves=avg_moves,
            min_duration=min(durations) if durations else 0.0,
            max_duration=max(durations) if durations else 0.0,
            metadata=metadata,
        )

        self._save_profile(profile)

        print(
            f"    {player_name} {version}: "
            f"avg={avg_duration:.3f}s, "
            f"total={total_duration:.2f}s, "
            f"avg_moves={avg_moves:.1f}"
        )

        # Show delta if not first version
        latest = self._get_latest_profile(player_name)
        if latest and latest.player_version != version:
            delta = avg_duration - latest.avg_duration
            delta_pct = (delta / latest.avg_duration * 100) if latest.avg_duration > 0 else 0.0
            print(f"    Delta: {delta:+.3f}s ({delta_pct:+.1f}%) -> {version}")

        return profile

    def profile_random_player(
        self,
        mode: int = 2,
        games: int = 50,
        opponent_cls: Type[Player] = GreedyPlayer,
    ) -> PlayerProfile:
        """
        Profile RandomPlayer against a fixed opponent (default: GreedyPlayer).

        Args:
            mode: Game mode (default: 2)
            games: Number of games to play (default: 50)
            opponent_cls: Opponent class to use (default: GreedyPlayer)

        Returns:
            PlayerProfile with performance metrics
        """
        player_name = "RandomPlayer"
        runner = GameRunner()
        opponent_name = opponent_cls.__name__

        durations = []
        move_counts = []
        total_start = time.time()

        print(f"  Profiling {player_name} against {opponent_name} ({games} games)...")

        for i in range(games):
            # Alternate colors for balanced testing
            if i % 2 == 0:
                white_cls = RandomPlayer
                black_cls = opponent_cls
                white_name = "RandomPlayer"
                black_name = opponent_name
            else:
                white_cls = opponent_cls
                black_cls = RandomPlayer
                white_name = opponent_name
                black_name = "RandomPlayer"

            result = runner.run_game(mode, white_cls, black_cls, white_name, black_name)
            durations.append(result.duration)
            move_counts.append(result.moves)

        total_duration = time.time() - total_start

        avg_duration = sum(durations) / len(durations) if durations else 0.0
        avg_moves = sum(move_counts) / len(move_counts) if move_counts else 0.0

        # Determine version:
        # - Prefer static VERSION on RandomPlayer if defined
        # - Otherwise use automatic performance-based versioning
        from src.players.random import RandomPlayer as RP  # Local import to avoid cycles

        version = getattr(RP, "VERSION", None)
        if not isinstance(version, str):
            version = self._determine_version(player_name, avg_duration)

        profile = PlayerProfile(
            player_name=player_name,
            player_version=version,
            timestamp=time.time(),
            mode=mode,
            games_played=games,
            avg_duration=avg_duration,
            total_duration=total_duration,
            avg_moves=avg_moves,
            min_duration=min(durations) if durations else 0.0,
            max_duration=max(durations) if durations else 0.0,
            metadata={},
        )

        self._save_profile(profile)

        print(
            f"    {player_name} {version}: "
            f"avg={avg_duration:.3f}s, "
            f"total={total_duration:.2f}s, "
            f"avg_moves={avg_moves:.1f}"
        )

        # Show delta if not first version
        latest = self._get_latest_profile(player_name)
        if latest and latest.player_version != version:
            delta = avg_duration - latest.avg_duration
            delta_pct = (delta / latest.avg_duration * 100) if latest.avg_duration > 0 else 0.0
            print(f"    Delta: {delta:+.3f}s ({delta_pct:+.1f}%) -> {version}")

        return profile

    def profile_all_players(
        self,
        player_classes: List[Type[Player]],
        mode: int = 2,
        games: int = 50,
        include_random: bool = False,
    ) -> Dict[str, PlayerProfile]:
        """
        Profile all players against RandomPlayer.

        Args:
            player_classes: List of player classes to profile
            mode: Game mode
            games: Number of games per player
            include_random: If True, also profile RandomPlayer against a fixed opponent

        Returns:
            Dictionary mapping player names to their profiles
        """
        print(f"\n{'='*70}")
        print(f"Performance Profiling (Mode {mode}, {games} games per player)")
        print(f"{'='*70}")

        profiles = {}

        # Filter out HumanPlayer, optionally include RandomPlayer
        players_to_profile = [
            cls
            for cls in player_classes
            if "Human" not in cls.__name__ and ("Random" not in cls.__name__ or include_random)
        ]

        # Profile non-Random players against RandomPlayer
        for player_cls in players_to_profile:
            if "Random" in player_cls.__name__:
                continue  # Handle RandomPlayer separately
            try:
                # Profile with RandomPlayer as white
                profile = self.profile_player(player_cls, mode=mode, games=games, random_as_white=True)
                profiles[player_cls.__name__] = profile
            except Exception as e:
                print(f"  Error profiling {player_cls.__name__}: {e}")
                import traceback

                traceback.print_exc()

        # Profile RandomPlayer if requested
        if include_random:
            try:
                profile = self.profile_random_player(mode=mode, games=games)
                profiles["RandomPlayer"] = profile
            except Exception as e:
                print(f"  Error profiling RandomPlayer: {e}")
                import traceback

                traceback.print_exc()

        print(f"\n{'='*70}")
        print("Profiling complete")
        print(f"{'='*70}\n")

        return profiles

