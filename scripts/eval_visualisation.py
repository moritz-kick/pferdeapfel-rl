"""Visualization script for evaluation results."""

import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.discovery import discover_players
from src.evaluation.profiler import PerformanceProfiler
from src.evaluation.storage import ResultStorage


def print_table(title: str, headers: List[str], rows: List[List[str]]):
    """Print a formatted ASCII table."""
    # Calculate widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    # Print
    print(f"\n### {title} ###")

    # Header
    header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))

    # Rows
    for row in rows:
        print(" | ".join(f"{str(val):<{w}}" for val, w in zip(row, widths)))
    print("-" * len(header_str))


def analyze_profiler_data(mode: int, profiler: PerformanceProfiler):
    """Analyze and display profiler performance data for a specific mode."""
    profiles_by_player = profiler.load_profiles(mode=mode)
    
    if not profiles_by_player:
        return
    
    # Get latest profiles for each player
    latest_profiles = {}
    for player_name, profiles in profiles_by_player.items():
        if profiles:
            latest_profiles[player_name] = profiles[0]  # Already sorted newest first
    
    if not latest_profiles:
        return
    
    # 1. Performance Speed Comparison
    headers = ["Player", "Version", "Avg Duration (s)", "Total Duration (s)", "Games", "Avg Moves"]
    rows = []
    for player_name in sorted(latest_profiles.keys()):
        profile = latest_profiles[player_name]
        rows.append([
            player_name,
            profile.player_version,
            f"{profile.avg_duration:.8f}",
            f"{profile.total_duration:.2f}",
            profile.games_played,
            f"{profile.avg_moves:.1f}"
        ])
    print_table(f"Mode {mode} - Performance Profiling (vs RandomPlayer)", headers, rows)
    
    # 2. Search Statistics (for players with search metadata)
    search_players = []
    for player_name, profile in latest_profiles.items():
        metadata = profile.metadata or {}
        if "avg_nodes_evaluated" in metadata or "avg_search_depth" in metadata:
            search_players.append((player_name, profile))
    
    if search_players:
        headers = ["Player", "Version", "Avg Nodes", "Total Nodes", "Avg Depth", "Min Depth", "Max Depth"]
        rows = []
        for player_name, profile in sorted(search_players, key=lambda x: x[1].metadata.get("avg_nodes_evaluated", 0)):
            metadata = profile.metadata
            avg_nodes = metadata.get("avg_nodes_evaluated", 0)
            total_nodes = metadata.get("total_nodes_evaluated", 0)
            avg_depth = metadata.get("avg_search_depth", 0)
            min_depth = metadata.get("min_search_depth", 0)
            max_depth = metadata.get("max_search_depth", 0)
            
            rows.append([
                player_name,
                profile.player_version,
                f"{avg_nodes:,.0f}" if avg_nodes > 0 else "-",
                f"{total_nodes:,}" if total_nodes > 0 else "-",
                f"{avg_depth:.1f}" if avg_depth > 0 else "-",
                f"{min_depth}" if min_depth > 0 else "-",
                f"{max_depth}" if max_depth > 0 else "-"
            ])
        print_table(f"Mode {mode} - Search Statistics", headers, rows)
    
    # 3. Version History (if multiple versions exist)
    version_history_players = []
    for player_name, profiles in profiles_by_player.items():
        if len(profiles) > 1:
            version_history_players.append((player_name, profiles))
    
    if version_history_players:
        headers = ["Player", "Version", "Avg Duration (s)", "Games", "Change from Previous"]
        rows = []
        for player_name, profiles in sorted(version_history_players):
            # Sort by version number (v1, v2, v3...)
            sorted_profiles = sorted(profiles, key=lambda p: int(p.player_version[1:]) if p.player_version[1:].isdigit() else 0)
            
            for i, profile in enumerate(sorted_profiles):
                change_str = "-"
                if i > 0:
                    prev_profile = sorted_profiles[i-1]
                    if prev_profile.avg_duration > 0:
                        delta = profile.avg_duration - prev_profile.avg_duration
                        delta_pct = (delta / prev_profile.avg_duration) * 100
                        change_str = f"{delta:+.4f}s ({delta_pct:+.1f}%)"
                
                rows.append([
                    player_name if i == 0 else "",  # Only show name for first version
                    profile.player_version,
                    f"{profile.avg_duration:.8f}",
                    profile.games_played,
                    change_str
                ])
        
        print_table(f"Mode {mode} - Version History", headers, rows)


def analyze_mode(mode: int, storage: ResultStorage, all_players: List[str]):
    """Analyze and print stats for a specific mode."""
    results = storage.load_results(mode)
    if not results:
        print(f"\nNo results for Mode {mode}")
        return

    # Aggregate stats
    class PlayerStats:
        def __init__(self, name):
            self.name = name
            self.games = 0
            self.wins = 0
            self.draws = 0
            self.losses = 0

            self.white_games = 0
            self.white_wins = 0
            self.white_losses = 0

            self.black_games = 0
            self.black_wins = 0
            self.black_losses = 0

            self.errors = 0
            self.total_moves = 0
            self.total_time = 0.0

        @property
        def win_rate(self):
            return self.wins / self.games if self.games > 0 else 0.0

        @property
        def white_win_rate(self):
            return self.white_wins / self.white_games if self.white_games > 0 else 0.0

        @property
        def black_win_rate(self):
            return self.black_wins / self.black_games if self.black_games > 0 else 0.0

        @property
        def avg_moves(self):
            return self.total_moves / self.games if self.games > 0 else 0.0

        @property
        def avg_time(self):
            return self.total_time / self.games if self.games > 0 else 0.0

    stats = {p: PlayerStats(p) for p in all_players}

    for r in results:
        # Ensure stats exist (if player not in list but in logs)
        if r.white_player not in stats:
            stats[r.white_player] = PlayerStats(r.white_player)
        if r.black_player not in stats:
            stats[r.black_player] = PlayerStats(r.black_player)

        p1 = stats[r.white_player]
        p2 = stats[r.black_player]

        p1.games += 1
        p1.white_games += 1
        p2.games += 1
        p2.black_games += 1

        p1.total_moves += r.moves
        p2.total_moves += r.moves
        p1.total_time += r.duration
        p2.total_time += r.duration  # Double count duration for simplicity per player

        if r.white_error:
            p1.errors += 1
        if r.black_error:
            p2.errors += 1

        if r.winner == "white":
            p1.wins += 1
            p1.white_wins += 1
            p2.losses += 1
            p2.black_losses += 1
        elif r.winner == "black":
            p2.wins += 1
            p2.black_wins += 1
            p1.losses += 1
            p1.white_losses += 1
        else:
            p1.draws += 1
            p2.draws += 1

    # Sort by wins desc
    sorted_stats = sorted(stats.values(), key=lambda s: s.wins, reverse=True)

    # 1. Main Subtable
    headers = ["Rank", "Player", "Games", "Wins", "Draws", "Losses", "Win Rate", "Errors"]
    rows = []
    for i, s in enumerate(sorted_stats, 1):
        rows.append([i, s.name, s.games, s.wins, s.draws, s.losses, f"{s.win_rate:.1%}", s.errors])
    print_table(f"Mode {mode} - Overall Standings", headers, rows)

    # 2. Side Stats (White/Black)
    headers = ["Player", "W Games", "W Win%", "B Games", "B Win%", "Avg Moves", "Avg Time (s)", "Cap(W/L)", "Stck(W/L)"]
    rows = []
    for s in sorted_stats:
        # Calculate sub-stats
        cap_wins = 0
        cap_losses = 0
        stuck_wins = 0
        stuck_losses = 0

        # We need to re-scan results or store them better.
        for r in results:
            term = r.metadata.get("termination", "unknown")
            if r.white_player == s.name:
                if r.winner == "white":
                    if term == "capture":
                        cap_wins += 1
                    elif term == "stuck":
                        stuck_wins += 1
                elif r.winner == "black":
                    if term == "capture":
                        cap_losses += 1
                    elif term == "stuck":
                        stuck_losses += 1
            elif r.black_player == s.name:
                if r.winner == "black":
                    if term == "capture":
                        cap_wins += 1
                    elif term == "stuck":
                        stuck_wins += 1
                elif r.winner == "white":
                    if term == "capture":
                        cap_losses += 1
                    elif term == "stuck":
                        stuck_losses += 1

        rows.append(
            [
                s.name,
                s.white_games,
                f"{s.white_win_rate:.1%}",
                s.black_games,
                f"{s.black_win_rate:.1%}",
                f"{s.avg_moves:.1f}",
                f"{s.avg_time:.2f}",
                f"{cap_wins}/{cap_losses}",
                f"{stuck_wins}/{stuck_losses}",
            ]
        )
    print_table(f"Mode {mode} - Detailed Stats", headers, rows)

    # 3. Matchup Subsubtables
    # Collect unique pairs that have played
    matchups = {}  # (white, black) -> {white_wins, black_wins, draws}

    for r in results:
        key = (r.white_player, r.black_player)
        if key not in matchups:
            matchups[key] = {"white_games": 0, "white_wins": 0, "black_wins": 0, "draws": 0}

        m = matchups[key]
        m["white_games"] += 1

        if r.winner == "white":
            m["white_wins"] += 1
        elif r.winner == "black":
            m["black_wins"] += 1
        else:
            m["draws"] += 1

    # Identify unique pairings (unordered)
    # Include self-matchups (e.g., RandomPlayer vs RandomPlayer)
    pairs = set()
    for w, b in matchups.keys():
        pairs.add(tuple(sorted((w, b))))

    sorted_pairs = sorted(list(pairs))

    for p1, p2 in sorted_pairs:
        # Prepare rows for this matchup
        # Row 1: p1 is White, p2 is Black
        # Row 2: p2 is White, p1 is Black

        rows = []

        # p1 (White) vs p2 (Black)
        k1 = (p1, p2)
        if k1 in matchups:
            stats = matchups[k1]
            rows.append(
                [
                    f"{p1} (W) vs {p2} (B)",
                    stats["white_games"],
                    stats["white_wins"],
                    stats["draws"],
                    stats["black_wins"],
                    f"{stats['white_wins'] / stats['white_games']:.1%}" if stats["white_games"] else "0.0%",
                ]
            )
        else:
            rows.append([f"{p1} (W) vs {p2} (B)", 0, 0, 0, 0, "-"])

        # p2 (White) vs p1 (Black)
        k2 = (p2, p1)
        if k2 in matchups:
            stats = matchups[k2]
            rows.append(
                [
                    f"{p2} (W) vs {p1} (B)",
                    stats["white_games"],
                    stats["white_wins"],
                    stats["draws"],
                    stats["black_wins"],
                    f"{stats['white_wins'] / stats['white_games']:.1%}" if stats["white_games"] else "0.0%",
                ]
            )
        else:
            rows.append([f"{p2} (W) vs {p1} (B)", 0, 0, 0, 0, "-"])

        headers = ["Matchup", "Games", "White Wins", "Draws", "Black Wins", "White Win%"]
        print_table(f"Mode {mode} - Matchup: {p1} vs {p2}", headers, rows)


def main():
    print("Generating Evaluation Report...")

    # Discover all potential players for consistent naming
    classes = discover_players()
    all_names = [c.__name__ for c in classes]

    storage = ResultStorage()
    profiler = PerformanceProfiler()

    # Iterate modes
    # If user wants 1, 2, 3
    for mode in [1, 2, 3]:
        # First show profiler analysis
        analyze_profiler_data(mode, profiler)
        
        # Then show evaluation results
        analyze_mode(mode, storage, all_names)


if __name__ == "__main__":
    main()
