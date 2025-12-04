#!/usr/bin/env python3
"""
Plotting utilities for PPO self-play training results.

This script provides various visualizations for monitoring and analyzing
the AlphaZero-style self-play training progress, including:
- Win rate progression over generations
- Per-color (white/black) performance analysis
- Smoothed training curves
- Generation advancement tracking

Usage:
    python scripts/plot_training.py                    # Plot all available data
    python scripts/plot_training.py --save             # Save plots to files
    python scripts/plot_training.py --window 10        # Custom smoothing window
    python scripts/plot_training.py --tensorboard      # Also show tensorboard data
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Try to import SB3 plotting utilities
try:
    from stable_baselines3.common.monitor import load_results
    from stable_baselines3.common.results_plotter import ts2xy, window_func

    HAS_SB3_PLOTTING = True
except ImportError:
    HAS_SB3_PLOTTING = False
    print("Warning: SB3 plotting utilities not available. Install stable-baselines3 for full functionality.")


def load_self_play_history(log_dir: Path) -> Optional[pd.DataFrame]:
    """Load self-play training history from CSV file."""
    history_path = log_dir / "self_play_history.csv"
    if not history_path.exists():
        print(f"No self-play history found at {history_path}")
        return None

    df = pd.read_csv(history_path)
    print(f"Loaded {len(df)} evaluation records from {history_path}")
    return df


def load_monitor_data(log_dir: Path) -> Optional[pd.DataFrame]:
    """Load monitor data from SB3 Monitor wrapper."""
    if not HAS_SB3_PLOTTING:
        return None

    try:
        df = load_results(str(log_dir))
        if len(df) > 0:
            print(f"Loaded {len(df)} episodes from monitor files")
            return df
    except Exception as e:
        print(f"Could not load monitor data: {e}")
    return None


def smooth_data(x: np.ndarray, y: np.ndarray, window: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Apply smoothing to data using rolling mean."""
    if len(x) < window:
        return x, y

    if HAS_SB3_PLOTTING:
        return window_func(x, y, window, np.mean)
    else:
        # Simple rolling mean fallback
        kernel = np.ones(window) / window
        y_smooth = np.convolve(y, kernel, mode="valid")
        x_smooth = x[window - 1 :]
        return x_smooth, y_smooth


def plot_win_rate_progression(
    df: pd.DataFrame,
    window: int = 10,
    ax: Optional[plt.Axes] = None,
    show_threshold: bool = True,
) -> plt.Axes:
    """
    Plot win rate progression over timesteps.

    Args:
        df: DataFrame with self-play history
        window: Smoothing window size
        ax: Optional matplotlib axes
        show_threshold: Whether to show the win threshold line

    Returns:
        The matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    timesteps = df["timesteps"].values / 1e6  # Convert to millions
    win_rate = df["win_rate"].values * 100  # Convert to percentage

    # Plot raw data
    ax.scatter(timesteps, win_rate, s=15, alpha=0.4, label="Raw Win Rate", color="steelblue")

    # Plot smoothed data
    if len(timesteps) >= window:
        t_smooth, wr_smooth = smooth_data(timesteps, win_rate, window)
        ax.plot(t_smooth, wr_smooth, linewidth=2, label=f"Smoothed ({window}-eval window)", color="darkblue")

    # Add threshold line
    if show_threshold:
        ax.axhline(y=55, color="red", linestyle="--", alpha=0.7, label="Win Threshold (55%)")
        ax.axhline(y=50, color="gray", linestyle=":", alpha=0.5, label="Random (50%)")

    ax.set_xlabel("Timesteps (millions)", fontsize=12)
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_title("Self-Play Win Rate vs Best Model", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    return ax


def plot_generation_progression(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot generation (model version) progression over timesteps.

    Args:
        df: DataFrame with self-play history
        ax: Optional matplotlib axes

    Returns:
        The matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    timesteps = df["timesteps"].values / 1e6
    generations = df["generation"].values

    # Find where generation changes (model became new best)
    became_best = df["became_best"].values

    # Plot generation line
    ax.step(timesteps, generations, where="post", linewidth=2, color="darkgreen", label="Current Generation")

    # Mark points where model became best
    best_mask = became_best == True
    ax.scatter(
        timesteps[best_mask],
        generations[best_mask],
        s=50,
        color="gold",
        edgecolor="darkgreen",
        linewidth=1.5,
        zorder=5,
        label="New Best Model",
    )

    ax.set_xlabel("Timesteps (millions)", fontsize=12)
    ax.set_ylabel("Generation", fontsize=12)
    ax.set_title("Model Generation Progression", fontsize=14)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Set y-axis to show integer generations
    max_gen = int(generations.max()) + 1
    ax.set_yticks(range(0, max_gen + 1, max(1, max_gen // 10)))

    return ax


def plot_per_color_performance(
    df: pd.DataFrame,
    window: int = 10,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot win rate breakdown by color (white vs black).

    Args:
        df: DataFrame with self-play history (must have white_win_rate and black_win_rate columns)
        window: Smoothing window size
        ax: Optional matplotlib axes

    Returns:
        The matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Check if per-color data is available
    if "white_win_rate" not in df.columns or "black_win_rate" not in df.columns:
        ax.text(
            0.5,
            0.5,
            "Per-color data not available\n(older format)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("Per-Color Performance (N/A)", fontsize=14)
        return ax

    timesteps = df["timesteps"].values / 1e6
    white_wr = df["white_win_rate"].values * 100
    black_wr = df["black_win_rate"].values * 100

    # Plot raw data with lower opacity
    ax.scatter(timesteps, white_wr, s=10, alpha=0.3, color="white", edgecolor="gray", label="_nolegend_")
    ax.scatter(timesteps, black_wr, s=10, alpha=0.3, color="black", label="_nolegend_")

    # Plot smoothed data
    if len(timesteps) >= window:
        t_smooth, w_smooth = smooth_data(timesteps, white_wr, window)
        _, b_smooth = smooth_data(timesteps, black_wr, window)
        ax.plot(t_smooth, w_smooth, linewidth=2, color="orange", label="White Win Rate")
        ax.plot(t_smooth, b_smooth, linewidth=2, color="purple", label="Black Win Rate")

    # Add threshold lines
    ax.axhline(y=40, color="red", linestyle="--", alpha=0.5, label="Min Balance Threshold (40%)")
    ax.axhline(y=50, color="gray", linestyle=":", alpha=0.3)

    ax.set_xlabel("Timesteps (millions)", fontsize=12)
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_title("Per-Color Performance (Balance Check)", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    return ax


def plot_evaluation_stats(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot wins/losses/draws distribution over time.

    Args:
        df: DataFrame with self-play history
        ax: Optional matplotlib axes

    Returns:
        The matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    timesteps = df["timesteps"].values / 1e6
    wins = df["wins"].values
    losses = df["losses"].values
    draws = df["draws"].values if "draws" in df.columns else np.zeros_like(wins)

    # Stacked area plot
    ax.fill_between(timesteps, 0, wins, alpha=0.6, color="green", label="Wins")
    ax.fill_between(timesteps, wins, wins + draws, alpha=0.6, color="gray", label="Draws")
    ax.fill_between(timesteps, wins + draws, wins + draws + losses, alpha=0.6, color="red", label="Losses")

    ax.set_xlabel("Timesteps (millions)", fontsize=12)
    ax.set_ylabel("Games", fontsize=12)
    ax.set_title("Evaluation Game Outcomes", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    return ax


def plot_monitor_rewards(
    df: pd.DataFrame,
    window: int = 100,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot episode rewards from monitor data.

    Args:
        df: DataFrame from SB3 Monitor wrapper
        window: Smoothing window size
        ax: Optional matplotlib axes

    Returns:
        The matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    if HAS_SB3_PLOTTING:
        x, y = ts2xy(df, "timesteps")
    else:
        x = df.index.values
        y = df["r"].values if "r" in df.columns else df["reward"].values

    # Convert to millions
    x = x / 1e6

    # Plot raw data
    ax.scatter(x, y, s=2, alpha=0.2, color="steelblue", label="Raw Rewards")

    # Plot smoothed data
    if len(x) >= window:
        x_smooth, y_smooth = smooth_data(x, y, window)
        ax.plot(x_smooth, y_smooth, linewidth=2, color="darkblue", label=f"Smoothed ({window}-ep window)")

    ax.set_xlabel("Timesteps (millions)", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("Training Episode Rewards", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    return ax


def plot_comprehensive_dashboard(
    log_dir: Path,
    window: int = 10,
    save_path: Optional[Path] = None,
) -> None:
    """
    Create a comprehensive training dashboard with all available plots.

    Args:
        log_dir: Path to logs directory
        window: Smoothing window size for win rate plots
        save_path: Optional path to save the figure
    """
    # Load data
    history_df = load_self_play_history(log_dir)
    monitor_df = load_monitor_data(log_dir)

    if history_df is None:
        print("No training data available to plot.")
        return

    # Determine layout based on available data
    has_per_color = "white_win_rate" in history_df.columns
    has_monitor = monitor_df is not None and len(monitor_df) > 0

    # Create figure with appropriate layout
    if has_per_color:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        ax_winrate = axes[0, 0]
        ax_generation = axes[0, 1]
        ax_color = axes[1, 0]
        ax_stats = axes[1, 1]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        ax_winrate = axes[0, 0]
        ax_generation = axes[0, 1]
        ax_stats = axes[1, 0]
        ax_color = axes[1, 1]

    # Plot all available data
    plot_win_rate_progression(history_df, window=window, ax=ax_winrate)
    plot_generation_progression(history_df, ax=ax_generation)
    plot_evaluation_stats(history_df, ax=ax_stats)

    if has_per_color:
        plot_per_color_performance(history_df, window=window, ax=ax_color)
    elif has_monitor:
        plot_monitor_rewards(monitor_df, window=100, ax=ax_color)
    else:
        ax_color.text(
            0.5,
            0.5,
            "No additional data available",
            ha="center",
            va="center",
            transform=ax_color.transAxes,
            fontsize=12,
        )
        ax_color.set_title("Additional Metrics (N/A)", fontsize=14)

    # Add overall title
    total_timesteps = history_df["timesteps"].max()
    final_gen = history_df["generation"].max()
    total_evals = len(history_df)

    fig.suptitle(
        f"PPO Self-Play Training Dashboard\n"
        f"Total: {total_timesteps / 1e6:.1f}M steps | Generation {final_gen} | {total_evals} evaluations",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved dashboard to {save_path}")

    plt.show()


def plot_single_metric(
    log_dir: Path,
    metric: str,
    window: int = 10,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot a single metric from training data.

    Args:
        log_dir: Path to logs directory
        metric: One of 'winrate', 'generation', 'color', 'stats', 'rewards'
        window: Smoothing window size
        save_path: Optional path to save the figure
    """
    history_df = load_self_play_history(log_dir)
    monitor_df = load_monitor_data(log_dir)

    fig, ax = plt.subplots(figsize=(12, 6))

    if metric == "winrate" and history_df is not None:
        plot_win_rate_progression(history_df, window=window, ax=ax)
    elif metric == "generation" and history_df is not None:
        plot_generation_progression(history_df, ax=ax)
    elif metric == "color" and history_df is not None:
        plot_per_color_performance(history_df, window=window, ax=ax)
    elif metric == "stats" and history_df is not None:
        plot_evaluation_stats(history_df, ax=ax)
    elif metric == "rewards" and monitor_df is not None:
        plot_monitor_rewards(monitor_df, window=100, ax=ax)
    else:
        print(f"Metric '{metric}' not available or data not found.")
        plt.close(fig)
        return

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


def print_training_summary(log_dir: Path) -> None:
    """Print a text summary of training progress."""
    df = load_self_play_history(log_dir)
    if df is None:
        return

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    # Basic stats
    total_timesteps = df["timesteps"].max()
    final_gen = df["generation"].max()
    total_evals = len(df)
    became_best_count = df["became_best"].sum()

    print(f"Total Timesteps:     {total_timesteps:,}")
    print(f"Total Evaluations:   {total_evals}")
    print(f"Final Generation:    {final_gen}")
    print(f"Times Became Best:   {became_best_count}")

    # Win rate stats
    win_rates = df["win_rate"].values * 100
    print(f"\nWin Rate Stats:")
    print(f"  Current:  {win_rates[-1]:.1f}%")
    print(f"  Mean:     {win_rates.mean():.1f}%")
    print(f"  Max:      {win_rates.max():.1f}%")
    print(f"  Min:      {win_rates.min():.1f}%")

    # Recent trend (last 10 evals)
    if len(win_rates) >= 10:
        recent = win_rates[-10:]
        earlier = win_rates[-20:-10] if len(win_rates) >= 20 else win_rates[:10]
        trend = recent.mean() - earlier.mean()
        trend_str = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
        print(f"  Trend:    {trend_str} {trend:+.1f}% (last 10 vs previous 10)")

    # Per-color stats (if available)
    if "white_win_rate" in df.columns:
        white_wr = df["white_win_rate"].values * 100
        black_wr = df["black_win_rate"].values * 100
        print(f"\nPer-Color Performance (latest):")
        print(f"  As White: {white_wr[-1]:.1f}%")
        print(f"  As Black: {black_wr[-1]:.1f}%")
        balance = abs(white_wr[-1] - black_wr[-1])
        balance_str = "✅ Balanced" if balance < 15 else "⚠️ Imbalanced"
        print(f"  Balance:  {balance_str} ({balance:.1f}% difference)")

    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot PPO self-play training results.")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="data/logs/ppo_self_play",
        help="Path to logs directory (default: data/logs/ppo_self_play)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["all", "winrate", "generation", "color", "stats", "rewards"],
        default="all",
        help="Which metric to plot (default: all = dashboard)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Smoothing window size (default: 10)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save plots to files instead of displaying",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/plots",
        help="Directory to save plots (default: data/plots)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print text summary of training progress",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Launch TensorBoard for real-time monitoring",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    log_dir = Path(args.log_dir)

    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        return

    # Print summary if requested
    if args.summary:
        print_training_summary(log_dir)
        return

    # Launch TensorBoard if requested
    if args.tensorboard:
        import subprocess

        print(f"Launching TensorBoard for {log_dir}...")
        subprocess.run(["tensorboard", "--logdir", str(log_dir)])
        return

    # Prepare save path if needed
    save_path = None
    if args.save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.metric == "all":
            save_path = output_dir / "training_dashboard.png"
        else:
            save_path = output_dir / f"training_{args.metric}.png"

    # Plot
    if args.metric == "all":
        plot_comprehensive_dashboard(log_dir, window=args.window, save_path=save_path)
    else:
        plot_single_metric(log_dir, args.metric, window=args.window, save_path=save_path)


if __name__ == "__main__":
    main()
