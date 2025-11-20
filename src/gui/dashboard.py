"""Lightweight dashboard for visualizing evaluation results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def load_results(csv_path: Path) -> Tuple[List[str], List[float]]:
    """Return agent labels and win rates from a results CSV."""
    agents: List[str] = []
    win_rates: List[float] = []
    if not csv_path.exists():
        return agents, win_rates

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            agents.append(row.get("agent", "agent"))
            try:
                win_rates.append(float(row.get("win_rate", 0.0)))
            except ValueError:
                win_rates.append(0.0)
    return agents, win_rates


def plot_win_rates(csv_path: Path) -> None:
    """Plot a simple bar chart of win rates from evaluation results."""
    agents, win_rates = load_results(csv_path)
    if not agents:
        print(f"No results found at {csv_path}")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(agents, win_rates, color="#4C72B0")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win rate")
    ax.set_title("RL Agent Performance vs Random")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_win_rates(Path("src/evaluation/results.csv"))
