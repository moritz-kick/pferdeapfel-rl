"""Summarize TensorBoard logs from Stable Baselines3 runs for quick sharing/LLM input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError as exc:  # pragma: no cover - optional dep at runtime
    msg = "tensorboard is required for tb_summary; install with `pip install tensorboard`."
    raise ImportError(msg) from exc

ScalarSeries = Dict[str, List[Tuple[int, float]]]


def _load_scalars(event_file: Path) -> ScalarSeries:
    """Load all scalar tags from a TensorBoard event file."""
    acc = event_accumulator.EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    acc.Reload()
    scalars: ScalarSeries = {}
    for tag in acc.Tags().get("scalars", []):
        scalars[tag] = [(int(event.step), float(event.value)) for event in acc.Scalars(tag)]
    return scalars


def _latest(series: Iterable[Tuple[int, float]]) -> Tuple[int, float] | None:
    series = list(series)
    if not series:
        return None
    return max(series, key=lambda item: item[0])


def _best(series: Iterable[Tuple[int, float]], mode: str = "max") -> Tuple[int, float] | None:
    series = list(series)
    if not series:
        return None
    key_func = (lambda item: item[1]) if mode == "max" else (lambda item: -item[1])
    best_step, best_val = max(series, key=key_func)
    return best_step, best_val


def _at_percentile(series: List[Tuple[int, float]], pct: float) -> Tuple[int, float] | None:
    """Return the value at a given percentile index (by step order)."""
    if not series:
        return None
    pct_clamped = max(0.0, min(1.0, pct))
    idx = int(round((len(series) - 1) * pct_clamped))
    return series[idx]


def _slope_last(series: List[Tuple[int, float]], window: int = 10) -> float | None:
    """Approximate slope over the last `window` points."""
    if len(series) < 2:
        return None
    recent = series[-window:]
    first_step, first_val = recent[0]
    last_step, last_val = recent[-1]
    if last_step == first_step:
        return None
    return (last_val - first_val) / float(last_step - first_step)


def summarize_run(run_dir: Path) -> dict[str, object]:
    """Produce a lightweight summary for a single run directory."""
    event_files = sorted(run_dir.glob("events.*"), key=lambda p: p.stat().st_mtime)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {run_dir}")

    scalars: ScalarSeries = {}
    for event_file in event_files:
        for tag, series in _load_scalars(event_file).items():
            scalars.setdefault(tag, []).extend(series)

    # Deduplicate by step to keep the last value per step
    for tag, series in scalars.items():
        seen: dict[int, float] = {}
        for step, val in series:
            seen[step] = val
        scalars[tag] = sorted(seen.items(), key=lambda item: item[0])

    ep_rew = scalars.get("rollout/ep_rew_mean", [])
    ep_len = scalars.get("rollout/ep_len_mean", [])
    value_loss = scalars.get("train/value_loss", [])
    entropy = scalars.get("train/entropy_loss", [])
    pg_loss = scalars.get("train/policy_gradient_loss", [])
    lr = scalars.get("train/learning_rate", [])

    max_step = max((step for series in scalars.values() for step, _ in series), default=0)
    latest_rew = _latest(ep_rew)
    best_rew = _best(ep_rew, mode="max")
    start_rew = _at_percentile(ep_rew, 0.0)
    mid_rew = _at_percentile(ep_rew, 0.5)
    reward_slope = _slope_last(ep_rew, window=10)

    start_val_loss = _at_percentile(value_loss, 0.0)
    end_val_loss = _latest(value_loss)
    val_loss_slope = _slope_last(value_loss, window=10)

    start_entropy = _at_percentile(entropy, 0.0)
    end_entropy = _latest(entropy)
    entropy_slope = _slope_last(entropy, window=10)

    return {
        "run": run_dir.name,
        "steps": max_step,
        "final": {
            "ep_rew_mean": latest_rew,
            "ep_len_mean": _latest(ep_len),
            "value_loss": _latest(value_loss),
            "entropy_loss": _latest(entropy),
            "policy_gradient_loss": _latest(pg_loss),
            "learning_rate": _latest(lr),
        },
        "best": {"ep_rew_mean": best_rew},
        "progress": {
            "ep_rew_mean": {
                "start": start_rew,
                "mid": mid_rew,
                "final": latest_rew,
                "delta": None
                if start_rew is None or latest_rew is None
                else (latest_rew[1] - start_rew[1]),
                "slope_last": reward_slope,
            },
            "value_loss": {
                "start": start_val_loss,
                "final": end_val_loss,
                "delta": None
                if start_val_loss is None or end_val_loss is None
                else (end_val_loss[1] - start_val_loss[1]),
                "slope_last": val_loss_slope,
            },
            "entropy_loss": {
                "start": start_entropy,
                "final": end_entropy,
                "delta": None
                if start_entropy is None or end_entropy is None
                else (end_entropy[1] - start_entropy[1]),
                "slope_last": entropy_slope,
            },
        },
        "tags_available": sorted(scalars.keys()),
    }


def format_text_summary(summary: dict[str, object]) -> str:
    """Format a concise human/LLM-friendly text summary."""
    run = summary.get("run", "unknown-run")
    steps = int(summary.get("steps") or 0)
    final = summary.get("final", {}) or {}
    best = summary.get("best", {}) or {}
    progress = summary.get("progress", {}) or {}
    rew_prog = progress.get("ep_rew_mean", {}) or {}
    val_prog = progress.get("value_loss", {}) or {}
    ent_prog = progress.get("entropy_loss", {}) or {}

    def _fmt(entry):
        if entry is None:
            return "n/a"
        step, val = entry
        return f"{val:.3f} @ step {step:,}"

    def _fmt_scalar(val: float | None) -> str:
        if val is None:
            return "n/a"
        return f"{val:.4f}"

    lines = [
        f"Run: {run}",
        f"Total steps seen: {steps:,}",
        f"Episode reward mean: start {_fmt(rew_prog.get('start'))}, mid {_fmt(rew_prog.get('mid'))}, final {_fmt(final.get('ep_rew_mean'))}, best {_fmt(best.get('ep_rew_mean'))}",
        f"Reward delta (final-start): {_fmt_scalar(rew_prog.get('delta'))}, recent slope (last window): {_fmt_scalar(rew_prog.get('slope_last'))} per step",
        f"Episode length mean: {_fmt(final.get('ep_len_mean'))}",
        f"Value loss: start {_fmt(val_prog.get('start'))}, final {_fmt(final.get('value_loss'))}, delta {_fmt_scalar(val_prog.get('delta'))}, recent slope {_fmt_scalar(val_prog.get('slope_last'))} per step",
        f"Entropy loss: start {_fmt(ent_prog.get('start'))}, final {_fmt(final.get('entropy_loss'))}, delta {_fmt_scalar(ent_prog.get('delta'))}, recent slope {_fmt_scalar(ent_prog.get('slope_last'))} per step",
        f"Policy gradient loss (final): {_fmt(final.get('policy_gradient_loss'))}",
        f"Learning rate (final): {_fmt(final.get('learning_rate'))}",
        "Suggested LLM prompt: Summarize convergence, note reward trend direction/plateaus, and flag correlating shifts in value/entropy losses.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TensorBoard logs for PPO/DQN runs.")
    parser.add_argument("--logdir", type=Path, default=Path("data/logs/rl"), help="Root TensorBoard log directory.")
    parser.add_argument(
        "--run",
        type=str,
        help="Specific run directory name under logdir. Defaults to the most recent run.",
    )
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format.")
    parser.add_argument("--output", type=Path, help="Optional path to write the summary to.")
    args = parser.parse_args()

    if not args.logdir.exists():
        raise FileNotFoundError(f"Logdir {args.logdir} does not exist.")

    run_dirs = [p for p in args.logdir.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {args.logdir}")

    if args.run:
        chosen = next((p for p in run_dirs if p.name == args.run), None)
        if chosen is None:
            raise FileNotFoundError(f"Run '{args.run}' not found in {args.logdir}")
    else:
        chosen = max(run_dirs, key=lambda p: p.stat().st_mtime)

    summary = summarize_run(chosen)
    if args.format == "json":
        output = json.dumps(summary, indent=2)
    else:
        output = format_text_summary(summary)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)

    print(output)


if __name__ == "__main__":
    main()
