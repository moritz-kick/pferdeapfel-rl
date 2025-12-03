"""Training utilities for Pferdeäpfel RL agents."""

from __future__ import annotations

__all__ = [
    "SelfPlayCallback",
    "OpponentUpdateCallback",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "SelfPlayCallback":
        from src.training.train_ppo_self_play import SelfPlayCallback
        return SelfPlayCallback
    elif name == "OpponentUpdateCallback":
        from src.training.train_ppo_self_play import OpponentUpdateCallback
        return OpponentUpdateCallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
