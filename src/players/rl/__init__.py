"""RL player utilities and discovery."""

from __future__ import annotations

import importlib
import pkgutil
import logging
from typing import Dict, Type

from src.players.base import Player

logger = logging.getLogger(__name__)


def discover_rl_players() -> Dict[str, Type[Player]]:
    """Dynamically import RL player classes available under this package."""
    players: Dict[str, Type[Player]] = {}
    package_path = __path__  # type: ignore[name-defined]

    for module_info in pkgutil.iter_modules(package_path):
        if module_info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{__name__}.{module_info.name}")
        except Exception as exc:  # pragma: no cover - discovery should not crash
            logger.warning("Skipping RL module %s: %s", module_info.name, exc)
            continue

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, Player) and attr is not Player:
                players[attr.__name__] = attr
    return players


__all__ = ["discover_rl_players"]
