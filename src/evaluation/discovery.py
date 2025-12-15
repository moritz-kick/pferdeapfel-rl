"""Player discovery module."""

import importlib
import inspect
import pkgutil
from typing import List, Type

from src.players.base import Player


def discover_players(package_name: str = "src.players") -> List[Type[Player]]:
    """
    Discover all Player subclasses in the given package.

    Args:
        package_name: Dot-separated package path (e.g. "src.players")

    Returns:
        List of Player subclasses found in the package.
    """
    player_classes: List[Type[Player]] = []

    # Import the package
    package = importlib.import_module(package_name)

    # Iterate over all modules in the package
    if hasattr(package, "__path__"):
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            full_module_name = f"{package_name}.{module_name}"
            try:
                module = importlib.import_module(full_module_name)

                # Inspect module members
                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, Player)
                        and obj is not Player
                        and not inspect.isabstract(obj)
                    ):
                        # Avoid duplicates (e.g. imported into __init__)
                        if obj not in player_classes:
                            player_classes.append(obj)

            except Exception as e:
                print(f"Error importing module {full_module_name}: {e}")

    return player_classes
