"""Training-time wrappers and helpers."""

from __future__ import annotations

from sb3_contrib.common.wrappers import ActionMasker


class ForwardingActionMasker(ActionMasker):
    """ActionMasker that forwards missing attributes to the wrapped env."""

    def __getattr__(self, name):
        return getattr(self.env, name)


__all__ = ["ForwardingActionMasker"]
