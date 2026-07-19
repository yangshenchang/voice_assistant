"""Lightweight in-memory context tracker.

Since nanobot manages conversation history server-side, we only need
to track creation timestamps locally for the context-timeout check.
"""

from datetime import datetime, timezone
from typing import Dict


class ContextTracker:
    """Tracks context creation timestamps in memory."""

    def __init__(self):
        self._timestamps: Dict[str, datetime] = {}

    def touch(self, context_id: str):
        """Record/update the timestamp for a context."""
        self._timestamps[context_id] = datetime.now(timezone.utc)

    def get_age(self, context_id: str) -> float:
        """Return age of context in seconds, or 0 if unknown."""
        ts = self._timestamps.get(context_id)
        if ts is None:
            return 0
        return (datetime.now(timezone.utc) - ts).total_seconds()

    def remove(self, context_id: str):
        """Forget a context."""
        self._timestamps.pop(context_id, None)
