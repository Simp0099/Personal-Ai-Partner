"""Deterministic local-first intent routing (Phase 2).

``handle(text, dispatch)`` runs before any provider call. See router.py.
"""

from jarvis.intents.router import LocalIntentResult, handle, match_only

__all__ = ["LocalIntentResult", "handle", "match_only"]
