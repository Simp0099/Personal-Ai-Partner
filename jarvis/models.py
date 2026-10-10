"""Configuration-driven model registry.

A model is a configuration entry, not code. Adding, removing, re-prioritising or
disabling a model is a `config.yaml` edit -- no Python changes.

Every entry records verified facts about the model:

``verified``
    Whether the model id, availability, context length and capability claims
    were actually checked against the provider. Unverified models are never
    advertised as supported.
``verified_on`` / ``notes``
    Provenance for the claims above.
``free``
    Whether the model is usable at no cost, which the router prefers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from jarvis.logger import logger
from jarvis.providers.base import Capabilities

# Long-context threshold used to set the `long_context` capability and for
# task routing. Deliberately conservative: 100k tokens.
LONG_CONTEXT_THRESHOLD = 100_000


@dataclass
class ModelSpec:
    """One configured, routable model."""

    key: str
    provider: str
    model_id: str
    enabled: bool = True
    priority: int = 50
    free: bool = False
    context_window: int = 128_000
    max_output_tokens: int = 4096
    capabilities: Capabilities = field(default_factory=Capabilities)
    verified: bool = False
    verified_on: str = ""
    notes: str = ""
    tier: str = "fallback"
    # Free-form task affinity hints, e.g. {"coding": 10}
    task_affinity: Dict[str, int] = field(default_factory=dict)

    def supports(self, capability: str) -> bool:
        return bool(getattr(self.capabilities, capability, False))

    def can_fit(self, estimated_tokens: int) -> bool:
        """Whether the request plausibly fits in this model's context window.

        Leaves headroom for the system prompt and the model's own reply.
        """
        reserve = self.max_output_tokens
        return estimated_tokens + reserve <= self.context_window

    def status_snapshot(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "provider": self.provider,
            "model_id": self.model_id,
            "enabled": self.enabled,
            "free": self.free,
            "verified": self.verified,
            "priority": self.priority,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "capabilities": self.capabilities.as_dict(),
            "notes": self.notes,
        }


class ModelRegistry:
    """Holds every configured :class:`ModelSpec`."""

    def __init__(self, specs: Optional[List[ModelSpec]] = None):
        self._specs: Dict[str, ModelSpec] = {}
        for spec in specs or []:
            self._specs[spec.key] = spec

    # -- construction ------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelRegistry":
        """Build a registry from the ``models:`` block of config.yaml.

        Malformed entries are reported rather than dropped silently: a silently
        ignored model looks identical to a model that was never configured, and
        that distinction matters when diagnosing "why didn't my model get used".
        """
        specs: List[ModelSpec] = []

        for key, raw in (config or {}).items():
            if not isinstance(raw, dict):
                logger.warning(
                    f"[MODELS] entry '{key}' is {type(raw).__name__}, expected a "
                    f"mapping; skipped"
                )
                continue

            if not raw.get("provider") or not raw.get("model"):
                logger.warning(
                    f"[MODELS] entry '{key}' is missing 'provider' or 'model'; skipped"
                )
                continue

            caps_raw = raw.get("capabilities") or {}
            context_window = int(raw.get("context_window", 128_000))
            capabilities = Capabilities(
                reasoning=bool(caps_raw.get("reasoning", False)),
                coding=bool(caps_raw.get("coding", False)),
                conversation=bool(caps_raw.get("conversation", True)),
                tool_calling=bool(caps_raw.get("tool_calling", False)),
                vision=bool(caps_raw.get("vision", False)),
                structured_output=bool(caps_raw.get("structured_output", False)),
                long_context=bool(
                    caps_raw.get("long_context", context_window >= LONG_CONTEXT_THRESHOLD)
                ),
                streaming=bool(caps_raw.get("streaming", True)),
            )

            specs.append(ModelSpec(
                key=key,
                provider=str(raw.get("provider", "")),
                model_id=str(raw.get("model", "")),
                enabled=bool(raw.get("enabled", True)),
                priority=int(raw.get("priority", 50)),
                free=bool(raw.get("free", False)),
                context_window=context_window,
                max_output_tokens=int(raw.get("max_output_tokens", 4096)),
                capabilities=capabilities,
                verified=bool(raw.get("verified", False)),
                verified_on=str(raw.get("verified_on", "")),
                notes=str(raw.get("notes", "")),
                tier=str(raw.get("tier", "fallback")),
                task_affinity={str(k): int(v) for k, v in (raw.get("task_affinity") or {}).items()},
            ))

        return cls(specs)

    # -- access ------------------------------------------------------------

    def all(self) -> List[ModelSpec]:
        return list(self._specs.values())

    def get(self, key: str) -> Optional[ModelSpec]:
        return self._specs.get(key)

    def enabled(self) -> List[ModelSpec]:
        """Specs that are enabled and usable (provider + model id present)."""
        return [
            s for s in self._specs.values()
            if s.enabled and s.provider and s.model_id
        ]

    def for_provider(self, provider: str) -> List[ModelSpec]:
        return [s for s in self._specs.values() if s.provider == provider]

    def replace(self, specs: List[ModelSpec]) -> None:
        """Replace the registry contents (used by tests and /api/models reload)."""
        self._specs = {s.key: s for s in specs}

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, key: object) -> bool:
        return key in self._specs