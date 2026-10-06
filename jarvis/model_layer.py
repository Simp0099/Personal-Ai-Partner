"""Model-layer wiring.

Assembles the provider adapters, model registry, health tracker and router from
configuration, and hands out a single shared :class:`ModelLayer` to the Brain.

This is the only place that knows how configuration maps onto objects, so the
Brain, the router and the tests all share one source of truth.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

from jarvis.classify import Classification
from jarvis.config import (
    MODELS_CONFIG,
    PROVIDERS_CONFIG,
    ROUTING_CONFIG,
    LLM_TEMPERATURE,
    LLM_MAX_OUTPUT_TOKENS,
    DEBUG_INPUT,
)
from jarvis.health import HealthTracker
from jarvis.logger import logger
from jarvis.models import ModelRegistry, ModelSpec
from jarvis.providers import build_provider
from jarvis.providers.base import Capabilities, ModelProvider, ProviderError, estimate_tokens
from jarvis.router import ModelRouter, RoutingConfig, RoutingWeights, ScoredModel


def _build_routing_config(raw: Dict[str, Any]) -> RoutingConfig:
    """Translate the ``routing:`` block into a :class:`RoutingConfig`."""
    raw = raw or {}
    weights = RoutingWeights(
        capability=float(raw.get("capability_weight", 0.30)),
        task=float(raw.get("task_weight", 0.25)),
        reliability=float(raw.get("reliability_weight", 0.25)),
        latency=float(raw.get("latency_weight", 0.10)),
        priority=float(raw.get("priority_weight", 0.10)),
    )
    return RoutingConfig(
        enabled=bool(raw.get("enabled", True)),
        weights=weights,
        cooldown_seconds=float(raw.get("cooldown_seconds", 60.0)),
        max_attempts=int(raw.get("max_attempts", 4)),
        probe_cooling_models=bool(raw.get("probe_cooling_models", True)),
        prefer_free=bool(raw.get("prefer_free", True)),
        latency_reference_ms=float(raw.get("latency_reference_ms", 4000.0)),
    )


class ModelLayer:
    """The provider-agnostic model layer: registry + providers + health + router."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        providers: Optional[Dict[str, ModelProvider]] = None,
        health: Optional[HealthTracker] = None,
        router: Optional[ModelRouter] = None,
        routing_config: Optional[RoutingConfig] = None,
        temperature: float = LLM_TEMPERATURE,
        max_output_tokens: int = LLM_MAX_OUTPUT_TOKENS,
    ):
        self.registry = registry if registry is not None else ModelRegistry()
        self.providers: Dict[str, ModelProvider] = providers or {}
        self.health = health if health is not None else HealthTracker()
        if routing_config is None:
            routing_config = _build_routing_config(ROUTING_CONFIG)
        self.health.base_backoff = routing_config.cooldown_seconds
        self.router = router if router is not None else ModelRouter(
            self.registry, self.health, routing_config
        )
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, env: Optional[Dict[str, str]] = None) -> "ModelLayer":
        """Build the layer from config.yaml plus environment credentials."""
        env = dict(os.environ if env is None else env)

        registry = ModelRegistry.from_config(MODELS_CONFIG)

        providers: Dict[str, ModelProvider] = {}
        for name, settings in (PROVIDERS_CONFIG or {}).items():
            if not isinstance(settings, dict):
                continue
            if not settings.get("enabled", True):
                logger.info(f"[MODELS] provider '{name}' disabled by config")
                continue
            provider = build_provider(name, settings, env)
            if provider is None:
                logger.warning(f"[MODELS] unknown provider type for '{name}'; skipped")
                continue
            if not provider.is_configured():
                # Not an error: a provider without credentials is simply
                # unavailable, and the router will route around it.
                logger.info(
                    f"[MODELS] provider '{name}' has no credentials "
                    f"({settings.get('api_key_env', 'no key env set')}); unavailable"
                )
            providers[name] = provider

        layer = cls(registry=registry, providers=providers)
        layer._log_summary()
        return layer

    def _log_summary(self) -> None:
        """Log which models are usable. Never logs credentials."""
        usable = [
            f"{s.key}({s.provider})"
            for s in self.registry.enabled()
            if (p := self.providers.get(s.provider)) and p.is_configured()
        ]
        total = len(self.registry.enabled())
        logger.info(f"[MODELS] {len(usable)}/{total} enabled models usable: {usable or 'none'}")

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def classify(
        self,
        user_message: str,
        *,
        conversation_tokens: int = 0,
        tools_available: bool = True,
    ) -> Classification:
        return self.router.classify_request(
            user_message,
            conversation_tokens=conversation_tokens,
            tools_available=tools_available,
        )

    def plan(
        self,
        classification: Classification,
        *,
        context_tokens: int = 0,
    ) -> List[ScoredModel]:
        """Ranked fallback chain for a classified request."""
        return self.router.select(
            classification, context_tokens=context_tokens, providers=self.providers
        )

    def spec_for(self, key: str) -> Optional[ModelSpec]:
        return self.registry.get(key)

    # ------------------------------------------------------------------
    # Health bookkeeping
    # ------------------------------------------------------------------

    def record_success(self, spec: ModelSpec, latency: float) -> None:
        self.health.record_success(spec.key, latency, provider=spec.provider)

    def record_failure(self, spec: ModelSpec, error: ProviderError, latency: float = 0.0) -> None:
        self.health.record_failure(
            spec.key, error.kind, latency=latency, provider=spec.provider
        )

    def options_for(self, spec: ModelSpec, timeout: float = 60.0) -> Dict[str, Any]:
        """Per-model generation options."""
        return {
            "temperature": self.temperature,
            "max_output_tokens": min(self.max_output_tokens, spec.max_output_tokens)
            or self.max_output_tokens,
            "timeout": timeout,
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Model status snapshot for the debug endpoint. No credentials."""
        return {
            "routing_enabled": self.router.config.enabled,
            "weights": {
                "capability": self.router.config.weights.capability,
                "task": self.router.config.weights.task,
                "reliability": self.router.config.weights.reliability,
                "latency": self.router.config.weights.latency,
                "priority": self.router.config.weights.priority,
            },
            "max_attempts": self.router.config.max_attempts,
            "cooldown_seconds": self.router.config.cooldown_seconds,
            "models": self.router.status_snapshot(self.providers),
        }


# ---------------------------------------------------------------------------
# Process-wide shared layer
# ---------------------------------------------------------------------------

_shared: Optional[ModelLayer] = None
_shared_lock = threading.Lock()


def get_model_layer() -> ModelLayer:
    """Return the process-wide model layer, building it on first use."""
    global _shared
    if _shared is None:
        with _shared_lock:
            if _shared is None:
                _shared = ModelLayer.from_config()
    return _shared


def set_model_layer(layer: Optional[ModelLayer]) -> None:
    """Replace the shared layer (used by tests)."""
    global _shared
    with _shared_lock:
        _shared = layer


__all__ = [
    "Capabilities",
    "Classification",
    "HealthTracker",
    "ModelLayer",
    "ModelRegistry",
    "ModelRouter",
    "ModelSpec",
    "ProviderError",
    "RoutingConfig",
    "RoutingWeights",
    "ScoredModel",
    "estimate_tokens",
    "get_model_layer",
    "set_model_layer",
    "DEBUG_INPUT",
]