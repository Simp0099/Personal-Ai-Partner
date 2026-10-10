"""Capability-aware model router.

The router answers two questions:

1. **Which model should serve this request?** A deterministic suitability score
   over capability fit, task fit, context fit, measured reliability, measured
   latency and configured priority.
2. **What is the ordered fallback chain?** Same scoring, filtered so that every
   remaining candidate can still satisfy the request's hard requirements.

Deliberate properties:

* **Deterministic.** No randomness, no model calls, no ML. The same inputs and
  the same health state always produce the same ranking, which is what makes it
  testable.
* **Hard requirements are filters, not penalties.** A model that cannot call
  tools is *excluded* from a tool-required request, never merely scored lower.
* **Latency cannot buy its way past capability or reliability.** Latency is one
  bounded weight among several.
* **Sequential, never raced.** One request goes to one model. Fallback happens
  only after that model actually fails.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from jarvis.classify import Classification, TaskType, classify
from jarvis.health import HealthStats, HealthTracker
from jarvis.models import ModelRegistry, ModelSpec
from jarvis.providers.base import Capabilities, estimate_tokens


@dataclass
class RoutingWeights:
    """Tunable scoring weights.

    They sum to 1.0 so scores stay comparable across releases.
    """

    capability: float = 0.30
    task: float = 0.25
    reliability: float = 0.25
    latency: float = 0.10
    priority: float = 0.10

    def __post_init__(self) -> None:
        total = self.capability + self.task + self.reliability + self.latency + self.priority
        if total > 0:
            # Normalise so mis-set weights still produce a 0..1 score.
            self.capability /= total
            self.task /= total
            self.reliability /= total
            self.latency /= total
            self.priority /= total


@dataclass
class RoutingConfig:
    """Router behaviour settings."""

    enabled: bool = True
    weights: RoutingWeights = field(default_factory=RoutingWeights)
    cooldown_seconds: float = 60.0
    max_attempts: int = 4
    #: Consecutive 429s after which a model is treated as quota-exhausted.
    daily_request_cap: Optional[int] = None
    #: When every candidate is in cooldown, allow the least-bad one anyway
    #: rather than failing outright. A stale cooldown should not deadlock.
    probe_cooling_models: bool = True
    latency_reference_ms: float = 4000.0


TIER_RANK = {"primary": 0, "fallback": 1, "emergency": 2}


@dataclass
class ScoredModel:
    """A candidate plus why it scored what it did."""

    spec: ModelSpec
    score: float
    reasons: Dict[str, float] = field(default_factory=dict)
    health: Optional[HealthStats] = None

    @property
    def key(self) -> str:
        return self.spec.key

    def explain(self) -> str:
        parts = ", ".join(f"{k}={v:+.2f}" for k, v in sorted(self.reasons.items()))
        return f"{self.spec.key} score={self.score:.3f} [{parts}]"


# Capability needed per task type. Missing a *required* capability excludes the
# model outright; `preferred` only affects scoring.
_TASK_REQUIREMENTS: Dict[TaskType, Dict[str, List[str]]] = {
    TaskType.CODING: {"required": ["coding"], "preferred": ["reasoning", "long_context"]},
    TaskType.REASONING: {"required": ["reasoning"], "preferred": ["long_context"]},
    TaskType.LONG_CONTEXT: {"required": ["long_context"], "preferred": ["reasoning"]},
    TaskType.STRUCTURED_OUTPUT: {"required": ["structured_output"], "preferred": []},
    TaskType.RESEARCH: {"required": [], "preferred": ["tool_calling", "reasoning"]},
    TaskType.CREATIVE: {"required": [], "preferred": []},
    TaskType.TOOL_USE: {"required": ["tool_calling"], "preferred": []},
    TaskType.SIMPLE_QUESTION: {"required": [], "preferred": []},
    TaskType.CASUAL: {"required": [], "preferred": []},
}

#: How much a *preferred* capability contributes when present.
_PREFERRED_BONUS = 0.5


class ModelRouter:
    """Selects models for requests using capability, health and latency."""

    def __init__(
        self,
        registry: ModelRegistry,
        health: HealthTracker,
        config: Optional[RoutingConfig] = None,
    ):
        self.registry = registry
        self.health = health
        self.config = config or RoutingConfig()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_request(
        self,
        user_message: str,
        *,
        conversation_tokens: int = 0,
        tools_available: bool = True,
        vision_required: bool = False,
    ) -> Classification:
        """Classify a request, resolving the long-context threshold from config."""
        smallest_window = min(
            (s.context_window for s in self.registry.enabled()), default=100_000
        )
        return classify(
            user_message,
            tools_available=tools_available,
            conversation_tokens=conversation_tokens,
            context_threshold=max(50_000, smallest_window // 2),
            vision_required=vision_required,
        )

    def select(
        self,
        classification: Classification,
        *,
        context_tokens: int = 0,
        providers: Optional[Dict[str, Any]] = None,
        exclude: Optional[Sequence[str]] = None,
    ) -> List[ScoredModel]:
        """Return the ranked fallback chain for a request.

        The first entry is the model to try; the rest are fallbacks that also
        satisfy the request's hard requirements.
        """
        exclude_set = set(exclude or ())
        candidates = self._eligible(classification, context_tokens, providers or {})

        scored = [self._score(spec, classification, context_tokens) for spec in candidates]
        scored = [s for s in scored if s.key not in exclude_set]

        # Best first. Tier first, then score, priority, key — deterministic.
        scored.sort(key=lambda s: (TIER_RANK.get(s.spec.tier, 1), -s.score,
                                   -s.spec.priority, s.spec.key))
        return scored

    def best(
        self,
        classification: Classification,
        *,
        context_tokens: int = 0,
        providers: Optional[Dict[str, Any]] = None,
    ) -> Optional[ScoredModel]:
        """Top-ranked model, or None when nothing is eligible."""
        ranked = self.select(classification, context_tokens=context_tokens, providers=providers)
        return ranked[0] if ranked else None

    # ------------------------------------------------------------------
    # Eligibility (hard requirements -> exclusion)
    # ------------------------------------------------------------------

    def _eligible(
        self,
        classification: Classification,
        context_tokens: int,
        providers: Dict[str, Any],
    ) -> List[ModelSpec]:
        """Filter to models that can actually serve this request.

        Models in cooldown are dropped unless every model is cooling down, in
        which case the least-recently-failed is returned so recovery is possible.
        """
        pool: List[ModelSpec] = []

        for spec in self.registry.enabled():
            provider = providers.get(spec.provider)
            if provider is None or not provider.is_configured():
                continue
            if not spec.can_fit(context_tokens):
                continue
            if classification.tool_required and not spec.supports("tool_calling"):
                # Hard requirement: a tool-required request must never reach a
                # model that cannot execute tools.
                continue
            if classification.vision_required and not spec.supports("vision"):
                # Hard requirement: sending an image to a text-only model would
                # either be rejected or silently answered from the text alone,
                # which is worse than an honest failure.
                continue
            if not self._meets_task_requirements(spec, classification):
                continue
            pool.append(spec)

        if not pool:
            return []

        available = [s for s in pool if self.health.available(s.key)]
        if available or not self.config.probe_cooling_models:
            return available

        # Everything is cooling down. Re-admit the model whose cooldown expires
        # soonest so the system can recover instead of deadlocking -- but never a
        # quota-exhausted one: probing it just burns the rest of a daily budget
        # that cannot recover before the reset anyway.
        usable = [s for s in pool if not self._quota_exhausted(s.key)]
        pool = usable or pool
        return [min(pool, key=lambda s: (self.health.stats(s.key).cooldown_remaining(), s.key))]

    def _quota_exhausted(self, key: str) -> bool:
        """Whether a model is parked until its provider's daily reset."""
        stat = self.health.peek(key)
        return bool(stat and stat.quota_exhausted_at is not None)

    def _meets_task_requirements(self, spec: ModelSpec, classification: Classification) -> bool:
        """Check required capabilities for the classified task."""
        for task in classification.all_types:
            requirements = _TASK_REQUIREMENTS.get(task)
            if not requirements:
                continue
            for capability in requirements.get("required", []):
                if not spec.supports(capability):
                    return False
        return True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(
        self,
        spec: ModelSpec,
        classification: Classification,
        context_tokens: int,
    ) -> ScoredModel:
        """Compute the suitability score for one candidate."""
        reasons: Dict[str, float] = {}
        weights = self.config.weights

        capability_score = self._capability_score(spec, classification, context_tokens, reasons)
        task_score = self._task_score(spec, classification, reasons)
        health = self.health.peek(spec.key)
        reliability_score = self._reliability_score(spec, health, reasons)
        latency_score = self._latency_score(spec, health, reasons)
        priority_score = self._priority_score(spec, health, reasons)

        total = (
            capability_score * weights.capability
            + task_score * weights.task
            + reliability_score * weights.reliability
            + latency_score * weights.latency
            + priority_score * weights.priority
        )

        return ScoredModel(
            spec=spec,
            score=round(total, 6),
            reasons=reasons,
            health=health,
        )

    def _capability_score(
        self,
        spec: ModelSpec,
        classification: Classification,
        context_tokens: int,
        reasons: Dict[str, float],
    ) -> float:
        """Reward capabilities the task actually benefits from.

        This is the dominant term, and it is bounded in [0, 1], so a model that
        merely lacks an optional capability is not punished out of contention.
        """
        wanted: List[str] = ["conversation"]
        for task in classification.all_types:
            requirements = _TASK_REQUIREMENTS.get(task)
            if not requirements:
                continue
            wanted.extend(requirements.get("required", []))
            wanted.extend(requirements.get("preferred", []))

        if classification.tool_required:
            wanted.append("tool_calling")
        if context_tokens > 32_000:
            wanted.append("long_context")

        unique = list(dict.fromkeys(wanted))
        if not unique:
            return 1.0

        hits = 0.0
        for capability in unique:
            if spec.supports(capability):
                hits += 1.0
        value = hits / len(unique)

        # Preference for verified models: an unverified model should not
        # outrank an equally capable verified one by accident.
        if not spec.verified:
            value *= 0.9
            reasons["unverified_penalty"] = -0.02

        reasons["capability"] = value
        return value

    def _task_score(
        self,
        spec: ModelSpec,
        classification: Classification,
        reasons: Dict[str, float],
    ) -> float:
        """Reward models with declared affinity for the classified task."""
        primary = classification.task_type.value
        affinity = 0.0
        if primary in spec.task_affinity:
            affinity += float(spec.task_affinity[primary]) / 100.0
        for task in classification.secondary:
            if task.value in spec.task_affinity:
                affinity += 0.5 * (float(spec.task_affinity[task.value]) / 100.0)

        value = max(0.0, min(1.0, 0.5 + affinity))
        reasons["task"] = value
        return value

    def _reliability_score(
        self,
        spec: ModelSpec,
        health: Optional[HealthStats],
        reasons: Dict[str, float],
    ) -> float:
        """Measured success rate, with an optimistic prior when unmeasured.

        The prior matters: a brand-new model must be able to earn traffic, so an
        unmeasured model starts at 0.8 rather than 0.
        """
        if health is None or health.attempts == 0:
            value = 0.8
        else:
            value = max(0.0, 1.0 - health.failure_rate)
            # A model still in cooldown should not be preferred, but it is
            # normally excluded upstream; this covers the recovery probe case.
            if not health.is_available():
                value *= 0.5

        reasons["reliability"] = value
        return value

    def _latency_score(
        self,
        spec: ModelSpec,
        health: Optional[HealthStats],
        reasons: Dict[str, float],
    ) -> float:
        """Measured latency mapped into [0, 1] against a reference.

        Bounded so latency can nudge the ranking but never dominate: it carries
        its own weight and cannot compensate for missing capability.
        """
        latency_ms = None
        if health is not None and health.success_count:
            latency_ms = health.recent_average_latency * 1000.0

        if latency_ms is None or latency_ms <= 0:
            value = 0.5
        else:
            reference = max(1.0, self.config.latency_reference_ms)
            # 0 latency -> 1.0, reference latency or worse -> 0.0
            value = max(0.0, min(1.0, 1.0 - (latency_ms / (2 * reference))))

        reasons["latency"] = value
        return value

    def _priority_score(
        self,
        spec: ModelSpec,
        health: Optional[HealthStats],
        reasons: Dict[str, float],
    ) -> float:
        """Configured preference, adjusted by observed behaviour.

        Rate-limited models are demoted here rather than by silently reordering
        config, so the operator's declared intent stays visible.
        """
        value = max(0.0, min(1.0, spec.priority / 100.0))

        if health is not None and health.rate_limit_count:
            demotion = min(0.5, 0.1 * health.rate_limit_count)
            value -= demotion
            reasons["rate_limit_demotion"] = -round(demotion, 3)

        reasons["priority"] = max(0.0, value)
        return max(0.0, value)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status_snapshot(
        self,
        providers: Optional[Dict[str, Any]] = None,
        health: Optional[HealthTracker] = None,
    ) -> List[Dict[str, Any]]:
        """Per-model status for the debug endpoint. No credentials included."""
        health = health or self.health
        now_ts = time.time()

        rows: List[Dict[str, Any]] = []
        for spec in self.registry.all():
            stats = health.peek(spec.key)
            provider = (providers or {}).get(spec.provider)

            configured = bool(provider and provider.is_configured())
            reachable = None
            if configured and provider:
                reachable = provider.probe()

            row = spec.status_snapshot()
            row.update({
                "configured": configured,
                "provider_available": reachable,
                "health": stats.status(now_ts) if stats else "unknown",
                "average_latency": round(stats.average_latency, 3) if stats else 0.0,
                "failure_rate": round(stats.failure_rate, 3) if stats else 0.0,
                "cooldown_remaining": round(stats.cooldown_remaining(now_ts), 1) if stats else 0.0,
                "last_error_kind": stats.last_error_kind if stats else None,
                "success_count": stats.success_count if stats else 0,
                "failure_count": stats.failure_count if stats else 0,
            })
            rows.append(row)

        return rows