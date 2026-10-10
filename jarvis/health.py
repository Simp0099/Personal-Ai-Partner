"""Runtime model health tracking and automatic cooldown.

In-memory only, by design. Nothing here needs to survive a restart: if the
process restarts, the correct default assumption is "everything is healthy
until proven otherwise".

The tracker's job is to stop the router from repeatedly hammering an endpoint
that is already known to be failing, and to let the router learn real latency
so a fast *reliable* model outranks a fast flaky one.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from jarvis.providers.base import ErrorKind

# Exponential backoff bounds for consecutive failures.
MIN_BACKOFF_SECONDS = 15.0
MAX_BACKOFF_SECONDS = 900.0

#: Cooldown applied once a model's daily request cap is hit. Reaching the cap
#: is categorically different from one transient 429: the key resets at the
#: provider's daily boundary, so a 60s backoff just burns more of the quota.
DAILY_QUOTA_COOLDOWN_SECONDS = 24 * 3600.0


@dataclass
class HealthStats:
    """Per-model runtime statistics."""

    model_id: str
    provider: str = ""

    success_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0
    rate_limit_count: int = 0
    auth_error_count: int = 0

    total_latency: float = 0.0
    #: Rolling window of recent latencies, newest last.
    recent_latencies: List[float] = field(default_factory=list)

    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    last_error_kind: Optional[str] = None
    cooldown_until: Optional[float] = None
    consecutive_failures: int = 0
    #: Set when `daily_request_cap` was reached. Distinguishes "this key is done
    #: until the provider's daily reset" from a transient rate limit.
    quota_exhausted_at: Optional[float] = None
    daily_request_cap: Optional[int] = None
    requests_this_window: int = 0
    window_started_at: float = field(default_factory=time.time)

    # -- derived ----------------------------------------------------------

    @property
    def attempts(self) -> int:
        return self.success_count + self.failure_count

    @property
    def average_latency(self) -> float:
        if not self.success_count:
            return 0.0
        return self.total_latency / self.success_count

    @property
    def recent_average_latency(self) -> float:
        if not self.recent_latencies:
            return self.average_latency
        return sum(self.recent_latencies) / len(self.recent_latencies)

    @property
    def failure_rate(self) -> float:
        if not self.attempts:
            return 0.0
        return self.failure_count / self.attempts

    def is_available(self, now_ts: Optional[float] = None) -> bool:
        """True unless the model is inside an active cooldown."""
        if self.cooldown_until is None:
            return True
        return (now_ts if now_ts is not None else time.time()) >= self.cooldown_until

    def cooldown_remaining(self, now_ts: Optional[float] = None) -> float:
        if self.cooldown_until is None:
            return 0.0
        remaining = self.cooldown_until - (now_ts if now_ts is not None else time.time())
        return max(0.0, remaining)

    def status(self, now_ts: Optional[float] = None) -> str:
        """Human-readable health label for the status endpoint."""
        if not self.is_available(now_ts):
            return "cooldown"
        if self.attempts == 0:
            return "unknown"
        if self.success_count == 0:
            return "unhealthy"
        if self.failure_rate >= 0.5:
            return "degraded"
        if self.failure_rate > 0:
            return "flaky"
        return "healthy"

    def snapshot(self, now_ts: Optional[float] = None) -> Dict[str, object]:
        """Serialisable view for the status endpoint. Never includes secrets."""
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "status": self.status(now_ts),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "timeout_count": self.timeout_count,
            "rate_limit_count": self.rate_limit_count,
            "failure_rate": round(self.failure_rate, 3),
            "average_latency": round(self.average_latency, 3),
            "recent_average_latency": round(self.recent_average_latency, 3),
            "last_error_kind": self.last_error_kind,
            "cooldown_remaining": round(self.cooldown_remaining(now_ts), 1),
        }


class HealthTracker:
    """Thread-safe store of :class:`HealthStats` keyed by model id."""

    def __init__(
        self,
        base_backoff: float = 60.0,
        max_backoff: float = MAX_BACKOFF_SECONDS,
        daily_request_cap: Optional[int] = None,
    ):
        self._stats: Dict[str, HealthStats] = {}
        self._lock = threading.RLock()
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        #: Per-model request budget before the key is parked until the daily
        #: reset. None disables the cap entirely.
        self.daily_request_cap = daily_request_cap

    def set_daily_request_cap(self, cap: Optional[int]) -> None:
        self.daily_request_cap = cap
        with self._lock:
            for stat in self._stats.values():
                stat.daily_request_cap = cap

    def stats(self, model_id: str) -> HealthStats:
        """Get (or create) stats for a model."""
        with self._lock:
            stat = self._stats.get(model_id)
            if stat is None:
                stat = HealthStats(model_id=model_id)
                self._stats[model_id] = stat
            return stat

    def peek(self, model_id: str) -> Optional[HealthStats]:
        """Get stats without creating an entry."""
        with self._lock:
            return self._stats.get(model_id)

    def record_success(self, model_id: str, latency: float, provider: str = "") -> None:
        with self._lock:
            stat = self.stats(model_id)
            if provider and not stat.provider:
                stat.provider = provider
            stat.success_count += 1
            stat.consecutive_failures = 0
            stat.total_latency += max(0.0, latency)
            stat.recent_latencies.append(max(0.0, latency))
            if len(stat.recent_latencies) > 20:
                stat.recent_latencies = stat.recent_latencies[-20:]
            stat.last_success = time.time()
            stat.cooldown_until = None
            stat.quota_exhausted_at = None

    def record_failure(
        self,
        model_id: str,
        kind: ErrorKind,
        latency: float = 0.0,
        provider: str = "",
    ) -> None:
        """Record a failure and apply cooldown when warranted.

        Auth/invalid-request errors do **not** trigger cooldown: retrying the
        same misconfigured model cannot help, and hiding that behind backoff
        would mask a genuine configuration problem.
        """
        now_ts = time.time()
        with self._lock:
            stat = self.stats(model_id)
            if provider and not stat.provider:
                stat.provider = provider
            stat.failure_count += 1
            stat.consecutive_failures += 1
            stat.last_failure = now_ts
            stat.last_error_kind = kind.value
            if latency:
                stat.total_latency += max(0.0, latency)

            if kind is ErrorKind.TIMEOUT:
                stat.timeout_count += 1
            elif kind is ErrorKind.RATE_LIMIT:
                stat.rate_limit_count += 1
            elif kind is ErrorKind.AUTH:
                stat.auth_error_count += 1

            if kind in (ErrorKind.AUTH, ErrorKind.INVALID_REQUEST, ErrorKind.UNSUPPORTED):
                return

            # A rate limit that repeats past the configured budget is a spent
            # daily quota, not a throttle: park the model until the reset rather
            # than retrying it for the next few hours.
            if kind is ErrorKind.RATE_LIMIT and self._quota_spent(stat, now_ts):
                stat.quota_exhausted_at = now_ts
                stat.cooldown_until = now_ts + DAILY_QUOTA_COOLDOWN_SECONDS
                return

            backoff = min(
                self.max_backoff,
                self.base_backoff * (2 ** (stat.consecutive_failures - 1)),
            )
            stat.cooldown_until = now_ts + backoff

    def _quota_spent(self, stat: HealthStats, now_ts: float) -> bool:
        """Whether this rate limit was the configured request budget running out.

        The count is consecutive 429s only. Successes reset it, so one unlucky
        burst on an otherwise healthy model never parks the model for a day.
        """
        if not self.daily_request_cap:
            return False
        stat.daily_request_cap = self.daily_request_cap
        return stat.rate_limit_count >= self.daily_request_cap

    def seconds_until_quota_reset(self, model_id: str) -> float:
        """Seconds until a quota-exhausted model is worth retrying (0 if not)."""
        stat = self.peek(model_id)
        if stat is None or stat.quota_exhausted_at is None:
            return 0.0
        return stat.cooldown_remaining()

    def available(self, model_id: str, now_ts: Optional[float] = None) -> bool:
        with self._lock:
            stat = self._stats.get(model_id)
            if stat is None:
                return True
            return stat.is_available(now_ts)

    def in_cooldown(self, model_ids: List[str], now_ts: Optional[float] = None) -> List[str]:
        """Subset of ``model_ids`` currently cooling down."""
        now_ts = now_ts if now_ts is not None else time.time()
        with self._lock:
            return [
                mid for mid in model_ids
                if (s := self._stats.get(mid)) is not None and not s.is_available(now_ts)
            ]

    def reset(self, model_id: Optional[str] = None) -> None:
        """Clear stats for one model, or all of them."""
        with self._lock:
            if model_id is None:
                self._stats.clear()
            else:
                self._stats.pop(model_id, None)

    def snapshot(self) -> Dict[str, Dict[str, object]]:
        now_ts = time.time()
        with self._lock:
            return {mid: s.snapshot(now_ts) for mid, s in self._stats.items()}