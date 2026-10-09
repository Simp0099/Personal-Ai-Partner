"""End-to-end latency tracing (Phase 9: measurement only).

One `TurnTrace` per assistant turn, correlated by trace ID. Spans use a
monotonic clock and record {duration ms, status, attrs} -- never prompts,
bodies, audio, or secrets. Summaries go through the Phase 1 logger at DEBUG,
so normal user output stays clean.

Tracing never changes behavior: span overhead is two clock reads, failures
inside spans re-raise, and every method degrades to a no-op rather than
raising when diagnostics themselves break.
"""

from __future__ import annotations

import contextvars
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from jarvis.logger import logger

OK = "ok"
FAIL = "fail"
CANCELLED = "cancelled"
SKIPPED = "skipped"


class Span:
    """One timed stage. Use as a context manager; exceptions mark fail."""

    def __init__(self, trace: "TurnTrace", name: str, attrs: Dict[str, Any]):
        self._trace = trace
        self.name = name
        self.attrs = dict(attrs)
        self.ms: float = 0.0
        self.status: str = OK
        self._start: float = 0.0

    def __enter__(self) -> "Span":
        self._start = self._trace._clock()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            self.ms = (self._trace._clock() - self._start) * 1000.0
            if exc_type is not None:
                self.status = FAIL
        except Exception:  # noqa: BLE001 - diagnostics never break the turn
            pass
        return False  # never swallow

    def complete(self, status: str = OK) -> "Span":
        try:
            self.ms = (self._trace._clock() - self._start) * 1000.0
            self.status = status
        except Exception:  # noqa: BLE001
            pass
        return self


class TurnTrace:
    """Spans for one assistant turn. Thread/task-safe via contextvars."""

    def __init__(self, trace_id: Optional[str] = None, clock=None):
        self.trace_id = trace_id or f"turn_{uuid.uuid4().hex[:8]}"
        self._clock = clock or time.perf_counter
        self._spans: List[Span] = []
        self.route: str = ""
        self._finished = False

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        """Time one stage. Skipped stages are simply never opened."""
        span = Span(self, name, attrs)
        self._spans.append(span)
        with span:
            yield span

    def note(self, key: str, value: Any) -> None:
        """Attach a scalar observation (e.g. wake_frame_ms) without a span."""
        try:
            span = Span(self, key, {})
            span.ms = float(value)
            self._spans.append(span)
        except Exception:  # noqa: BLE001
            pass

    def cancel_open(self, reason: str = "") -> None:
        """Mark unfinished business cancelled (stale/interrupted turns)."""
        try:
            for span in self._spans:
                if span.ms == 0.0 and span.status == OK:
                    span.status = CANCELLED
                    if reason:
                        span.attrs["reason"] = reason
        except Exception:  # noqa: BLE001
            pass

    @property
    def spans(self) -> List[Span]:
        return list(self._spans)

    def total_ms(self) -> float:
        try:
            return sum(s.ms for s in self._spans if s.name != "turn_total")
        except Exception:  # noqa: BLE001
            return 0.0

    def finish(self, route: str = "") -> Dict[str, Any]:
        """Emit one compact DEBUG summary. Idempotent; never raises."""
        try:
            if self._finished:
                return self.to_dict()
            self._finished = True
            if route:
                self.route = route
            summary = self.to_dict()
            stages = " ".join(
                f"{s['stage']}={s['ms']:.1f}ms:{s['status']}" for s in summary["stages"]
            )
            logger.debug(f"trace {self.trace_id} route={self.route or '?'} "
                         f"total={summary['total_ms']:.1f}ms stages=[{stages}]")
            return summary
        except Exception:  # noqa: BLE001
            return {}

    def to_dict(self) -> Dict[str, Any]:
        try:
            return {
                "trace_id": self.trace_id,
                "route": self.route,
                "total_ms": round(self.total_ms(), 1),
                "stages": [
                    {"stage": s.name, "ms": round(s.ms, 1),
                     "status": s.status, "attrs": dict(s.attrs)}
                    for s in self._spans
                ],
            }
        except Exception:  # noqa: BLE001
            return {"trace_id": self.trace_id, "route": "", "total_ms": 0.0,
                    "stages": []}


_active: contextvars.ContextVar[Optional[TurnTrace]] = contextvars.ContextVar(
    "jarvis_turn_trace", default=None
)


def current_trace() -> Optional[TurnTrace]:
    """The trace bound to this thread/task, if any."""
    try:
        return _active.get()
    except Exception:  # noqa: BLE001
        return None


@contextmanager
def use_trace(trace: TurnTrace) -> Iterator[TurnTrace]:
    """Bind `trace` for nested calls (one turn, no duplicate events)."""
    token = _active.set(trace)
    try:
        yield trace
    finally:
        _active.reset(token)


@contextmanager
def span_of(name: str, **attrs: Any) -> Iterator[Optional[Span]]:
    """Open a span on the active trace, or a no-op when untraced.

    Lets hot paths instrument unconditionally: with no bound trace this
    costs one function call and changes nothing.
    """
    trace = current_trace()
    if trace is None:
        yield None
    else:
        with trace.span(name, **attrs) as span:
            yield span


def new_trace(prefix: str = "turn") -> TurnTrace:
    return TurnTrace(trace_id=f"{prefix}_{uuid.uuid4().hex[:8]}")
