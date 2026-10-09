"""Regression tests for the wake-listener lifecycle guard.

`start()` must never spawn a duplicate worker -- not while listening, and
not while a previous worker is still releasing audio in its callback or
cleanup. Mocked audio throughout; no microphone hardware.
"""

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.wake_word import WakeWordListener  # noqa: E402


class FakeStream:
    def __init__(self):
        self.stopped = False
        self.closed = False
        self.terminated = False

    def read(self, n, exception_on_overflow=False):
        return np.zeros(n, dtype=np.int16).tobytes()

    def stop_stream(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakeAudio:
    def __init__(self):
        self.terminated = False
        self.streams = 0

    def open(self, **kwargs):
        self.streams += 1
        self.stream = FakeStream()
        return self.stream

    def terminate(self):
        self.terminated = True


@pytest.fixture()
def _fake_audio(monkeypatch):
    mod = MagicMock()
    mod.PyAudio.side_effect = lambda: FakeAudio()
    mod.paInt16 = 8
    monkeypatch.setitem(sys.modules, "pyaudio", mod)
    return mod


class FakeEngine:
    def __init__(self, fire_at=0):
        self.n = 0
        self.fire_at = fire_at
        self.threshold = 0.5
        self.model = "test"
        self.closed = 0

    def load(self):
        return True

    def score(self, pcm):
        self.n += 1
        return 0.9 if self.n > self.fire_at else 0.0

    def close(self):
        self.closed += 1


def _listener(engine, on_wake=None):
    listener = WakeWordListener(on_wake=on_wake or (lambda: None))
    listener.engine = engine
    return listener


class TestStartGuard:
    def test_restart_after_join_releases_then_rearms(self, _fake_audio):
        wakes = []
        listener = _listener(FakeEngine(fire_at=1), on_wake=lambda: wakes.append(1))
        listener.start()
        listener.join()
        first = listener._thread
        assert first is not None and not first.is_alive()
        listener.start()
        listener.join()
        assert listener._thread is not first
        assert len(wakes) == 2

    def test_start_during_callback_spawns_no_second_worker(self, _fake_audio):
        entered = threading.Event()
        release = threading.Event()

        def _slow_wake():
            entered.set()
            assert release.wait(5)

        listener = _listener(FakeEngine(fire_at=0), on_wake=_slow_wake)
        listener.start()
        assert entered.wait(5)
        first = listener._thread
        assert first.is_alive()
        listener.start()  # re-arm attempt while worker still in callback
        time.sleep(0.1)
        assert listener._thread is first
        release.set()
        listener.join()
        listener.stop()

    def test_double_start_while_listening(self, _fake_audio):
        listener = _listener(FakeEngine(fire_at=10 ** 9))
        listener.start()
        first = listener._thread
        listener.start()
        assert listener._thread is first
        listener.stop()

    def test_concurrent_starts_spawn_one_worker(self, _fake_audio):
        listener = _listener(FakeEngine(fire_at=10 ** 9))
        barrier = threading.Barrier(8)
        seen = []

        def _go():
            barrier.wait()
            listener.start()
            seen.append(listener._thread)

        threads = [threading.Thread(target=_go) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(set(map(id, seen))) == 1
        listener.stop()
