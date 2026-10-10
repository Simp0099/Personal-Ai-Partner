"""Keep the real-microphone check on the shared, single-reader audio path."""

from scripts import phase4_live_check


def test_wake_live_check_scores_frames_from_microphone_sink(monkeypatch):
    class FakeEngine:
        model = "hey_jarvis"
        threshold = 0.5

        def __init__(self):
            self.calls = 0

        def load(self):
            return True

        def score(self, pcm):
            self.calls += 1
            assert pcm == b"frame"
            return 0.1

        def close(self):
            pass

    class FakeMicrophone:
        def __init__(self):
            self.sink = None

        def add_sink(self, sink):
            self.sink = sink

        def start(self):
            assert self.sink is not None
            for _ in range(30):
                self.sink(b"frame")
            return True

        def stop(self):
            pass

    engine = FakeEngine()
    microphone = FakeMicrophone()
    phase4_live_check.RESULTS.clear()
    monkeypatch.setattr(phase4_live_check, "WakeWordEngine", lambda: engine)

    phase4_live_check.check_wake_engine(True, microphone=microphone)

    assert engine.calls == 30
    assert phase4_live_check.RESULTS[0][1] is True
    assert phase4_live_check.RESULTS[1][1] is None
