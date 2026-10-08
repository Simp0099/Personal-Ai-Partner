"""Phase 3 tests: webcam perception, the shared perception layer, and quiet behaviour.

The properties that matter, and that the architecture exists to guarantee:

* the camera is opt-in and never opened on its own,
* unchanged frames cost nothing — no model call, no observation,
* a changed frame produces labelled observations, never a spoken reply,
* the camera is released on shutdown, repeatably,
* nothing perceived becomes memory without being asked for.

Runs offline: synthetic frames, the existing mock providers, and a temporary
SQLite file. No camera is opened and no network call is made.
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis import behavior, memory, vision, vision_change, webcam  # noqa: E402
from jarvis.brain import JarvisBrain, BrainError, _build_context_block  # noqa: E402
from jarvis.classify import classify  # noqa: E402
from jarvis.providers.base import image_part  # noqa: E402
from tests.mock_providers import build_layer, reply  # noqa: E402

import base64

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """Keep every test off the real database, the shared context, and the console."""
    import tempfile

    import jarvis.brain as brain_module
    from jarvis import speech

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        spare = Path(tmp.name)
    conn = memory.init_memory_db(spare)
    monkeypatch.setattr(brain_module, "get_memory_conn", lambda: conn)
    for name in ("thinking", "tool_call", "tool_result", "memory", "info"):
        monkeypatch.setattr(
            brain_module.StatusIndicator, name, staticmethod(lambda *a, **k: None)
        )
    monkeypatch.setattr(brain_module.StatusIndicator, "shutdown", staticmethod(lambda: None))
    monkeypatch.setattr(speech, "speak", lambda *a, **k: pytest.fail("perception spoke"))
    vision.reset_visual_context()
    yield conn
    vision.reset_visual_context()
    conn.close()
    spare.unlink(missing_ok=True)


@pytest.fixture
def db():
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        path = Path(tmp.name)
    conn = memory.init_memory_db(path)
    yield conn
    conn.close()
    path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Synthetic frames and cameras
# ---------------------------------------------------------------------------

def _frame(value: int, size=(64, 48)) -> np.ndarray:
    """A flat frame of uniform brightness."""
    return np.full((size[1], size[0], 3), value, dtype=np.uint8)


def _frame_with_block(size=(64, 48)) -> np.ndarray:
    """A dark frame with a bright rectangle in it — a person appearing."""
    frame = _frame(10, size)
    frame[10:38, 15:45] = 220
    return frame


# A desk scene, so the change-detection tests have a realistic frame to work
# with rather than flat colour. Proportions match a 640x480 webcam frame.
_SCENE_SIZE = (640, 480)


def _scene():
    from PIL import Image, ImageDraw

    width, height = _SCENE_SIZE
    img = Image.new("RGB", (width, height), (238, 236, 232))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 330, width, height], fill=(120, 92, 60))       # desk plane
    d.rectangle([180, 110, 460, 300], fill=(40, 42, 48))          # monitor
    d.rectangle([190, 120, 450, 290], fill=(200, 200, 200))        # screen
    d.rectangle([280, 300, 360, 340], fill=(30, 30, 30))          # stand
    return np.asarray(img, dtype=np.uint8)


def _person_at_desk(glasses=False, offset=0):
    """The desk with someone sitting at it. `offset` shifts them, for drift."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(_scene())
    d = ImageDraw.Draw(img)
    d.ellipse([240 + offset, 150, 400 + offset, 260], fill=(214, 176, 150))
    d.polygon([(215 + offset, 400), (320 + offset, 250), (425 + offset, 400)],
              fill=(70, 90, 160))
    d.rectangle([430, 250, 530, 320], fill=(30, 30, 30))          # laptop
    if glasses:
        d.ellipse([252, 190, 288, 212], outline=(20, 20, 20), width=3)
        d.ellipse([352, 190, 388, 212], outline=(20, 20, 20), width=3)
        d.line([(288, 200), (352, 200)], fill=(20, 20, 20), width=3)
    return np.asarray(img, dtype=np.uint8)


def _desk():
    return _scene()


class FakeCamera:
    """A camera that hands out a scripted sequence of frames."""

    def __init__(self, frames=None, *, fail_after=None, open_error=None):
        self.frames = list(frames or [_frame(10)])
        self.fail_after = fail_after
        self.open_error = open_error
        self.opened = 0
        self.released = 0
        self.reads = 0
        self.is_open = False

    def open(self):
        if self.open_error:
            raise self.open_error
        self.opened += 1
        self.is_open = True

    def read(self):
        if self.fail_after is not None and self.reads >= self.fail_after:
            return None
        self.reads += 1
        if not self.frames:
            return None
        return self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]

    def release(self):
        self.released += 1
        self.is_open = False


class Recorder:
    """Stands in for the vision model. Records prompts, returns canned text."""

    def __init__(self, text="observed: The user is seated at the desk."):
        self.text = text
        self.calls = []

    def __call__(self, prompt, attachment):
        self.calls.append((prompt, attachment))
        if isinstance(self.text, Exception):
            raise self.text
        return self.text

    @property
    def count(self):
        return len(self.calls)


def _perception(analyze, **kwargs):
    """A perception loop with no real camera and no waiting."""
    kwargs.setdefault("interval", 3600)
    kwargs.setdefault("cooldown", 0)
    kwargs.setdefault("camera", FakeCamera([_person_at_desk()]))
    return webcam.WebcamPerception(analyze=analyze, **kwargs)


def _vision_layer(text="ok"):
    return build_layer(
        models=[{"key": "m", "provider": "p", "model": "mm",
                 "capabilities": {"reasoning": True, "tool_calling": True, "vision": True}}],
        behaviours={"mm": reply(text)},
    )


# ============================================================================
# Opt-in
# ============================================================================

class TestWebcamIsOptIn:
    def test_webcam_is_disabled_by_default(self):
        from jarvis.config import WEBCAM_ENABLED

        assert WEBCAM_ENABLED is False, "the camera must be off until asked for"

    def test_config_declares_the_opt_in(self):
        import yaml

        config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
        assert config["vision"]["webcam"]["enabled"] is False
        assert "enabled" in config["vision"], "the vision block must be togglable"

    def test_start_does_nothing_when_disabled(self, monkeypatch):
        monkeypatch.setattr(webcam, "VISION_ENABLED", True)
        monkeypatch.setattr(webcam, "WEBCAM_ENABLED", False)
        brain = JarvisBrain(model_layer=_vision_layer())
        assert webcam.start_webcam_perception(brain) is None
        assert webcam.get_webcam_perception() is None

    def test_importing_the_module_opens_no_camera(self):
        """Import side effects are how a camera gets activated invisibly."""
        import inspect

        # Constructed at import time and never assigned at module scope: the
        # camera class only builds a handle inside open().
        assert webcam._instance is None
        assert webcam.get_webcam_perception() is None


# ============================================================================
# Lifecycle
# ============================================================================

class TestWebcamLifecycle:
    def test_start_opens_the_camera_and_runs_a_worker(self):
        camera = FakeCamera()
        perception = _perception(Recorder(), camera=camera)
        assert perception.start() is True
        try:
            assert camera.opened == 1
            assert perception.is_running() is True
            assert perception.available is True
        finally:
            perception.stop()
        assert perception.is_running() is False

    def test_stop_releases_the_camera(self):
        camera = FakeCamera()
        perception = _perception(Recorder(), camera=camera)
        perception.start()
        perception.stop()
        assert camera.released == 1
        assert perception.available is False

    def test_stop_joins_the_worker_thread(self):
        camera = FakeCamera()
        perception = _perception(Recorder(), camera=camera, interval=0.01)
        perception.start()
        assert perception.stop(timeout=5.0) is None
        assert not [t for t in threading.enumerate() if t.name == "webcam-perception"]

    def test_stop_is_idempotent(self):
        perception = _perception(Recorder())
        perception.start()
        perception.stop()
        perception.stop()
        assert perception.is_running() is False

    def test_stop_without_start_is_safe(self):
        perception = _perception(Recorder())
        perception.stop()
        assert perception.is_running() is False

    def test_repeated_start_stop_cycles_do_not_leak(self):
        """The failure that matters: repeated cycles leaking threads or handles."""
        before = threading.active_count()
        for _ in range(5):
            camera = FakeCamera()
            perception = _perception(Recorder(), camera=camera, interval=0.01)
            perception.start()
            time.sleep(0.02)
            perception.stop()
            assert camera.released == 1
        assert threading.active_count() <= before + 1

    def test_loop_samples_on_its_interval(self):
        camera = FakeCamera()
        perception = _perception(Recorder(), camera=camera, interval=0.01, cooldown=0)
        perception.start()
        time.sleep(0.1)
        perception.stop()
        assert camera.reads > 1, "the worker did not keep sampling"

    def test_missing_opencv_is_reported_not_raised(self, monkeypatch):
        import sys as _sys

        monkeypatch.setitem(_sys.modules, "cv2", None)
        camera = webcam.OpenCVCamera()
        with pytest.raises(webcam.WebcamUnavailable):
            camera.open()
        assert camera.is_open is False

    def test_release_without_open_is_safe(self):
        webcam.OpenCVCamera().release()


# ============================================================================
# Camera failures do not break anything
# ============================================================================

class TestCameraFailures:
    def test_unavailable_camera_is_not_fatal(self):
        camera = FakeCamera(open_error=webcam.WebcamUnavailable("no camera"))
        perception = _perception(Recorder(), camera=camera)
        assert perception.start() is False
        assert perception.available is False
        assert "no camera" in perception.last_error
        assert perception.is_running() is False

    def test_unavailable_camera_lets_text_work(self):
        """A missing camera must not take the assistant down with it."""
        camera = FakeCamera(open_error=webcam.WebcamUnavailable("in use"))
        brain = JarvisBrain(model_layer=_vision_layer("text answer"))
        perception = webcam.WebcamPerception(analyze=lambda *a: "", camera=camera)
        assert perception.start() is False
        assert brain.ask("what time is it?") == "text answer"

    def test_failed_read_does_not_crash_the_loop(self):
        camera = FakeCamera(fail_after=0)
        perception = _perception(Recorder(), camera=camera)
        perception.start()
        try:
            perception.tick()
            assert perception.frames_sampled == 0
            assert "read failed" in perception.last_error
            # And the loop keeps going.
            time.sleep(0.05)
            assert perception.is_running()
        finally:
            perception.stop()

    def test_a_failing_frame_does_not_kill_the_worker(self):
        analyze = Recorder(RuntimeError("vision unavailable"))
        perception = _perception(analyze, camera=FakeCamera(), interval=0.01, cooldown=0)
        perception.start()
        time.sleep(0.08)
        assert perception.is_running(), "one bad frame stopped the worker"
        assert perception.analyses == 0
        perception.stop()

    def test_vision_failure_records_an_error_not_an_observation(self):
        perception = _perception(Recorder(RuntimeError("no vision model")))
        perception.start()
        try:
            with pytest.raises(RuntimeError):
                perception.tick()
            assert perception.context.observations() == [], "failure was disguised"
        finally:
            perception.stop()

    def test_start_reports_honestly_when_no_vision_model_exists(self):
        """Do not open a camera that could never use a frame."""
        blind = build_layer(
            models=[{"key": "m", "provider": "p", "model": "mm",
                     "capabilities": {"reasoning": True, "vision": False}}],
        )
        brain = JarvisBrain(model_layer=blind)
        with pytest.raises(BrainError):
            brain.ask("describe this", images=[image_part(PNG_1PX)], ephemeral=True)


# ============================================================================
# Sampling, change detection, cost control
# ============================================================================

class TestChangeDetection:
    def test_identical_frames_score_zero(self):
        detector = vision_change.ChangeDetector()
        frame = _frame_with_block()
        detector.score(frame)
        assert detector.score(frame) == 0.0

    def test_different_frames_score_highly(self):
        detector = vision_change.ChangeDetector()
        detector.score(_scene())
        assert detector.score(_person_at_desk()) > 0.02
        detector.reset()
        detector.score(_scene())
        assert detector.score(np.full_like(_scene(), 255)) > 0.9

    def test_a_person_arriving_is_detected_at_the_default_threshold(self):
        """Regressed once: the default threshold sat just above this."""
        detector = vision_change.ChangeDetector()
        detector.score(_desk())
        assert detector.score(_person_at_desk()) >= detector.threshold

    def test_lighting_drift_is_not_a_scene_change(self):
        """Regressed once, and worse than the false negative it replaced.

        Averaging brightness change cannot tell these apart: a window shade
        opening moves the frame mean by 0.0235 and a person sitting down by
        0.0240. As a fraction of materially-changed cells they are 0.00 and
        0.054.
        """
        detector = vision_change.ChangeDetector()
        base = _desk()
        dimmer = np.clip(base.astype(np.int16) + 24, 0, 255).astype(np.uint8)
        detector.score(base)
        assert detector.score(dimmer) < detector.threshold

    def test_sensor_noise_is_not_a_scene_change(self):
        rng = np.random.default_rng(7)
        base = _desk()
        detector = vision_change.ChangeDetector()
        detector.score(base)
        noisy = np.clip(base.astype(np.int16) + rng.normal(0, 8, base.shape), 0, 255)
        assert detector.score(noisy.astype(np.uint8)) < detector.threshold

    def test_fine_changes_are_below_the_default_threshold(self):
        """The documented ceiling: glasses are ~1% of the frame.

        Asserted so the ceiling stays a decision rather than a surprise.
        """
        detector = vision_change.ChangeDetector()
        detector.score(_person_at_desk())
        assert detector.score(_person_at_desk(glasses=True)) < detector.threshold

    def test_a_lower_threshold_catches_fine_changes(self):
        """The documented upgrade path actually works."""
        detector = vision_change.ChangeDetector(threshold=0.005)
        detector.score(_person_at_desk())
        assert detector.score(_person_at_desk(glasses=True)) >= detector.threshold

    def test_first_frame_is_always_a_change(self):
        detector = vision_change.ChangeDetector()
        assert detector.score(_frame(128)) == 1.0

    def test_change_requires_crossing_the_threshold(self):
        detector = vision_change.ChangeDetector(threshold=0.5)
        assert detector.changed(_frame(10)) is True   # no baseline yet
        assert detector.changed(_frame(11)) is False, "sensor noise crossed the threshold"
        detector.reset()
        detector.score(_scene())
        assert detector.score(_person_at_desk()) < 0.5, "a real change was not detected"
        assert detector.score(np.full_like(_scene(), 255)) > 0.5

    def test_brightness_alone_is_not_a_change(self):
        """Mean-centring: illumination is cancelled, structure is not."""
        detector = vision_change.ChangeDetector()
        base = _scene()
        brighter = np.clip(base.astype(np.int16) + 30, 0, 255).astype(np.uint8)
        detector.score(base)
        assert detector.changed(brighter) is False
        assert detector.changed(_person_at_desk()) is True

    def test_baseline_tracks_the_current_scene(self):
        """Slow drift must accumulate instead of being pinned to a stale frame."""
        tracking = vision_change.ChangeDetector(threshold=0.005)
        original = tracking.signature(_person_at_desk())
        tracking._previous = original
        steps = [tracking.score(_person_at_desk(offset=n)) for n in (2, 6, 14)]

        pinned = vision_change.ChangeDetector(threshold=0.005)
        pinned._previous = original  # never advanced past the first frame
        total = pinned.score(_person_at_desk(offset=14))

        assert sum(steps) >= total, "advancing the baseline lost drift"
        assert steps[-1] < total, "the newest change reads smaller than the whole drift"

    def test_reset_forgets_the_baseline(self):
        detector = vision_change.ChangeDetector()
        detector.score(_frame(0))
        detector.reset()
        assert detector.score(_frame(0)) == 1.0

    def test_detector_uses_no_model_and_no_network(self):
        """The whole point: change detection cannot cost a model call."""
        import inspect

        source = inspect.getsource(vision_change)
        for banned in ("requests", "urllib", "ask(", "openai", "gemini"):
            assert banned not in source, f"change detector reached for {banned}"


class TestSamplingAndCostControl:
    def test_unchanged_frames_are_discarded_without_a_vision_call(self):
        analyze = Recorder()
        camera = FakeCamera([_frame_with_block()])
        perception = _perception(analyze, camera=camera)

        for _ in range(20):
            perception.tick()

        # Only the first frame, which has no baseline to compare against.
        assert analyze.count == 1, "an unchanged scene was analysed anyway"
        assert perception.frames_sampled == 20
        assert perception.frames_discarded == 19, "the first frame has no baseline"

    def test_meaningful_change_triggers_vision(self):
        analyze = Recorder()
        camera = FakeCamera([_frame_with_block()])
        perception = _perception(analyze, camera=camera)

        result = perception.tick()          # baseline scene: no prior to compare
        assert result is not None

        camera.frames = [_desk()]           # the user has left the frame
        result = perception.tick()

        assert analyze.count == 2
        assert perception.analyses == 2
        assert result.text == "The user is seated at the desk."

    def test_analysis_is_capped_by_the_cooldown(self):
        analyze = Recorder()
        camera = FakeCamera()
        now = {"t": 1000.0}
        perception = _perception(
            analyze, camera=camera, cooldown=120.0, clock=lambda: now["t"]
        )

        perception.tick()
        assert analyze.count == 1

        camera.frames = [_desk()]
        perception.tick()
        assert analyze.count == 1, "a second call inside the cooldown"

        now["t"] += 121.0
        camera.frames = [_person_at_desk()]
        perception.tick()
        assert analyze.count == 2, "cooldown never released"

    def test_discard_rate_is_reported(self):
        perception = _perception(Recorder(), camera=FakeCamera([_person_at_desk()]))
        perception.tick()
        perception.tick()
        perception.tick()
        status = perception.status()
        assert status["frames_sampled"] == 3
        assert status["frames_discarded"] == 2
        assert status["vision_analyses"] == 1

    def test_identical_observation_only_refreshes_the_context(self):
        analyze = Recorder("observed: The user is seated at the desk.")
        camera = FakeCamera()
        perception = _perception(analyze, camera=camera)

        perception.tick()
        camera.frames = [_desk()]
        perception.tick()

        context = vision.get_visual_context()
        assert len(context.observations()) == 1, "duplicate observation recorded"


# ============================================================================
# Scene understanding: observation vs inference
# ============================================================================

class TestPerceptionGrounding:
    @pytest.mark.parametrize("line, kind", [
        ("observed: A laptop is open on the desk.", vision.OBSERVATION),
        ("observed: The user is visible in the frame.", vision.OBSERVATION),
        ("inferred: The user appears to be working.", vision.INFERENCE),
        ("unclear: Text on the second monitor is unreadable.", vision.UNCERTAINTY),
    ])
    def test_model_labels_are_honoured(self, line, kind):
        parsed = vision.parse_observations(line)
        assert parsed[0].kind == kind
        assert not parsed[0].text.lower().startswith(("observed", "inferred", "unclear"))

    def test_unlabelled_hedged_claim_is_an_inference(self):
        parsed = vision.parse_observations("The user seems to be coding.")
        assert parsed[0].kind == vision.INFERENCE

    def test_unlabelled_plain_claim_is_an_observation(self):
        parsed = vision.parse_observations("Glasses are visible.")
        assert parsed[0].kind == vision.OBSERVATION

    def test_bullets_and_blank_lines_are_handled(self):
        parsed = vision.parse_observations("\n- observed: A cup is on the desk.\n\n- **inferred:** Coffee.\n")
        assert [o.text for o in parsed] == ["A cup is on the desk.", "Coffee."]
        assert [o.kind for o in parsed] == [vision.OBSERVATION, vision.INFERENCE]

    def test_empty_description_yields_nothing(self):
        assert vision.parse_observations("") == []
        assert vision.parse_observations("   \n  ") == []

    def test_speculation_is_prohibited_by_directive(self):
        directive = behavior.perception_directive()
        for banned in ("identity", "emotion", "health", "intention", "age, gender"):
            assert banned in directive

    def test_directive_forbids_catalogue_spam(self):
        assert "walls" in behavior.perception_directive().lower()

    def test_scene_prompt_carries_the_previous_frame(self):
        prompt = vision.scene_instruction([vision.Observation("A laptop is open")])
        assert "A laptop is open" in prompt
        assert "report only what is different" in prompt.lower()

    def test_scene_prompt_without_history_is_the_plain_one(self):
        assert vision.scene_instruction() == vision.SCENE_INSTRUCTION

    @pytest.mark.parametrize("before, after, expect", [
        ("observed: The frame is empty.", "observed: The user is seated at a desk.",
         "The user is seated at a desk."),
        ("observed: The user is seated at a desk.", "observed: The frame is empty.",
         "The frame is empty."),
        ("observed: Glasses are visible.", "inferred: The desk looks like a dev setup.",
         "The desk looks like a dev setup."),
    ])
    def test_new_observations_replace_the_old_scene(self, before, after, expect):
        context = vision.VisualContext()
        context.update(vision.parse_observations(before))
        context.update(vision.parse_observations(after))
        assert [o.text for o in context.observations()] == [expect]

    def test_context_caps_a_chatty_analysis(self):
        context = vision.VisualContext(max_observations=2)
        context.update(vision.parse_observations(
            "observed: one\nobserved: two\nobserved: three\nobserved: four"
        ))
        assert [o.text for o in context.observations()] == ["three", "four"]


# ============================================================================
# Routing: camera frames reach a vision-capable model
# ============================================================================

class TestPerceptionRouting:
    def test_webcam_analysis_routes_to_a_vision_model(self):
        layer = build_layer(
            models=[
                {"key": "blind", "provider": "p1", "model": "m1", "priority": 100,
                 "free": True, "capabilities": {"reasoning": True, "vision": False}},
                {"key": "sees", "provider": "p2", "model": "m2", "priority": 10,
                 "capabilities": {"reasoning": True, "vision": True}},
            ],
            behaviours={"m1": reply("blind"), "m2": reply("observed: a person at a desk")},
        )
        brain = JarvisBrain(model_layer=layer)
        perception = webcam.WebcamPerception(
            analyze=lambda prompt, image: brain.ask(prompt, images=[image], ephemeral=True),
            camera=FakeCamera(), interval=3600, cooldown=0,
        )
        perception.start()
        try:
            perception.tick()
        finally:
            perception.stop()
        assert brain.last_model_key == "sees"
        assert perception.analyses == 1

    def test_frame_travels_as_the_existing_neutral_attachment(self):
        layer = _vision_layer("observed: a person at a desk")
        brain = JarvisBrain(model_layer=layer)
        brain.ask("describe", images=[image_part(PNG_1PX)], ephemeral=True)
        sent = [p for p in layer.providers["p"].sent if p.get("kind") == "user"][-1]
        assert sent["images"][0]["mime"] == "image/png"

    def test_a_frame_is_downscaled_before_it_is_sent(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        part = vision.frame_to_image_part(frame)
        assert part["kind"] == "image"
        assert part["mime"] == "image/jpeg"
        assert len(part["data"]) < 200_000, "a full-resolution frame was sent as-is"


# ============================================================================
# Quiet perception
# ============================================================================

class TestQuietPerception:
    def test_a_perception_turn_is_not_recorded_in_history(self):
        brain = JarvisBrain(model_layer=_vision_layer("observed: a person at a desk"))
        brain.ask("what changed?", images=[image_part(PNG_1PX)], ephemeral=True)
        assert brain._history == [], "camera perception entered the conversation"

    def test_a_perception_turn_does_not_read_the_user_conversation(self):
        layer = _vision_layer("observed: a person at a desk")
        brain = JarvisBrain(model_layer=layer)
        brain.ask("my dentist is called Dr Rao")
        brain.ask("describe", images=[image_part(PNG_1PX)], ephemeral=True)
        opened = layer.providers["p"].sessions[-1]
        assert opened.history == [], "the camera analysis was given the user's history"

    def test_a_perception_turn_gets_no_tools(self):
        """Otherwise the camera can call save_memory, which is the one thing
        webcam perception must never be able to do."""
        layer = _vision_layer("observed: a person at a desk")
        JarvisBrain(model_layer=layer).ask(
            "describe", images=[image_part(PNG_1PX)], ephemeral=True
        )
        assert layer.providers["p"].sessions[-1].tools == []

    def test_a_normal_turn_still_has_its_tools(self):
        layer = _vision_layer("ok")
        JarvisBrain(model_layer=layer).ask("hello")
        assert layer.providers["p"].sessions[-1].tools

    def test_perception_does_not_speak(self, monkeypatch):
        import jarvis.speech as speech

        spoken = []
        monkeypatch.setattr(speech, "speak", lambda text: spoken.append(text))
        monkeypatch.setattr("jarvis.webcam.speak", speech.speak, raising=False)

        perception = _perception(Recorder(), camera=FakeCamera(), interval=0.01)
        perception.start()
        time.sleep(0.06)
        perception.stop()
        assert spoken == []

    def test_a_full_cycle_updates_context_without_any_reply(self):
        perception = _perception(
            Recorder("observed: The user has entered the frame and is at the desk."),
            camera=FakeCamera(),
        )
        perception.start()
        try:
            perception.tick()
        finally:
            perception.stop()
        context = vision.get_visual_context()
        assert [o.text for o in context.observations()] == [
            "The user has entered the frame and is at the desk."
        ]
        assert context.is_fresh()

    def test_context_survives_into_the_next_conversation(self):
        """A new Brain still sees the camera's current view."""
        vision.get_visual_context().update(
            vision.parse_observations("observed: The user is seated at the desk.")
        )
        layer = _vision_layer("ok")
        JarvisBrain(model_layer=layer).ask("what can you see?")
        assert "seated at the desk" in layer.providers["p"].sessions[-1].system_prompt


# ============================================================================
# Current visual context
# ============================================================================

class TestVisualContext:
    def test_context_reaches_the_prompt(self):
        vision.get_visual_context().update(
            vision.parse_observations("observed: A laptop is open on the desk.")
        )
        block = _build_context_block("what can you see?", classify("what can you see?"), [])
        assert "## Current Visual Context" in block
        assert "A laptop is open" in block

    def test_context_carries_the_kinds(self):
        vision.get_visual_context().update(
            vision.parse_observations("observed: Glasses are visible.")
        )
        vision.get_visual_context().update(
            vision.parse_observations("inferred: The user appears to be working.")
        )
        block = vision.format_context_block()
        assert "[inference]" in block

    def test_empty_context_costs_nothing(self):
        assert vision.format_context_block() == ""
        block = _build_context_block("hello", classify("hello"), [])
        assert "## Current Visual Context" not in block

    def test_context_expires(self):
        context = vision.VisualContext(ttl=60.0)
        context.update(vision.parse_observations("observed: The user is present."))
        assert context.is_fresh()
        later = time.time() + 61
        assert context.observations(now=later) == []
        assert context.is_fresh(now=later) is False

    def test_stale_context_is_not_offered_to_the_model(self):
        context = vision.VisualContext(ttl=1.0)
        context.update(vision.parse_observations("observed: The user is present."))
        vision.get_visual_context()._observations = context._observations
        vision.get_visual_context()._changed_at = time.time() - 3600
        assert vision.format_context_block() == ""

    def test_context_is_not_long_term_memory(self):
        """No observation history is ever dumped into the prompt wholesale."""
        context = vision.VisualContext(max_observations=2)
        for n in range(20):
            context.update(vision.parse_observations(f"observed: scene {n}"))
        block = vision.format_context_block(context)
        assert "scene 19" in block
        assert "scene 0" not in block

    def test_context_block_is_labelled_as_live_not_remembered(self):
        vision.get_visual_context().update(
            vision.parse_observations("observed: The user is present.")
        )
        block = vision.format_context_block()
        assert "not a stored memory" in block
        assert "## Relevant Long-Term Memory" not in block

    def test_user_can_query_the_camera(self):
        vision.get_visual_context().update(
            vision.parse_observations("observed: The user is seated at the desk.")
        )
        layer = _vision_layer("You are at your desk.")
        brain = JarvisBrain(model_layer=layer)
        brain.ask("what am I doing right now?")
        prompt = layer.providers["p"].sessions[-1].system_prompt
        assert "## Current Visual Context" in prompt
        assert "seated at the desk" in prompt

    def test_memory_still_reaches_a_turn_that_has_visual_context(self):
        vision.get_visual_context().update(
            vision.parse_observations("observed: The user is seated at the desk.")
        )
        layer = _vision_layer("ok")
        JarvisBrain(model_layer=layer).ask("what database does atlas use")
        prompt = layer.providers["p"].sessions[-1].system_prompt
        assert "## This turn" in prompt


# ============================================================================
# Perception never becomes memory
# ============================================================================

class TestPerceptionIsNotMemory:
    def test_webcam_observations_never_reach_memory(self, _isolated):
        perception = _perception(
            Recorder("observed: The user was wearing a black shirt at the desk."),
            camera=FakeCamera(),
        )
        perception.start()
        try:
            perception.tick()
        finally:
            perception.stop()
        assert memory.get_memory_count(_isolated) == 0
        assert memory.recall_all(_isolated) == []

    def test_repeated_perception_still_writes_nothing(self, _isolated):
        camera = FakeCamera()
        perception = _perception(Recorder("observed: The user is at the desk."), camera=camera)
        perception.start()
        try:
            for scene in (_desk(), _person_at_desk(), _desk(), _person_at_desk(offset=9)):
                camera.frames = [scene]
                perception.tick()
        finally:
            perception.stop()
        assert memory.get_memory_count(_isolated) == 0

    def test_perception_never_calls_a_memory_tool(self):
        """Proven structurally: the perception session carries no tools."""
        layer = _vision_layer("observed: a person at a desk")
        JarvisBrain(model_layer=layer).ask(
            "describe", images=[image_part(PNG_1PX)], ephemeral=True
        )
        assert not layer.providers["p"].sessions[-1].tools


# ============================================================================
# Privacy
# ============================================================================

class TestPerceptionPrivacy:
    def test_frames_are_never_persisted(self, tmp_path):
        """The loop must not write anything to disk."""
        import inspect

        source = inspect.getsource(webcam)
        for banned in ("write_bytes", "write_text", "savefig", "imwrite",
                       "mkdir", "tempfile", "Path("):
            assert banned not in source, f"webcam loop wrote to disk via {banned}"

    def test_status_reports_no_persistence(self):
        perception = _perception(Recorder())
        assert perception.status()["frames_persisted"] == 0

    def test_status_contains_no_image_data(self):
        perception = _perception(Recorder(), camera=FakeCamera())
        perception.start()
        try:
            perception.tick()
        finally:
            perception.stop()
        import json

        blob = json.dumps(perception.status())
        assert "iVBORw0KGgo" not in blob
        assert "data" not in perception.status()["context"]["observations"][0]

    def test_credentials_visible_in_a_frame_are_never_stored(self, db):
        assert memory.remember(
            db, "the screenshot shows AWS_SECRET_ACCESS_KEY=wJalr", source="visual"
        ) is False
        assert memory.recall_all(db) == []

    def test_secret_refusal_covers_webcam_sourced_facts(self, db):
        """Same guard for a fact that came from a camera observation."""
        assert memory.remember(
            db, "the api_key visible on the second monitor is sk-abc123456", source="visual"
        ) is False


# ============================================================================
# Visual memory still uses the one existing system
# ============================================================================

class TestVisualMemoryUnchanged:
    def test_explicit_visual_memory_still_works(self, db):
        assert memory.remember(
            db, "Atlas runs MySQL 8.0", "project", source="visual"
        ) is True
        assert [f for f, _ in memory.recall_relevant(db, "what database does atlas use")] == [
            "Atlas runs MySQL 8.0"
        ]

    def test_supersession_still_works(self, db):
        memory.remember(db, "Atlas uses Postgres", "project", subject="project:atlas")
        memory.remember(db, "Atlas uses MySQL 8.0", "project", subject="project:atlas")
        assert memory.recall_all(db) == ["Atlas uses MySQL 8.0"]

    def test_forgetting_still_works(self, db):
        memory.remember(db, "Atlas uses MySQL", "project")
        assert memory.forget_matching(db, "MySQL")
        assert memory.recall_relevant(db, "what database does atlas use") == []

    def test_there_is_still_exactly_one_memory_store(self):
        import inspect

        for module in (vision, vision_change, webcam):
            assert inspect.getsource(module).count("CREATE TABLE") == 0
        assert inspect.getsource(memory).count("CREATE TABLE") == 1

    def test_current_evidence_must_beat_stale_memory(self):
        """A screenshot contradicting a stored fact says so and asks."""
        directive = behavior.vision_directive(1)
        assert "contradicts a stored memory" in directive
        assert "do not silently overwrite" in directive.lower()


# ============================================================================
# Model switching
# ============================================================================

class TestModelSwitching:
    def test_visual_context_is_intact_after_a_switch(self):
        """A mid-conversation switch to another provider must keep it."""
        from jarvis.providers.base import ErrorKind
        from tests.mock_providers import error

        vision.get_visual_context().update(
            vision.parse_observations("observed: A laptop is open on the desk.")
        )
        layer = build_layer(
            models=[
                {"key": "a", "provider": "p1", "model": "m1", "priority": 100,
                 "capabilities": {"reasoning": True, "vision": True}},
                {"key": "b", "provider": "p2", "model": "m2", "priority": 10,
                 "capabilities": {"reasoning": True, "vision": True}},
            ],
            behaviours={"m1": reply("A"), "m2": reply("B")},
        )
        brain = JarvisBrain(model_layer=layer)
        brain.ask("what is happening here?", images=[image_part(PNG_1PX)])
        assert brain.last_model_key == "a"

        # Second turn is served by the other provider, from the replayed history.
        layer.providers["p1"]._behaviours["m1"] = error(ErrorKind.SERVER, "503")
        brain.ask("and what does the error say?")
        assert brain.last_model_key == "b"
        assert "A laptop is open" in layer.providers["p2"].sessions[-1].system_prompt

    def test_a_perception_turn_does_not_leak_its_directive_into_the_next_turn(self):
        """The ephemeral turn borrows the conversation's turn context, not the reverse."""
        layer = _vision_layer("observed: a person at a desk")
        brain = JarvisBrain(model_layer=layer)
        brain.ask("describe", images=[image_part(PNG_1PX)], ephemeral=True)
        brain.ask("hello")
        prompt = layer.providers["p"].sessions[-1].system_prompt
        assert "## Webcam Perception" not in prompt
        assert "## This turn" in prompt

    def test_perception_keeps_routing_to_vision_after_a_switch(self):
        layer = build_layer(
            models=[
                {"key": "blind", "provider": "p1", "model": "m1", "priority": 100,
                 "free": True, "capabilities": {"reasoning": True, "vision": False}},
                {"key": "sees", "provider": "p2", "model": "m2", "priority": 10,
                 "capabilities": {"reasoning": True, "vision": True}},
            ],
            behaviours={"m1": reply("blind"), "m2": reply("observed: a person")},
        )
        brain = JarvisBrain(model_layer=layer)
        brain.ask("what is happening here?", images=[image_part(PNG_1PX)], ephemeral=True)
        brain.ask("what is happening here?", images=[image_part(PNG_1PX)], ephemeral=True)
        assert brain.last_model_key == "sees"

    def test_a_text_turn_after_an_image_turn_still_routes_to_a_vision_model(self):
        """Regressed once, and it broke every conversation that contained an image.

        History is replayed to whichever model serves the next turn, so a text
        turn following an image turn still *sends* that image. Routing only looked
        at the current turn's attachments, picked a text-only model, and the
        provider rejected the replayed image outright:
        `404 No endpoints found that support image input`.
        """
        layer = build_layer(
            models=[
                {"key": "blind", "provider": "p1", "model": "m1", "priority": 100,
                 "free": True, "capabilities": {"reasoning": True, "vision": False}},
                {"key": "sees", "provider": "p2", "model": "m2", "priority": 10,
                 "capabilities": {"reasoning": True, "vision": True}},
            ],
            behaviours={"m1": reply("blind"), "m2": reply("saw it")},
        )
        brain = JarvisBrain(model_layer=layer)
        brain.ask("what is this?", images=[image_part(PNG_1PX)])
        assert brain.ask("and what does the error say?") == "saw it"
        assert brain.last_model_key == "sees", "a replayed image was sent to a blind model"

    def test_the_replayed_image_really_is_resent(self):
        """Proves the routing fix is needed, rather than defensive."""
        layer = _vision_layer("ok")
        brain = JarvisBrain(model_layer=layer)
        brain.ask("what is this?", images=[image_part(PNG_1PX)])
        brain.ask("and now?")
        assert layer.providers["p"].sessions[-1].history[0].get("images")

    def test_a_text_only_conversation_still_prefers_the_text_model(self):
        """The fix must not send every conversation to the paid vision model."""
        layer = build_layer(
            models=[
                {"key": "blind", "provider": "p1", "model": "m1", "priority": 100,
                 "free": True, "capabilities": {"reasoning": True, "vision": False}},
                {"key": "sees", "provider": "p2", "model": "m2", "priority": 10,
                 "capabilities": {"reasoning": True, "vision": True}},
            ],
            behaviours={"m1": reply("free"), "m2": reply("paid")},
        )
        brain = JarvisBrain(model_layer=layer)
        brain.ask("hello")
        brain.ask("tell me a joke")
        assert brain.last_model_key == "blind"


# ============================================================================
# Shutdown
# ============================================================================

class TestShutdown:
    def test_start_then_shutdown_releases_everything(self):
        camera = FakeCamera()
        perception = _perception(Recorder(), camera=camera, interval=0.01)
        perception.start()
        time.sleep(0.05)
        perception.stop()
        assert camera.released == 1
        assert camera.is_open is False
        assert perception.is_running() is False

    def test_shutdown_during_analysis_is_clean(self):
        """Stopping while the loop is mid-cycle must not raise."""
        slow = Recorder()
        original = slow.__call__

        def lagging(prompt, attachment):
            time.sleep(0.05)
            return original(prompt, attachment)

        perception = _perception(lagging, camera=FakeCamera(), interval=0.01, cooldown=0)
        perception.start()
        time.sleep(0.03)
        perception.stop(timeout=5.0)
        assert perception.is_running() is False

    def test_no_worker_thread_survives(self):
        before = {t.name for t in threading.enumerate()}
        for _ in range(3):
            perception = _perception(Recorder(), camera=FakeCamera(), interval=0.01)
            perception.start()
            time.sleep(0.02)
            perception.stop()
        assert "webcam-perception" not in {t.name for t in threading.enumerate()} - before

    def test_stop_webcam_perception_is_safe_when_nothing_started(self):
        webcam.stop_webcam_perception()
        assert webcam.get_webcam_perception() is None

    def test_disabled_shutdown_path_does_nothing(self):
        webcam.stop_webcam_perception()
        assert webcam.get_webcam_perception() is None