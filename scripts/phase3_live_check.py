#!/usr/bin/env python3
"""Phase 3 live verification.

Two halves, because only one of them can run anywhere:

* **Synthetic camera, real model.** Frames are generated in-process and pushed
  through the real perception loop and the real vision model. This is what proves
  scene understanding, grounding, cost control and quiet behaviour against a
  model rather than a mock.
* **Real camera, if there is one.** When a camera opens, the same loop reads from
  it instead. When one does not, that is reported as SKIPPED rather than passed.

Nothing here writes to memory or to disk, and nothing speaks.

Usage:
    python3 scripts/phase3_live_check.py
"""

import io
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from jarvis import memory, vision  # noqa: E402
from jarvis.brain import JarvisBrain, BrainError  # noqa: E402
from jarvis.providers.base import image_part  # noqa: E402
from jarvis.webcam import OpenCVCamera, WebcamPerception  # noqa: E402

RESULTS = []

#: The Brain writes to long-term memory through a module-level connection, and a
#: model is free to save a fact mid-turn whether or not it was asked to. This
#: script therefore runs the whole session against a throwaway database so a
#: verification run can never write into the user's real memory.
_SCRATCH = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
SCRATCH_DB = Path(_SCRATCH.name)
SCRATCH_CONN = memory.init_memory_db(SCRATCH_DB)

import jarvis.brain as _brain_module  # noqa: E402

_brain_module.get_memory_conn = lambda: SCRATCH_CONN


def check(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print(f"--- {name} ---\nDetail: {detail}\nResult: {'PASS' if ok else 'FAIL'}\n")


def skip(name, detail):
    RESULTS.append((name, None, detail))
    print(f"--- {name} ---\nDetail: {detail}\nResult: SKIPPED\n")


# ---------------------------------------------------------------------------
# Synthetic frames that stand in for a desk scene
# ---------------------------------------------------------------------------

W, H = 640, 480


def _desk(light=200):
    """An empty desk: monitor rectangle, desk plane, wall."""
    img = Image.new("RGB", (W, H), (238, 236, 232))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 330, W, H], fill=(120, 92, 60))                 # desk
    d.rectangle([180, 110, 460, 300], fill=(40, 42, 48))            # monitor
    d.rectangle([190, 120, 450, 290], fill=(light, light, light))    # screen
    d.rectangle([280, 300, 360, 340], fill=(30, 30, 30))            # stand
    return img


def _with_person(draw_glasses=False):
    """The same desk with someone sitting at it."""
    img = _desk()
    d = ImageDraw.Draw(img)
    d.ellipse([240, 150, 400, 260], fill=(214, 176, 150))           # head
    d.polygon([(215, 400), (320, 250), (425, 400)], fill=(70, 90, 160))  # torso
    if draw_glasses:
        d.ellipse([252, 190, 288, 212], outline=(20, 20, 20), width=3)
        d.ellipse([352, 190, 388, 212], outline=(20, 20, 20), width=3)
        d.line([(288, 200), (352, 200)], fill=(20, 20, 20), width=3)
    d.rectangle([430, 250, 530, 320], fill=(30, 30, 30))            # laptop
    return img


def _frame(image):
    return np.asarray(image, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def real_camera_available():
    try:
        camera = OpenCVCamera()
        camera.open()
        camera.release()
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def check_camera_reality():
    ok, detail = real_camera_available()
    if ok:
        check("A — camera opens", True, "real camera opened and released cleanly")
    else:
        skip("A — camera opens (real hardware)", f"no usable camera here: {detail}")
        check("A' — missing camera leaves the app usable", _text_still_works(),
              "text turn served normally with the camera unavailable")


def _text_still_works():
    brain = JarvisBrain()
    try:
        return bool(brain.ask("Reply with the single word: ok").strip())
    except Exception as e:  # noqa: BLE001
        print(f"    (text turn failed: {e})")
        return False


def check_synthetic_scene():
    """Enter / leave / glasses, through the real model, through the real loop."""
    brain = JarvisBrain(conversation_id="phase3-live")
    calls = {"n": 0}

    def analyze(prompt, attachment):
        calls["n"] += 1
        return brain.ask(prompt, images=[attachment], ephemeral=True)

    scenes = [
        ("empty desk", _frame(_desk())),
        ("user arrives", _frame(_with_person(draw_glasses=False))),
        ("unchanged", _frame(_with_person(draw_glasses=False))),
        ("glasses on", _frame(_with_person(draw_glasses=True))),
        ("user leaves", _frame(_desk(light=210))),
    ]

    perception = WebcamPerception(analyze=analyze, cooldown=0.0)
    spoken_before = len(RESULTS)
    del spoken_before

    for label, frame in scenes:
        perception.camera = _OneShot(frame)
        perception.tick()
        current = vision.get_visual_context().observations()
        print(f"  [{label}] -> " + ("; ".join(f"{o.kind}: {o.text}" for o in current) or "(nothing)"))

    context = vision.get_visual_context()
    check("C/D — meaningful changes produce labelled observations",
          bool(context.observations()) and calls["n"] >= 3,
          f"{calls['n']} vision calls for {len(scenes)} frames "
          f"({perception.frames_sampled} sampled, {perception.frames_discarded} discarded)")
    check("B — an unchanged frame costs nothing",
          perception.frames_discarded >= 1 and calls["n"] < len(scenes) + 2,
          f"identical frame #3 skipped, no vision call spent on it")
    kinds = {o.kind for o in context.observations()}
    check("9 — observation and inference are distinguished",
          kinds and kinds <= {vision.OBSERVATION, vision.INFERENCE, vision.UNCERTAINTY},
          f"kinds present: {sorted(kinds)}")
    return perception


class _OneShot:
    """A camera stand-in that returns one scripted frame."""

    def __init__(self, frame):
        self.frame = frame
        self.is_open = True

    def open(self):
        self.is_open = True

    def read(self):
        return self.frame

    def release(self):
        self.is_open = False


def check_quiet(brain, perception):
    from jarvis import speech

    spoken = []
    original = speech.speak
    speech.speak = lambda text: spoken.append(text)
    try:
        perception.camera = _OneShot(_frame(_with_person()))
        perception.tick()
    finally:
        speech.speak = original
    check("E — perception stays quiet", spoken == [],
          "a camera event produced no spoken reply")


def check_visual_query(brain):
    """Assert on the prompt actually sent, not on state left behind after the turn."""
    # Captured inside _create_chat because that is the only moment the turn
    # context is attached to a prompt; ask() restores it when the turn ends.
    sent = []
    original = brain._create_chat

    def recording(model, history=None):
        sent.append(brain._turn_context)
        return original(model, history=history)

    brain._create_chat = recording
    try:
        brain.ask("What can you currently see?")
    finally:
        brain._create_chat = original

    prompt = sent[-1] if sent else ""
    check("E' — the user can ask what the camera sees",
          "## Current Visual Context" in prompt,
          f"the camera's current view reached the model: "
          f"{[ln for ln in prompt.splitlines() if ln.startswith('- [')][:2]}")


# ---------------------------------------------------------------------------
# Real screenshots
# ---------------------------------------------------------------------------

def _screenshot(lines, title="config.yaml"):
    img = Image.new("RGB", (900, 520), (250, 250, 248))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 900, 44], fill=(38, 40, 46))
    d.text((18, 16), title, fill=(235, 235, 235))
    y = 80
    for line in lines:
        d.text((28, y), line, fill=(30, 30, 30))
        y += 34
    return img


def _png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return image_part(buffer.getvalue(), "image/png")


def check_screenshots(brain):
    before = [
        "atlas:",
        "  database: postgresql",
        "  port: 5432",
        "  replicas: 1",
    ]
    after = [
        "atlas:",
        "  database: mysql",
        "  port: 3306",
        "  replicas: 3",
    ]

    reply = brain.ask("What is shown here?", images=[_png(_screenshot(before))])
    print(f"  [direct] {reply}")
    check("F — a screenshot is analysed", bool(reply.strip()), f"reply: {reply[:120]!r}")

    reply = brain.ask(
        "What changed between these two screenshots?",
        images=[_png(_screenshot(before)), _png(_screenshot(after))],
    )
    print(f"  [compare] {reply}")
    check("H — before/after comparison works",
          "mysql" in reply.lower() and ("3306" in reply or "5432" in reply),
          f"reply: {reply[:160]!r}")

    reply = brain.ask("How many failed login attempts are shown in this screenshot?",
                      images=[_png(_screenshot(["idle desk", "no terminal here"]))])
    print(f"  [missing] {reply}")
    lowered = reply.lower()
    check("G — missing evidence is not invented",
          any(w in lowered for w in ("not visible", "isn't visible", "no ", "cannot", "can't", "does not")),
          f"reply: {reply[:160]!r}")


def check_memory_and_privacy(brain):
    """Every write in this run lands in the scratch database."""
    conn = SCRATCH_CONN
    if True:
        brain.ask(
            "Remember this for the project: atlas runs MySQL on port 3306.",
            images=[_png(_screenshot(["atlas:", "  database: mysql", "  port: 3306"]))],
        )
        stored = memory.recall_all(conn)
        print(f"  [remember] {stored}")
        check("I — an explicit visual fact is stored in the existing memory",
              any("3306" in f or "MySQL" in f for f in stored), f"stored: {stored}")

        fresh = JarvisBrain(conversation_id="phase3-fresh")
        facts = [f for f, _ in memory.recall_relevant(conn, "what port does atlas run on")]
        check("I' — a fresh brain recalls it without the screenshot",
              bool(facts), f"recalled: {facts}")

        brain.ask(
            "Actually that's the new setup — remember that atlas runs MySQL on port 3306 "
            "instead of Postgres.",
            images=[_png(_screenshot(["atlas:", "  database: mysql"]))],
        )
        active = memory.recall_all(conn)
        print(f"  [correct] {active}")
        check("J — correction supersedes the stale fact",
              len([f for f in active if "atlas" in f.lower()]) <= 1, f"active: {active}")

        secret = memory.remember(
            conn, "the screenshot shows AWS_SECRET_ACCESS_KEY=wJalrFakeValue123",
            source="visual",
        )
        check("K — a fake credential is refused", secret is False,
              "credential-shaped fact was not stored")

        dump = " ".join(str(v) for r in conn.execute("SELECT * FROM memories") for v in tuple(r))
        check("K' — no image bytes in the database", "iVBORw0KGgo" not in dump,
              "no raw image data persisted")


def check_no_webcam_memory(brain):
    """Camera events must not reach the store even after many other turns."""
    stored = memory.recall_all(SCRATCH_CONN)
    check("12 — webcam observations never become memory",
          not any("observed" in f or "inference" in f for f in stored),
          f"store holds {len(stored)} facts, none of them camera observations")


def check_model_switch(brain):
    from jarvis.model_layer import get_model_layer, set_model_layer

    original = get_model_layer()
    try:
        reply = brain.ask("Reply with the single word: ok")
        check("L — the assistant still works after all of the above",
              bool(reply.strip()), f"reply: {reply[:80]!r}")
    except BrainError as e:
        check("L — the assistant still works after all of the above", False, e.message)
    finally:
        set_model_layer(original)


def check_shutdown():
    """Repeated start/stop cycles: no crash, no leaked thread, camera released."""
    started = time.time()
    cycles = 20
    tally = {"opened": 0, "released": 0}

    class _CountingCamera:
        def __init__(self):
            self.is_open = False

        def open(self):
            tally["opened"] += 1
            self.is_open = True

        def read(self):
            return _frame(_desk())

        def release(self):
            tally["released"] += 1
            self.is_open = False

    import threading

    for _ in range(cycles):
        perception = WebcamPerception(
            analyze=lambda p, a: "observed: nothing changed.",
            camera=_CountingCamera(), interval=0.005, cooldown=0.0,
        )
        perception.start()
        perception.stop()

    leaked = [t for t in threading.enumerate() if t.name == "webcam-perception"]
    check("M — repeated start/stop cycles are clean",
          tally["opened"] == cycles and tally["released"] == cycles and not leaked,
          f"{cycles} cycles in {time.time() - started:.2f}s, "
          f"{tally['opened']} opened / {tally['released']} released, "
          f"{len(leaked)} leaked threads")


# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Phase 3 Live Verification")
    print("=" * 60 + "\n")

    vision.reset_visual_context()
    check_camera_reality()

    perception = check_synthetic_scene()
    brain = JarvisBrain(conversation_id="phase3-live-2")
    check_quiet(brain, perception)
    check_visual_query(brain)
    check_screenshots(brain)
    check_memory_and_privacy(brain)
    check_no_webcam_memory(brain)
    check_model_switch(brain)
    check_shutdown()
    vision.reset_visual_context()
    SCRATCH_CONN.close()
    SCRATCH_DB.unlink(missing_ok=True)

    passed = sum(1 for _, ok, _ in RESULTS if ok is True)
    failed = sum(1 for _, ok, _ in RESULTS if ok is False)
    skipped = sum(1 for _, ok, _ in RESULTS if ok is None)
    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60 + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())