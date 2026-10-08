"""Phase 3 tests: vision input and visual memory.

Covers the four properties the architecture depends on:

* images reach a vision-capable model through the provider abstraction,
* routing never hands an image to a text-only model,
* a vision turn keeps the Phase 2 behavioral and memory context, and
* durable visual facts use the Phase 2 memory system (one system, not two).

Runs offline against the existing mock providers and a temporary SQLite file.
"""

import base64
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis import behavior, memory  # noqa: E402
from jarvis.brain import (  # noqa: E402
    JarvisBrain,
    BrainError,
    _build_context_block,
    _build_system_prompt,
    save_memory,
)
from jarvis.classify import classify  # noqa: E402
from jarvis.config import JARVIS_SYSTEM_PROMPT  # noqa: E402
from jarvis.providers.base import (  # noqa: E402
    IMAGE_TOKEN_COST,
    MAX_IMAGE_BYTES,
    estimate_tokens,
    image_part,
    user_message,
)
from jarvis.providers.gemini import _to_gemini_content  # noqa: E402
from jarvis.providers.openai_compat import _openai_user_parts  # noqa: E402
from tests.mock_providers import build_layer, reply  # noqa: E402

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        path = Path(tmp.name)
    conn = memory.init_memory_db(path)
    yield conn
    conn.close()
    path.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """Keep every test off the user's real database."""
    import jarvis.brain as brain_module

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        spare = Path(tmp.name)
    conn = memory.init_memory_db(spare)
    monkeypatch.setattr(brain_module, "get_memory_conn", lambda: conn)
    monkeypatch.setattr(brain_module.StatusIndicator, "thinking", staticmethod(lambda: None))
    monkeypatch.setattr(brain_module.StatusIndicator, "tool_call", staticmethod(lambda n, a: None))
    monkeypatch.setattr(brain_module.StatusIndicator, "tool_result", staticmethod(lambda n, r: None))
    monkeypatch.setattr(brain_module.StatusIndicator, "memory", staticmethod(lambda m: None))
    yield conn
    conn.close()
    spare.unlink(missing_ok=True)


def _vision_layer(text="ok", vision=True):
    return build_layer(
        models=[{"key": "m", "provider": "p", "model": "mm",
                 "capabilities": {"reasoning": True, "tool_calling": True, "vision": vision}}],
        behaviours={"mm": reply(text)},
    )


def _img(n=1):
    return [image_part(PNG_1PX + bytes([i]), "image/png") for i in range(n)]


# ============================================================================
# Neutral attachment format
# ============================================================================

class TestAttachmentFormat:
    def test_text_only_message_is_unchanged(self):
        """Phase 0 contract: a text turn must look exactly as it always did."""
        assert user_message("hello") == {"role": "user", "content": "hello"}

    def test_images_ride_alongside_text(self):
        msg = user_message("look", _img())
        assert msg["content"] == "look"
        assert len(msg["images"]) == 1

    def test_image_part_carries_bytes_and_mime(self):
        part = image_part(PNG_1PX, "image/png")
        assert part == {"kind": "image", "mime": "image/png", "data": PNG_1PX}

    def test_unknown_mime_is_refused(self):
        with pytest.raises(ValueError):
            image_part(PNG_1PX, "application/pdf")

    def test_empty_image_is_refused(self):
        with pytest.raises(ValueError):
            image_part(b"", "image/png")

    def test_oversized_image_is_refused(self):
        with pytest.raises(ValueError):
            image_part(b"x" * (MAX_IMAGE_BYTES + 1), "image/png")

    def test_images_cost_real_tokens_for_routing(self):
        """Otherwise a screenshot routes to a model that cannot hold it."""
        without = estimate_tokens([{"role": "user", "content": "hi"}])
        with_img = estimate_tokens([{"role": "user", "content": "hi", "images": _img(2)}])
        assert with_img - without == 2 * IMAGE_TOKEN_COST

    def test_text_estimate_unchanged(self):
        assert estimate_tokens([{"role": "user", "content": "12345678"}]) == 2


# ============================================================================
# Provider rendering (provider abstraction)
# ============================================================================

class TestProviderRendering:
    def test_openai_renders_data_urls_in_order(self):
        parts = _openai_user_parts("compare these", _img(2))
        assert parts[0] == {"type": "text", "text": "compare these"}
        assert [p["type"] for p in parts[1:]] == ["image_url", "image_url"]
        assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_openai_image_only_turn_has_no_text_part(self):
        parts = _openai_user_parts("", _img(1))
        assert len(parts) == 1
        assert parts[0]["type"] == "image_url"

    def test_gemini_renders_inline_blobs(self):
        from google.genai import types

        contents = _to_gemini_content(
            [{"role": "user", "content": "look", "images": _img(2)}], types
        )
        assert len(contents) == 1
        parts = contents[0].parts
        assert parts[0].text == "look"
        assert parts[1].inline_data.mime_type == "image/png"
        assert parts[2].inline_data.data is not None

    def test_gemini_text_only_turn_is_unchanged(self):
        from google.genai import types

        contents = _to_gemini_content([{"role": "user", "content": "hi"}], types)
        assert contents[0].parts[0].text == "hi"
        assert len(contents[0].parts) == 1

    def test_both_providers_accept_the_same_neutral_payload(self):
        """One representation, two renderings -- the whole point of the seam."""
        msg = user_message("what is this", _img(2))
        openai_parts = _openai_user_parts(msg["content"], msg["images"])
        from google.genai import types
        gemini_parts = _to_gemini_content([msg], types)[0].parts
        assert len(openai_parts) == 3
        assert len(gemini_parts) == 3


# ============================================================================
# Vision-aware routing
# ============================================================================

class TestVisionRouting:
    def test_classification_flags_vision(self):
        assert classify("what is this?", vision_required=True).vision_required is True
        assert classify("what is this?").vision_required is False

    def test_text_only_request_is_unaffected(self):
        assert classify("hello").vision_required is False

    def test_image_routes_to_vision_model(self):
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
        brain.ask("what is this?", images=_img())
        assert brain.last_model_key == "sees", "image routed to a text-only model"

    def test_text_only_still_prefers_the_free_model(self):
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
        assert brain.last_model_key == "blind", "vision model hijacked a text turn"

    def test_no_vision_model_fails_honestly(self):
        layer = _vision_layer(vision=False)
        with pytest.raises(BrainError) as exc:
            JarvisBrain(model_layer=layer).ask("what is this?", images=_img())
        msg = exc.value.message.lower()
        assert "not been processed" in msg or "can't look at images" in msg
        assert "i see" not in msg

    def test_falls_back_across_providers(self):
        from jarvis.providers.base import ErrorKind
        from tests.mock_providers import error as mock_error

        layer = build_layer(
            models=[
                {"key": "dead", "provider": "p1", "model": "m1", "priority": 100,
                 "capabilities": {"reasoning": True, "vision": True}},
                {"key": "sees", "provider": "p2", "model": "m2", "priority": 10,
                 "capabilities": {"reasoning": True, "vision": True}},
            ],
            behaviours={"m1": mock_error(ErrorKind.SERVER, "503"),
                        "m2": reply("saw it")},
        )
        brain = JarvisBrain(model_layer=layer)
        assert brain.ask("what is this?", images=_img()) == "saw it"
        assert brain.last_model_key == "sees"


# ============================================================================
# Turn shape
# ============================================================================

class TestVisionTurns:
    def test_image_only_turn_is_allowed(self):
        brain = JarvisBrain(model_layer=_vision_layer("described"))
        assert brain.ask("", images=_img()) == "described"

    def test_empty_turn_without_image_still_rejected(self):
        with pytest.raises(ValueError):
            JarvisBrain(model_layer=_vision_layer()).ask("   ")

    def test_images_reach_the_provider_in_order(self):
        layer = _vision_layer("ok")
        JarvisBrain(model_layer=layer).ask("compare", images=_img(3))
        session = layer.providers["p"].sessions[-1].session
        sent = [p for p in layer.providers["p"].sent if p.get("kind") == "user"][-1]
        assert len(sent["images"]) == 3
        assert sent["text"] == "compare"

    def test_text_only_payload_has_no_images_key(self):
        """Phase 0 integrity: unchanged payload shape for text turns."""
        layer = _vision_layer("ok")
        JarvisBrain(model_layer=layer).ask("hello")
        sent = [p for p in layer.providers["p"].sent if p.get("kind") == "user"][-1]
        assert set(sent) == {"kind", "text"}

    def test_images_survive_a_model_switch(self):
        """A session opened on a different provider must replay the image.

        Asserted at the switch mechanism itself (`_create_chat`) rather than by
        coaxing the health tracker into a switch, which is not what is under
        test here.
        """
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
        brain.ask("first", images=_img())

        # Same conversation, different provider: this is what a model switch does.
        brain._session = None
        brain._create_chat("b", history=brain._history)

        replayed = layer.providers["p2"].sessions[-1].history
        assert any(t.get("images") for t in replayed), "visual context lost on switch"

    def test_history_records_the_image_with_its_turn(self):
        layer = _vision_layer("ok")
        brain = JarvisBrain(model_layer=layer)
        brain.ask("look", images=_img())
        user_turn = [t for t in brain._history if t.get("role") == "user"][0]
        assert len(user_turn["images"]) == 1


# ============================================================================
# Vision + Phase 2 behavior
# ============================================================================

class TestVisionBehavior:
    def test_vision_directive_requires_grounding(self):
        d = behavior.vision_directive(1)
        assert "Stay grounded" in d
        assert "Never claim to have seen an image" in d

    def test_vision_directive_counts_images(self):
        assert "2 images" in behavior.vision_directive(2)
        assert "1 image." in behavior.vision_directive(1)

    def test_vision_turn_gets_visual_context_block(self, _isolated):
        block = _build_context_block("what is this", classify("what is this"), [], _img())
        assert "## Visual Input" in block

    def test_text_turn_gets_no_visual_block(self, _isolated):
        block = _build_context_block("hello", classify("hello"), [])
        assert "## Visual Input" not in block

    def test_vision_turn_keeps_memory_and_behavior(self, _isolated):
        memory.remember(_isolated, "Project atlas uses Postgres", "project")
        block = _build_context_block(
            "why is atlas failing", classify("why is atlas failing"), [], _img()
        )
        assert "## Relevant Long-Term Memory" in block, "memory lost on a vision turn"
        assert "## This turn" in block, "behavior lost on a vision turn"
        assert "## Visual Input" in block

    def test_identity_stays_separate_from_visual_memory(self, _isolated):
        memory.remember(_isolated, "Atlas uses MySQL", "project", source="visual")
        assert _build_system_prompt() == JARVIS_SYSTEM_PROMPT


# ============================================================================
# Visual memory
# ============================================================================

class TestVisualMemory:
    def test_image_alone_stores_nothing(self, _isolated):
        layer = _vision_layer("described it")
        brain = JarvisBrain(model_layer=layer)
        brain.ask("what does this show?", images=_img())
        assert memory.get_memory_count(_isolated) == 0

    def test_explicit_remember_intent_for_visual(self):
        p = behavior.memory_posture("Remember this for the project", has_images=True)
        assert "durable fact, not the image" in p
        assert "credentials" in p

    def test_visual_remember_still_gets_subject_keys(self):
        """Regressed once: the visual path skipped the supersession hint."""
        p = behavior.memory_posture(
            "remember this", subjects=["project:atlas-db"], has_images=True
        )
        assert "project:atlas-db" in p

    def test_visual_facts_use_the_existing_memory_system(self, db):
        memory.remember(db, "Atlas runs MySQL 8.0", "project",
                        subject="project:atlas", source="visual")
        assert [f for f, _ in memory.recall_relevant(db, "what database does atlas use")] == [
            "Atlas runs MySQL 8.0"
        ]

    def test_visual_memory_is_retrievable_later(self, db):
        memory.remember(db, "Atlas database host is db-atlas-01", "project",
                        subject="project:atlas", source="visual")
        hits = [f for f, _ in memory.recall_relevant(db, "what is the atlas database host")]
        assert hits == ["Atlas database host is db-atlas-01"]

    def test_visual_memory_can_be_superseded(self, db):
        memory.remember(db, "Atlas uses Postgres", "project", subject="project:atlas-db")
        memory.remember(db, "Atlas uses MySQL 8.0", "project", subject="project:atlas-db")
        active = memory.recall_all(db)
        assert active == ["Atlas uses MySQL 8.0"]

    def test_visual_memory_can_be_forgotten(self, db):
        memory.remember(db, "Atlas uses MySQL", "project", subject="project:atlas")
        forgotten = memory.forget_matching(db, "MySQL")
        assert forgotten
        assert memory.recall_relevant(db, "what database does atlas use") == []

    def test_irrelevant_visual_memory_is_not_injected(self, db):
        memory.remember(db, "Atlas architecture diagram shows three services", "project")
        assert memory.recall_relevant(db, "what is the capital of france") == []

    def test_visual_memory_survives_model_switching(self, _isolated):
        layer = build_layer(
            models=[
                {"key": "a", "provider": "p1", "model": "m1", "priority": 100,
                 "capabilities": {"reasoning": True, "vision": True}},
                {"key": "b", "provider": "p2", "model": "m2", "priority": 10,
                 "capabilities": {"reasoning": True, "vision": True}},
            ],
            behaviours={"m1": reply("A"), "m2": reply("B")},
        )
        memory.remember(_isolated, "Atlas uses MySQL 8.0", "project", subject="project:atlas")
        JarvisBrain(model_layer=layer).ask("which database does atlas use")
        for provider in layer.providers.values():
            for record in provider.sessions:
                assert "MySQL" in record.system_prompt

    def test_there_is_only_one_memory_table(self):
        """Visual memory must not create a second store."""
        import inspect

        src = inspect.getsource(memory)
        assert src.count("CREATE TABLE") == 1, "a second table was introduced"

    def test_no_image_bytes_are_persisted(self, db):
        memory.remember(db, "Atlas uses MySQL", "project", source="visual")
        blob = " ".join(str(v) for r in db.execute("SELECT * FROM memories")
                        for v in tuple(r))
        assert "iVBORw0KGgo" not in blob, "raw image data reached the database"


# ============================================================================
# Privacy
# ============================================================================

class TestVisualPrivacy:
    @pytest.mark.parametrize("fact", [
        "the api_key is sk-abcdef123456",
        "Atlas password is hunter2",
        "access_token: abcdef123456",
        "AWS_SECRET_ACCESS_KEY=xyz",
    ])
    def test_credentials_are_never_stored(self, db, fact):
        assert memory.remember(db, fact, "project", source="visual") is False
        assert memory.recall_all(db) == []

    def test_secret_refusal_is_reported_honestly(self, _isolated):
        out = save_memory("Atlas api_key is sk-abcdef123456", "project", source="explicit")
        assert "didn't store" in out
        assert "Nothing has been saved" in out

    def test_normal_facts_still_store(self, db):
        assert memory.remember(db, "Atlas uses MySQL 8.0", "project") is True

    def test_secret_detection_is_case_insensitive(self):
        assert memory.contains_secret("API KEY: abc123456") is True
        assert memory.contains_secret("Password: abc123456") is True

    def test_ordinary_words_are_not_secrets(self):
        assert memory.contains_secret("Atlas uses MySQL on port 3306") is False