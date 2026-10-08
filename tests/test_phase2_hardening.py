"""Phase 2 hardening regression tests.

Covers four defects found at the end of Phase 2:

1. Retrieval was blind to paraphrase ("what theme do I like" vs
   "prefers dark mode").
2. Stale legacy facts stayed active beside the fact that replaced them.
3. A brain turn must not leak daemon threads (native-teardown race at exit).
4. An empty model reply after a tool loop returned the idle fallback
   "Standing by, Boss.", hiding the fact that tools had actually run.
"""

import sys
import tempfile
import threading
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis import memory  # noqa: E402
from jarvis.brain import JarvisBrain, TOOL_REGISTRY, _report_executed  # noqa: E402
from jarvis.providers.base import ModelResponse, ToolCall  # noqa: E402
from tests.mock_providers import build_layer, error as mock_error, reply  # noqa: E402
from jarvis.providers.base import ErrorKind  # noqa: E402


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        path = Path(tmp.name)
    conn = memory.init_memory_db(path)
    yield conn
    conn.close()
    path.unlink(missing_ok=True)


def _hits(conn, query):
    return [f for f, _ in memory.recall_relevant(conn, query)]


# ============================================================================
# 1. Semantic / paraphrase retrieval
# ============================================================================

class TestParaphraseRetrieval:
    def test_exact_match_still_retrieves(self, db):
        memory.remember(db, "Project atlas uses Postgres", "project")
        assert _hits(db, "which database does atlas use") == ["Project atlas uses Postgres"]

    def test_paraphrase_retrieves(self, db):
        """The defect: zero shared characters, but same meaning."""
        memory.remember(db, "User prefers dark mode.", "preference")
        assert _hits(db, "What theme do I like?") == ["User prefers dark mode."]

    def test_paraphrase_variant_retrieves(self, db):
        memory.remember(db, "User prefers dark mode.", "preference")
        assert _hits(db, "what colour scheme") == ["User prefers dark mode."]

    def test_morphological_variant_retrieves(self, db):
        memory.remember(db, "User prefers Postgres for storage", "preference")
        assert _hits(db, "which datastore is preferred") == [
            "User prefers Postgres for storage"
        ]

    def test_unrelated_query_does_not_retrieve(self, db):
        """The guard rail: semantic matching must not become a catch-all."""
        memory.remember(db, "User prefers dark mode.", "preference")
        assert _hits(db, "What database are we using?") == []

    def test_sibling_project_does_not_leak(self, db):
        """Two memories sharing a concept are still distinct."""
        memory.remember(db, "Project atlas uses Postgres", "project")
        memory.remember(db, "Project borealis uses DynamoDB", "project")
        hits = _hits(db, "which database does atlas use")
        assert hits == ["Project atlas uses Postgres"]
        assert not any("borealis" in h for h in hits)

    def test_concept_only_loses_to_specific_match(self, db):
        memory.remember(db, "Deploys use a manual approval step", "project")
        memory.remember(db, "Deploys run through GitHub Actions", "project")
        hits = _hits(db, "how do deploys use github actions")
        assert "Deploys run through GitHub Actions" in hits

    def test_retrieval_is_still_bounded(self, db):
        for i in range(40):
            memory.remember(db, f"atlas database note {i}", "project")
        assert len(memory.recall_relevant(db, "atlas database", limit=5)) == 5

    def test_retrieval_is_deterministic(self, db):
        for f in ["atlas uses postgres", "atlas ships in march", "atlas owner is sam"]:
            memory.remember(db, f, "project")
        q = "what about atlas"
        assert memory.recall_relevant(db, q) == memory.recall_relevant(db, q)

    def test_relevance_is_ordered_by_score(self, db):
        memory.remember(db, "atlas database", "project")
        memory.remember(db, "atlas database and atlas deployment and atlas api", "project")
        hits = _hits(db, "atlas")
        assert hits[0] == "atlas database and atlas deployment and atlas api"

    def test_scoring_components(self):
        q = memory.tokenize("theme")
        exact = memory._score(q, memory.tokenize("theme"))[0]
        concept = memory._score(q, memory.tokenize("dark mode"))[0]
        unrelated = memory._score(q, memory.tokenize("postgres server"))[0]
        assert exact > concept >= memory._RELEVANCE_THRESHOLD
        assert unrelated < memory._RELEVANCE_THRESHOLD

    def test_no_network_or_model_dependency(self):
        """Retrieval must stay local: no embedding or chat call per turn."""
        import inspect
        src = inspect.getsource(memory)
        for forbidden in ("requests.", "httpx", "genai", "openai", "urllib"):
            assert forbidden not in src, f"retrieval reached for {forbidden}"

    def test_empty_query_retrieves_nothing(self, db):
        memory.remember(db, "User prefers dark mode.", "preference")
        assert memory.recall_relevant(db, "") == []

    def test_graceful_when_store_is_empty(self, db):
        assert memory.recall_relevant(db, "anything at all") == []


# ============================================================================
# 2. Stale memory reconciliation
# ============================================================================

class TestStaleMemoryReconciliation:
    def _legacy(self, conn, fact, category="fact"):
        conn.execute(
            "INSERT INTO memories (fact, category, active) VALUES (?, ?, 1)",
            (fact, category),
        )
        conn.commit()

    def _reopen(self, conn):
        """Re-run migration/reconciliation, as a restart would."""
        memory.init_memory_db(Path(conn.execute("PRAGMA database_list").fetchone()[2]))

    def test_stale_name_is_retired(self, db):
        self._legacy(db, "User's name is Ravi.")
        self._legacy(db, "User's name is Alex")
        self._reopen(db)
        assert "User's name is Ravi." not in memory.recall_all(db)
        assert "User's name is Alex" in memory.recall_all(db)

    def test_newest_claim_wins(self, db):
        self._legacy(db, "favourite language is Go")
        self._legacy(db, "favourite language is Rust")
        self._reopen(db)
        assert _hits(db, "what language do I like") == ["favourite language is Rust"]

    def test_reconciliation_is_generic_not_name_specific(self, db):
        """No hardcoded person or fact: any 'X is Y' then 'X is Z' resolves."""
        self._legacy(db, "The deploy target is staging")
        self._legacy(db, "The deploy target is production")
        self._reopen(db)
        active = memory.recall_all(db)
        assert "The deploy target is production" in active
        assert "The deploy target is staging" not in active

    def test_unrelated_facts_are_untouched(self, db):
        self._legacy(db, "User's name is Alex")
        self._legacy(db, "User's timezone is IST")
        self._legacy(db, "Project atlas uses Postgres")
        self._reopen(db)
        active = memory.recall_all(db)
        assert len(active) == 3

    def test_distinct_topics_do_not_collide(self, db):
        """Two rows that merely look alike must both survive."""
        self._legacy(db, "The auth service is done")
        self._legacy(db, "The billing service is pending")
        self._reopen(db)
        assert len(memory.recall_all(db)) == 2

    def test_reconciliation_is_idempotent(self, db):
        self._legacy(db, "User's name is Ravi.")
        self._legacy(db, "User's name is Alex")
        self._reopen(db)
        first = memory.recall_all(db)
        self._reopen(db)
        assert memory.recall_all(db) == first

    def test_skeleton_needs_a_value_shape(self, db):
        """A long sentence tail is not a 'value', so nothing is merged."""
        self._legacy(db, "The meeting is on friday to discuss the roadmap")
        self._legacy(db, "The standup is on monday to discuss the roadmap")
        self._reopen(db)
        assert len(memory.recall_all(db)) == 2

    def test_explicit_writes_still_win_after_reconcile(self, db):
        self._legacy(db, "User's name is Ravi.")
        memory.remember(db, "User's name is Alex", "fact", subject="user:name",
                        source="explicit")
        self._reopen(db)
        assert "User's name is Ravi." not in memory.recall_all(db)


# ============================================================================
# 3. Shutdown lifecycle
# ============================================================================

class TestShutdownLifecycle:
    def test_no_daemon_thread_survives_a_turn(self, monkeypatch):
        import jarvis.brain as brain_module
        layer = build_layer(
            models=[{"key": "m", "provider": "p", "model": "mm",
                     "capabilities": {"reasoning": True}}],
            behaviours={"mm": reply("ok")},
        )
        before = threading.active_count()
        JarvisBrain(model_layer=layer).ask("hello")
        assert threading.active_count() == before


# ============================================================================
# 4. Empty response after a tool loop
# ============================================================================

def _tool_layer(second_text=None, tool="tell_time", tools=None):
    """A model that calls a tool once, then returns `second_text`."""
    calls = [ToolCall(id="1", name=tool, arguments={})] if tools is None else tools

    def script(history, payload):
        if payload.get("kind") == "tool_results":
            return ModelResponse(text=second_text)
        return ModelResponse(text=None, tool_calls=calls)

    return build_layer(
        models=[{"key": "m", "provider": "p", "model": "mm",
                 "capabilities": {"reasoning": True, "tool_calling": True}}],
        behaviours={"mm": script},
    )


class TestEmptyToolLoopResponse:
    def test_successful_tool_with_normal_reply(self):
        reply_text = JarvisBrain(model_layer=_tool_layer("It is noon.")).ask("check my inbox")
        assert reply_text == "It is noon."

    def test_successful_tool_with_empty_reply_surfaces_the_result(self):
        """The defect: this used to return 'Standing by, Boss.'"""
        out = JarvisBrain(model_layer=_tool_layer(None)).ask("check my inbox")
        assert out != "Standing by, Boss."
        assert "tell_time" in out
        assert "Standing by" not in out

    def test_failed_tool_with_normal_reply_discloses_failure(self):
        def boom():
            raise RuntimeError("device gone")

        original = TOOL_REGISTRY["tell_time"]
        TOOL_REGISTRY["tell_time"] = boom
        try:
            out = JarvisBrain(model_layer=_tool_layer("All set.")).ask("check my inbox")
        finally:
            TOOL_REGISTRY["tell_time"] = original
        assert "All set." in out
        assert "did not complete" in out

    def test_failed_tool_with_empty_reply_never_reads_as_success(self):
        def boom():
            raise RuntimeError("device gone")

        original = TOOL_REGISTRY["tell_time"]
        TOOL_REGISTRY["tell_time"] = boom
        try:
            out = JarvisBrain(model_layer=_tool_layer(None)).ask("check my inbox")
        finally:
            TOOL_REGISTRY["tell_time"] = original
        assert "did not complete" in out
        assert "Standing by" not in out

    def test_multiple_tools_with_empty_reply(self):
        calls = [ToolCall(id="1", name="tell_time", arguments={}),
                 ToolCall(id="2", name="tell_joke", arguments={})]
        out = JarvisBrain(model_layer=_tool_layer(None, tools=calls)).ask("do both")
        assert "tell_time" in out and "tell_joke" in out
        assert "Standing by" not in out

    def test_no_tool_at_all_still_uses_idle_fallback(self):
        """Unchanged: a genuinely empty first response is honestly idle."""
        layer = build_layer(
            models=[{"key": "m", "provider": "p", "model": "mm",
                     "capabilities": {"reasoning": True}}],
            behaviours={"mm": reply("")},
        )
        assert JarvisBrain(model_layer=layer).ask("hello") == "Standing by, Boss."

    def test_empty_response_does_not_kill_the_request(self):
        """A retry must still be able to produce a real answer."""
        layer = build_layer(
            models=[
                {"key": "bad", "provider": "p1", "model": "m1", "priority": 100,
                 "capabilities": {"reasoning": True}},
                {"key": "good", "provider": "p2", "model": "m2", "priority": 10,
                 "capabilities": {"reasoning": True}},
            ],
            behaviours={"m1": mock_error(ErrorKind.SERVER, "503"),
                        "m2": reply("Recovered.")},
        )
        assert JarvisBrain(model_layer=layer).ask("hello") == "Recovered."

    def test_report_helper_labels_failures(self):
        out = _report_executed([
            ("tell_time", "12:00"),
            ("send_email", "Error executing 'send_email': smtp refused"),
        ])
        assert "- tell_time: 12:00" in out
        assert "send_email did not complete" in out

    def test_report_helper_handles_empty_output(self):
        assert "(no output)" in _report_executed([("thing", "")])