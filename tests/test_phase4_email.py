"""Phase 4 tests: email sends only after explicit confirmation.

The SMTP layer is faked (no real mail ever); listen/speak are stubbed at the
email_tool seam. GMAIL_* are stubbed so tests never depend on a real .env.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jarvis.brain as brain_module  # noqa: E402
from jarvis.brain import JarvisBrain  # noqa: E402
from jarvis.logger import configure_logging  # noqa: E402
from jarvis.providers.base import ModelResponse, ToolCall, new_tool_call_id  # noqa: E402
from jarvis.tools import email_tool  # noqa: E402
from tests.mock_providers import build_layer  # noqa: E402

BODY = "THIS_IS_PRIVATE_EMAIL_BODY_12345"
RECIPIENT = "alice@example.com"
SUBJECT = "Meeting Tomorrow"


class _FakeSMTP:
    """Recording SMTP stand-in. Instances are tracked on the class."""
    instances = []

    def __init__(self, *args, **kwargs):
        self.sent = []
        _FakeSMTP.instances.append(self)

    def ehlo(self): ...
    def starttls(self): ...
    def login(self, *args): ...
    def send_message(self, msg):
        self.sent.append(msg)
    def quit(self): ...


@pytest.fixture()
def _mail(monkeypatch):
    """Fake creds, fake SMTP, stubbed voice. Yields (smtp_instances, answers)."""
    _FakeSMTP.instances = []
    monkeypatch.setattr(email_tool, "GMAIL_ADDRESS", "jarvis@example.com")
    monkeypatch.setattr(email_tool, "GMAIL_APP_PASSWORD", "FAKEPASS123")
    monkeypatch.setattr(email_tool.smtplib, "SMTP", _FakeSMTP)
    monkeypatch.setattr(email_tool, "speak", lambda *a: None)
    answers = {"reply": "none"}
    monkeypatch.setattr(email_tool, "listen", lambda *a: answers["reply"])
    return _FakeSMTP.instances, answers


@pytest.fixture()
def _quiet():
    configure_logging(debug=False)
    yield
    configure_logging(debug=False)


def _sends():
    return [m for inst in _FakeSMTP.instances for m in inst.sent]


def _send():
    return email_tool.send_email(RECIPIENT, SUBJECT, BODY)


class TestConfirmationGate:
    def test_unconfirmed_send_never_executes(self, _mail, _quiet):
        _, answers = _mail
        answers["reply"] = "none"
        assert _send() is False
        assert _sends() == []

    @pytest.mark.parametrize("yes", ["yes", "YES", " yes ", "y", "yeah",
                                     "go ahead", "send it", "confirm"])
    def test_explicit_yes_sends_once(self, _mail, _quiet, yes):
        _, answers = _mail
        answers["reply"] = yes
        assert _send() is True
        sent = _sends()
        assert len(sent) == 1
        assert sent[0]["To"] == RECIPIENT
        assert sent[0]["Subject"] == SUBJECT
        assert BODY in sent[0].as_string()

    @pytest.mark.parametrize("no", ["no", "n", "nope"])
    def test_no_does_not_send(self, _mail, _quiet, no):
        _, answers = _mail
        answers["reply"] = no
        assert _send() is False
        assert _sends() == []

    @pytest.mark.parametrize("vague", ["maybe", "sure?", "I guess", "probably",
                                       "okay?", "", "sure", "ok", "okay",
                                       "yes? no", "cancel", "stop",
                                       "don't send", "don't send it"])
    def test_ambiguous_and_cancel_never_send(self, _mail, _quiet, vague):
        _, answers = _mail
        answers["reply"] = vague
        assert _send() is False
        assert _sends() == []

    def test_parse_confirmation_unit(self):
        assert email_tool.parse_confirmation("yes") is True
        assert email_tool.parse_confirmation("  Yes, send it. ") is True
        assert email_tool.parse_confirmation("no") is False
        assert email_tool.parse_confirmation("okay") is False
        assert email_tool.parse_confirmation("") is False
        assert email_tool.parse_confirmation(None) is False


class TestLoggingPrivacy:
    def test_body_never_logged(self, _mail, _quiet, caplog):
        _, answers = _mail
        answers["reply"] = "yes"
        with caplog.at_level("INFO", logger="jarvis"):
            _send()
        text = "\n".join(r.message for r in caplog.records)
        assert RECIPIENT in text
        assert SUBJECT in text
        assert BODY not in text

    def test_unconfirmed_logged_without_body(self, _mail, _quiet, caplog):
        _, answers = _mail
        answers["reply"] = "no"
        with caplog.at_level("INFO", logger="jarvis"):
            _send()
        text = "\n".join(r.message for r in caplog.records)
        assert BODY not in text

    def test_secrets_never_logged(self, _mail, _quiet, caplog):
        import smtplib as _real_smtp
        _, answers = _mail
        answers["reply"] = "yes"

        class _Boom(_FakeSMTP):
            def send_message(self, msg):
                raise _real_smtp.SMTPException("smtp refused")

        import smtplib
        smtplib.SMTP = _Boom
        try:
            with caplog.at_level("DEBUG", logger="jarvis"):
                assert _send() is False
        finally:
            smtplib.SMTP = _FakeSMTP
        text = "\n".join(r.message for r in caplog.records)
        assert "FAKEPASS123" not in text
        assert "smtp refused" in text  # raw error logged internally


class TestNoDuplicateSend:
    def test_repeated_tool_call_sends_once(self, _mail, _quiet):
        _, answers = _mail
        answers["reply"] = "yes"
        state = {"n": 0}

        def _script(history, payload):
            if payload.get("kind") == "tool_results":
                state["n"] += 1
                if state["n"] < 2:  # model re-requests the same send
                    return ModelResponse(text=None, tool_calls=[_call()])
                return ModelResponse(text="Sent.")
            return ModelResponse(text=None, tool_calls=[_call()])

        def _call():
            return ToolCall(id=new_tool_call_id(), name="send_email",
                            arguments={"to_address": RECIPIENT,
                                       "subject": SUBJECT, "message": BODY})

        layer = build_layer(
            models=[{"key": "m", "model": "m", "priority": 90,
                     "capabilities": {"reasoning": True, "tool_calling": True}}],
            behaviours={"m": _script},
        )
        assert JarvisBrain(model_layer=layer).ask("email alice the report") == "Sent."
        assert len(_sends()) == 1  # second request reused the cached result


class TestAllPathsProtected:
    def test_registry_funnels_to_gated_send(self, _mail, _quiet, monkeypatch):
        seen = []
        monkeypatch.setattr(email_tool, "send_email",
                            lambda t, s, c: seen.append((t, s, c)) or True)
        brain = JarvisBrain(model_layer=build_layer(
            models=[{"key": "m", "model": "m", "priority": 90,
                     "capabilities": {"reasoning": True, "tool_calling": True}}],
            behaviours={"m": lambda h, p: ModelResponse(text="ok")},
        ))
        out = brain._execute_tool_call("send_email", {
            "to_address": RECIPIENT, "subject": SUBJECT, "message": BODY})
        assert seen == [(RECIPIENT, SUBJECT, BODY)]
        assert "success" in out.lower()

    def test_interactive_handler_is_gated(self, _mail, _quiet, monkeypatch):
        _, answers = _mail
        monkeypatch.setattr(email_tool, "CONTACTS", {"mom": "mom@example.com"})
        answers["reply"] = "no"
        email_tool.handle_email_command("send mom the update")
        assert _sends() == []
        answers["reply"] = "yes"
        email_tool.handle_email_command("tell mom hello")
        assert len(_sends()) == 1
        assert _sends()[0]["To"] == "mom@example.com"
