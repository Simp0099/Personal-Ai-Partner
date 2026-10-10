"""Email confirmation gate, after the tool stopped owning the conversation.

The gate used to live inside the tool: it spoke a question and called
listen(), which meant a confirmation prompt in the middle of a model turn,
unreachable in text mode. It now lives where the answer actually is -- in the
conversation -- which makes these tests about the tool refusing honestly.
"""

import smtplib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.tools import email_tool  # noqa: E402

RECIPIENT = "alice@example.com"
SUBJECT = "Meeting Tomorrow"
BODY = "THIS_IS_PRIVATE_EMAIL_BODY_12345"


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
    _FakeSMTP.instances = []
    monkeypatch.setattr(email_tool, "GMAIL_ADDRESS", "jarvis@example.com")
    monkeypatch.setattr(email_tool, "GMAIL_APP_PASSWORD", "FAKEPASS123")
    monkeypatch.setattr(email_tool.smtplib, "SMTP", _FakeSMTP)
    return _FakeSMTP.instances


def _sends():
    return [m for inst in _FakeSMTP.instances for m in inst.sent]


class TestConfirmationGate:
    def test_unconfirmed_never_sends(self, _mail):
        out = email_tool.send_email(RECIPIENT, SUBJECT, BODY)
        assert _sends() == []
        assert "not sent" in out.lower()

    def test_unconfirmed_tells_the_caller_what_to_do(self, _mail):
        out = email_tool.send_email(RECIPIENT, SUBJECT, BODY)
        assert "confirmed=true" in out

    def test_confirmed_sends_once(self, _mail):
        out = email_tool.send_email(RECIPIENT, SUBJECT, BODY, confirmed=True)
        sent = _sends()
        assert len(sent) == 1
        assert sent[0]["To"] == RECIPIENT
        assert sent[0]["Subject"] == SUBJECT
        assert BODY in sent[0].as_string()
        assert "sent" in out.lower()

    def test_missing_credentials_never_sends(self, _mail, monkeypatch):
        monkeypatch.setattr(email_tool, "GMAIL_ADDRESS", "")
        out = email_tool.send_email(RECIPIENT, SUBJECT, BODY, confirmed=True)
        assert _sends() == []
        assert "not configured" in out.lower()

    def test_malformed_recipient_never_sends(self, _mail):
        for bad in ("not-an-email", "a@b", "", "@example.com", "a b@example.com"):
            assert _sends() == []
            out = email_tool.send_email(bad, SUBJECT, BODY, confirmed=True)
            assert "not a valid email address" in out
        assert _sends() == []

    def test_smtp_failure_reports_failure_not_success(self, _mail, monkeypatch):
        class _Boom(_FakeSMTP):
            def send_message(self, msg):
                raise smtplib.SMTPException("smtp refused")

        monkeypatch.setattr(email_tool.smtplib, "SMTP", _Boom)
        out = email_tool.send_email(RECIPIENT, SUBJECT, BODY, confirmed=True)
        assert "could not send" in out.lower()

    def test_connection_closed_even_when_send_fails(self, _mail, monkeypatch):
        """A failed send must not leak the SMTP connection."""
        closed = {"quit": 0}

        class _Boom(_FakeSMTP):
            def send_message(self, msg):
                raise smtplib.SMTPException("refused")

            def quit(self):
                closed["quit"] += 1

        monkeypatch.setattr(email_tool.smtplib, "SMTP", _Boom)
        email_tool.send_email(RECIPIENT, SUBJECT, BODY, confirmed=True)
        assert closed["quit"] == 1


class TestLoggingPrivacy:
    def test_body_never_logged(self, _mail, caplog):
        import logging

        caplog.set_level(logging.INFO, logger="jarvis")
        email_tool.send_email(RECIPIENT, SUBJECT, BODY, confirmed=True)
        text = "\n".join(r.message for r in caplog.records)
        assert RECIPIENT in text
        assert SUBJECT in text
        assert BODY not in text

    def test_password_never_logged(self, _mail, caplog, monkeypatch):
        import logging

        class _Boom(_FakeSMTP):
            def send_message(self, msg):
                raise smtplib.SMTPException("login FAKEPASS123 rejected")

        monkeypatch.setattr(email_tool.smtplib, "SMTP", _Boom)
        caplog.set_level(logging.DEBUG, logger="jarvis")
        email_tool.send_email(RECIPIENT, SUBJECT, BODY, confirmed=True)
        assert "FAKEPASS123" not in caplog.text


class TestConfirmationParsing:
    @pytest.mark.parametrize("yes", ["yes", "YES", " yes ", "y", "yeah",
                                     "go ahead", "send it", "confirm"])
    def test_affirmative(self, yes):
        assert email_tool.parse_confirmation(yes) is True

    @pytest.mark.parametrize("no", ["no", "nope", "cancel", "stop", "maybe",
                                    "sure?", "okay", "ok", "", "none", None,
                                    "don't send"])
    def test_everything_else_is_refused(self, no):
        assert email_tool.parse_confirmation(no) is False
