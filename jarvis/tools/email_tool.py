"""Secure email tool for JARVIS 2.0.

SMTP over TLS, credentials from environment variables only -- never stored,
never logged.

Two properties matter more than the feature:

* **The body is never logged.** Recipient and subject may be; content never is.
* **Nothing is sent without an explicit confirmation** on every path. There is
  no timeout, no default-yes, and no partial match: silence is not consent.
"""

import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from jarvis.config import GMAIL_ADDRESS, GMAIL_APP_PASSWORD, CONTACTS
from jarvis.logger import logger
from jarvis.providers.base import redact

#: Explicit affirmative replies. Anything else -- no, cancel, "sure?", "ok",
#: empty, "none" from a non-interactive session -- is unconfirmed.
AFFIRMATIVE = frozenset({
    "yes", "y", "yeah", "yep", "yes please", "send it", "do it",
    "confirm", "confirmed", "go ahead", "send",
})

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def parse_confirmation(reply) -> bool:
    """True only for a clearly affirmative confirmation. Strict by design."""
    text = " ".join(str(reply or "").strip().lower().split()).rstrip("?!.")
    return (text in AFFIRMATIVE or text.startswith("yes ")
            or text.startswith("yes,"))


def _safe(message: str) -> str:
    """Error text safe to log and to return.

    Key-shaped strings are redacted by prefix, but an SMTP error can echo this
    account's own app password verbatim, and that string looks like ordinary
    words. Removing the literal credential is the only thing that catches it.
    """
    text = redact(message)
    secret = str(GMAIL_APP_PASSWORD or "")
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return text


def resolve_contact(query: str):
    """Email address for a contact alias mentioned in `query`, or None."""
    lowered = (query or "").lower()
    for name, address in (CONTACTS or {}).items():
        if name.lower() in lowered:
            return address
    return None


def send_email(to_address: str, subject: str, content: str,
               confirmed: bool = False) -> str:
    """Send an email, if the caller already has the user's confirmation.

    The confirmation gate is on ``confirmed``, not on a prompt read here: a
    tool cannot ask the user a question the conversation has not already
    answered. The Brain sets ``confirmed`` only after the user has said yes in
    their own turn, which is also the only place the answer is available.

    Args:
        to_address: Recipient.
        subject: Subject line.
        content: Message body.
        confirmed: True only when the user explicitly approved this send.

    Returns:
        A short status string. Never raises.
    """
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        logger.warning("send_email called without GMAIL_ADDRESS/GMAIL_APP_PASSWORD set.")
        return "Email is not configured: GMAIL_ADDRESS and GMAIL_APP_PASSWORD are missing from .env."

    recipient = (to_address or "").strip()
    if not _EMAIL_RE.match(recipient):
        logger.warning(f"Refusing to send to a malformed address: {to_address!r}")
        return f"'{to_address}' is not a valid email address, so nothing was sent."

    if not confirmed:
        logger.info(f"Email to {recipient} NOT sent: no explicit confirmation.")
        return ("This email was not sent. Ask the user to confirm first, then "
                "call send_email again with confirmed=true.")

    logger.info(f"Sending email: recipient={recipient} subject={subject!r}")
    try:
        msg = MIMEMultipart()
        msg["From"] = GMAIL_ADDRESS
        msg["To"] = recipient
        msg["Subject"] = subject or ""
        msg.attach(MIMEText(content or "", "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=15)
        try:
            server.ehlo()
            server.starttls()
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.send_message(msg)
        finally:
            server.quit()

        logger.info(f"Email sent: recipient={recipient} subject={subject!r}")
        return f"Email sent to {recipient}."
    except Exception as e:  # noqa: BLE001
        detail = _safe(str(e))
        # No exc_info: the traceback would carry the unredacted exception text.
        logger.error(f"Email send failed for {recipient}: {detail}")
        return f"I could not send the email to {recipient}: {detail}"
