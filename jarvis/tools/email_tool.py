"""Secure Email tool for JARVIS 2.0.

Uses smtplib with TLS and loads credentials strictly from environment variables.
Never stores or logs plain text passwords.

Phase 8: All errors caught and logged gracefully.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from jarvis.speech import speak, listen
from jarvis.config import GMAIL_ADDRESS, GMAIL_APP_PASSWORD, CONTACTS
from jarvis.logger import logger

# Explicit affirmative replies. Anything else -- no, cancel, stop, ambiguous,
# empty, "none" (non-interactive) -- is unconfirmed and must NOT send.
AFFIRMATIVE = frozenset({
    "yes", "y", "yeah", "yep", "yes please", "send it", "do it",
    "confirm", "confirmed", "go ahead", "send",
})


def parse_confirmation(reply) -> bool:
    """True only for a clearly affirmative confirmation. Strict by design."""
    text = " ".join(str(reply or "").strip().lower().split()).rstrip("?!.")
    return (text in AFFIRMATIVE or text.startswith("yes ")
            or text.startswith("yes,"))


def send_email(to_address: str, subject: str, content: str) -> bool:
    """Send an email using SMTP over TLS with credentials loaded from environment.

    Safety gate: explicit user confirmation is required BEFORE the irreversible
    SMTP dispatch, on every path that reaches this function (tool registry,
    LLM tool loop, interactive handler). Unconfirmed means unsent.
    Recipient and subject may be logged; the body never is.
    """
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        speak("Gmail credentials are not configured in your .env file.")
        logger.warning("Missing GMAIL_ADDRESS or GMAIL_APP_PASSWORD in .env.")
        return False

    logger.info(f"Email send requested: recipient={to_address} subject={subject!r}")
    speak(f"Send this email to {to_address}? (yes/no)")
    if not parse_confirmation(listen("Answer (yes/no): ")):
        logger.info(f"Email send unconfirmed: recipient={to_address} subject={subject!r}")
        speak("Email not sent.")
        return False

    try:
        msg = MIMEMultipart()
        msg["From"] = GMAIL_ADDRESS
        msg["To"] = to_address
        msg["Subject"] = subject
        msg.attach(MIMEText(content, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=15)
        server.ehlo()
        server.starttls()
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.send_message(msg)
        server.quit()

        logger.info(f"Email sent: recipient={to_address} subject={subject!r}")
        speak(f"Email successfully sent to {to_address}!")
        return True
    except Exception as e:
        logger.error(f"Email error for {to_address}: {e}", exc_info=True)
        speak("Sorry, I was unable to send the email.")
        return False


def handle_email_command(query: str) -> None:
    """Interactive command handler for email sending."""
    target_address = None

    # Check known contact aliases from config.yaml
    for contact_name, email in CONTACTS.items():
        if contact_name.lower() in query.lower():
            target_address = email
            speak(f"Found contact {contact_name}.")
            break

    if not target_address:
        speak("Whom should I send the email to? Please speak the email address.")
        spoken_address = listen()
        if spoken_address == "none":
            speak("Email canceled.")
            return
        target_address = spoken_address.replace(" at the rate ", "@").replace(" at ", "@").replace(" ", "").lower()

    speak("What is the message content?")
    content = listen()
    if content == "none":
        speak("Email canceled.")
        return

    subject = "Message from JARVIS Assistant"
    send_email(target_address, subject, content)
