"""Safe local application launching (Phase 3).

Allowlisted macOS apps only, via argument-list ``subprocess`` — never a
shell. The app name is DATA: it is resolved against ``ALLOWED_APPS`` first,
and only the resolved allowlist entry ever reaches the OS. Anything else
(not allowlisted, empty, uninstalled) is a controlled local message.
"""

import platform
import subprocess

from jarvis.config import ALLOWED_APPS
from jarvis.logger import logger


def normalize_app_name(name) -> str:
    """Harmless differences only: surrounding whitespace + case."""
    if not isinstance(name, str):
        return ""
    return " ".join(name.split())


def resolve_allowed(name: str):
    """Canonical allowlist entry for `name`, or None. Exact match, casefolded."""
    wanted = normalize_app_name(name).casefold()
    if not wanted:
        return None
    for entry in ALLOWED_APPS or []:
        if normalize_app_name(entry).casefold() == wanted:
            return entry
    return None


def launch_app(app_name: str) -> str:
    """Launch an allowlisted macOS application. Never raises to the caller.

    Returns text and stays silent: the conversation layer speaks the reply.
    """
    requested = normalize_app_name(app_name)
    if not requested:
        return "I need an application name to open. For example: Open Safari."

    resolved = resolve_allowed(requested)
    if resolved is None:
        logger.info(f"App launch rejected (not allowlisted): {requested!r}")
        return f"{requested} isn't an approved application, so I won't open it."

    if platform.system() != "Darwin":
        logger.warning(f"App launch unsupported on {platform.system()}: {resolved!r}")
        return f"I can only open applications on macOS. {resolved} was not launched."

    argv = ["open", "-a", resolved]
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        logger.error("App launch failed: macOS 'open' command not found.")
        return "The macOS launcher isn't available, so nothing was opened."
    except subprocess.TimeoutExpired:
        logger.error(f"App launch timed out: {resolved!r}")
        return f"Opening {resolved} timed out."
    except Exception as e:  # noqa: BLE001 -- controlled local failure, logged
        logger.error(f"App launch failed for {resolved!r}: {e}", exc_info=True)
        return f"I couldn't open {resolved}."

    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()
        logger.error(f"App launch failed for {resolved!r}: rc={completed.returncode} {detail}")
        return f"{resolved} isn't installed on this Mac, so I couldn't open it."

    logger.info(f"App launch succeeded: {resolved!r}")
    return f"{resolved} is now open."
