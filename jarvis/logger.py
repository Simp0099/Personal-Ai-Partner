"""JARVIS 2.0 Centralized Logging with Rotating File Handler.

Configures Python's logging module to output to both console and a rotating
log file (jarvis.log). All modules should import `logger` from here instead
of using print() statements.

Usage:
    from jarvis.logger import logger
    logger.info("Message")
    logger.error("Error occurred", exc_info=True)
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _get_config_value(key: str, default=None):
    """Lazily fetch a value from jarvis.config to avoid circular imports."""
    try:
        from jarvis import config
        return getattr(config, key, default)
    except Exception:
        return default


# Console verbosity for the "jarvis" logger. Normal mode keeps the terminal
# to user-facing print() output only; diagnostics always go to the rotating
# file. --debug lowers the console handler to DEBUG via configure_logging().
# Nothing in the codebase logs above ERROR, so CRITICAL silences the console
# without detaching the handler (keeps handler identity stable for tests).
_CONSOLE_QUIET_LEVEL = logging.CRITICAL
_CONSOLE_DEBUG_LEVEL = logging.DEBUG

_DEBUG = False


def debug_mode() -> bool:
    """True when --debug opened console diagnostics for this process."""
    return _DEBUG


class _LiveStdout:
    """Stream proxy that resolves sys.stdout on every write.

    The console handler is created once per process, but test runners and
    hosts may swap sys.stdout (pytest capture, daemonization). A proxy keeps
    console diagnostics following the live stdout without rebinding, and can
    never dangle on a closed capture object.
    """

    def write(self, data):
        return sys.stdout.write(data)

    def flush(self):
        try:
            sys.stdout.flush()
        except Exception:
            pass


def _console_handler(logger: logging.Logger):
    """The stdout StreamHandler, or None. File handlers never match."""
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, RotatingFileHandler
        ):
            return handler
    return None


def _file_handler(logger: logging.Logger):
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            return handler
    return None


def _formatter() -> logging.Formatter:
    return logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _setup_logger() -> logging.Logger:
    """Configure and return the JARVIS logger with rotating file handler.

    Returns:
        Configured logger instance with console and file handlers.
    """
    # Lazy import config values to avoid circular import
    data_dir = _get_config_value("DATA_DIR", Path(__file__).resolve().parent.parent / "data")
    log_file = _get_config_value("LOG_FILE", data_dir / "jarvis.log")
    log_max_bytes = _get_config_value("LOG_MAX_BYTES", 1_000_000)
    log_backup_count = _get_config_value("LOG_BACKUP_COUNT", 3)

    logger = logging.getLogger("jarvis")
    # Capture everything; the handlers gate where it surfaces. The file
    # always receives DEBUG+; the console level is owned by configure_logging().
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers if module is reloaded
    if logger.handlers:
        return logger

    # Console handler: quiet by default (normal mode shows user-facing
    # print() output only). configure_logging(debug=True) opens it.
    console_handler = logging.StreamHandler(_LiveStdout())
    console_handler.setLevel(_CONSOLE_QUIET_LEVEL)
    console_handler.setFormatter(_formatter())
    logger.addHandler(console_handler)

    # Rotating file handler
    try:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=log_max_bytes,
            backupCount=log_backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(_formatter())
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not create log file: {e}")

    return logger


def configure_logging(debug: bool = False, log_file=None) -> logging.Logger:
    """Set console verbosity; never duplicates handlers.

    Normal mode (debug=False): console stays silent, file keeps DEBUG+.
    Debug mode (debug=True): console shows DEBUG+ alongside the file.
    Optional log_file redirects the file handler (tests use a tmp path).
    Idempotent: repeated calls reuse the same two handlers.
    """
    global _DEBUG
    _DEBUG = bool(debug)

    logger = logging.getLogger("jarvis")
    logger.setLevel(logging.DEBUG)

    console = _console_handler(logger)
    if console is None:
        console = logging.StreamHandler(_LiveStdout())
        console.setFormatter(_formatter())
        logger.addHandler(console)
    console.setLevel(_CONSOLE_DEBUG_LEVEL if _DEBUG else _CONSOLE_QUIET_LEVEL)

    if log_file is not None:
        target = str(log_file)
        current = _file_handler(logger)
        if current is not None and getattr(current, "baseFilename", None) == target:
            pass
        else:
            if current is not None:
                logger.removeHandler(current)
                try:
                    current.close()
                except Exception:
                    pass
            path = Path(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            replacement = RotatingFileHandler(
                target, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
            )
            replacement.setLevel(logging.DEBUG)
            replacement.setFormatter(_formatter())
            logger.addHandler(replacement)
    elif _file_handler(logger) is None:
        _setup_logger()

    return logger


# Global logger instance
logger = _setup_logger()


class StatusIndicator:
    """Lightweight CLI status visualizer.

    Prints clear state messages during runtime to make interaction seamless.

    Usage:
        status = StatusIndicator()
        status.listening()
        status.thinking()
        status.speaking()
        status.tool_call("search_google", {"query": "Python"})
    """

    # Status symbols for quick visual scanning
    SYMBOLS = {
        "listening": "[Listening...]",
        "thinking": "[Thinking, Boss...]",
        "speaking": "[Speaking...]",
        "tool": "[Tool]",
        "wake": "[Wake Word]",
        "error": "[Error]",
        "info": "[Info]",
        "memory": "[Memory]",
        "ready": "[Ready]",
        "shutdown": "[Shutdown]",
    }

    @classmethod
    def listening(cls):
        """Indicate JARVIS is listening for user input."""
        logger.info(f"{cls.SYMBOLS['listening']} Waiting for user input...")

    @classmethod
    def thinking(cls):
        """Indicate JARVIS is processing with Gemini."""
        logger.info(f"{cls.SYMBOLS['thinking']} Processing with Gemini...")

    @classmethod
    def speaking(cls):
        """Indicate JARVIS is speaking a response."""
        logger.info(f"{cls.SYMBOLS['speaking']} Generating audio output...")

    @classmethod
    def tool_call(cls, name: str, args: dict):
        """Indicate a tool is being executed."""
        logger.info(f"{cls.SYMBOLS['tool']} Executing: {name}({args})")

    @classmethod
    def tool_result(cls, name: str, result: str):
        """Indicate a tool has returned a result."""
        logger.debug(f"{cls.SYMBOLS['tool']} {name} returned: {result[:100]}")

    @classmethod
    def wake_detected(cls, confidence: float):
        """Indicate wake word was detected."""
        logger.info(f"{cls.SYMBOLS['wake']} Detected! (confidence: {confidence:.2f})")

    @classmethod
    def error(cls, message: str):
        """Indicate an error occurred."""
        logger.error(f"{cls.SYMBOLS['error']} {message}")

    @classmethod
    def info(cls, message: str):
        """General info message."""
        logger.info(f"{cls.SYMBOLS['info']} {message}")

    @classmethod
    def memory(cls, action: str):
        """Indicate memory operation."""
        logger.info(f"{cls.SYMBOLS['memory']} {action}")

    @classmethod
    def ready(cls):
        """Indicate JARVIS is ready."""
        logger.info(f"{cls.SYMBOLS['ready']} System initialized and ready.")

    @classmethod
    def shutdown(cls):
        """Indicate JARVIS is shutting down."""
        logger.info(f"{cls.SYMBOLS['shutdown']} Going offline.")
