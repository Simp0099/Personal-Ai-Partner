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
    log_level = _get_config_value("LOG_LEVEL", "INFO")

    logger = logging.getLogger("jarvis")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Prevent duplicate handlers if module is reloaded
    if logger.handlers:
        return logger

    # Format
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
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
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not create log file: {e}")

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
