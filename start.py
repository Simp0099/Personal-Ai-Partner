#!/usr/bin/env python3
"""JARVIS 2.0 — Unified Startup Entrypoint (Phase 9).

Initializes logging, loads configs, checks environment keys,
boots the background wake-word listener, and brings Kyuoko Hori fully online.

Usage:
    python3 start.py              # Wake word mode (default)
    python3 start.py --text       # Text mode (skip wake word)
    python3 start.py --test       # Run integration tests
    python3 start.py --no-wake    # Voice mode without wake word
    python3 start.py --status     # Show system status
"""

import sys
import os
import argparse
import platform
from pathlib import Path

# Ensure project root is on Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_environment():
    """Verify all required environment variables and dependencies are present."""
    from jarvis.config import GEMINI_API_KEY, LLM_MODEL, ASSISTANT_NAME
    from jarvis.logger import logger

    print(f"\n{'='*60}")
    print(f"  JARVIS 2.0 — System Startup")
    print(f"{'='*60}\n")

    # Check Python version
    py_version = platform.python_version()
    print(f"[Check] Python version: {py_version}")
    if tuple(map(int, py_version.split(".")[:2])) < (3, 10):
        print(f"  WARNING: Python 3.10+ recommended")
    else:
        print(f"  OK")

    # Check .env file
    env_file = PROJECT_ROOT / ".env"
    print(f"\n[Check] .env file: {'Found' if env_file.exists() else 'NOT FOUND'}")
    if not env_file.exists():
        print(f"  ERROR: Copy .env.example to .env and fill in your keys")
        return False

    # Check GEMINI_API_KEY
    print(f"\n[Check] GEMINI_API_KEY: {'Configured' if GEMINI_API_KEY else 'MISSING'}")
    if not GEMINI_API_KEY:
        print(f"  ERROR: Add GEMINI_API_KEY to .env file")
        return False

    # Check config.yaml
    config_file = PROJECT_ROOT / "config.yaml"
    print(f"[Check] config.yaml: {'Found' if config_file.exists() else 'NOT FOUND'}")

    # Check data directory
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Check] data directory: {data_dir}")

    # Check log file
    log_file = data_dir / "jarvis.log"
    print(f"[Check] log file: {log_file}")

    # Check wake word model
    try:
        import openwakeword
        model_info = openwakeword.MODELS.get("hey_jarvis")
        if model_info:
            model_path = model_info["model_path"].replace(".tflite", ".onnx")
            model_exists = os.path.exists(model_path)
            print(f"[Check] Wake word model: {'Found' if model_exists else 'NOT FOUND'}")
            if not model_exists:
                print(f"  Note: Run 'python3 -m jarvis.main --test' to download models")
    except ImportError:
        print(f"[Check] openwakeword: Not installed")

    # Check Chatterbox TTS (active voice engine) and its reference audio
    try:
        from jarvis.speech import _resolve_chatterbox_reference
        _resolve_chatterbox_reference()
        print(f"[Check] Chatterbox TTS: reference audio found")
    except Exception as e:
        print(f"[Check] Chatterbox TTS: {e}")

    try:
        import chatterbox  # noqa: F401
        print(f"[Check] Chatterbox package: Installed")
    except ImportError:
        print(f"[Check] Chatterbox package: Not installed (pip install chatterbox-tts)")

    # System info
    print(f"\n[Info] OS: {platform.system()} {platform.release()}")
    print(f"[Info] Machine: {platform.machine()}")
    print(f"[Info] Assistant: {ASSISTANT_NAME}")
    print(f"[Info] LLM Model: {LLM_MODEL}")

    print(f"\n{'='*60}")
    print(f"  All checks passed. Starting {ASSISTANT_NAME}...")
    print(f"{'='*60}\n")

    return True


def show_status():
    """Display current system status."""
    from jarvis.config import (
        ASSISTANT_NAME, GREETING_NAME, LLM_MODEL,
        WAKE_WORD_MODEL, WAKE_WORD_THRESHOLD,
        MAX_HISTORY_MESSAGES, TTS_ENGINE,
    )
    from jarvis.logger import logger

    print(f"\n{'='*60}")
    print(f"  JARVIS 2.0 — System Status")
    print(f"{'='*60}\n")

    print(f"  Assistant:     {ASSISTANT_NAME}")
    print(f"  User:          {GREETING_NAME}")
    print(f"  LLM Model:     {LLM_MODEL}")
    print(f"  TTS Engine:    {TTS_ENGINE}")
    print(f"  Wake Word:     {WAKE_WORD_MODEL} (threshold: {WAKE_WORD_THRESHOLD})")
    print(f"  Max History:   {MAX_HISTORY_MESSAGES} messages")

    # Memory stats
    try:
        from jarvis import memory
        conn = memory.init_memory_db()
        count = memory.get_memory_count(conn)
        print(f"  Long-term Mem: {count} facts stored")
        conn.close()
    except Exception:
        print(f"  Long-term Mem: Not available")

    # Log file
    from jarvis.config import LOG_FILE
    if LOG_FILE.exists():
        size = LOG_FILE.stat().st_size
        print(f"  Log File:      {LOG_FILE} ({size} bytes)")
    else:
        print(f"  Log File:      Not yet created")

    print(f"\n{'='*60}\n")


def main():
    """Parse arguments and start JARVIS."""
    parser = argparse.ArgumentParser(description="JARVIS 2.0 — AI Assistant")
    parser.add_argument("--text", action="store_true", help="Text mode (skip wake word)")
    parser.add_argument("--test", action="store_true", help="Run integration tests")
    parser.add_argument("--no-wake", action="store_true", help="Voice mode without wake word")
    parser.add_argument("--status", action="store_true", help="Show system status")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.test:
        from tests.test_integration import run_all_tests
        run_all_tests()
        return

    # Check environment before starting
    if not check_environment():
        print("\nStartup aborted. Please fix the issues above.\n")
        sys.exit(1)

    # Start the appropriate mode
    if args.text:
        from jarvis.main import run_text_mode
        run_text_mode()
    elif args.no_wake:
        from jarvis.main import run_assistant
        try:
            run_assistant()
        except KeyboardInterrupt:
            print("\nSession ended by user.")
            sys.exit(0)
    else:
        from jarvis.main import run_wake_word_mode
        try:
            run_wake_word_mode()
        except KeyboardInterrupt:
            print("\nSession ended by user.")
            sys.exit(0)


if __name__ == "__main__":
    main()
