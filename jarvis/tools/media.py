"""Media tools for JARVIS 2.0 (Screenshots, Music, Alarm).

Cross-platform implementations using pathlib and native OS openers.

Phase 8: All errors caught and logged gracefully.
"""

import sys
import subprocess
import datetime
from jarvis.speech import speak, listen
from jarvis.config import SCREENSHOT_DIR
from jarvis.logger import logger


def open_file_or_directory(path) -> None:
    """Open a file or directory using the native operating system viewer."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(path)], check=False)
        elif sys.platform == "win32":
            import os
            os.startfile(str(path))
    except Exception as e:
        logger.error(f"Open error: {e}", exc_info=True)


def take_screenshot(filename: str = None) -> str:
    """Capture a screenshot and save it to the project's data directory."""
    try:
        import pyautogui

        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
        elif not filename.endswith(".png"):
            filename = f"{filename}.png"

        save_path = SCREENSHOT_DIR / filename
        image = pyautogui.screenshot()
        image.save(str(save_path))

        speak("Screenshot captured and saved.")
        open_file_or_directory(save_path)
        return str(save_path)
    except Exception as e:
        logger.error(f"Screenshot error: {e}", exc_info=True)
        speak("Unable to capture screenshot on this display.")
        return ""


def handle_screenshot_command() -> None:
    """Interactive screenshot command handler."""
    speak("What should I name the screenshot? Or say default.")
    name = listen()
    if name == "none" or "default" in name.lower():
        take_screenshot()
    else:
        clean_name = name.replace(" ", "_")
        take_screenshot(clean_name)


def play_music(song_name: str = None) -> None:
    """Play a song using YouTube playback."""
    target_song = song_name
    if not target_song:
        speak("Which song would you like to hear?")
        target_song = listen()
        if target_song == "none":
            speak("Playback canceled.")
            return

    try:
        import pywhatkit
        speak(f"Playing {target_song} now.")
        pywhatkit.playonyt(target_song)
    except Exception as e:
        logger.error(f"Music fallback error: {e}", exc_info=True)
        import webbrowser
        import urllib.parse
        encoded = urllib.parse.quote_plus(target_song)
        webbrowser.open(f"https://www.youtube.com/results?search_query={encoded}")
        speak(f"Opened YouTube search for {target_song}.")
