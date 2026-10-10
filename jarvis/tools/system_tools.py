"""System information tools for JARVIS 2.0.

Provides local system introspection: time/date, directory contents, and
resource status (CPU, memory, disk). These are the Phase 2 verification
tools that confirm the Gemini tool-calling loop works end-to-end.
"""

import os
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from jarvis.config import ALLOWED_FILE_ROOTS


def get_system_time() -> str:
    """Get the current system date and time.

    Returns:
        A human-readable string with the current date and time.
    """
    now = datetime.now()
    date_str = now.strftime("%A, %B %d, %Y")
    time_str = now.strftime("%I:%M:%S %p")
    result = f"Today is {date_str}. The current time is {time_str}."
    return result


def get_directory_contents(directory: str = ".") -> str:
    """List the contents of a local directory.

    Args:
        directory: Path to the directory to list. Defaults to the current directory.

    Returns:
        A formatted string listing files and subdirectories.
    """
    try:
        dir_path = Path(directory).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        return f"Directory '{directory}' does not exist or cannot be resolved."

    if not any(dir_path == root or root in dir_path.parents for root in ALLOWED_FILE_ROOTS):
        return f"Access denied: '{directory}' is outside the allowed directories."

    if not dir_path.exists():
        return f"Directory '{directory}' does not exist."

    if not dir_path.is_dir():
        return f"'{directory}' is not a directory."

    try:
        entries = sorted(dir_path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    except PermissionError:
        return f"Permission denied accessing '{directory}'."

    if not entries:
        return f"The directory '{dir_path.name}' is empty."

    files = []
    dirs = []
    for entry in entries:
        if entry.name.startswith("."):
            continue  # Skip hidden files for brevity
        try:
            target = entry.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if not any(target == root or root in target.parents for root in ALLOWED_FILE_ROOTS):
            continue
        if entry.is_dir():
            dirs.append(f"[DIR]  {entry.name}/")
        else:
            try:
                size = entry.stat().st_size
            except OSError:
                continue
            size_str = _format_size(size)
            files.append(f"[FILE] {entry.name} ({size_str})")

    lines = [f"Contents of {dir_path}:"]
    lines.extend(dirs)
    lines.extend(files)
    lines.append(f"\nTotal: {len(dirs)} directories, {len(files)} files.")

    return "\n".join(lines)


def get_system_status() -> str:
    """Check local system status: OS, CPU, memory, and disk usage.

    Returns:
        A formatted string with key system metrics.
    """
    lines = []

    # OS Info
    lines.append(f"OS: {platform.system()} {platform.release()}")
    lines.append(f"Machine: {platform.machine()}")
    lines.append(f"Processor: {platform.processor() or 'Unknown'}")

    # CPU
    try:
        cpu_count = os.cpu_count()
        lines.append(f"CPU Cores: {cpu_count}")
    except Exception:
        pass

    # Memory (cross-platform)
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines.append(f"Memory: {result.stdout.strip().split(chr(10))[0]}")
        elif platform.system() == "Linux":
            result = subprocess.run(
                ["free", "-h"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                mem_line = result.stdout.strip().split("\n")[1]
                lines.append(f"Memory: {mem_line}")
    except Exception:
        pass

    # Disk usage for current directory
    try:
        total, used, free = shutil.disk_usage(".")
        total_gb = total / (1024**3)
        used_gb = used / (1024**3)
        free_gb = free / (1024**3)
        percent_used = (used / total) * 100
        lines.append(
            f"Disk: {used_gb:.1f} GB used / {total_gb:.1f} GB total "
            f"({percent_used:.1f}% used, {free_gb:.1f} GB free)"
        )
    except Exception:
        pass

    # Uptime
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["uptime"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines.append(f"Uptime: {result.stdout.strip()}")
    except Exception:
        pass

    return "\n".join(lines)


def _format_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"
