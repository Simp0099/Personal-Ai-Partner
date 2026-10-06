"""JARVIS 2.0 Long-Term Memory System.

Provides persistent storage for user facts, preferences, and notes across sessions.
Uses SQLite for lightweight, zero-config local storage.

Schema:
    memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fact TEXT NOT NULL,
        category TEXT DEFAULT 'general',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )

Phase 8: All errors caught and logged gracefully.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from jarvis.config import DATA_DIR
from jarvis.logger import logger

# Ensure the data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

MEMORY_DB_PATH = DATA_DIR / "jarvis_memory.db"


def init_memory_db(db_path: Path = None) -> sqlite3.Connection:
    """Initialize the memory database and create tables if they don't exist.

    Args:
        db_path: Optional custom path for the database file.

    Returns:
        An open SQLite connection with row factory set.
    """
    path = db_path or MEMORY_DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def remember(conn: sqlite3.Connection, fact: str, category: str = "general") -> bool:
    """Save a fact to long-term memory.

    Args:
        conn: Active database connection.
        fact: The fact or note to store.
        category: Optional category for grouping (e.g. 'preference', 'fact', 'note').

    Returns:
        True if the fact was saved successfully.
    """
    try:
        conn.execute(
            "INSERT INTO memories (fact, category) VALUES (?, ?)",
            (fact.strip(), category.strip().lower())
        )
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to save fact: {e}", exc_info=True)
        return False


def recall_all(conn: sqlite3.Connection, limit: int = 50) -> List[str]:
    """Retrieve all stored facts, most recent first.

    Args:
        conn: Active database connection.
        limit: Maximum number of facts to return.

    Returns:
        List of fact strings.
    """
    try:
        cursor = conn.execute(
            "SELECT fact FROM memories ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        return [row["fact"] for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to recall facts: {e}", exc_info=True)
        return []


def recall_by_category(conn: sqlite3.Connection, category: str) -> List[str]:
    """Retrieve facts filtered by category.

    Args:
        conn: Active database connection.
        category: The category to filter by.

    Returns:
        List of matching fact strings.
    """
    try:
        cursor = conn.execute(
            "SELECT fact FROM memories WHERE category = ? ORDER BY created_at DESC",
            (category.lower(),)
        )
        return [row["fact"] for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to recall category '{category}': {e}", exc_info=True)
        return []


def search_memories(conn: sqlite3.Connection, query: str) -> List[str]:
    """Search facts containing the query string (case-insensitive).

    Args:
        conn: Active database connection.
        query: Search term to look for in stored facts.

    Returns:
        List of matching fact strings.
    """
    try:
        cursor = conn.execute(
            "SELECT fact FROM memories WHERE LOWER(fact) LIKE ? ORDER BY created_at DESC",
            (f"%{query.lower()}%",)
        )
        return [row["fact"] for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Memory search failed: {e}", exc_info=True)
        return []


def forget(conn: sqlite3.Connection, fact_id: int) -> bool:
    """Delete a specific fact by its ID.

    Args:
        conn: Active database connection.
        fact_id: The ID of the fact to delete.

    Returns:
        True if the fact was deleted.
    """
    try:
        cursor = conn.execute("DELETE FROM memories WHERE id = ?", (fact_id,))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Failed to delete fact: {e}", exc_info=True)
        return False


def clear_all_memories(conn: sqlite3.Connection) -> bool:
    """Delete all stored memories. Use with caution.

    Args:
        conn: Active database connection.

    Returns:
        True if all memories were cleared.
    """
    try:
        conn.execute("DELETE FROM memories")
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to clear memories: {e}", exc_info=True)
        return False


def get_memory_count(conn: sqlite3.Connection) -> int:
    """Get the total number of stored facts.

    Args:
        conn: Active database connection.

    Returns:
        Count of stored facts.
    """
    try:
        cursor = conn.execute("SELECT COUNT(*) as count FROM memories")
        return cursor.fetchone()["count"]
    except Exception:
        return 0


def format_facts_for_prompt(facts: List[str]) -> str:
    """Format a list of facts for injection into the system prompt.

    Args:
        facts: List of fact strings.

    Returns:
        A formatted string ready to append to the system prompt.
    """
    if not facts:
        return ""
    lines = ["\n\nThings you know about the user (from long-term memory):"]
    for i, fact in enumerate(facts, 1):
        lines.append(f"  {i}. {fact}")
    return "\n".join(lines)
