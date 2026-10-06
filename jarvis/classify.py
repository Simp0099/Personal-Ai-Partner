"""Deterministic task classification.

Purpose: tell the router *what kind of request this is* so it can pick a model
with the right capabilities. Deliberately not an LLM call -- routing has to be
fast, free, and predictable, and a model call to decide which model to call
would be circular.

Every signal here is a keyword, pattern, or request-metadata check, so
classification is instantaneous and unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class TaskType(str, Enum):
    """Request categories the router scores against."""

    CASUAL = "casual_conversation"
    SIMPLE_QUESTION = "simple_question"
    REASONING = "reasoning"
    CODING = "coding"
    LONG_CONTEXT = "long_context"
    TOOL_USE = "tool_use"
    STRUCTURED_OUTPUT = "structured_output"
    CREATIVE = "creative"
    RESEARCH = "research"


@dataclass
class Classification:
    """The router's read of one request."""

    task_type: TaskType = TaskType.CASUAL
    #: Secondary types also plausibly apply, with lower weight.
    secondary: List[TaskType] = field(default_factory=list)
    tool_required: bool = False
    estimated_tokens: int = 0
    #: Signals that fired, for diagnostics and debugging.
    signals: List[str] = field(default_factory=list)

    @property
    def all_types(self) -> List[TaskType]:
        return [self.task_type, *self.secondary]


# --- Keyword sets ----------------------------------------------------------
# Kept explicit and reviewable rather than clever.

_CODING = {
    "code", "function", "bug", "debug", "python", "javascript", "typescript",
    "java", "c++", "rust", "sql", "html", "css", "api", "regex", "compile",
    "error", "stacktrace", "traceback", "refactor", "algorithm", "script",
    "class", "method", "variable", "loop", "array", "dict", "json", "yaml",
    "repo", "git", "deploy", "docker", "kubernetes", "database", "query",
}

_REASONING = {
    "why", "explain", "reason", "because", "prove", "proof", "therefore",
    "logic", "deduce", "infer", "implication", "compare", "contrast",
    "trade-off", "tradeoff", "difference between", "pros and cons", "analyze",
    "analyse", "evaluate", "argue", "critique", "root cause", "solve",
    "calculate", "compute", "derive", "strategy", "plan out",
}

_RESEARCH = {
    "research", "find out", "look up", "search", "investigate", "compare",
    "review", "summary of", "overview of", "history of", "latest news",
    "current", "recent", "market", "competitor", "paper", "study",
}

_CREATIVE = {
    "write", "story", "poem", "song", "essay", "blog", "joke", "fiction",
    "character", "dialogue", "imagine", "brainstorm", "name ideas", "slogan",
    "tagline", "haiku", "script for", "screenplay", "lyrics",
}

_STRUCTURED = {
    "json", "yaml", "xml", "csv", "table", "markdown table", "bullet list",
    "numbered list", "schema", "format as", "output as", "structured",
    "list of", "return only", "answer only with", "exactly", "template",
}

_TOOL = {
    "weather", "temperature", "search", "google", "youtube", "wikipedia",
    "screenshot", "play", "music", "email", "send", "open", "launch",
    "nasa", "apod", "directory", "files", "system status", "look up",
    "browse", "download", "install", "run", "execute", "check my",
    "my computer", "my system", "current time", "what time",
}

_CASUAL = {
    "hi", "hello", "hey", "yo", "howdy", "sup", "bye", "goodbye",
    "how are you", "how's it going", "whats up", "what's up", "thank you",
    "thanks", "good morning", "good evening", "good afternoon", "nice to meet",
}

_QUESTION_RE = re.compile(r"\?\s*$")
_CODE_FENCE_RE = re.compile(r"```")
_URL_RE = re.compile(r"https?://")


def _contains_any(text: str, words: Set[str]) -> List[str]:
    """Whole-word (or phrase) matches of `words` in `text`."""
    hits: List[str] = []
    for word in words:
        pattern = r"\b" + re.escape(word).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, text):
            hits.append(word)
    return hits


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token)."""
    return len(text) // 4


def classify(
    user_message: str,
    *,
    tools_available: bool = True,
    conversation_tokens: int = 0,
    context_threshold: int = 100_000,
) -> Classification:
    """Classify a user request.

    Args:
        user_message: The user's message.
        tools_available: Whether any tools are offered at all. A request is
            only ``tool_required`` when the model actually has tools, otherwise
            the router would exclude every model for no reason.
        conversation_tokens: Existing conversation size, used to detect
            long-context work.
        context_threshold: Token count above which a task counts as long-context.

    Returns:
        A :class:`Classification`.
    """
    text = (user_message or "").strip()
    lowered = text.lower()
    result = Classification()

    if not text:
        return result

    tokens = estimate_tokens(text)
    result.estimated_tokens = tokens

    def add(task: TaskType, signal: str) -> None:
        result.signals.append(signal)
        if task == result.task_type:
            return
        if result.task_type is TaskType.CASUAL and not result.signals[:-1]:
            result.task_type = task
        elif task not in result.secondary:
            result.secondary.append(task)

    # -- tool requirement --------------------------------------------------
    tool_hits = _contains_any(lowered, _TOOL)
    # Imperative phrasing about the user's own machine/accounts needs a tool.
    if tools_available and (tool_hits or _URL_RE.search(text)):
        result.tool_required = True
        result.signals.append(f"tool:{','.join(tool_hits[:3])}" if tool_hits else "tool:url")
        if TaskType.TOOL_USE not in result.secondary:
            result.secondary.append(TaskType.TOOL_USE)

    # -- long context ------------------------------------------------------
    if conversation_tokens + tokens > context_threshold:
        add(TaskType.LONG_CONTEXT, "long_context:conversation_size")

    # -- coding ------------------------------------------------------------
    # Checked before structured output: "write a Python function that returns
    # JSON" is primarily a coding task that happens to mention JSON, and
    # routing it as a structured-output task would favour the wrong models.
    coding_hits = _contains_any(lowered, _CODING)
    code_fence = bool(_CODE_FENCE_RE.search(text))
    if coding_hits or code_fence:
        add(TaskType.CODING, f"coding:{','.join(coding_hits[:3])}" if coding_hits else "coding:fence")

    # -- structured output -------------------------------------------------
    struct_hits = _contains_any(lowered, _STRUCTURED)
    if struct_hits and not code_fence:
        add(TaskType.STRUCTURED_OUTPUT, f"structured:{','.join(struct_hits[:3])}")

    # -- reasoning ---------------------------------------------------------
    reasoning_hits = _contains_any(lowered, _REASONING)
    if reasoning_hits or _QUESTION_RE.search(text):
        add(TaskType.REASONING, f"reasoning:{','.join(reasoning_hits[:3])}")

    # -- research ----------------------------------------------------------
    research_hits = _contains_any(lowered, _RESEARCH)
    if research_hits:
        add(TaskType.RESEARCH, f"research:{','.join(research_hits[:3])}")

    # -- creative ----------------------------------------------------------
    creative_hits = _contains_any(lowered, _CREATIVE)
    if creative_hits:
        add(TaskType.CREATIVE, f"creative:{','.join(creative_hits[:3])}")

    # -- casual vs simple question ----------------------------------------
    casual_hits = _contains_any(lowered, _CASUAL)
    if casual_hits and len(text.split()) <= 8:
        result.signals.append(f"casual:{','.join(casual_hits[:2])}")
        if result.task_type is TaskType.CASUAL and not result.secondary:
            result.task_type = TaskType.CASUAL
    elif _QUESTION_RE.search(text) and len(text.split()) <= 12:
        if result.task_type is TaskType.CASUAL:
            result.task_type = TaskType.SIMPLE_QUESTION
            result.signals.append("simple_question:short_form")

    return result


#: Task types a model is scored on, used by tests and the status endpoint.
ALL_TASK_TYPES = [t.value for t in TaskType]