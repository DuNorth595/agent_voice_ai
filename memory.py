#!/usr/bin/env python3
"""
memory.py — placeholder memory layer for the voice bridge.

This module is intentionally **minimal and generic**. It defines the
public symbols (`build_memory_context`, `SOUL_PATH`, `SAM_INDEX_PATH`,
`FACT_STORE_DB`, `LONG_TERM_MEMORY_DIR`) that the bridge expects to import, but
it does not hardcode any personal identity, family names, or project
folders. Each path is configurable via environment variables.

For a useful voice agent, fork this module and inject your own memory
sources. Two common patterns:

1. **Personality layer.** Load a markdown file describing the agent's
   voice, opinions, and identity. Read it from `MEMORY_SOUL_PATH` and
   include it in the system prompt. See the placeholder below for the
   minimal version.

2. **Long-term context layer.** Load an index of facts about the user
   (people, projects, recurring topics) from `MEMORY_INDEX_PATH`. When
   a user query hits a relevant keyword, pull the matching file under
   `MEMORY_LIFE_DIR` and inject it.

Both layers should be capped (see `MAX_*_CHARS` below) so they don't
blow the model context window.

Env vars (all optional — defaults are sensible):
    MEMORY_SOUL_PATH        personality/voice rules markdown
                            (default: ~/.config/agent-voice-ai/SOUL.md)
    MEMORY_INDEX_PATH       index of memory files
                            (default: ~/.config/agent-voice-ai/MEMORY_INDEX.md)
    MEMORY_LIFE_DIR         directory of long-term memory files
                            (default: ~/.local/share/agent-voice-ai/life)
    MEMORY_FACT_STORE_DB    SQLite db of operational facts
                            (default: ~/.local/share/agent-voice-ai/facts.db)
    MEMORY_SESSIONS_DIR     session history directory
                            (default: ~/.local/share/agent-voice-ai/sessions)

The defaults sit under XDG-style paths and contain nothing by default.
Override them in `.env` if you want to wire in your own files.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("ava.memory")


# ---- Paths (all env-driven, XDG defaults) ----

def _env_path(name: str, default: Path) -> Path:
    val = os.environ.get(name)
    return Path(val).expanduser() if val else default


def _xdg_config() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / "agent-voice-ai"


def _xdg_data() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / "agent-voice-ai"


SOUL_PATH = _env_path("MEMORY_SOUL_PATH", _xdg_config() / "SOUL.md")
SAM_INDEX_PATH = _env_path("MEMORY_INDEX_PATH", _xdg_config() / "MEMORY_INDEX.md")
LONG_TERM_MEMORY_DIR = _env_path("MEMORY_LIFE_DIR", _xdg_data() / "life")
FACT_STORE_DB = _env_path("MEMORY_FACT_STORE_DB", _xdg_data() / "facts.db")
SESSIONS_DIR = _env_path("MEMORY_SESSIONS_DIR", _xdg_data() / "sessions")

# Cap each layer to keep the system prompt sane.
MAX_SOUL_CHARS = 3500
MAX_INDEX_CHARS = 2500
MAX_FACTS_CHARS = 2400
MAX_LIFE_CHARS = 1800


# ---- Read helpers ----

def _safe_read(path: Path, max_chars: int = 0) -> str:
    """Read a file, optionally truncating."""
    try:
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "\n[…truncated…]"
        return text.strip()
    except Exception as e:
        log.warning("memory: failed to read %s: %s", path, e)
        return ""


def _read_fact_store_for_query(query: str, max_chars: int = MAX_FACTS_CHARS) -> str:
    """FTS5 search the fact_store for facts relevant to the query.

    Returns empty string if the db doesn't exist or the query has no
    usable tokens. The schema (`facts` + `facts_fts`) matches what
    `fact_store_add` writes — see tools.py.
    """
    if not FACT_STORE_DB.exists():
        return ""
    stop = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
        "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
        "how", "its", "may", "new", "now", "old", "see", "way", "who", "boy",
        "did", "let", "say", "she", "too", "use", "what", "when", "your",
        "from", "this", "that", "with", "have", "will", "just",
    }
    tokens = [
        t for t in re.findall(r"[A-Za-z0-9_]+", query)
        if len(t) >= 3 and t.lower() not in stop
    ]
    if not tokens:
        return ""
    quoted = " OR ".join(f'"{t}"' for t in tokens[:8])
    try:
        conn = sqlite3.connect(str(FACT_STORE_DB), timeout=2.0)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT f.fact_id, f.category, substr(f.content, 1, 400) AS snippet "
            "FROM facts_fts ft JOIN facts f ON f.fact_id = ft.rowid "
            "WHERE facts_fts MATCH ? "
            "ORDER BY ft.rank LIMIT 6",
            (quoted,),
        ).fetchall()
        conn.close()
    except Exception as e:
        log.warning("memory: fact_store FTS query failed: %s", e)
        return ""
    if not rows:
        return ""
    out = ["## fact_store — operational facts matching your query\n"]
    for r in rows:
        out.append(f"- [id {r['fact_id']} | {r['category']}] {r['snippet']}")
    text = "\n".join(out)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[…truncated…]"
    return text


def _read_life_memory_for_query(query: str, max_chars: int = MAX_LIFE_CHARS) -> str:
    """Generic stub: returns empty unless the user has wired up their own.

    A fork that wants long-term memory should override this function.
    The recommended pattern is:

        1. Maintain an index file (MEMORY_INDEX_PATH) that maps trigger
           words to file paths under MEMORY_LIFE_DIR.
        2. On a query that hits a trigger, read the matched file and
           include it in the system prompt (capped at max_chars).

    The placeholder below reads the index file (if present) and returns
    the file(s) it lists. No trigger keywords are hardcoded — fork the
    function to add your own.
    """
    if not LONG_TERM_MEMORY_DIR.exists() or not SAM_INDEX_PATH.exists():
        return ""
    try:
        # Parse the index file: lines of "keyword: path/to/file.md"
        index_lines = SAM_INDEX_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
        q = query.lower()
        files = []
        for line in index_lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                kw, rel = line.split(":", 1)
                if kw in q.split():
                    rel = rel.strip()
                    f = (LONG_TERM_MEMORY_DIR / rel).resolve()
                    # Path-traversal guard: must stay under LONG_TERM_MEMORY_DIR
                    if LONG_TERM_MEMORY_DIR.resolve() in f.parents and f.exists():
                        files.append(f)
            if len(files) >= 3:
                break
    except Exception as e:
        log.warning("memory: index parse failed: %s", e)
        return ""
    if not files:
        return ""
    out = ["## long-term memory — relevant context\n"]
    remaining = max_chars
    for f in files[:3]:
        chunk_size = min(remaining, 800)
        text = _safe_read(f, chunk_size)
        if text:
            try:
                rel = f.relative_to(LONG_TERM_MEMORY_DIR)
            except ValueError:
                rel = f
            out.append(f"\n### {rel}\n{text}\n")
            remaining -= len(text)
            if remaining <= 100:
                break
    return "\n".join(out).strip()


# ---- Public entry point ----

def build_memory_context(user_text: str) -> str:
    """Build the memory block to inject into the system prompt.

    Order: SOUL.md (identity) → index → FTS facts → long-term memory.
    Each layer is optional and capped. Returns an empty string if no
    memory sources are configured — the bridge still works without
    memory, it just won't have identity/context.
    """
    parts: list[str] = []
    soul = _safe_read(SOUL_PATH, MAX_SOUL_CHARS)
    if soul:
        parts.append(f"## SOUL.md — your identity & voice rules\n{soul}\n")
    index = _safe_read(SAM_INDEX_PATH, MAX_INDEX_CHARS)
    if index:
        parts.append(f"## MEMORY_INDEX.md — your memory index\n{index}\n")
    facts = _read_fact_store_for_query(user_text)
    if facts:
        parts.append(facts + "\n")
    life = _read_life_memory_for_query(user_text)
    if life:
        parts.append(life + "\n")
    return "\n".join(parts).strip()


# ---- Session history (disk-persisted) ----

_SESSIONS_LOCK = threading.Lock()
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _session_path(session_id: str) -> Path:
    # Sanitize session_id (it comes from the client) — allow only safe chars
    safe = re.sub(r"[^A-Za-z0-9_\-]", "_", session_id)[:64]
    return SESSIONS_DIR / f"{safe}.jsonl"


def load_history(session_id: str, max_turns: int = 10) -> list[dict]:
    """Load persisted history for a session. Returns list of {role, content}."""
    path = _session_path(session_id)
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        with _SESSIONS_LOCK:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(__import__("json").loads(line))
                    except Exception:
                        continue
    except Exception as e:
        log.warning("memory: failed to load history for %s: %s", session_id, e)
        return []
    # Each turn = 1 user + 1 assistant; keep last max_turns * 2
    return out[-(max_turns * 2):]


def append_turn(session_id: str, role: str, content: str) -> None:
    """Append one turn to the on-disk session log (newline-delimited JSON)."""
    import json
    path = _session_path(session_id)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "role": role,
        "content": content,
    }
    try:
        with _SESSIONS_LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("memory: failed to persist turn for %s: %s", session_id, e)


def clear_session(session_id: str) -> None:
    path = _session_path(session_id)
    try:
        path.unlink(missing_ok=True)
    except Exception as e:
        log.warning("memory: failed to clear session for %s: %s", session_id, e)


def session_count() -> int:
    """How many sessions on disk. Useful for diagnostics."""
    return len(list(SESSIONS_DIR.glob("*.jsonl")))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    print("=== ava.memory: placeholder memory layer ===")
    print(f"  SOUL_PATH     = {SOUL_PATH} (exists: {SOUL_PATH.exists()})")
    print(f"  INDEX_PATH    = {SAM_INDEX_PATH} (exists: {SAM_INDEX_PATH.exists()})")
    print(f"  LONG_TERM_DIR  = {LONG_TERM_MEMORY_DIR} (exists: {LONG_TERM_MEMORY_DIR.exists()})")
    print(f"  FACT_STORE_DB = {FACT_STORE_DB} (exists: {FACT_STORE_DB.exists()})")
    print(f"  SESSIONS_DIR  = {SESSIONS_DIR} (exists: {SESSIONS_DIR.exists()})")
    print()
    print("To wire in real memory:")
    print("  1. Set MEMORY_SOUL_PATH to your agent's voice/personality markdown")
    print("  2. Set MEMORY_INDEX_PATH to a file of 'keyword: path.md' lines")
    print("  3. Populate MEMORY_LIFE_DIR with the referenced files")
    print()
    ctx = build_memory_context("hello")
    print(f"  build_memory_context('hello') → {len(ctx)} chars (empty if no config)")