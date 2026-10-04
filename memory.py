#!/usr/bin/env python3
"""
memory.py — Sam's memory layer for the voice bridge.

Loads Sam's actual memory (not a fake placeholder) before each LLM call:
  1. SOUL.md — Sam's identity/personality rules
  2. SAM_MEMORY_INDEX.md — the high-signal index
  3. fact_store FTS5 search on the user's question (operational facts)
  4. LIFE_MEMORY file search for identity/family/project queries
  5. Session history persisted to disk (survives restarts)

This is what makes the voice bridge "really me" — same memory, same facts,
same identity, different modality.
"""
from __future__ import annotations

import os
import re
import json
import logging
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("sam.memory")

HOME = Path.home()
LIFE_MEMORY = HOME / "Desktop" / "LIFE_MEMORY"
HERMES_DIR = HOME / ".hermes"
SOUL_PATH = HERMES_DIR / "SOUL.md"
SAM_INDEX_PATH = LIFE_MEMORY / "SAM_MEMORY_INDEX.md"
FACT_STORE_DB = HERMES_DIR / "memory_store.db"
SESSIONS_DIR = HERMES_DIR / "voice-bridge-sessions"

# Identity triggers — when the user asks about these, we MUST search LIFE_MEMORY,
# not hallucinate. Cardinal rule from life-memory-lookup skill.
IDENTITY_TRIGGERS = re.compile(
    r"\b("
    r"justin|sara|noah|logan|taylor|mason|kona|mugo|"
    r"kd0kah|callsign|birthday|wife|kids|family|"
    r"anniversary|p25|smartconnect|k1lnx|fne|dvm|"
    r"voice.?agent|sam|jarvis"
    r")\b",
    re.IGNORECASE,
)

# Max chars to inject from each memory source into the system prompt
MAX_SOUL_CHARS = 3500
MAX_INDEX_CHARS = 2500
MAX_FACTS_CHARS = 2400
MAX_LIFE_CHARS = 1800


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
    """FTS5 search the fact_store for facts relevant to the query."""
    if not FACT_STORE_DB.exists():
        return ""
    # Extract search terms (skip stopwords, keep >=3 char tokens)
    stop = {"the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
            "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
            "how", "its", "may", "new", "now", "old", "see", "way", "who", "boy",
            "did", "let", "say", "she", "too", "use", "what", "when", "your", "from",
            "this", "that", "with", "have", "will", "just"}
    tokens = [t for t in re.findall(r"[A-Za-z0-9_]+", query) if len(t) >= 3 and t.lower() not in stop]
    if not tokens:
        return ""

    quoted = " OR ".join(f'"{t}"' for t in tokens[:8])  # cap to 8 terms
    try:
        conn = sqlite3.connect(str(FACT_STORE_DB), timeout=2.0)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT f.fact_id, f.category, substr(f.content, 1, 400) as snippet "
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
    """If the query hits an identity trigger, pull the relevant LIFE_MEMORY files."""
    if not IDENTITY_TRIGGERS.search(query):
        return ""
    if not LIFE_MEMORY.exists():
        return ""

    q = query.lower()
    # Map identity triggers to specific files (most-direct hit first)
    file_hints = []
    if "justin" in q or re.search(r"\b(i|me|my|myself)\b", q):
        file_hints.append(LIFE_MEMORY / "PEOPLE" / "justin.md")
    if "sam" in q or "jarvis" in q or "you" in q:
        file_hints.append(LIFE_MEMORY / "SAM" / "IDENTITY_LITE.md")
        file_hints.append(SAM_INDEX_PATH)
    if "p25" in q or "fne" in q or "k1lnx" in q or "dvm" in q or "smartconnect" in q or "kd0kah" in q:
        # Add the most likely project file based on keywords
        if "p25 page" in q or "page" in q:
            file_hints.append(LIFE_MEMORY / "PROJECTS" / "P25_PAGE.md")
        elif "p25 edge" in q or "edge" in q:
            file_hints.append(LIFE_MEMORY / "PROJECTS" / "P25_EDGE.md")
        else:
            file_hints.append(LIFE_MEMORY / "PROJECTS" / "KD0KAH_TRUNKING.md")
    if "sara" in q or "wife" in q or "anniversary" in q or "family" in q or "kids" in q:
        file_hints.append(LIFE_MEMORY / "PEOPLE" / "sara.md")

    # Dedupe + filter to existing
    seen, files = set(), []
    for f in file_hints:
        if f and f.exists() and str(f) not in seen:
            seen.add(str(f))
            files.append(f)

    if not files:
        return ""

    out = ["## LIFE_MEMORY — relevant personal/project context\n"]
    remaining = max_chars
    for f in files[:3]:  # cap to 3 files
        chunk_size = min(remaining, 800)
        text = _safe_read(f, chunk_size)
        if text:
            rel = f.relative_to(HOME)
            out.append(f"\n### {rel}\n{text}\n")
            remaining -= len(text)
            if remaining <= 100:
                break

    return "\n".join(out).strip()


def build_memory_context(user_text: str) -> str:
    """Build the memory block to inject into the system prompt.

    Order matters: SOUL.md first (identity), then index, then FTS facts, then
    LIFE_MEMORY file content. Cap total to ~10K chars so we don't blow the
    30K-system-prompt budget the model imposes.
    """
    parts = []
    soul = _safe_read(SOUL_PATH, MAX_SOUL_CHARS)
    if soul:
        parts.append(f"## SOUL.md — your identity & voice rules\n{soul}\n")
    index = _safe_read(SAM_INDEX_PATH, MAX_INDEX_CHARS)
    if index:
        parts.append(f"## SAM_MEMORY_INDEX.md — your high-signal index\n{index}\n")
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
    out = []
    try:
        with _SESSIONS_LOCK:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        log.warning("memory: failed to load history for %s: %s", session_id, e)
        return []
    # Each turn = 1 user + 1 assistant; keep last max_turns * 2
    return out[-(max_turns * 2):]


def append_turn(session_id: str, role: str, content: str) -> None:
    """Append one turn to the on-disk session log (newline-delimited JSON)."""
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
        log.warning("memory: failed to clear session %s: %s", session_id, e)


def session_count() -> int:
    """How many sessions on disk. Useful for diagnostics."""
    return len(list(SESSIONS_DIR.glob("*.jsonl")))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    print("=== testing memory.build_memory_context ===")
    ctx = build_memory_context("What's my wife's name?")
    print(ctx[:1500])
    print("...")
    print(f"  total chars: {len(ctx)}")
    print(f"  sessions on disk: {session_count()}")
