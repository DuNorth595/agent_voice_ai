#!/usr/bin/env python3
"""
tools.py — voice-Sam's tool implementations.

Each tool is a plain function: (args: dict) -> (result: str | dict).

Tools are organized by tier:
- READ_ONLY: safe, no side effects (search_files, read_file, web_fetch)
- LOCAL_WRITE: visible, reversible (fact_store_add, delegate_task, cron_create)
- EXTERNAL_WRITE: irreversible, requires confirmation (send_message)

Confirmation flow is enforced at the agent loop level in brain.py, not here.
Tools themselves execute whatever they're asked — the loop wraps them.
"""
from __future__ import annotations

import os
import re
import json
import sqlite3
import subprocess
import urllib.request
import urllib.parse
import urllib.error
import logging
from pathlib import Path
from typing import Callable, Any

log = logging.getLogger("sam.tools")

HOME = Path.home()
# Configurable directories — see memory.py for the same env-driven pattern.
# Override via .env or your shell. Defaults follow XDG conventions and
# contain nothing out of the box.
_HERMES_DIR = Path(os.environ.get("HERMES_DIR") or (HOME / ".hermes"))
FACT_STORE_DB = Path(
    os.environ.get("MEMORY_FACT_STORE_DB")
    or (HOME / ".local" / "share" / "agent-voice-ai" / "facts.db")
)
LONG_TERM_MEMORY_DIR = Path(
    os.environ.get("MEMORY_LIFE_DIR")
    or (HOME / ".local" / "share" / "agent-voice-ai" / "life")
)

# Tier classifications — used by brain.py to decide whether to gate execution
READ_ONLY = "read_only"
LOCAL_WRITE = "local_write"
EXTERNAL_WRITE = "external_write"


# ---------- helpers ----------

def _load_env() -> dict:
    """Read .env from a configurable location. Falls back to ~/.hermes/.env
    if HERMES_DIR is set or if the file exists there. Override with the
    AVA_ENV_FILE env var to point at a different file."""
    env = {}
    env_file = Path(os.environ.get("AVA_ENV_FILE") or _HERMES_DIR / ".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def _search_stopwords() -> set:
    return {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
        "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
        "how", "its", "may", "new", "now", "old", "see", "way", "who", "boy",
        "did", "let", "say", "she", "too", "use", "what", "when", "your", "from",
        "this", "that", "with", "have", "will", "just",
    }


# ---------- READ_ONLY tools ----------

def search_files(args: dict) -> dict:
    """Search file contents under a path. Args: query, path (default $MEMORY_LIFE_DIR
    or ~/.local/share/agent-voice-ai/life), limit (default 20). If MEMORY_LIFE_DIR
    is not set and the default directory doesn't exist, falls back to the home dir.
    Scoped to keep searches fast — pass an explicit `path` for big trees."""
    query = args.get("query", "").strip()
    if not query:
        return {"error": "query is required"}
    search_path = args.get("path") or (str(LONG_TERM_MEMORY_DIR) if LONG_TERM_MEMORY_DIR.exists() else str(HOME))
    search_path = os.path.expanduser(search_path)
    limit = min(int(args.get("limit", 20)), 50)

    stop = _search_stopwords()
    tokens = [t for t in re.findall(r"[A-Za-z0-9_]+", query) if len(t) >= 3 and t.lower() not in stop]
    if not tokens:
        return {"matches": [], "note": "no searchable tokens in query"}

    # rg: use a single alternation pattern via -e so tokens aren't mistaken for paths.
    pattern = "|".join(tokens[:8])
    try:
        proc = subprocess.run(
            ["rg", "--json", "-i", "-l",
             "-g", "!*.pyc", "-g", "!.git/*", "-g", "!node_modules/*",
             "-g", "!venv*", "-g", "!.venv*", "-g", "!.cache/*",
             "--max-count", "3",
             "-e", pattern, search_path],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode not in (0, 1):
            return {"matches": [], "note": f"ripgrep exited {proc.returncode}: {proc.stderr.strip()[:200]}"}
        paths = [p for p in proc.stdout.strip().splitlines() if p]
        return {"matches": paths[:limit], "count": len(paths)}
    except FileNotFoundError:
        return {"matches": [], "note": "ripgrep not installed"}
    except subprocess.TimeoutExpired:
        return {"matches": [], "note": "search timed out after 15s"}


def read_file(args: dict) -> dict:
    """Read a file (truncated to keep prompt sane). Args: path, max_lines (default 200)."""
    path = args.get("path", "").strip()
    if not path:
        return {"error": "path is required"}
    max_lines = min(int(args.get("max_lines", 200)), 2000)
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = HOME / p
    if not p.exists():
        return {"error": f"file not found: {p}"}
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        truncated = len(lines) > max_lines
        if truncated:
            lines = lines[:max_lines]
        return {
            "path": str(p),
            "total_lines": len(text.splitlines()),
            "content": "\n".join(lines),
            "truncated": truncated,
        }
    except Exception as e:
        return {"error": f"read failed: {e}"}


def web_fetch(args: dict) -> dict:
    """Fetch a URL and return the body. Args: url, max_bytes (default 50000)."""
    url = args.get("url", "").strip()
    if not url:
        return {"error": "url is required"}
    max_bytes = min(int(args.get("max_bytes", 50000)), 200000)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "sam-voice-bridge/0.4"})
        with urllib.request.urlopen(req, timeout=20) as r:
            body = r.read(max_bytes + 1)
            truncated = len(body) > max_bytes
            if truncated:
                body = body[:max_bytes]
            return {
                "url": url,
                "status": r.status,
                "content_type": r.headers.get("Content-Type", ""),
                "body": body.decode("utf-8", errors="replace"),
                "truncated": truncated,
            }
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"error": f"fetch failed: {e}"}


# ---------- LOCAL_WRITE tools ----------

def fact_store_add(args: dict) -> dict:
    """Add a fact to Sam's memory. Args: content (required), category (default 'general'),
    tags (comma-separated, optional)."""
    content = args.get("content", "").strip()
    if not content:
        return {"error": "content is required"}
    category = args.get("category", "general").strip() or "general"
    tags = args.get("tags", "").strip()

    if not FACT_STORE_DB.exists():
        return {"error": f"fact_store db missing: {FACT_STORE_DB}"}

    try:
        conn = sqlite3.connect(str(FACT_STORE_DB), timeout=5.0)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO facts (category, content, tags, trust, created_at, updated_at) "
            "VALUES (?, ?, ?, 0.5, datetime('now'), datetime('now'))",
            (category, content, tags),
        )
        new_id = cur.lastrowid
        # Update FTS index if it exists
        try:
            cur.execute("INSERT INTO facts_fts(rowid, content, category, tags) VALUES (?, ?, ?, ?)",
                        (new_id, content, category, tags))
        except sqlite3.OperationalError as e:
            log.warning("FTS update skipped: %s", e)
        conn.commit()
        conn.close()
        return {"ok": True, "fact_id": new_id, "category": category, "stored_chars": len(content)}
    except Exception as e:
        return {"error": f"fact_store_add failed: {e}"}


def delegate_task(args: dict) -> dict:
    """Hand off a long task to a sub-agent (background). Args: goal (required),
    context (optional background). NOTE: this is a thin wrapper around the
    delegate_task tool I have at the platform level — but voice-bridge doesn't
    have that tool, so we shell out via cron_create as the proper async path.

    For voice Sam, this is best modeled as 'create a cron job to do X and notify
    me on completion.' Returns the cron job id.
    """
    goal = args.get("goal", "").strip()
    if not goal:
        return {"error": "goal is required"}
    # Defer to cron_create — same result, available from this session
    return cron_create({
        "name": f"voice-delegate: {goal[:50]}",
        "schedule": "now",
        "prompt": goal,
        "deliver": "telegram",
    })


def cron_create(args: dict) -> dict:
    """Schedule a cron job via the Hermes scheduler. Args: name, schedule (cron or 'now'),
    prompt (required), deliver ('origin'|'local'|'telegram'|'all', default 'telegram')."""
    prompt = args.get("prompt", "").strip()
    if not prompt:
        return {"error": "prompt is required"}
    name = args.get("name", "voice-sam-job").strip()
    schedule = args.get("schedule", "now").strip()
    deliver = args.get("deliver", "telegram").strip()

    # Use the hermes CLI to schedule. hermes-cron-script-args skill has the
    # canonical reference — but the basic command shape is:
    #   hermes cron create <name> <schedule> --prompt "<prompt>" --deliver <target>
    cmd = ["hermes", "cron", "create", "--name", name, "--schedule", schedule,
           "--prompt", prompt, "--deliver", deliver]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            return {"error": f"hermes cron create failed: {proc.stderr.strip()[:500]}"}
        return {"ok": True, "name": name, "schedule": schedule, "deliver": deliver,
                "output": proc.stdout.strip()[:500]}
    except FileNotFoundError:
        return {"error": "hermes CLI not on PATH"}
    except Exception as e:
        return {"error": f"cron_create failed: {e}"}


# ---------- EXTERNAL_WRITE tools ----------

def send_message(args: dict) -> dict:
    """Send a Telegram message. Args: text (required), chat_id (optional — defaults
    to TELEGRAM_HOME_CHANNEL), reply_to_message_id (optional)."""
    text = args.get("text", "").strip()
    if not text:
        return {"error": "text is required"}
    chat_id = str(args.get("chat_id") or _load_env().get("TELEGRAM_HOME_CHANNEL") or "").strip()
    if not chat_id:
        return {"error": "no chat_id and no TELEGRAM_HOME_CHANNEL in env"}

    token = _load_env().get("TELEGRAM_BOT_TOKEN")
    if not token:
        return {"error": "TELEGRAM_BOT_TOKEN missing from $AVA_ENV_FILE or ~/.hermes/.env"}

    payload = {"chat_id": chat_id, "text": text}
    rid = args.get("reply_to_message_id")
    if rid:
        payload["reply_to_message_id"] = int(rid)

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            resp = json.loads(r.read().decode("utf-8"))
        if not resp.get("ok"):
            return {"error": f"telegram returned not-ok: {resp}"}
        result = resp.get("result", {})
        return {
            "ok": True,
            "message_id": result.get("message_id"),
            "chat_id": result.get("chat", {}).get("id"),
            "text_sent": text,
            "ts": result.get("date"),
        }
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:500]}"}
    except Exception as e:
        return {"error": f"send_message failed: {e}"}


# ---------- Registry ----------

TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "search_files": {
        "fn": search_files,
        "tier": READ_ONLY,
        "description": "Search file contents by regex/keyword. Args: query (str, required), path (str, default $MEMORY_LIFE_DIR), limit (int, default 20).",
    },
    "read_file": {
        "fn": read_file,
        "tier": READ_ONLY,
        "description": "Read a file's contents. Args: path (str, required), max_lines (int, default 200).",
    },
    "web_fetch": {
        "fn": web_fetch,
        "tier": READ_ONLY,
        "description": "Fetch a URL and return its body. Args: url (str, required), max_bytes (int, default 50000).",
    },
    "fact_store_add": {
        "fn": fact_store_add,
        "tier": LOCAL_WRITE,
        "description": "Add a fact to Sam's persistent memory (fact_store). Args: content (str, required), category (str, default 'general'), tags (str, comma-separated, optional).",
    },
    "cron_create": {
        "fn": cron_create,
        "tier": LOCAL_WRITE,
        "description": "Schedule a Hermes cron job. Args: name (str), schedule (cron or 'now'), prompt (str, required), deliver ('origin'|'local'|'telegram'|'all', default 'telegram').",
    },
    "delegate_task": {
        "fn": delegate_task,
        "tier": LOCAL_WRITE,
        "description": "Hand off a long task to a sub-agent via a one-shot cron job. Args: goal (str, required), context (str, optional).",
    },
    "send_message": {
        "fn": send_message,
        "tier": EXTERNAL_WRITE,
        "description": "Send a Telegram message. Args: text (str, required), chat_id (str/int, optional — defaults to TELEGRAM_HOME_CHANNEL).",
    },
}


def anthropic_tools() -> list[dict]:
    """Build the tools[] payload for the Anthropic API. Schema is permissive
    (additionalProperties=true) because we infer param names from descriptions."""
    out = []
    for name, spec in TOOL_REGISTRY.items():
        props = _infer_properties(spec["description"])
        required = [k for k, v in props.items() if v.get("hint") == "required"]
        # strip our private hint field
        clean_props = {k: {"type": v["type"]} for k, v in props.items()}
        out.append({
            "name": name,
            "description": spec["description"],
            "input_schema": {
                "type": "object",
                "properties": clean_props,
                "required": required,
                "additionalProperties": True,
            },
        })
    return out


def _infer_properties(description: str) -> dict:
    """Pull 'Args: name (type, default ...)' hints out of the description."""
    props = {}
    m = re.search(r"Args:\s*(.+)$", description)
    if not m:
        return props
    arg_blob = m.group(1)
    for piece in re.split(r",\s*(?=[a-z_]+\s*\()", arg_blob):
        am = re.match(r"([a-z_]+)\s*\(([^)]+)\)", piece.strip())
        if not am:
            continue
        name = am.group(1)
        meta = am.group(2)
        hint = "required" if "required" in meta.lower() else "optional"
        type_hint = "string"
        if "int" in meta.lower():
            type_hint = "integer"
        elif "bool" in meta.lower():
            type_hint = "boolean"
        props[name] = {"type": type_hint, "hint": hint}
    return props


def execute(name: str, args: dict) -> dict:
    """Run a tool by name. Result is always a dict."""
    spec = TOOL_REGISTRY.get(name)
    if not spec:
        return {"error": f"unknown tool: {name}"}
    try:
        return spec["fn"](args)
    except Exception as e:
        log.exception("tool %s raised", name)
        return {"error": f"{name} raised: {e}"}


def tier_of(name: str) -> str | None:
    spec = TOOL_REGISTRY.get(name)
    return spec["tier"] if spec else None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    # Quick smoke
    print("--- registry ---")
    for n, s in TOOL_REGISTRY.items():
        print(f"  {n}: {s['tier']}")
    print("--- anthropic tools shape ---")
    print(json.dumps(anthropic_tools()[0], indent=2))
    print("--- smoke: read_file (will skip if no SOUL.md at the configured path) ---")
    soul_path = os.environ.get("MEMORY_SOUL_PATH", str(Path.home() / ".config" / "agent-voice-ai" / "SOUL.md"))
    print(json.dumps(read_file({"path": soul_path, "max_lines": 5}), indent=2)[:400])
    print("--- smoke: search_files ---")
    print(json.dumps(search_files({"query": "example", "limit": 5}), indent=2)[:400])