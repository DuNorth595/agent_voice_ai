"""
Skill loading for voice-bridge.

Loads the SKILL.md files that are relevant to the current query and
injects them into the system prompt. Two strategies:

1. ALWAYS-LOAD: skills that should be present every turn
   (saving-cross-session-memory-context, life-memory-lookup)
2. TRIGGER-LOAD: skills surfaced by keyword heuristics

Each loaded skill is summarized to the prompt as: name + a short
description + a brief excerpt of the procedure. The full text is
NOT inlined — too big. If the model needs more, it can call a
"load_skill" tool to get the full SKILL.md text on demand.

Source paths:
  ~/.hermes/skills/<name>/SKILL.md
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

SKILLS_ROOT = Path.home() / ".hermes" / "skills"

ALWAYS_LOAD = (
    "saving-cross-session-memory-context",
    "life-memory-lookup",
)

# Trigger keyword -> skill name
TRIGGER_KEYWORDS: dict[str, tuple[str, ...]] = {
    "morning intel": ("briefings",),
    "morning brief": ("briefings",),
    "ai intel": ("briefings",),
    "market brief": ("briefings",),
    "kd0kah": ("kd0kah-fire-monitor", "kd0kah-alert-cadence"),
    "fire monitor": ("kd0kah-fire-monitor",),
    "alert cadence": ("kd0kah-alert-cadence",),
    "p25": ("radio-systems",),
    "mmdvm": ("radio-systems",),
    "apx": ("radio-systems",),
    "launchagent": ("launchagent-env-drift", "hermes-mac-ops"),
    "launchd": ("launchagent-env-drift", "hermes-mac-ops"),
    "cron": ("hermes-cron-script-args",),
    "telegram bot": ("intel-brief-telegram-token-rotation",),
    "token rotation": ("intel-brief-telegram-token-rotation",),
    "memory drift": ("memory-drift-soft-fail",),
    "memory tool": ("saving-cross-session-memory-context", "life-memory-lookup"),
    "life memory": ("life-memory-lookup",),
    "fact_store": ("life-memory-lookup",),
    "ollama": ("inference-sh",),
    "whisper": ("voice-widget-piper-swap", "voice-widget-first-word-cutoff"),
    "piper": ("voice-widget-piper-swap",),
    "browser": ("browser-tool-stability-fix",),
    "smart-home": ("smart-home",),
    "home assistant": ("smart-home",),
    "xrpl": ("xrpl-mpt-distribution",),
    "xlm": ("xlm-rwa-dashboard-polish",),
    "stellar": ("xlm-rwa-dashboard-polish",),
    "post workout": ("sam-post-workout-shake",),
    "shake": ("sam-post-workout-shake",),
    "screen lock": ("ios-app-development",),
    "ios app": ("ios-app-development",),
}


def _read_skill(name: str) -> str | None:
    path = SKILLS_ROOT / name / "SKILL.md"
    if not path.exists():
        return None
    try:
        return path.read_text()
    except OSError:
        return None


def _parse_frontmatter(text: str) -> tuple[str, str]:
    """Returns (name, description) from YAML frontmatter.

    Handles multi-line YAML descriptions with literal block style (|).
    """
    name = ""
    description = ""
    if not text.startswith("---\n"):
        return name, description
    try:
        end = text.index("\n---\n", 4)
        block = text[4:end]
    except ValueError:
        return name, description

    lines = block.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("name:"):
            name = line.split(":", 1)[1].strip()
            i += 1
            continue
        if line.startswith("description:"):
            value = line.split(":", 1)[1].strip()
            if value == "|" or value == ">":
                i += 1
                buf: list[str] = []
                while i < len(lines) and (lines[i].startswith("  ") or lines[i] == ""):
                    buf.append(lines[i].lstrip())
                    if i + 1 < len(lines) and lines[i + 1] and not lines[i + 1].startswith(" "):
                        break
                    i += 1
                description = " ".join(b for b in buf if b)
            else:
                description = value
            i += 1
            continue
        i += 1
    return name, description


def _short_section(text: str, heading: str, max_chars: int = 220) -> str:
    """Return the body of the FIRST `## <heading>` section, capped tightly."""
    pattern = re.compile(rf"^##\s+{re.escape(heading)}\s*$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return ""
    start = match.end()
    next_heading = re.search(r"^##\s+", text[start:], re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(text)
    body = text[start:end].strip()
    body = re.sub(r"\n{2,}", "\n", body)
    return body[:max_chars]


def _first_section(text: str, heading: str, max_chars: int = 800) -> str:
    """Return the body of the first `## <heading>` section, capped."""
    pattern = re.compile(rf"^##\s+{re.escape(heading)}\s*$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return ""
    start = match.end()
    next_heading = re.search(r"^##\s+", text[start:], re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(text)
    body = text[start:end].strip()
    return body[:max_chars]


def _summarize(text: str, max_chars: int = 260) -> str:
    """Pull name + description + a compact useful excerpt."""
    name, description = _parse_frontmatter(text)
    excerpt = (
        _short_section(text, "Trigger Conditions", max_chars=180)
        or _short_section(text, "Procedure", max_chars=200)
        or _short_section(text, "Triggers", max_chars=160)
    )
    excerpt = excerpt.strip()
    full = f"### {name}\n{description}"
    if excerpt:
        full = f"{full}\n\n{excerpt}"
    return full[:max_chars]


def _triggered_skills(query: str) -> tuple[str, ...]:
    q = query.lower()
    hits: set[str] = set()
    for keyword, names in TRIGGER_KEYWORDS.items():
        if keyword in q:
            hits.update(names)
    return tuple(hits)


def build_skills_context(query: str, budget_chars: int = 2600) -> str:
    """Build the skills block to inject into the system prompt.

    Layout:
      ### Active skills (ALWAYS-LOAD)
      ### Relevant skills (TRIGGER-LOAD by current query)
    """
    sections: list[str] = []
    used_chars = 0

    sections.append("### Active skills (loaded every turn)")
    used_chars += len(sections[-1])
    for name in ALWAYS_LOAD:
        text = _read_skill(name)
        if not text:
            continue
        summary = _summarize(text, max_chars=420)
        block = f"\n--- {name} ---\n{summary}"
        if used_chars + len(block) > budget_chars:
            break
        sections.append(block)
        used_chars += len(block)

    triggered = _triggered_skills(query)
    triggered = tuple(n for n in triggered if n not in ALWAYS_LOAD)
    if triggered:
        sections.append("\n### Relevant skills (matched by current query)")
        used_chars += len(sections[-1])
        for name in triggered:
            text = _read_skill(name)
            if not text:
                continue
            summary = _summarize(text, max_chars=320)
            block = f"\n--- {name} ---\n{summary}"
            if used_chars + len(block) > budget_chars:
                sections.append(f"\n(more skills available; budget reached)")
                used_chars += 60
                break
            sections.append(block)
            used_chars += len(block)

    return "\n".join(sections)


def load_full_skill(name: str) -> str:
    """Return the full SKILL.md text. Used by the load_skill tool."""
    text = _read_skill(name)
    if text is None:
        return f"Skill '{name}' not found at {SKILLS_ROOT / name / 'SKILL.md'}"
    return text


def list_skill_names() -> list[str]:
    """Return all installed skill names."""
    if not SKILLS_ROOT.exists():
        return []
    return sorted(p.name for p in SKILLS_ROOT.iterdir() if p.is_dir() and (p / "SKILL.md").exists())


def render_skill_index() -> str:
    """Compact index of available skills — useful for one-shot discovery."""
    names = list_skill_names()
    if not names:
        return "(no skills installed)"
    return "\n".join(f"- {n}" for n in names)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "context":
        query = " ".join(sys.argv[2:]) or "memory tool life memory"
        print(build_skills_context(query))
    elif len(sys.argv) > 1 and sys.argv[1] == "list":
        print(render_skill_index())
    else:
        print("usage: python -m skills_bridge context <query>")
        print("       python -m skills_bridge list")