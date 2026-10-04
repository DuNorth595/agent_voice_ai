# agent-voice-ai

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)

> **A push-to-talk voice loop that gives an LLM agent a body.**
> The agent listens through your microphone, thinks in your chosen model,
> and replies in a local TTS voice — with sentence-by-sentence audio
> streaming so it feels conversational, not stilted.

Reachable from any HTTPS-capable browser (iPhone Safari works) on the same network. Designed to be self-hosted on a single Mac (or any Unix box) with no cloud dependencies beyond the model API itself.

---

## Cover Page

| | |
|---|---|
| **Package** | `agent-voice-ai` |
| **Version** | 0.9.1 (beta) |
| **Author** | Justin Douglas |
| **Organization** | S_DevLabs (Strategic Development Labs) |
| **Contact** | S_DevLabs@outlook.com |
| **License** | MIT |
| **Python** | ≥ 3.11 |
| **Runtime deps** | `fastapi`, `uvicorn`, `requests`, `numpy` |
| **Optional deps** | `kokoro`, `soundfile` (alt TTS/STT), `mcp` (MCP server wrapper) |
| **System deps** | `whisper.cpp` (STT), `piper` (TTS), `ffmpeg` (webm → wav) |

**What this is in one sentence:**
A self-hosted voice loop that wraps any tool-using LLM agent — the agent speaks and listens, you talk back, the conversation stays on your hardware.

**What this is NOT:**
A replacement for your agent's brain. A cloud TTS dependency. A turnkey "AI girlfriend." A consumer app.
This is plumbing. You bring the agent; we provide the ears and mouth.

```
   ┌──────────┐  wss://  ┌──────────────────┐  HTTP   ┌──────────────┐
   │ Browser  │ ───────▶ │ agent_voice_ai   │ ──────▶ │ LLM         │
   │ PWA      │          │ FastAPI          │         │ (Anthropic) │
   │ orb+mic  │ ◀─────── │ port 8770        │ ◀────── │ streaming   │
   └──────────┘  audio   └────┬─────────────┘  text    └────┬─────────┘
                              │                            │
                        whisper.cpp                tools.py (voice-smart)
                        (STT)                      - read-only: search_files,
                                                    read_file, web_extract,
                                                    vision_analyze
                                                  - reversible: todo, fact_store
                                                  - external write: send_message
                                                       (Telegram, with hold)
                              │
                        Piper libritts_r (TTS, sentence-streaming)
                              │
                        ffmpeg (webm → wav transcode for STT)
```

---

## Features

- **Push-to-talk over WebSocket** — single `WSS /ws` carries audio chunks in, text deltas + WAV chunks out. State machine: idle → listening → thinking → speaking.
- **Sentence-streaming TTS** — first audio chunk lands while later sentences are still being synthesized. Feels like a real conversation.
- **Anthropic prompt caching** — repeated turns drop from ~2.5 s to ~700 ms first-token.
- **Voice-smart tool subset** — read-only tools auto-invoke; irreversible external writes (Telegram) hold for explicit confirmation.
- **iOS-friendly** — gesture-unlock pattern for Safari's AbortError, `latencyHint: "interactive"`, webm→wav transcode for whisper.cpp.
- **Self-hosted, no per-minute fees** — runs entirely on your hardware. The only network call is the model API.

---

## Install

### 1. External services (must be running before the bridge starts)

```bash
# whisper.cpp — STT server
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp && make
./server -m models/ggml-base.en.bin --host 0.0.0.0 --port 8080

# piper — TTS CLI
# See https://github.com/rhasspy/piper — needs ONNX voice model
pip install piper-tts
# Download a voice (e.g. libritts_r medium):
# https://huggingface.co/rhasspy/piper-voices

# ffmpeg — for webm → wav transcode
brew install ffmpeg   # macOS
```

### 2. Python deps

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env — at minimum, set BRAIN_API_KEY.
```

### 4. TLS (optional but required for iPhone)

```bash
mkdir -p certs
# Self-signed cert works for personal use; for production use a proper
# cert (e.g. via Tailscale, mkcert, or Let's Encrypt).
openssl req -x509 -newkey rsa:4096 -nodes -keyout certs/server.key \
  -out certs/server.crt -days 365 \
  -subj "/CN=agent-voice-ai.local" \
  -addext "subjectAltName=DNS:agent-voice-ai.local,IP:127.0.0.1"
```

### 5. Run

```bash
python agent_voice_ai.py
# → open https://localhost:8770/ in a browser on the same machine
# → open https://<your-host>:8770/ from iPhone (accept the cert warning)
```

For autostart on macOS, see `examples/com.agent-voice-ai.bridge.plist` (install with `launchctl bootstrap`).

---

## Configuration reference

All settings are environment variables (or entries in `.env`):

| Variable | Default | Purpose |
|---|---|---|
| `BRAIN_API_KEY` | — | Required. API key for the brain. Falls back to `MINIMAX_API_KEY` for backward compat. |
| `BRAIN_MODEL` | `MINIMAX-M3` | Model identifier. |
| `BRAIN_API_URL` | `https://api.minimax.io/v1/messages` | Anthropic-compatible endpoint. |
| `TELEGRAM_BOT_TOKEN` | — | Optional. Enables `send_message` tool. |
| `TELEGRAM_HOME_CHANNEL` | — | Optional. Numeric chat_id for home channel. |
| `AVA_HOST` | `0.0.0.0` | Bind address. |
| `AVA_PORT` | `8770` | Bind port. |
| `AVA_TLS_CERT` | `certs/server.crt` | TLS cert path. |
| `AVA_TLS_KEY` | `certs/server.key` | TLS key path. |
| `AVA_HTTPS` | `1` (if certs exist) | Set `0` to force plain HTTP (localhost only). |
| `AVA_TTS_PROVIDER` | `piper` | `piper` or `kokoro`. |
| `AVA_CORS_ORIGINS` | `*` | Comma-separated list of allowed CORS origins. |
| `AVA_VERBOSE` | `0` | Set `1` for DEBUG logging. |
| `WHISPER_URL` | `http://127.0.0.1:8178/inference` | whisper.cpp server endpoint. |
| `PIPER_BIN` | `piper` | Piper CLI binary. |
| `PIPER_MODEL_DIR` | `~/.local/share/piper/voices` | Directory holding Piper ONNX voice files. |
| `PIPER_MODEL` | `en_US-libritts_r-medium` | Voice model filename (without `.onnx`). |
| `KOKORO_URL` | `http://127.0.0.1:8080` | Kokoro TTS server endpoint. |
| `KOKORO_VOICE` | `af_heart` | Kokoro voice name. |
| `MEMORY_SOUL_PATH` | `~/.config/agent-voice-ai/SOUL.md` | Personality/voice rules markdown. |
| `MEMORY_INDEX_PATH` | `~/.config/agent-voice-ai/MEMORY_INDEX.md` | Long-term memory index. |
| `MEMORY_LIFE_DIR` | `~/.local/share/agent-voice-ai/life` | Directory of memory files. |
| `MEMORY_FACT_STORE_DB` | `~/.local/share/agent-voice-ai/facts.db` | SQLite db of operational facts. |
| `MEMORY_SESSIONS_DIR` | `~/.local/share/agent-voice-ai/sessions` | Session history directory. |
| `AVA_SKILLS_ROOT` | `~/.hermes/skills` | Root directory containing skill subdirs. |
| `AVA_ALWAYS_LOAD_SKILLS` | (empty) | Comma-separated skills to always inject. |
| `AVA_TRIGGER_KEYWORDS_JSON` | (empty) | Override the default keyword→skill trigger map. |

---

## Architecture notes

### Wire protocol — `WSS /ws`

**Client → bridge:**
- `{"type":"start"}` — begin a turn (bridge enters listening state)
- Raw binary frames — webm Opus audio chunks (client uses `MediaRecorder` with `timeslice=100`)
- `{"type":"stop"}` — end of turn; bridge transcribes → brain → streams reply

**Bridge → client:**
- `{"type":"state","value":"idle|listening|thinking|speaking|tap-to-unlock|error"}`
- `{"type":"stt","text":"..."}` — final transcript of what user said
- `{"type":"text","delta":"..."}` — streaming reply text (sentence-by-sentence)
- Binary frames — WAV audio (one per completed sentence)
- `{"type":"transcript","role":"user|assistant","text":"..."}` — append to chat
- `{"type":"pending_confirm","action":"...","preview":"...","chat_id":"..."}` — tool held, awaits "do it" / "cancel" on next user turn
- `{"type":"done","ts":"..."}` — turn complete
- `{"type":"error","message":"..."}` — fatal error; client should reload

### Tiered confirmation protocol

Tools are bucketed by reversibility:

| Tier | Examples | Behavior |
|---|---|---|
| read-only | `search_files`, `read_file`, `web_extract`, `vision_analyze`, `fact_store` (read) | Auto-invoke + narrate. No hold. |
| reversible | `todo` (write), `fact_store` (add) | Auto-invoke + narrate. User can undo. |
| irreversible external | `send_message` (Telegram) | Hold. Bridge emits `pending_confirm`; waits for "do it" / "cancel" on next user turn. **Auto-confirmed** if user's most recent message contained a pre-approval phrase. |

**Pre-approval phrases** (case-insensitive, matched against `user_text`):
- "don't need to confirm", "no confirmation needed", "just send it", "fire away", "skip the confirm", "no need to ask"

**The invariant:** narration is not action. The agent must invoke the tool to make it happen. Pre-approval makes the tool auto-confirm; it never lets the agent skip the tool.

### iOS / Safari quirks

The browser side of this stack is fragile in well-known ways. The implementation handles all of them:

1. **AbortError on first play** — Safari rejects `audio.play()` until the user taps after the device locks. The bridge bails out, sets `_wsWaitingForGesture = true`, and shows a pulsing "tap anywhere to resume audio" cue. Re-prime on `pointerdown` / `touchstart`.
2. **NotAllowedError** — same shape, slightly different code; handled identically.
3. **webm rejection by whisper.cpp** — whisper.cpp's middleware checks for `audio/wav` content-type and 400s on raw webm. ffmpeg subprocess transcodes webm→wav before STT.
4. **AudioContext gesture requirement** — first mic access needs a user tap; the orb click is the unlock.
5. **No service worker** — the PWA is just an HTTPS page; refreshes do a full reload. State persists via localStorage only.

---

## File map

```
agent_voice_ai.py     — FastAPI app, / HTML, /ws handler, TTS, sentence-streaming
brain.py              — LLM streaming wrapper with Anthropic prompt caching
whisper_stt.py        — whisper.cpp batch client (HTTP POST with WAV body)
piper_tts.py          — Piper CLI wrapper (libritts_r voice, 22050 Hz mono WAV)
kokoro_tts.py         — Fallback TTS (Kokoro KPipeline)
tools.py              — Voice-smart tool subset (no terminal/write/patch)
skills_bridge.py      — Skill registry exposed to the voice agent
telegram_sync.py      — Background poller for incoming Telegram messages
memory.py             — Persistent memory (fact_store, MEMORY.md)
.env.example          — Configuration placeholders
requirements.txt      — Python dependencies
examples/
  com.agent-voice-ai.bridge.plist  — macOS launchd supervisor template
```

---

## License

MIT — see `LICENSE`.

Copyright © 2026 Justin Douglas / SDevLabs.

---

## For another agent — picking up this codebase

> **If you are an LLM agent reading this codebase to help the user maintain or extend it, read this section first.** It distills the load-bearing invariants and the failure modes the human has hit.

### What this project is, in one paragraph

A self-hosted push-to-talk voice agent bridge. The user runs whisper.cpp + piper TTS locally on their own hardware (any Unix box — Mac, Linux, a NUC, a Raspberry Pi 5), and the LLM call goes to any Anthropic-compatible API. The bridge serves a single-page HTML/JS frontend over HTTPS that can be reached from a phone browser (iOS Safari works). The agent has a voice-tuned tool subset — read-only tools fire freely, local writes fire and report, irreversible external writes (Telegram) hold for explicit confirmation. The repo is at v0.9.1.

### Load-bearing invariants

1. **Voice is a modality, not a different entity.** When the agent has a personality / SOUL.md / memory, the voice bridge reuses the same — don't fork personality per surface.
2. **Narration is not action.** The agent must call the tool to make it happen. If the agent narrates "I sent it" without invoking `send_message`, that is a bug. See v0.8c bug history.
3. **Pre-approval does not skip the tool.** It auto-confirms the tool. Look for phrases like "fire away" / "no confirmation needed" and execute `send_message` immediately instead of holding.
4. **iOS Safari is fragile.** All audio paths assume the user may need to tap to unlock. The "tap anywhere to resume" cue is the safety net. Never silent-fail on audio errors.
5. **The bridge is supervised by launchd.** Don't run duplicate copies manually — they fight for port 8770.

### How to make changes safely

1. **Edit `agent_voice_ai.py`.** It's the main module; everything else is a helper.
2. **Restart via launchd:** `launchctl kickstart -k gui/$(id -u)/com.agent-voice-ai.bridge`
3. **Verify the served content:** `curl -s --insecure https://localhost:8770/ | grep <feature>` — the patch tool reports success but the served file may not reflect it. Always grep.
4. **No automated tests yet.** The tiered confirmation protocol is the safety-critical bit and should be the first thing tested when adding tests.

### Failure modes the user has hit (do not reintroduce)

- **"I said I sent but I didn't."** Pre-v0.8c: the brain narrated without calling the tool. Fixed by `_has_pre_approval` + system-prompt rewrite. If you see the agent narrate without invoking, that's regression.
- **iOS AbortError silent spin.** Pre-v0.9.1: `_wsPlayQueue` re-called itself in a tight loop on AbortError. Fixed by `_wsWaitingForGesture` early-return. If you see audio spinning silently, that's regression.
- **whisper.cpp HTTP 400 on webm.** Pre-v0.8b: STT got raw webm bytes. Fixed by ffmpeg transcode. If STT starts 400-ing, that's regression.
- **Bridge silently dies.** Pre-launchd: bridge ran in a shell session and died with it. Fixed by launchd `KeepAlive=true`. If the bridge stops responding, check `launchctl list | grep agent-voice-ai`.
- **Patch reports success but file unchanged.** Pattern that's happened in this codebase. Always grep the served content after restart.

### Things to NOT do without asking the user

- **Push to a public remote.** The repo isn't git-tracked by default. If a fork is published, treat credentials carefully — see `.env.example` for what should never be committed.
- **Add new tools to `tools.py`.** The voice-smart subset is curated; adding `terminal` or `write_file` defeats the safety model. Confirm with the user first.
- **Add a wake-word.** Out of scope until openWakeWord / Porcupine is wired in. See the `[unreleased]` → `### Planned` section in CHANGELOG.md.
- **Touch `certs/`.** TLS certs are operator-managed. The bridge auto-detects their presence.

### Where to read more

- `CHANGELOG.md` — version history (what changed, what broke, what was learned)
- `SECURITY.md` — threat model, secret-handling rules, known limitations
- `SYNC.md` — pre-release sync checklist (doc / test / version drift detector)
- `CONTRIBUTING.md` — Conventional Commits, code style, two-layer secret defense

### Common tasks

| User says | You probably want to |
|---|---|
| "Bump to v0.X" | Update `version=` in FastAPI app + version display in HTML, restart via launchd, grep `/` to confirm |
| "Bridge is down" | `launchctl list \| grep agent-voice-ai` → check PID/exit; if -15 it was killed; `kickstart -k` to bounce |
| "Audio is buggy on iPhone" | Check browser console for AbortError; orb should show "tap anywhere to resume"; if not, see iOS quirks section |
| "Add a tool" | Edit `tools.py`, add to agent's tool list in `brain.py` system prompt, decide tier (read-only / reversible / irreversible external) |
| "I said sent but didn't" | Verify `_has_pre_approval` is wired and system prompt tells the brain to call the tool even when pre-approved |