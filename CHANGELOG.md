# Changelog

All notable changes to **agent-voice-ai** are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), dates are ISO 8601 (YYYY-MM-DD).

## [0.9.1] — 2026-10-04

### Fixed
- iOS Safari AbortError / NotAllowedError silent spin. `_wsPlayQueue` no longer retries in a tight loop on first-failure; sets `_wsWaitingForGesture = true` and surfaces a pulsing yellow "tap anywhere to resume audio" cue. Confirmed against Safari and Chrome.

### Added
- New `tap-to-unlock` and `error` states on the bridge status indicator.

## [0.9] — 2026-10-04

### Added
- **Hide-chat toggle.** Eye icon in the chat panel header. `Cmd+H` / `Ctrl+H` keyboard shortcut. State persists in `localStorage` (`sb_hide_chat_v1`).

### Notes
- Saved as baseline. Future changes build on 0.9 unless otherwise noted.

## [0.8c] — 2026-10-04

### Fixed
- **"I said I sent but I didn't" bug.** Pre-0.8c, the agent could narrate "I sent it" without invoking the `send_message` tool when the user pre-approved the action. Fixed two ways: (a) explicit pre-approval detector in `brain.py` (`_has_pre_approval` recognizes phrases like "fire away" / "no confirmation needed"); (b) system prompt rewritten to require tool invocation regardless of pre-approval state.
- iOS audio-unlock AbortError handling — added `NotAllowedError` as a sibling of `AbortError`, with proper "tap to unlock" state transition.

## [0.8b] — 2026-10-04

### Fixed
- whisper.cpp returned HTTP 400 on raw webm bytes from `MediaRecorder`. Added ffmpeg subprocess transcode (webm → wav) in the WS stop handler before invoking `whisper_stt.transcribe`. Confirmed by round-trip test with Piper-generated speech.

## [0.8] — 2026-10-03

### Added
- **WebSocket `/ws` push-to-talk path.** Replaces the legacy HTTP `/api/listen` + `/api/chat` round-trip. Audio chunks in, text deltas + WAV chunks out. State machine: idle → listening → thinking → speaking.
- **Sentence-streaming TTS.** `_split_sentences` + `_stream_brain_to_sentences` feed Piper sentence-by-sentence so the user hears sentence 1 while sentence 2 is being synthesized.
- **iPhone audio tuning.** `AudioContext` with `latencyHint: "interactive"`, `MediaRecorder` with `timeslice=100`.
- **Anthropic prompt caching.** System prompt sent as content blocks with `cache_control: ephemeral`. First-token time drops from ~2.5 s cold to ~700 ms warm on subsequent turns. Cache hits/misses logged.
- **launchd supervision.** Bridge now runs under `com.agent-voice-ai.bridge` (label may differ in your install) with `KeepAlive=true` + `ThrottleInterval=10`. Survives shell sessions, auto-restarts on crash.

### Notes
- The legacy HTTP paths (`/api/listen`, `/api/chat`) remain in place for non-browser callers.

## [unreleased]

### Planned
- Streaming STT (replace batch whisper.cpp with faster-whisper).
- Barge-in (interrupt the agent mid-response).
- Wake-word detection (openWakeWord or Porcupine).
- Automated test coverage, starting with the tiered confirmation protocol.
- WebSocket auth (currently relies on network-level access control — Tailscale or similar).

---

[0.9.1]: #091--2026-10-04
[0.9]: #09--2026-10-04
[0.8c]: #08c--2026-10-04
[0.8b]: #08b--2026-10-04
[0.8]: #08--2026-10-03
[unreleased]: #unreleased