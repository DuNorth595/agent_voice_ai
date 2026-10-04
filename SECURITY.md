# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| `0.9.x` (latest) | ✅ |
| `0.8.x` | ⚠️ Critical fixes only |
| `< 0.8` | ❌ No longer supported |

## Reporting a vulnerability

**Do not file a public GitHub issue for security bugs.**

Email: `S_DevLabs@outlook.com` (PGP not currently configured — cleartext accepted for now, switch to a PGP-encrypted channel if you need one and I'll add a key). Expect an acknowledgement within 72 hours.

Please include:
- Reproduction steps (a failing test is the best case)
- Impact assessment (what an attacker can do)
- Whether your reproduction is local-only or requires network access
- Any logs that show the issue (redact any API keys / seeds)

## Threat model

`agent-voice-ai` is a **local server** that bridges a browser mic to an LLM API. It does not store user content, does not run third-party code on your behalf, and does not custody credentials. It listens on `127.0.0.1` by default; binding to `0.0.0.0` is opt-in (you typically pair it with a Tailscale or VPN layer for remote access).

| Asset | Where it lives | What this bridge does with it |
|---|---|---|
| LLM API key | Operator's `.env` file (or KMS, future) | Reads via `os.environ["BRAIN_API_KEY"]` (falls back to `MINIMAX_API_KEY`). **Never logs it. Never writes it to disk. Never includes it in error tracebacks.** |
| Telegram bot token | Operator's `.env` file | Reads via `os.environ["TELEGRAM_BOT_TOKEN"]`. Used only by `telegram_sync.py` to poll updates and push messages. **Never logs it.** |
| Chat ID | Operator's `.env` file | Read-only destination address for outbound messages. Not a secret on its own but should still not be committed. |
| TLS private key | `certs/server.key` | Used by FastAPI/uvicorn to terminate HTTPS. **Never logged.** Never leaves the host. |
| User audio | Browser → WebSocket → STT → LLM | Audio is transient: streamed in, transcribed, transcribed text sent to LLM, audio bytes discarded. Audio is **not persisted** to disk by default. |
| Conversation transcripts | LLM provider's API logs (Anthropic, etc.) | Subject to the LLM provider's own retention policy. **Not stored by this bridge.** |
| Browser session | localStorage on the operator's device | Holds UI prefs (`sb_hide_chat_v1`). Not a secret. |

## Secret-handling rules (for contributors)

1. **Never log a secret.** `print(api_key)`, `logger.info(token)`, `repr(env_dict)` — all forbidden.
   The cleanest pattern is to log a **hash or last-4-chars** for correlation:
   ```python
   key = os.environ["BRAIN_API_KEY"]  # or MINIMAX_API_KEY as a fallback
   logger.info("api_key loaded (%d chars, suffix=…%s)", len(key), key[-4:])
   ```
2. **Never commit a secret.** Two layers of defense:
   - **`.gitignore`** excludes `.env`, `certs/`, `*.pem`, `*.key`, `*.crt`. Verify before every commit:
     `git status --ignored | grep -E '\.env$|\.key$|\.pem$'`.
   - **Manual review** of `git diff --cached` for any string that looks like a token (`sk-ant-…`, `1234567890:AA…`, `-----BEGIN … PRIVATE KEY-----`).
   If you need a test secret, generate one inside the test (e.g. `os.environ["TEST_FAKE_KEY"] = "test_" + secrets.token_hex(16)`).
3. **Use environment variables, never hardcoded literals.** `os.environ["BRAIN_API_KEY"]`,
   not `client = Anthropic(api_key="sk-ant-...")`. The env var should be set in a shell
   that's not logged.
4. **No secrets in error messages.** When wrapping exceptions, log `e.args[0]` (the human
   message), not the full exception object. Upstream libraries may include the request body
   in tracebacks.
5. **Memory hygiene is out of scope.** Python strings are immutable; you can't reliably
   scrub a secret from process memory after use. Don't try — instead, minimize the secret's
   lifetime: load, use, drop the reference.

## Cryptographic choices

- **Transport encryption:** TLS 1.2+ (TLS 1.3 preferred) via `uvicorn` + `ssl.SSLContext`. Certificate and key are operator-supplied (`certs/server.crt`, `certs/server.key`). No custom crypto; no custom protocol.
- **WebSocket auth:** **None at the protocol level.** The bridge relies on network-level access control (Tailscale tailnet, VPN, or localhost-only binding). A future release will add a `?token=` query parameter or `Authorization` header — see [Known limitations](#known-limitations) #1.
- **No end-to-end encryption.** Audio and text travel in plaintext inside the TLS tunnel to the LLM provider. This is by design — the LLM needs to read the prompt to respond.
- **No signing / no verification.** This bridge does not produce or verify signed messages. Telegram messages are sent via the official Bot API over HTTPS.

## Dependencies

`agent_voice_ai` has **four runtime dependencies** (`fastapi`, `uvicorn`, `requests`, `numpy`) and three optional / system-side dependencies:

| Type | What it adds | Why it's optional |
|---|---|---|
| Runtime | `fastapi`, `uvicorn`, `requests`, `numpy` | Required to start the bridge. |
| Optional (Python) | `kokoro`, `soundfile` | Only needed if you swap Piper for Kokoro TTS. |
| Optional (system) | `whisper.cpp` | External STT server; the bridge calls it over HTTP. |
| Optional (system) | `piper` | External TTS binary; the bridge shells out to it. |
| Optional (system) | `ffmpeg` | External transcode binary; required for browser-side `audio/webm` → `audio/wav`. |

We pin minimum versions in `requirements.txt` (major-version floor) but do not pin patch versions. For supply-chain hardening, run `pip-audit` against your installed environment.

## Known limitations

1. **No WebSocket auth yet.** Today, anyone who can reach the bridge's port can stream audio in and read the served HTML. Mitigation: bind to `127.0.0.1` only, or front it with Tailscale / a reverse proxy that does auth. v0.10 will add an `AVA_WS_TOKEN` env var + `Authorization: Bearer *** check on `/ws`.
2. **No rate limiting.** A malicious client could spam the WS endpoint and rack up LLM API costs. Mitigation: same as #1 (network-level access control). v0.10 will add per-IP token-bucket limits.
3. **No CSRF on the HTML page.** The static page has no state to protect today, but if you add user accounts / settings persistence in a fork, defend against CSRF with SameSite cookies + a custom request header.
4. **Audio is not persisted, but LLM provider retention applies.** Anthropic's API retains prompts for 30 days by default (per their policy at the time of writing). If you need stricter retention, set `Anthropic-Vendor: …` headers or use a self-hosted LLM.
5. **launchd plist runs the bridge as your user.** If the bridge is compromised, the attacker has your user's permissions. Mitigation: run as a low-privilege service user (`_ava` or similar) — coming in v0.10.
6. **No HTTPS certificate validation for the LLM API call.** We trust the system trust store. If you MITM your own machine, you don't get a clean abort. (Standard library behavior, not something this bridge changes.)

## Acknowledgements

This policy is modeled on the [GitHub Security Lab's guide to writing a SECURITY.md](https://docs.github.com/en/code-security/getting-started/adding-a-security-policy-to-your-repository) and [xkcd 538](https://xkcd.com/538/).