#!/usr/bin/env python3
"""
agent_voice_ai.py — The agent-voice-ai bridge.

FastAPI server on port 8770 (configurable via AVA_PORT env).
Endpoints:
  GET  /              → minimal web UI (HTML)
  GET  /api/health    → health check
  POST /api/chat      → text in, text + audio out
  POST /api/listen    → audio in (WAV), text + audio out
  WS   /ws            → push-to-talk: audio chunks in, sentence-streaming text + audio out
  GET  /api/history   → get session history
  POST /api/clear     → clear session

Audio out: 24kHz mono WAV (Kokoro).
Audio in:  16kHz mono WAV (whisper-server default).

Run:    python agent_voice_ai.py
Debug:  AGENT_VOICE_VERBOSE=1 python agent_voice_ai.py
"""
import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import brain
try:
    import kokoro_tts  # optional fallback TTS provider; may be unavailable on this box
    _KOKORO_OK = True
except Exception as _kokoro_err:
    kokoro_tts = None  # type: ignore[assignment]
    _KOKORO_OK = False
    logging.getLogger("agent_voice_ai").warning("kokoro_tts unavailable: %s — fallback chain disabled", _kokoro_err)
import piper_tts
import whisper_stt
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

# ---- logging ----
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("AGENT_VOICE_VERBOSE") else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("agent_voice_ai.bridge")

# ---- app ----
app = FastAPI(title="agent-voice-ai", version="0.9.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("AVA_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

PORT = int(os.environ.get("AVA_PORT", "8770"))
HOST = os.environ.get("AVA_HOST", "0.0.0.0")  # bind 0.0.0.0 so any LAN interface (or VPN/Tailscale) can reach it


def _synth_with_fallback(reply_text: str) -> tuple[bytes, str]:
    """
    Synthesize reply_text → WAV bytes synchronously using the active TTS
    provider chain. Caller must be inside an async handler; we use
    run_in_executor at the call site to keep the event loop responsive.
    """
    tts_provider = os.environ.get("AGENT_VOICE_TTS_PROVIDER", "piper").lower()
    # Primary: Piper (the default voice-bridge TTS)
    if tts_provider == "piper" and piper_tts.is_available():
        return piper_tts.synth_to_wav_bytes(reply_text), "piper"
    # Fallback: Kokoro (only if it actually imported)
    if _KOKORO_OK and kokoro_tts is not None:
        return kokoro_tts.synth_to_wav_bytes(reply_text), "kokoro"
    raise RuntimeError("no TTS provider available (piper + kokoro both unavailable)")


# Pre-warm Kokoro on startup so the first /api/chat call doesn't pay the ~5s
# model-load penalty. Without this, the first request to a freshly-restarted
# bridge takes ~13s and the mobile client times out before the audio_url
# is returned (which is the "50% silent audio" bug we just diagnosed).
_warmup_done = False
def _warm_kokoro():
    global _warmup_done
    if _warmup_done:
        return
    if not _KOKORO_OK or kokoro_tts is None:
        return  # kokoro not available, nothing to warm
    try:
        # synth a throwaway string to load the model + voice
        kokoro_tts.synth_to_wav_bytes("ready.", voice=kokoro_tts.DEFAULT_VOICE)
        _warmup_done = True
        log.info("kokoro pre-warmed")
    except Exception as e:
        log.warning("kokoro warmup failed: %s — bridge will fall back at runtime", e)


# ---- web UI ----
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0a0e1a">
<title>sam</title>
<style>
  :root {
    --bg: #0a0e1a;
    --bg-elev: #131a2a;
    --text: #e8eef9;
    --text-dim: #6b7a99;
    --accent: #3b82f6;
    --accent-glow: rgba(59, 130, 246, 0.5);
    --state-idle: #2a3550;
    --state-listening: #3b82f6;
    --state-thinking: #f59e0b;
    --state-speaking: #10b981;
  }
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  html, body {
    margin: 0; padding: 0;
    background: var(--bg);
    color: var(--text);
    font: 16px/1.5 -apple-system, BlinkMacSystemFont, "SF Pro Display", system-ui, sans-serif;
    height: 100%; overflow: hidden;
  }
  body {
    display: flex; flex-direction: column;
    padding: env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left);
  }
  .header {
    display: flex; align-items: center; gap: 12px;
    padding: 16px 20px;
    border-bottom: 1px solid var(--bg-elev);
  }
  .header h1 {
    margin: 0; font-size: 18px; font-weight: 600; letter-spacing: 0.05em;
  }
  .header .status {
    font-size: 12px; color: var(--text-dim);
    margin-left: auto;
  }
  .header .status .dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: var(--state-idle);
    margin-right: 6px;
    vertical-align: middle;
  }
  .stage {
    flex: 1;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 20px;
    position: relative;
  }
  .orb {
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, var(--accent), #1e3a8a);
    box-shadow: 0 0 60px var(--accent-glow);
    display: flex; align-items: center; justify-content: center;
    transition: all 0.3s ease;
    cursor: pointer;
    user-select: none;
    -webkit-user-select: none;
  }
  .orb.listening {
    background: radial-gradient(circle at 30% 30%, #60a5fa, #3b82f6);
    box-shadow: 0 0 80px var(--accent-glow);
    animation: pulse 1.2s ease-in-out infinite;
  }
  .orb.thinking {
    background: radial-gradient(circle at 30% 30%, #fbbf24, #f59e0b);
    box-shadow: 0 0 80px rgba(245, 158, 11, 0.5);
    animation: spin 2s linear infinite;
  }
  .orb.speaking {
    background: radial-gradient(circle at 30% 30%, #34d399, #10b981);
    box-shadow: 0 0 80px rgba(16, 185, 129, 0.5);
  }
  @keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.08); }
  }
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  .orb-label {
    margin-top: 24px;
    font-size: 14px;
    color: var(--text-dim);
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }
  .transcript {
    flex: 0 0 auto;
    padding: 20px;
    overflow-y: auto;
    max-height: 40vh;
    border-top: 1px solid var(--bg-elev);
  }
  .turn { margin-bottom: 16px; }
  .turn .role {
    font-size: 11px; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 4px;
  }
  .turn.user .role { color: #60a5fa; }
  .turn.assistant .role { color: #34d399; }
  .turn .text { font-size: 15px; line-height: 1.5; }
  .empty { color: var(--text-dim); text-align: center; padding: 20px; font-style: italic; }
  .controls {
    padding: 16px 20px;
    border-top: 1px solid var(--bg-elev);
    display: flex; gap: 12px; align-items: center;
  }
  button {
    background: var(--accent);
    color: white;
    border: none;
    padding: 12px 20px;
    border-radius: 8px;
    font-size: 14px; font-weight: 500;
    cursor: pointer;
    -webkit-appearance: none;
  }
  button.secondary {
    background: var(--bg-elev);
    color: var(--text-dim);
  }
  button:active { opacity: 0.7; }
  input[type="text"] {
    flex: 1;
    background: var(--bg-elev);
    color: var(--text);
    border: 1px solid transparent;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 15px;
    outline: none;
  }
  input[type="text"]:focus { border-color: var(--accent); }
  .mic-pulse { display: none; }
  .mic-pulse.active { display: inline-block; animation: pulse 1s ease-in-out infinite; }

  /* v0.9 hide-chat */
  .chat-shell { position: relative; }
  .chat-bar { display: flex; justify-content: flex-end; align-items: center; margin: 0 0 6px 0; }
  .hide-btn {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
    color: var(--text); font-size: 14px; line-height: 1; padding: 6px 10px;
    border-radius: 8px; cursor: pointer; user-select: none;
  }
  .hide-btn:hover { background: rgba(255,255,255,0.12); }
  .hide-btn.shown { opacity: 0.5; }
  .chat-shell.hidden .transcript,
  .chat-shell.hidden .controls { display: none !important; }
  .chat-shell.hidden .chat-bar { margin-top: 0; }
  .chat-shell.hidden { padding-bottom: 0; }
  body.chat-hidden .chat-shell { display: none; }
</style>
</head>
<body>
  <div class="header">
    <h1>sam</h1>
    <div class="status"><span class="dot" id="status-dot"></span><span id="status-text">idle</span></div>
    <div class="version" id="build-version" style="font-size:10px;color:#4a5a7a;margin-top:2px;">v0.9.1</div>
  </div>

  <div class="stage">
    <div class="orb" id="orb"></div>
    <div class="orb-label" id="orb-label">tap to talk</div>
  </div>

  <div class="chat-shell" id="chat-shell">
    <div class="chat-bar">
      <button class="hide-btn" id="hide-btn" title="Hide chat (Cmd+H)" aria-label="Hide chat">👁</button>
    </div>
    <div class="transcript" id="transcript">
      <div class="empty">conversation will appear here</div>
    </div>

    <div class="controls">
      <input type="text" id="text-input" placeholder="message sam..." autocomplete="off">
      <button id="send-btn">Send</button>
      <button class="secondary" id="clear-btn" title="Clear conversation">×</button>
    <button class="secondary" id="debug-btn" title="View debug log">🪲</button>
    <button class="secondary" id="audio-test-btn" title="Test audio output (plays a 440Hz tone)">🔊</button>
  </div>
  <div id="debug-panel" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.85);z-index:2000;padding:20px;overflow:auto;font-family:monospace;font-size:12px;color:#a7f3d0;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
      <strong>agent-voice-ai debug log</strong>
      <button id="debug-close" style="background:#374151;color:#fff;border:none;padding:6px 12px;border-radius:6px;">close</button>
    </div>
    <pre id="debug-log" style="white-space:pre-wrap;line-height:1.4;"></pre>
  </div>

<script>
const SESSION_ID = "ava-" + (localStorage.getItem("ava-session") || (() => {
  const id = Math.random().toString(36).slice(2, 10);
  localStorage.setItem("ava-session", id);
  return id;
})());

const orb = document.getElementById("orb");
const orbLabel = document.getElementById("orb-label");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const transcript = document.getElementById("transcript");
const textInput = document.getElementById("text-input");
const hideBtn = document.getElementById("hide-btn");
const chatShell = document.getElementById("chat-shell");

// v0.9 hide-chat toggle (persisted)
const HIDE_KEY = "sb_hide_chat_v1";
function applyHide(hidden) {
  document.body.classList.toggle("chat-hidden", hidden);
  hideBtn.textContent = hidden ? "👁‍🗨" : "👁";
  hideBtn.title = hidden ? "Show chat (Cmd+H)" : "Hide chat (Cmd+H)";
}
applyHide(localStorage.getItem(HIDE_KEY) === "1");
hideBtn.addEventListener("click", () => {
  const next = !document.body.classList.contains("chat-hidden");
  localStorage.setItem(HIDE_KEY, next ? "1" : "0");
  applyHide(next);
});
document.addEventListener("keydown", (e) => {
  const mod = e.metaKey || e.ctrlKey;
  if (mod && (e.key === "h" || e.key === "H")) {
    e.preventDefault();
    hideBtn.click();
  }
});
const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");

let state = "idle";
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

function setState(s) {
  state = s;
  orb.className = "orb";
  if (s === "listening") {
    orb.classList.add("listening");
    orbLabel.textContent = "listening...";
    statusText.textContent = "listening";
    statusDot.style.background = "var(--state-listening)";
  } else if (s === "thinking") {
    orb.classList.add("thinking");
    orbLabel.textContent = "thinking...";
    statusText.textContent = "thinking";
    statusDot.style.background = "var(--state-thinking)";
  } else if (s === "speaking") {
    orb.classList.add("speaking");
    orbLabel.textContent = "speaking";
    statusText.textContent = "speaking";
    statusDot.style.background = "var(--state-speaking)";
  } else if (s === "tap-to-unlock") {
    orb.classList.add("tap-to-unlock");
    orbLabel.textContent = "tap anywhere to resume audio";
    statusText.textContent = "tap to unlock";
    statusDot.style.background = "#ffcc00";
    statusDot.style.animation = "pulse 0.6s ease-in-out infinite";
  } else if (s === "error") {
    orb.classList.add("error");
    orbLabel.textContent = "error — see log";
    statusText.textContent = "error";
    statusDot.style.background = "#ff5050";
  } else {
    orbLabel.textContent = "tap to talk";
    statusText.textContent = "idle";
    statusDot.style.background = "var(--state-idle)";
    statusDot.style.animation = "none";
  }
}

function addTurn(role, text) {
  const empty = transcript.querySelector(".empty");
  if (empty) empty.remove();
  const turn = document.createElement("div");
  turn.className = "turn " + role;
  const label = role === "user" ? "you" : (role === "sam" ? "sam" : role);
  const replyId = "reply-" + Math.random().toString(36).slice(2, 10);
  if (role === "sam") {
    turn.innerHTML = `<div class="role">${label}</div><div class="text" id="${replyId}"></div><button class="play-btn" data-target="${replyId}" style="margin-top:6px;background:#374151;color:#a7f3d0;border:none;padding:4px 10px;border-radius:6px;font-size:12px;display:none;">▶ replay</button>`;
  } else {
    turn.innerHTML = `<div class="role">${label}</div><div class="text"></div>`;
  }
  turn.querySelector(".text").textContent = text;
  transcript.appendChild(turn);
  transcript.scrollTop = transcript.scrollHeight;
  // wire replay button
  if (role === "sam") {
    const btn = turn.querySelector(".play-btn");
    if (btn) {
      btn.onclick = () => {
        const url = btn.dataset.audioUrl;
        if (url) playAudio(url, btn);
        else dbg("replay: no audio_url stored on this turn");
      };
    }
  }
}

// --- debug logger + audio helper ---
// iOS Safari blocks Audio.play() if it isn't called inside (or chained to) a
// user gesture. A fetch that takes 5+ seconds breaks the gesture chain, so the
// first chat/listen call after page load hits "NotAllowedError" silently. The
// fix is to create the Audio element lazily, then unlock it with a silent
// 1-frame WAV on the very first user interaction (the first tap that triggers
// startMic or sendText). After that, play() works for the lifetime of the page.
let _audio = null;
let _audioUnlocked = false;
function _getAudio() {
  if (!_audio) {
    _audio = new Audio();
    _audio.preload = "auto";
  }
  return _audio;
}
function _unlockAudio() {
  if (_audioUnlocked) return;
  _audioUnlocked = true;
  // Tiny silent WAV (44 byte header + 0 samples) — play and immediately pause.
  // This satisfies iOS Safari's "user gesture" requirement for future play() calls.
  const silentWav = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=";
  const a = _getAudio();
  a.src = silentWav;
  a.volume = 0;
  const p = a.play();
  if (p && p.then) p.then(() => { a.pause(); a.currentTime = 0; a.volume = 1; dbg("audio: iOS unlock succeeded"); }).catch(e => dbg("audio: iOS unlock rejected " + e.name));
}
let _dbgLog = [];
function dbg(msg) {
  const ts = new Date().toISOString().slice(11, 23);
  const line = "[" + ts + "] " + msg;
  _dbgLog.push(line);
  if (_dbgLog.length > 200) _dbgLog = _dbgLog.slice(-200);
  console.log(line);
  const pre = document.getElementById("debug-log");
  if (pre) { pre.textContent = _dbgLog.join("\\n"); pre.scrollTop = pre.scrollHeight; }
  // also POST to server-side log for long-term debug
  fetch("/api/debug", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ts: ts, msg: msg, ua: navigator.userAgent }) }).catch(() => {});
}
function playAudio(url, btn) {
  dbg("playAudio: " + url);
  setState("speaking");
  // Reuse the same <audio> element so iOS doesn't keep allocating new players
  const a = _getAudio();
  a.src = url;
  a.onended = () => { dbg("playAudio: ended"); setState("idle"); };
  a.onerror = (e) => { dbg("playAudio: ERROR " + (a.error ? a.error.code : "?") + " — " + (a.error ? a.error.message : "unknown")); setState("idle"); };
  a.oncanplay = () => dbg("playAudio: canplay (duration=" + a.duration.toFixed(2) + "s)");
  const p = a.play();
  if (p && p.catch) {
    p.then(() => dbg("playAudio: promise resolved"))
     .catch(e => { dbg("playAudio: PROMISE REJECTED " + e.name + ": " + e.message); setState("idle"); });
  }
}
document.addEventListener("DOMContentLoaded", () => {
  const dbgBtn = document.getElementById("debug-btn");
  const dbgClose = document.getElementById("debug-close");
  const dbgPanel = document.getElementById("debug-panel");
  const audioTestBtn = document.getElementById("audio-test-btn");
  if (dbgBtn) dbgBtn.onclick = () => { dbgPanel.style.display = "block"; dbg("debug panel opened"); };
  if (dbgClose) dbgClose.onclick = () => { dbgPanel.style.display = "none"; };
  if (audioTestBtn) audioTestBtn.onclick = () => {
    _unlockAudio();
    // 440Hz tone, 0.4s, generated as a data: WAV so no network needed.
    const sr = 22050, dur = 0.4, freq = 440;
    const n = Math.floor(sr * dur);
    const buf = new ArrayBuffer(44 + n * 2);
    const v = new DataView(buf);
    const ws = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o+i, s.charCodeAt(i)); };
    ws(0, "RIFF"); v.setUint32(4, 36+n*2, true); ws(8, "WAVE");
    ws(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
    v.setUint16(22, 1, true); v.setUint32(24, sr, true); v.setUint32(28, sr*2, true);
    v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    ws(36, "data"); v.setUint32(40, n*2, true);
    for (let i = 0; i < n; i++) {
      // 5ms attack/release envelope to avoid clicks
      let env = 1;
      if (i < sr*0.01) env = i/(sr*0.01);
      else if (i > n - sr*0.05) env = (n-i)/(sr*0.05);
      v.setInt16(44 + i*2, Math.sin(2*Math.PI*freq*i/sr) * env * 16000, true);
    }
    // Convert to base64 data URL
    const bytes = new Uint8Array(buf);
    let bin = "";
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    const dataUrl = "data:audio/wav;base64," + btoa(bin);
    const a = _getAudio();
    a.src = dataUrl;
    a.play().then(() => dbg("audioTest: played 440Hz chime"))
             .catch(e => dbg("audioTest: FAILED " + e.name + ": " + e.message));
  };
  dbg("page loaded; ua=" + navigator.userAgent.slice(0, 80));
});

async function startMic() {
  if (isRecording) {
    stopMic();
    return;
  }
  // Detect whether this browser supports mic access
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showHint("This browser doesn't support microphone access. Open this page in Safari or Chrome to use voice. Text input still works below.");
    return;
  }
  // Some embedded browsers (Telegram, in-app webviews) lack getUserMedia entirely.
  if (window.location.protocol !== "https:" && window.location.hostname !== "127.0.0.1" && window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1") {
    showHint("Mic requires HTTPS or localhost. You're on " + window.location.protocol + "//" + window.location.hostname);
    return;
  }
  try {
    // v0.8: AudioContext with latencyHint 'interactive' shaves capture buffer
    // latency on iOS Safari from ~250ms (default 'interactive') to ~80ms.
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 16000,
      latencyHint: "interactive",
    });
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, sampleRate: 16000, echoCancellation: true, noiseSuppression: true }
    });

    // Try WebSocket push-to-talk path first (v0.8). Falls back to HTTP /api/listen
    // if WS is unavailable.
    if (typeof WebSocket !== "undefined") {
      try {
        await startMicWS(stream);
        return;
      } catch (e) {
        dbg("ws push-to-talk unavailable, using legacy /api/listen: " + e.message);
      }
    }

    // Legacy fallback: MediaRecorder → webm blob → /api/listen
    mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    audioChunks = [];
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = async () => {
      const blob = new Blob(audioChunks, { type: "audio/webm" });
      // Convert webm to wav via a quick decode
      const ctx = new AudioContext({ sampleRate: 16000 });
      const arrayBuf = await blob.arrayBuffer();
      const decoded = await ctx.decodeAudioData(arrayBuf);
      const pcm = decoded.getChannelData(0);
      const wav = encodeWav(pcm, 16000);
      await sendAudio(wav);
      stream.getTracks().forEach(t => t.stop());
    };
    mediaRecorder.start();
    isRecording = true;
    setState("listening");
  } catch (e) {
    showHint("Mic access failed: " + e.message + ". You can type messages below.");
  }
}

// ---- v0.8 WebSocket push-to-talk ----
// Stream MediaRecorder webm chunks over a single WebSocket to /ws. On stop,
// the server runs STT, then streams the brain reply sentence-by-sentence
// with Piper audio chunks interleaved. Audio playback is queued so we hear
// sentence 1 while sentence 2 is still being synthesized.
let _ws = null;
let _wsSentences = [];     // queue of base64 WAVs waiting to play
let _wsPlaying = false;    // are we currently playing back-to-back?
let _wsCurrentReplyEl = null; // live reply DOM element being appended-to
let _wsAudioEl = null;     // dedicated <audio> for sentence-streaming playback

function _wsAudio() {
  if (!_wsAudioEl) {
    _wsAudioEl = new Audio();
    _wsAudioEl.preload = "auto";
  }
  return _wsAudioEl;
}

function _wsPlayQueue() {
  if (_wsPlaying || _wsSentences.length === 0) return;
  if (_wsWaitingForGesture) {
    setState("tap-to-unlock");
    return;
  }
  _wsPlaying = true;
  const wav_b64 = _wsSentences.shift();
  const dataUrl = "data:audio/wav;base64," + wav_b64;
  const a = _wsAudio();
  a.src = dataUrl;
  a.onended = () => {
    _wsPlaying = false;
    _wsPlayQueue();
  };
  a.onerror = () => {
    _wsPlaying = false;
    _wsSentences.unshift(wav_b64);
    _wsWaitingForGesture = true;
    setState("tap-to-unlock");
  };
  setState("speaking");
  const p = a.play();
  if (p && p.catch) p.catch((e) => {
    _wsPlaying = false;
    if (e && (e.name === "AbortError" || e.name === "NotAllowedError")) {
      _wsSentences.unshift(wav_b64);
      _wsWaitingForGesture = true;
      setState("tap-to-unlock");
      dbg("audio: " + e.name + " — queued, tap anywhere to resume");
    } else {
      _wsPlayQueue();
    }
  });
}

// If we hit an iOS audio-context unlock failure, the next pointerdown or
// touchstart anywhere on the page is a valid user gesture to retry.
let _wsWaitingForGesture = false;
function _wsGestureUnlock() {
  if (!_wsWaitingForGesture) return;
  _wsWaitingForGesture = false;
  // Re-prime the audio element with a silent play to satisfy the gesture,
  // then drain the queue.
  const a = _wsAudio();
  try {
    a.src = "";
    const p = a.play();
    if (p && p.then) p.then(() => { a.pause(); a.currentTime = 0; dbg("audio: gesture unlock ok"); _wsPlayQueue(); }).catch(() => { _wsPlayQueue(); });
    else _wsPlayQueue();
  } catch (e) { _wsPlayQueue(); }
}
document.addEventListener("pointerdown", _wsGestureUnlock, { passive: true });
document.addEventListener("touchstart", _wsGestureUnlock, { passive: true });

function _wsReset() {
  _wsSentences = [];
  _wsPlaying = false;
  _wsCurrentReplyEl = null;
  setState("idle");
}

async function startMicWS(stream) {
  // Decide on a mimeType the browser actually supports. iOS Safari often
  // refuses audio/webm; fallback to audio/mp4.
  const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
    ? "audio/webm;codecs=opus"
    : (MediaRecorder.isTypeSupported("audio/mp4") ? "audio/mp4" : "audio/webm");

  mediaRecorder = new MediaRecorder(stream, { mimeType: mime });
  audioChunks = [];

  // Open the WS
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  _ws = new WebSocket(proto + "//" + window.location.host + "/ws");
  _wsSentences = [];
  _wsPlaying = false;
  _wsCurrentReplyEl = null;

  await new Promise((resolve, reject) => {
    _ws.onopen = () => {
      _ws.send(JSON.stringify({ type: "start", session_id: SESSION_ID }));
      resolve();
    };
    _ws.onerror = e => { reject(new Error("ws connect failed")); };
    setTimeout(() => reject(new Error("ws timeout")), 3000);
  });

  // Stream chunks as MediaRecorder emits them. timeslice=100 → ~100ms chunks.
  mediaRecorder.ondataavailable = e => {
    if (e.data.size > 0) {
      audioChunks.push(e.data);
      // Send this chunk over WS as base64. We could send binary directly,
      // but JSON keeps the protocol uniform with the rest of /ws.
      const reader = new FileReader();
      reader.onload = () => {
        const b64 = reader.result.split(",")[1];
        if (_ws && _ws.readyState === 1) {
          _ws.send(JSON.stringify({
            type: "audio",
            mime: mime,
            data_b64: b64,
          }));
        }
      };
      reader.readAsDataURL(e.data);
    }
  };

  mediaRecorder.onstop = () => {
    if (_ws && _ws.readyState === 1) {
      _ws.send(JSON.stringify({ type: "stop" }));
    }
    stream.getTracks().forEach(t => t.stop());
  };

  // WS message handler
  _ws.onmessage = (e) => {
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }
    const t = msg.type;
    if (t === "state") {
      setState(msg.state);
    } else if (t === "transcript") {
      // STT result landed — show it as a "you" turn
      if (msg.final) {
        addTurn("user", msg.final);
        dbg("stt: " + msg.stt_ms + "ms → " + JSON.stringify(msg.final));
      }
    } else if (t === "text") {
      // Stream the reply text into the live DOM element so it appears
      // word-by-word as the brain produces it.
      if (!_wsCurrentReplyEl) {
        const empty = transcript.querySelector(".empty");
        if (empty) empty.remove();
        const turn = document.createElement("div");
        turn.className = "turn sam";
        turn.innerHTML = `<div class="role">sam</div><div class="text"></div>`;
        transcript.appendChild(turn);
        _wsCurrentReplyEl = turn.querySelector(".text");
      }
      _wsCurrentReplyEl.textContent += msg.text;
      transcript.scrollTop = transcript.scrollHeight;
    } else if (t === "audio") {
      // Queue sentence audio. _wsPlayQueue() picks the next one and chains.
      _wsSentences.push(msg.data_b64);
      _wsPlayQueue();
    } else if (t === "pending_confirm") {
      addTurn("sam", "⚠ " + msg.summary);
      setState("idle");
    } else if (t === "done") {
      dbg("ws done: " + msg.reply_len + " chars in " + msg.total_ms + "ms");
      _wsCurrentReplyEl = null;
    } else if (t === "pong") {
      // keepalive
    } else {
      dbg("ws unhandled msg type=" + t);
    }
  };

  _ws.onclose = () => {
    dbg("ws closed");
    _ws = null;
    if (state === "listening" || state === "thinking") _wsReset();
  };

  mediaRecorder.start(100);  // 100ms timeslice = ~10 chunks/sec streaming
  isRecording = true;
  setState("listening");
}

let _hintTimeout = null;
function showHint(msg) {
  let hint = document.getElementById("hint");
  if (!hint) {
    hint = document.createElement("div");
    hint.id = "hint";
    hint.style.cssText = "position:fixed;bottom:80px;left:20px;right:20px;background:#1e3a8a;color:#fff;padding:14px 18px;border-radius:12px;font-size:14px;line-height:1.4;z-index:1000;box-shadow:0 4px 20px rgba(0,0,0,0.4);";
    document.body.appendChild(hint);
  }
  hint.textContent = msg;
  hint.style.display = "block";
  if (_hintTimeout) clearTimeout(_hintTimeout);
  _hintTimeout = setTimeout(() => { hint.style.display = "none"; }, 8000);
}

function stopMic() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  isRecording = false;
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off+i, s.charCodeAt(i)); };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

async function sendAudio(wavBlob) {
  setState("thinking");
  const fd = new FormData();
  fd.append("audio", wavBlob, "input.wav");
  fd.append("session_id", SESSION_ID);
  dbg("sendAudio: " + wavBlob.size + " bytes");
  try {
    const r = await fetch("/api/listen", { method: "POST", body: fd });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    dbg("listen reply: user=" + JSON.stringify(data.user_text) + " sam=" + JSON.stringify(data.reply_text) + " audio=" + (data.audio_url || "NONE") + " size=" + (data.audio_size || 0));
    if (data.user_text) addTurn("user", data.user_text);
    if (data.reply_text) addTurn("sam", data.reply_text);
    if (data.audio_url) {
      const btn = transcript.querySelector(".turn:last-child .play-btn");
      if (btn) { btn.dataset.audioUrl = data.audio_url; btn.style.display = "inline-block"; }
      playAudio(data.audio_url, btn);
    } else {
      dbg("listen: no audio_url returned — server-side synth returned empty");
      setState("idle");
    }
  } catch (e) {
    dbg("sendAudio FAILED: " + e.message);
    showHint("send failed: " + e.message);
    setState("idle");
  }
}

async function sendText() {
  const text = textInput.value.trim();
  if (!text) return;
  textInput.value = "";
  addTurn("user", text);
  setState("thinking");
  try {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, session_id: SESSION_ID }),
    });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    if (data.reply_text) addTurn("sam", data.reply_text);
    renderPending(data.pending_confirm);
    renderToolLog(data.tool_log || []);
    if (data.audio_url) {
      const btn = transcript.querySelector(".turn:last-child .play-btn");
      if (btn) { btn.dataset.audioUrl = data.audio_url; btn.style.display = "inline-block"; }
      playAudio(data.audio_url, btn);
    } else {
      dbg("chat: no audio_url returned");
      setState("idle");
    }
  } catch (e) {
    alert("send failed: " + e.message);
    setState("idle");
  }
}

function renderPending(pending) {
  let el = document.getElementById("pending-banner");
  if (!el) {
    el = document.createElement("div");
    el.id = "pending-banner";
    el.style.cssText = "margin:10px 0;padding:10px 14px;border-radius:8px;background:#7c2d12;color:#fed7aa;font-size:13px;display:none;";
    transcript.parentNode.insertBefore(el, transcript.nextSibling);
  }
  if (!pending) { el.style.display = "none"; el.textContent = ""; return; }
  el.style.display = "block";
  el.innerHTML = `<strong>⚠ Held action:</strong> ${pending.summary}. <em>Say "do it" or "cancel".</em>`;
}

function renderToolLog(entries) {
  let el = document.getElementById("tool-log");
  if (!el) {
    el = document.createElement("div");
    el.id = "tool-log";
    el.style.cssText = "margin:10px 0;padding:8px 12px;border-radius:8px;background:#0f172a;color:#94a3b8;font-size:11px;font-family:monospace;";
    transcript.parentNode.insertBefore(el, transcript.nextSibling);
  }
  if (!entries.length) { el.style.display = "none"; el.textContent = ""; return; }
  el.style.display = "block";
  el.innerHTML = "<strong style='color:#cbd5e1'>tool calls:</strong><br>" +
    entries.map(e => {
      const color = e.tier === 'external_write' ? '#fb923c' : (e.tier === 'local_write' ? '#a3e635' : '#60a5fa');
      return `<span style="color:${color}">[${e.tier}]</span> ${e.tool}(${JSON.stringify(e.args).slice(0,60)}) → ${e.result_summary}`;
    }).join("<br>");
}

orb.addEventListener("click", () => { _unlockAudio(); startMic(); });
sendBtn.addEventListener("click", () => { _unlockAudio(); sendText(); });
textInput.addEventListener("keydown", e => { if (e.key === "Enter") { _unlockAudio(); sendText(); } });
clearBtn.addEventListener("click", async () => {
  await fetch("/api/clear", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: SESSION_ID }),
  });
  transcript.innerHTML = '<div class="empty">conversation cleared</div>';
});

setState("idle");
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(
        content=HTML,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


def _whisper_reachable() -> bool:
    try:
        r = requests.get("http://127.0.0.1:8178/", timeout=2)
        return r.status_code < 500
    except Exception:
        return False


@app.post("/api/chat")
async def chat(payload: dict):
    """Text in, text + audio out. Returns JSON with reply_text and audio URL.

    Agent-aware: streams text through the brain's agent loop, runs tools,
    and surfaces any pending confirmation request for irreversible actions.
    """
    text = (payload.get("text") or "").strip()
    session_id = payload.get("session_id") or f"ava-{uuid.uuid4().hex[:8]}"
    if not text:
        raise HTTPException(400, "text is required")

    log.info("chat: session=%s text=%r", session_id, text[:80])

    # Resolve any pending confirmation request before treating this as a new turn
    pending = brain.get_pending_confirm(session_id)
    if pending:
        decision = brain._looks_like_confirmation(text)
        if decision == "yes":
            result = brain.resolve_pending(session_id, "yes")
            log.info("chat: confirmation YES for session=%s → %s", session_id, result)
            reply_text = _format_confirmation_result(pending, result)
        elif decision == "no":
            cancelled = brain.resolve_pending(session_id, "no")
            log.info("chat: confirmation NO for session=%s", session_id)
            reply_text = f"Cancelled. I won't {pending['summary']}."
        else:
            # Not a yes/no — but there's a pending action. Clear it; treat
            # the reply as a fresh turn.
            log.info("chat: pending hold dropped, treating as new turn (session=%s)", session_id)
            brain.resolve_pending(session_id, "no")  # discard
            reply_text = await _brain_speak(text, session_id)
    else:
        reply_text = await _brain_speak(text, session_id)

    # Persist turn
    brain.append_turn(session_id, text, reply_text)

    # Synthesize
    wav_bytes, tts_provider = b"", "none"
    if reply_text.strip():
        loop = asyncio.get_event_loop()
        try:
            wav_bytes, tts_provider = await loop.run_in_executor(
                None, _synth_with_fallback, reply_text
            )
            log.info("chat: tts=%s bytes=%d", tts_provider, len(wav_bytes))
        except Exception as e:
            log.exception("synth failed: %s", e)
            wav_bytes, tts_provider = b"", "none"

    # Cache + URL
    audio_url = None
    if wav_bytes:
        import uuid as _uuid
        audio_id = f"{int(time.time()*1000)}-{_uuid.uuid4().hex[:6]}.wav"
        _AUDIO_CACHE[audio_id] = wav_bytes
        if len(_AUDIO_CACHE) > 50:
            oldest = next(iter(_AUDIO_CACHE))
            _AUDIO_CACHE.pop(oldest, None)
        audio_url = f"/api/audio/{audio_id}"

    # Surface pending confirm + tool log so the UI can show them
    held = brain.get_pending_confirm(session_id)
    return JSONResponse({
        "session_id": session_id,
        "user_text": text,
        "reply_text": reply_text,
        "audio_url": audio_url,
        "audio_size": len(wav_bytes) if wav_bytes else 0,
        "pending_confirm": held,
        "tool_log": brain.get_tool_log(session_id)[-10:],
    })


# ---- streaming TTS infrastructure ----
# v0.8: instead of waiting for the brain's full reply before synthesizing,
# we consume brain.stream_reply live, split text on sentence boundaries, and
# fire Piper per-sentence so the first WAV chunk can ship back to the client
# while the brain is still composing later ones.

import re as _re

_SENT_END = _re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])|(?<=[.!?])$")
_SENT_FLOOR = 12   # don't fire Piper on a sentence shorter than this (chars)
_SENT_CEIL = 180   # hard cap — Piper quality drops on very long inputs


def _split_sentences(buffer: str) -> tuple[list[str], str]:
    """Split *buffer* on sentence boundaries; keep the trailing partial.

    Returns (complete_sentences, remainder). We never split mid-sentence —
    Piper gets whole sentences, not fragments, so intonation stays clean.
    """
    parts = _SENT_END.split(buffer)
    # _SENT_END.split returns: [text_before, sep, sentence, sep, sentence, ...]
    # We want the trailing complete ones. Strategy: every split produces
    # either a sep (whitespace) or text. A "complete sentence" is text that
    # is followed by a sep. The very last element is whatever trailing
    # partial hasn't been split yet.
    sentences: list[str] = []
    remainder: str = ""
    # Re-join with explicit separator tracking
    # Simpler approach: split on the regex matches directly
    last_end = 0
    for m in _SENT_END.finditer(buffer):
        end = m.end()
        sent = buffer[last_end:end].strip()
        if sent:
            sentences.append(sent)
        last_end = end
    remainder = buffer[last_end:]
    return sentences, remainder


async def _stream_brain_to_sentences(
    user_text: str,
    session_id: str,
    on_sentence,
) -> str:
    """Consume brain.stream_reply live. For each completed sentence, call
    on_sentence(sentence_str) — that callback fires Piper and ships audio.

    Returns the full concatenated reply text (also passed to brain.append_turn
    by the caller). Non-text events (tool_call, pending_confirm, done) are
    ignored here — the caller wires them as needed.
    """
    history = brain.get_history(session_id)
    text_buf: list[str] = []
    pending: str = ""  # un-split tail
    t0 = time.time()
    loop = asyncio.get_event_loop()

    def _consume():
        return brain.stream_reply(
            user_text, history=history, session_id=session_id,
        )

    # Run the blocking sync generator in a thread; iterate as it produces.
    # We bridge sync-iter → async by polling the generator in chunks so the
    # event loop can interleave Piper synth + websocket sends.
    import queue
    q: queue.Queue = queue.Queue(maxsize=64)

    def _producer():
        try:
            for evt in _consume():
                q.put(("evt", evt))
        except Exception as e:
            q.put(("err", e))
        finally:
            q.put(("end", None))

    prod_task = loop.run_in_executor(None, _producer)

    while True:
        # Non-blocking poll so the event loop stays responsive
        try:
            item = q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue

        kind, payload = item
        if kind == "err":
            log.exception("brain stream error")
            text_buf.append(f"\n\nI hit an error: {payload}\n")
            break
        if kind == "end":
            break

        evt = payload
        etype = evt.get("type")
        if etype == "text":
            delta = evt.get("delta", "")
            if not delta:
                continue
            pending += delta
            # Split off any complete sentences
            sentences, pending = _split_sentences(pending)
            for sent in sentences:
                # Skip too-short fragments (mid-word matches), but don't
                # starve the pipeline — force-flush if pending is huge.
                if len(sent) < _SENT_FLOOR and len(pending) < _SENT_CEIL:
                    pending = sent + " " + pending
                    continue
                text_buf.append(sent)
                # Fire the sentence callback (Piper + ship audio)
                try:
                    await on_sentence(sent)
                except Exception as e:
                    log.warning("on_sentence failed: %s", e)
        # tool_call / pending_confirm / done / tool_result are ignored here —
        # the WS handler wires them separately if it cares.

    # Flush the trailing partial as a final sentence (even if short).
    if pending.strip():
        sentences, _ = _split_sentences(pending + ".")
        # The above returns sentences inside `pending` even though we passed
        # the whole thing; we just want the original pending back.
        text_buf.append(pending.strip())

    full = " ".join(text_buf).strip()
    log.info(
        "stream_brain: %.2fs → %d chars (%d sentences)",
        time.time() - t0, len(full), full.count(".!?") + full.count("."),
    )
    return full


async def _brain_speak(user_text: str, session_id: str) -> str:
    """Run the agent loop, collect text deltas, return concatenated reply.

    v0.8: still used by /api/chat and /api/listen for the legacy path.
    Streaming consumers should use _stream_brain_to_sentences.
    """
    history = brain.get_history(session_id)
    text_parts: list[str] = []
    t0 = time.time()
    loop = asyncio.get_event_loop()
    try:
        # stream_reply is a sync generator; run in a thread so the event loop
        # can still breathe while the API streams.
        def _consume():
            return list(brain.stream_reply(
                user_text, history=history, session_id=session_id,
            ))
        events = await loop.run_in_executor(None, _consume)
        for evt in events:
            if evt.get("type") == "text":
                text_parts.append(evt.get("delta", ""))
        reply_text = "".join(text_parts).strip()
        log.info("chat: brain %.2fs → %d chars (%d events)", time.time() - t0,
                 len(reply_text), len(events))
        return reply_text
    except Exception as e:
        log.exception("brain.stream_reply failed")
        return f"I hit an error: {e}"


def _format_confirmation_result(pending: dict, result: dict | None) -> str:
    """Build the spoken reply after a 'yes' confirmation."""
    if result is None:
        return "Sorry, the action dropped. Try again?"
    if result.get("cancelled"):
        return f"Cancelled."
    if "error" in result:
        return f"I tried but hit an error: {result['error']}"
    if pending["tool"] == "send_message" and result.get("ok"):
        mid = result.get("message_id", "?")
        return f"Sent. Message id {mid}."
    if pending["tool"] == "send_message":
        return f"I couldn't send it — check the bridge log."
    return "Done."


# Cache recent audio responses in memory for the /api/audio/... route
_AUDIO_CACHE: dict[str, bytes] = {}


# Server-side debug log: collects browser events POSTed from /api/debug so we
# can troubleshoot long-term audio dropouts on mobile.
_CLIENT_DEBUG_LOG: list[dict] = []


@app.post("/api/debug")
async def client_debug(payload: dict):
    """Receive debug events from the web UI; append to in-memory ring + log file."""
    msg = payload.get("msg", "")
    ts = payload.get("ts", "")
    ua = payload.get("ua", "")
    entry = {"ts": ts, "msg": msg, "ua": ua}
    _CLIENT_DEBUG_LOG.append(entry)
    if len(_CLIENT_DEBUG_LOG) > 500:
        del _CLIENT_DEBUG_LOG[: len(_CLIENT_DEBUG_LOG) - 500]
    log.info("client-debug: [%s] %s (ua=%s)", ts, msg, ua[:40])
    return {"ok": True, "logged": len(_CLIENT_DEBUG_LOG)}


@app.get("/api/debug")
async def get_client_debug():
    """Retrieve the last 100 client debug events. Useful when the user reports an issue."""
    return {"events": _CLIENT_DEBUG_LOG[-100:]}


@app.get("/api/audio/{audio_id}")
async def get_audio(audio_id: str):
    """Return a cached audio response (24kHz mono WAV)."""
    wav = _AUDIO_CACHE.get(audio_id)
    if not wav:
        raise HTTPException(404, "audio not found or expired")
    return Response(content=wav, media_type="audio/wav", headers={
        "Content-Disposition": f"inline; filename={audio_id}",
        "Cache-Control": "public, max-age=300",
    })


@app.post("/api/listen")
async def listen(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    """Audio in (WAV/webm/m4a), text + audio out. Returns JSON like /api/chat."""
    sid = session_id or f"ava-{uuid.uuid4().hex[:8]}"
    log.info("listen: session=%s file=%s content_type=%s", sid, audio.filename, audio.content_type)

    audio_bytes = await audio.read()
    log.info("listen: got %d bytes", len(audio_bytes))

    # Step 1: STT
    t0 = time.time()
    try:
        user_text = await asyncio.get_event_loop().run_in_executor(
            None, whisper_stt.transcribe, audio_bytes
        )
    except Exception as e:
        log.exception("whisper_stt.transcribe failed")
        raise HTTPException(500, f"stt error: {e}")
    log.info("listen: stt %.2fs → %r", time.time() - t0, user_text)

    if not user_text:
        return JSONResponse({
            "session_id": sid,
            "user_text": "",
            "reply_text": "I didn't catch that. Could you try again?",
            "audio_url": None,
        })

    # Step 2: handle confirmation OR speak (mirrors /api/chat)
    pending = brain.get_pending_confirm(sid)
    if pending:
        decision = brain._looks_like_confirmation(user_text)
        if decision == "yes":
            result = brain.resolve_pending(sid, "yes")
            log.info("listen: confirmation YES for session=%s → %s", sid, result)
            reply_text = _format_confirmation_result(pending, result)
        elif decision == "no":
            brain.resolve_pending(sid, "no")
            log.info("listen: confirmation NO for session=%s", sid)
            reply_text = f"Cancelled. I won't {pending['summary']}."
        else:
            brain.resolve_pending(sid, "no")
            log.info("listen: pending hold dropped, treating as new turn (session=%s)", sid)
            reply_text = await _brain_speak(user_text, sid)
    else:
        reply_text = await _brain_speak(user_text, sid)

    log.info("listen: brain → %d chars", len(reply_text))
    brain.append_turn(sid, user_text, reply_text)

    # Step 3: TTS
    wav_bytes, tts_provider = b"", "none"
    if reply_text.strip():
        loop = asyncio.get_event_loop()
        try:
            wav_bytes, tts_provider = await loop.run_in_executor(
                None, _synth_with_fallback, reply_text
            )
            log.info("listen: tts=%s bytes=%d", tts_provider, len(wav_bytes))
        except Exception as e:
            log.exception("synth failed: %s", e)

    audio_url = None
    if wav_bytes:
        import uuid as _uuid
        audio_id = f"{int(time.time()*1000)}-{_uuid.uuid4().hex[:6]}.wav"
        _AUDIO_CACHE[audio_id] = wav_bytes
        if len(_AUDIO_CACHE) > 50:
            oldest = next(iter(_AUDIO_CACHE))
            _AUDIO_CACHE.pop(oldest, None)
        audio_url = f"/api/audio/{audio_id}"

    return JSONResponse({
        "session_id": sid,
        "user_text": user_text,
        "reply_text": reply_text,
        "audio_url": audio_url,
        "audio_size": len(wav_bytes) if wav_bytes else 0,
        "pending_confirm": brain.get_pending_confirm(sid),
        "tool_log": brain.get_tool_log(sid)[-10:],
    })


@app.post("/api/clear")
async def clear(payload: dict):
    session_id = payload.get("session_id")
    if session_id:
        brain.clear_session(session_id)
    return {"cleared": session_id}


@app.get("/api/history")
async def history(session_id: str):
    return {"session_id": session_id, "history": brain.get_history(session_id)}


@app.get("/api/memory")
async def memory_context_endpoint(q: str = ""):
    """Show the memory context that would be injected for query `q`.

    Used for transparency — see exactly what the agent will "remember"
    before answering. No LLM call is made; this just builds the same
    memory context brain.py uses.
    """
    import memory
    ctx = memory.build_memory_context(q or "general context")
    sections = {}
    current = None
    buf = []
    for line in ctx.split("\n"):
        if line.startswith("## "):
            if current:
                sections[current] = "\n".join(buf).strip()
            current = line[3:].strip()
            buf = []
        else:
            buf.append(line)
    if current:
        sections[current] = "\n".join(buf).strip()
    return {
        "query": q,
        "total_chars": len(ctx),
        "sections": sections,
        "sessions_on_disk": memory.session_count(),
    }


@app.get("/api/context")
async def live_context_endpoint(q: str = ""):
    """Show what live cross-surface context would be loaded for query `q`.

    Combines:
      - skills_bridge.build_skills_context(q)  → active + triggered skills
      - telegram_sync.get_full_context()       → recent Telegram updates

    No LLM call; just inspection of the actual context blocks brain.py would inject.
    """
    import skills_bridge
    import telegram_sync

    query = q or "general"
    skills_ctx = skills_bridge.build_skills_context(query)
    tg_ctx = telegram_sync.get_full_context(limit_chars=1500)
    tg_state_file = (Path.home() / ".hermes" / "telegram_sync" / "state.json")
    cache_file = (Path.home() / ".hermes" / "telegram_sync" / "cache.json")
    return {
        "query": query,
        "skills_chars": len(skills_ctx),
        "skills_preview": skills_ctx[:1500],
        "telegram_chars": len(tg_ctx),
        "telegram_preview": tg_ctx[:1500],
        "telegram_state_exists": tg_state_file.exists(),
        "telegram_cache_exists": cache_file.exists(),
        "available_skill_count": len(skills_bridge.list_skill_names()),
    }


@app.get("/api/health")
async def health():
    """Bridge health + version + memory layer status."""
    import memory
    return {
        "status": "ok",
        "service": "sam-bridge",
        "version": app.version,
        "memory_layer": {
            "soul_md": bool(memory.SOUL_PATH.exists()),
            "sam_index": bool(memory.SAM_INDEX_PATH.exists()),
            "fact_store_db": memory.FACT_STORE_DB.exists(),
            "life_memory_dir": memory.LONG_TERM_MEMORY_DIR.exists(),
            "sessions_on_disk": memory.session_count(),
        },
        "tts_chain": "piper (libritts_r) -> kokoro (am_adam) fallback",
        "stt": "whisper.cpp on 127.0.0.1:8178",
        "telegram_poller": _telegram_poller_status(),
    }


def _telegram_poller_status() -> dict:
    """Inspect telegram_sync state for /api/health."""
    try:
        import telegram_sync
        return {
            "bot_token_loaded": bool(telegram_sync.BOT_TOKEN),
            "home_channel_loaded": bool(telegram_sync.HOME_CHANNEL_ID),
            "poller_running": telegram_sync._poller_thread is not None
                              and telegram_sync._poller_thread.is_alive(),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    """WebSocket for real-time push-to-talk.

    Wire protocol (v0.8):

      client → server:
        {"type": "start", "session_id": "..."}    open a turn
        {"type": "audio", "data_b64": "..."}       raw 16kHz mono s16le PCM chunk
        {"type": "stop"}                            end of utterance → run brain
        {"type": "text", "text": "...", "session_id": "..."}  text-only turn
        {"type": "confirm", "decision": "yes"|"no"}  resolve pending action
        {"type": "ping"}

      server → client:
        {"type": "state", "state": "listening"|"thinking"|"speaking"|"idle"}
        {"type": "transcript", "partial": "...", "final": "..."}  STT progress
        {"type": "text", "role": "sam", "text": "...", "stream": true}
        {"type": "audio", "format": "wav", "sample_rate": 22050,
         "data_b64": "..."}        one sentence of Piper audio at a time
        {"type": "pending_confirm", "tool": "...", "summary": "..."}
        {"type": "tool_call", ...}
        {"type": "done", "stop_reason": "...", "audio_chunks": N}
        {"type": "pong"}
    """
    await websocket.accept()
    log.info("ws: client connected")
    sid: str = f"ava-{uuid.uuid4().hex[:8]}"
    # audio_chunks holds base64-decoded webm/mp4 bytes from the browser
    # MediaRecorder. We accumulate them, then run whisper_stt.transcribe
    # on the concatenated blob on `stop`.
    audio_chunks: list[bytes] = []
    audio_mime: str = "audio/webm"
    loop = asyncio.get_event_loop()

    async def synth_and_ship(sentence: str) -> None:
        """Fire Piper on a single sentence and ship both the text delta
        and the WAV over WS so the UI can render text + queue audio."""
        try:
            # Send the text delta first so the UI shows it word-by-word
            await websocket.send_json({
                "type": "text", "role": "sam", "text": sentence + " ",
                "stream": True,
            })
            wav, provider = await loop.run_in_executor(
                None, _synth_with_fallback, sentence
            )
            if wav:
                import base64
                await websocket.send_json({
                    "type": "audio",
                    "format": "wav",
                    "sample_rate": piper_tts.SAMPLE_RATE,
                    "data_b64": base64.b64encode(wav).decode("ascii"),
                    "sentence": sentence,
                    "provider": provider,
                })
        except Exception as e:
            log.warning("ws synth failed for sentence %r: %s", sentence[:40], e)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = msg.get("type")

            if kind == "ping":
                await websocket.send_json({"type": "pong", "ts": time.time()})
                continue

            if kind == "start":
                sid = msg.get("session_id") or sid
                audio_chunks = []
                audio_mime = msg.get("mime") or "audio/webm"
                await websocket.send_json({
                    "type": "state", "state": "listening",
                    "session_id": sid,
                })
                continue

            if kind == "audio":
                # Browser sends base64-encoded webm/mp4 MediaRecorder chunks
                # at ~10/sec (timeslice=100ms). We accumulate; server-side
                # STT runs on the concatenated blob at `stop`.
                import base64
                try:
                    chunk = base64.b64decode(msg.get("data_b64", ""))
                    if chunk:
                        audio_chunks.append(chunk)
                except Exception as e:
                    log.warning("ws audio chunk decode failed: %s", e)
                continue

            if kind == "stop":
                # User released the button. Concatenate audio chunks,
                # run STT, then stream the brain reply back sentence-by-
                # sentence with Piper chunks interleaved.
                await websocket.send_json({"type": "state", "state": "thinking"})
                t0 = time.time()
                user_text = ""
                if audio_chunks:
                    blob = b"".join(audio_chunks)
                    log.info(
                        "ws stop: %d chunks, %d bytes, mime=%s",
                        len(audio_chunks), len(blob), audio_mime,
                    )
                    try:
                        # whisper.cpp server rejects non-WAV (HTTP 400). The
                        # browser sends webm/opus from MediaRecorder, so we
                        # transcode to 16k mono PCM/WAV with ffmpeg first.
                        import subprocess, tempfile
                        wav_bytes = blob
                        if "wav" not in audio_mime.lower():
                            with tempfile.NamedTemporaryFile(
                                suffix="\.webm", delete=False
                            ) as tf_in, tempfile.NamedTemporaryFile(
                                suffix=".wav", delete=False
                            ) as tf_out:
                                tf_in.write(blob)
                                tf_in.flush()
                                tf_out_path = tf_out.name
                            subprocess.run(
                                [
                                    "ffmpeg", "-y", "-loglevel", "error",
                                    "-i", tf_in.name,
                                    "-ac", "1", "-ar", "16000",
                                    "-f", "wav", tf_out_path,
                                ],
                                check=True,
                            )
                            with open(tf_out_path, "rb") as f:
                                wav_bytes = f.read()
                            os.unlink(tf_in.name)
                            os.unlink(tf_out_path)
                            log.info(
                                "ws transcoded webm→wav: %d → %d bytes",
                                len(blob), len(wav_bytes),
                            )
                        user_text = await loop.run_in_executor(
                            None, whisper_stt.transcribe, wav_bytes
                        )
                    except Exception as e:
                        log.exception("ws whisper failed")
                        user_text = ""
                audio_chunks = []
                stt_ms = int((time.time() - t0) * 1000)

                if not user_text.strip():
                    await websocket.send_json({
                        "type": "transcript", "partial": "", "final": "",
                        "stt_ms": stt_ms,
                    })
                    await websocket.send_json({
                        "type": "text", "role": "sam",
                        "text": "I didn't catch that. Try again?",
                    })
                    await websocket.send_json({
                        "type": "done", "stop_reason": "end_turn",
                        "reply_text": "I didn't catch that. Try again?",
                        "reply_len": 31,
                        "total_ms": int((time.time() - t0) * 1000),
                    })
                    await websocket.send_json({"type": "state", "state": "idle"})
                    continue

                await websocket.send_json({
                    "type": "transcript", "final": user_text, "stt_ms": stt_ms,
                })

                # Handle confirmation cycle on a held action
                pending = brain.get_pending_confirm(sid)
                if pending:
                    decision = brain._looks_like_confirmation(user_text)
                    if decision == "yes":
                        result = brain.resolve_pending(sid, "yes")
                        reply_text = _format_confirmation_result(pending, result)
                        await websocket.send_json({
                            "type": "text", "role": "sam", "text": reply_text,
                        })
                        await synth_and_ship(reply_text)
                        brain.append_turn(sid, user_text, reply_text)
                        await websocket.send_json({
                            "type": "done", "stop_reason": "end_turn",
                            "reply_text": reply_text, "reply_len": len(reply_text),
                            "total_ms": int((time.time() - t0) * 1000),
                        })
                        await websocket.send_json({"type": "state", "state": "idle"})
                        continue
                    elif decision == "no":
                        brain.resolve_pending(sid, "no")
                        reply_text = f"Cancelled. I won't {pending['summary']}."
                        await websocket.send_json({
                            "type": "text", "role": "sam", "text": reply_text,
                        })
                        await synth_and_ship(reply_text)
                        brain.append_turn(sid, user_text, reply_text)
                        await websocket.send_json({
                            "type": "done", "stop_reason": "end_turn",
                            "reply_text": reply_text, "reply_len": len(reply_text),
                            "total_ms": int((time.time() - t0) * 1000),
                        })
                        await websocket.send_json({"type": "state", "state": "idle"})
                        continue
                    else:
                        brain.resolve_pending(sid, "no")  # off-topic → drop

                # Stream the brain reply sentence-by-sentence.
                # Capture non-text events so we can relay them to the client.
                non_text_events: list[dict] = []

                async def on_sentence(sent: str):
                    await synth_and_ship(sent)

                # Run the streaming brain
                await websocket.send_json({"type": "state", "state": "speaking"})
                full = await _stream_brain_to_sentences(
                    user_text, sid, on_sentence,
                )
                brain.append_turn(sid, user_text, full)

                # Send any deferred tool/confirm events
                held = brain.get_pending_confirm(sid)
                if held:
                    log.info("ws: sending pending_confirm for sid=%s tool=%s", sid, held["tool"])
                    await websocket.send_json({
                        "type": "pending_confirm",
                        "tool": held["tool"], "summary": held["summary"],
                    })

                await websocket.send_json({
                    "type": "done",
                    "stop_reason": "end_turn",
                    "reply_text": full,
                    "reply_len": len(full),
                    "total_ms": int((time.time() - t0) * 1000),
                })
                await websocket.send_json({"type": "state", "state": "idle"})
                continue

            if kind == "text":
                # Pure text path (no audio). Reuse streaming brain.
                text = msg.get("text", "").strip()
                if not text:
                    continue
                sid = msg.get("session_id") or sid
                t0 = time.time()
                await websocket.send_json({"type": "state", "state": "thinking"})

                # Confirm-cycle handling
                pending = brain.get_pending_confirm(sid)
                if pending:
                    decision = brain._looks_like_confirmation(text)
                    if decision == "yes":
                        result = brain.resolve_pending(sid, "yes")
                        reply_text = _format_confirmation_result(pending, result)
                        await websocket.send_json({
                            "type": "text", "role": "sam", "text": reply_text,
                        })
                        await synth_and_ship(reply_text)
                        brain.append_turn(sid, text, reply_text)
                        await websocket.send_json({
                            "type": "done", "stop_reason": "end_turn",
                            "reply_text": reply_text, "reply_len": len(reply_text),
                            "total_ms": int((time.time() - t0) * 1000),
                        })
                        await websocket.send_json({"type": "state", "state": "idle"})
                        continue
                    elif decision == "no":
                        brain.resolve_pending(sid, "no")
                        reply_text = f"Cancelled. I won't {pending['summary']}."
                        await websocket.send_json({
                            "type": "text", "role": "sam", "text": reply_text,
                        })
                        await synth_and_ship(reply_text)
                        brain.append_turn(sid, text, reply_text)
                        await websocket.send_json({
                            "type": "done", "stop_reason": "end_turn",
                            "reply_text": reply_text, "reply_len": len(reply_text),
                            "total_ms": int((time.time() - t0) * 1000),
                        })
                        await websocket.send_json({"type": "state", "state": "idle"})
                        continue
                    else:
                        brain.resolve_pending(sid, "no")

                async def on_sentence_text(sent: str):
                    await synth_and_ship(sent)

                await websocket.send_json({"type": "state", "state": "speaking"})
                full = await _stream_brain_to_sentences(
                    text, sid, on_sentence_text,
                )
                brain.append_turn(sid, text, full)
                held = brain.get_pending_confirm(sid)
                if held:
                    await websocket.send_json({
                        "type": "pending_confirm",
                        "tool": held["tool"], "summary": held["summary"],
                    })
                await websocket.send_json({
                    "type": "done",
                    "stop_reason": "end_turn",
                    "reply_text": full,
                    "reply_len": len(full),
                    "total_ms": int((time.time() - t0) * 1000),
                })
                await websocket.send_json({"type": "state", "state": "idle"})
                continue

    except WebSocketDisconnect:
        log.info("ws: client disconnected")


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw s16le mono PCM in a minimal WAV header so whisper_stt.transcribe
    (which expects WAV/webm/m4a) accepts the rolling PCM buffer from the
    browser MediaStream."""
    import struct
    data_size = len(pcm_bytes)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,    # PCM
        1,    # mono
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    return header + pcm_bytes


if __name__ == "__main__":
    import uvicorn

    # Optional HTTPS — iOS Safari blocks getUserMedia on http:// origins,
    # so we serve HTTPS when cert files are present. Set AVA_HTTPS=0 to
    # force plain HTTP (e.g. for local dev).
    cert_path = os.environ.get("AGENT_VOICE_TLS_CERT", os.path.join(os.path.dirname(__file__), "certs", "server.crt"))
    key_path = os.environ.get("AGENT_VOICE_TLS_KEY", os.path.join(os.path.dirname(__file__), "certs", "server.key"))
    use_https = os.environ.get("AGENT_VOICE_HTTPS", "1") != "0" and os.path.exists(cert_path) and os.path.exists(key_path)

    if use_https:
        log.info("starting sam bridge (HTTPS) on %s:%d", HOST, PORT)
        log.info("open https://<your-host>:%d/  (TLS cert)", PORT)
        log.info("open https://<your-host>:%d/  (IP — cert won't match, use hostname on iOS)", PORT)
        # Pre-warm Kokoro in a background thread so the first /api/chat is fast.
        import threading
        threading.Thread(target=_warm_kokoro, daemon=True).start()
        # Prime Telegram context cache and start the poller so the voice agent
        # has fresh cross-surface context without a manual prompt.
        try:
            import telegram_sync
            threading.Thread(target=telegram_sync.prime_cache, daemon=True).start()
            if telegram_sync.start_poller():
                log.info("telegram_sync: poller started")
        except Exception as e:
            log.warning("telegram_sync startup failed: %s", e)
        uvicorn.run(
            app, host=HOST, port=PORT, log_level="info", access_log=False,
            ssl_certfile=cert_path, ssl_keyfile=key_path,
        )
    else:
        log.info("starting agent-voice-ai bridge (HTTP) on %s:%d", HOST, PORT)
        log.info("open http://localhost:%d/  (or http://localhost:%d/)", PORT, PORT)
        import threading
        try:
            import telegram_sync
            threading.Thread(target=telegram_sync.prime_cache, daemon=True).start()
            telegram_sync.start_poller()
        except Exception as e:
            log.warning("telegram_sync startup failed: %s", e)
        uvicorn.run(app, host=HOST, port=PORT, log_level="info", access_log=False)
