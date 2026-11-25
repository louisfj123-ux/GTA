#!/usr/bin/env python3
"""clip_agent.py - find and cut highlight clips from a Twitch VOD.

Pipeline
  1. Resolve the VOD's HLS streams with yt-dlp (no full download needed).
  2. Pull the chat replay from Twitch's GQL API.
  3. Stream the audio through ffmpeg and measure loudness per second.
  4. Score every second of the VOD from chat rate, hype keywords and loudness.
  5. Pick the top-N peaks (with a minimum gap between them) and cut each one
     with ffmpeg straight from the stream.
  6. Write clips/<channel>/<vod_id>/... plus highlights.json and highlights.csv.

Example
  python clip_agent.py --vod https://www.twitch.tv/videos/2932568279 --channel streamername

Offline / testing
  --video-file and --chat-file let you run on a local recording and a saved
  chat JSON instead of hitting Twitch.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("clip_agent")

GQL_URL = "https://gql.twitch.tv/gql"
# Public web client id used by twitch.tv itself and most open-source tools.
GQL_CLIENT_ID = "kd1unb4b3q4t58fwlpcbzcbnm76a8fp"
COMMENTS_QUERY_HASH = "b70a3591ff0f4e0313d126c6a1502d79a1c02baebb288227c582044aa76adf6a"

DEFAULT_KEYWORDS = [
    "lul", "lmao", "lmfao", "kekw", "omegalul", "pog", "poggers", "pogchamp",
    "pogu", "clip", "clip it", "clipit", "wtf", "omg", "holy", "no way",
    "monkas", "pepelaugh", "icant", "w", "ww", "www", "l", "gg", "insane",
    "lol", "haha", "😂", "💀", "!!!", "???",
]

# Rage: chat telling him to calm down, mirroring his screaming ("NOOOO"),
# or spamming NotLikeThis. Entries starting with "re:" are regular expressions.
RAGE_KEYWORDS = [
    r"re:\bno{3,}\b", "notlikethis", "chill", "calm down", "mad", "he mad", "bro is mad",
    "tweaking", "tweakin", "tweak", "crash out", "crashing out", "crashout", "bark", "barking",
    "rage", "raging", "angry", "tight", "malding", "screaming", "yelling", "throw", "threw",
    "controller", "monitor", "broke", "broken", "biblethump", "💀",
]

STYLES = {
    # weights: (chat volume, keywords, audio); relative_audio scores loudness
    # against the streamer's recent baseline so sudden yelling stands out.
    "hype": {"keywords": DEFAULT_KEYWORDS, "weights": (1.0, 1.0, 0.7), "relative_audio": False},
    "rage": {"keywords": RAGE_KEYWORDS, "weights": (0.3, 1.5, 1.0), "relative_audio": True},
    "funny": {"keywords": None, "weights": (0.4, 1.6, 0.6), "relative_audio": True},
}

# Funny: chat laughing, not cheering ("W" spam is praise, not comedy).
FUNNY_KEYWORDS = [
    "lol", "lmao", "lmfao", "kekw", "omegalul", "lul", "lulw", "pepelaugh", "icant", "i cant", "i can't",
    "dead", "im dead", "i'm dead", "crying", "bro what", "nah", "😂", "🤣", "💀", "😭",
    r"re:\b(ha){2,}h?\b", r"re:\blo+l\b", r"re:\blmf?a+o+\b",
]
STYLES["funny"]["keywords"] = FUNNY_KEYWORDS

AUDIO_RATE = 8000  # Hz; plenty for loudness and keeps the pipe small.


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
@dataclass
class ChatMessage:
    t: float      # seconds from VOD start
    text: str


@dataclass
class Highlight:
    rank: int
    peak: float
    start: float
    end: float
    score: float
    chat_msgs: int
    keyword_hits: int
    loudness_db: float
    top_words: list[str] = field(default_factory=list)
    file: str | None = None
    gameplay: float | None = None   # fraction of the clip with on-screen motion

    @property
    def timestamp(self) -> str:
        return hms(self.start)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def hms(seconds: float) -> str:
    s = int(max(0, seconds))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def parse_vod_id(vod: str) -> str:
    m = re.search(r"(?:videos/|^v?)(\d{5,})", vod.strip())
    if not m:
        raise SystemExit(f"Could not find a Twitch VOD id in {vod!r}")
    return m.group(1)


def require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise SystemExit(f"'{binary}' is required but was not found on PATH")
    return path


Z_CAP = 10.0  # stops one near-constant signal from drowning out the others


def robust_z(x: np.ndarray) -> np.ndarray:
    """z-score using median / MAD so a few huge spikes don't flatten the rest."""
    if x.size == 0:
        return x
    med = np.median(x)
    spread = max(np.median(np.abs(x - med)) * 1.4826, x.std())
    if spread < 1e-9:
        return np.zeros_like(x, dtype=float)
    return np.clip((x - med) / spread, -3.0, Z_CAP)


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window == 1 or x.size == 0:
        return x.astype(float)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def shift(x: np.ndarray, seconds: int) -> np.ndarray:
    """Shift a series earlier by `seconds` (chat reacts after the moment)."""
    if seconds <= 0:
        return x
    out = np.zeros_like(x)
    out[: max(0, x.size - seconds)] = x[seconds:]
    return out


def ffmpeg_net_args(source: str, headers: dict | None) -> list[str]:
    """Input options for reading a URL: request headers plus proxy.

    ffmpeg only honours `http_proxy` from the environment, so pass HTTPS_PROXY
    through explicitly for https sources.
    """
    if not source.startswith(("http://", "https://")):
        return []
    args = []
    if headers:
        args += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
    proxy = (os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
             or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy"))
    if proxy:
        args += ["-http_proxy", proxy]
    return args


# --------------------------------------------------------------------------- #
# Twitch: stream resolution and chat
# --------------------------------------------------------------------------- #
def resolve_vod(vod_url: str, quality: str) -> dict:
    """Return metadata plus direct HLS urls for video and audio."""
    try:
        import yt_dlp
    except ImportError:
        raise SystemExit("yt-dlp is required: pip install -r requirements.txt")

    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(vod_url, download=False)

    formats = info.get("formats") or []
    if not formats:
        raise SystemExit("yt-dlp returned no playable formats for this VOD")

    def height(f):
        return f.get("height") or 0

    video_formats = [f for f in formats if f.get("vcodec") not in (None, "none")]
    audio_only = [f for f in formats if f.get("vcodec") == "none" and f.get("acodec") != "none"]

    if quality == "best":
        video = max(video_formats, key=lambda f: (height(f), f.get("tbr") or 0))
    else:
        want = int(quality.rstrip("p"))
        under = [f for f in video_formats if height(f) <= want] or video_formats
        video = max(under, key=lambda f: (height(f), f.get("tbr") or 0))

    # Lowest-bitrate stream is fine for audio analysis.
    audio = audio_only[0] if audio_only else min(video_formats, key=lambda f: f.get("tbr") or 1e9)

    return {
        "id": str(info.get("id", "")).lstrip("v"),
        "title": info.get("title") or "",
        "uploader": info.get("uploader") or "",
        "uploader_id": info.get("uploader_id") or "",
        "duration": float(info.get("duration") or 0),
        "video_url": video["url"],
        "video_height": height(video),
        "video_width": int(video.get("width") or round(height(video) * 16 / 9)),
        "audio_url": audio["url"],
        # Smallest video stream is plenty for motion analysis.
        "motion_url": min(video_formats, key=lambda f: (height(f) or 1e9, f.get("tbr") or 1e9))["url"],
        "http_headers": video.get("http_headers") or {},
    }


def _gql(payload: list | dict, retries: int = 4) -> list | dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        GQL_URL,
        data=body,
        headers={"Client-ID": GQL_CLIENT_ID, "Content-Type": "application/json"},
        method="POST",
    )
    delay = 1.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            log.debug("GQL error %s, retrying in %.0fs", e, delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def _comments_page(vod_id: str, offset: float | None = None, cursor: str | None = None) -> dict:
    variables: dict = {"videoID": vod_id}
    if cursor:
        variables["cursor"] = cursor
    else:
        variables["contentOffsetSeconds"] = int(offset or 0)
    resp = _gql([{
        "operationName": "VideoCommentsByOffsetOrCursor",
        "variables": variables,
        "extensions": {"persistedQuery": {"version": 1, "sha256Hash": COMMENTS_QUERY_HASH}},
    }])
    item = resp[0] if isinstance(resp, list) else resp
    if item.get("errors"):
        raise RuntimeError(f"Twitch GQL error: {item['errors']}")
    video = (item.get("data") or {}).get("video") or {}
    return video.get("comments") or {}


def _fetch_chat_segment(vod_id: str, seg_start: float, seg_end: float) -> dict[str, ChatMessage]:
    """Fetch messages with seg_start <= t < seg_end, keyed by message id."""
    out: dict[str, ChatMessage] = {}
    comments = _comments_page(vod_id, offset=seg_start)
    while True:
        edges = comments.get("edges") or []
        last_t = seg_start
        for edge in edges:
            node = edge.get("node") or {}
            t = float(node.get("contentOffsetSeconds", 0))
            last_t = max(last_t, t)
            if seg_start <= t < seg_end:
                frags = (node.get("message") or {}).get("fragments") or []
                out[node.get("id") or f"{t}:{len(out)}"] = ChatMessage(
                    t, "".join(f.get("text", "") for f in frags))
        cursor = edges[-1].get("cursor") if edges else None
        if last_t >= seg_end or not cursor or not (comments.get("pageInfo") or {}).get("hasNextPage"):
            return out
        comments = _comments_page(vod_id, cursor=cursor)


def fetch_chat(vod_id: str, duration: float = 0, workers: int = 16, segment: int = 300) -> list[ChatMessage]:
    """Download the full chat replay for a VOD via Twitch GQL.

    The VOD is split into `segment`-second slices fetched in parallel; each
    slice starts from its own offset and pages forward with cursors.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if duration <= 0:  # unknown length: fall back to one sequential pass
        return sorted(_fetch_chat_segment(vod_id, 0, float("inf")).values(), key=lambda m: m.t)

    bounds = [(float(a), float(min(a + segment, duration + 1))) for a in range(0, int(duration) + 1, segment)]
    merged: dict[str, ChatMessage] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch_chat_segment, vod_id, a, b) for a, b in bounds]
        for fut in as_completed(futures):
            merged.update(fut.result())
            done += 1
            if done % max(1, len(bounds) // 10) == 0 or done == len(bounds):
                log.info("  chat: %d/%d segments, %d msgs", done, len(bounds), len(merged))
    return sorted(merged.values(), key=lambda m: m.t)


def load_chat_file(path: Path) -> list[ChatMessage]:
    """Load chat from JSON: either our own format or a TwitchDownloader export."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "comments" in data:  # TwitchDownloader
        out = []
        for c in data["comments"]:
            msg = c.get("message") or {}
            text = msg.get("body") or "".join(f.get("text", "") for f in msg.get("fragments") or [])
            out.append(ChatMessage(float(c.get("content_offset_seconds", 0)), text))
        return out
    return [ChatMessage(float(m["t"]), m.get("text", "")) for m in data]


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
def _loudness_chunk(source: str, headers: dict | None, start: float | None, length: float | None) -> np.ndarray:
    cmd = [require("ffmpeg"), "-nostdin", "-loglevel", "error"]
    cmd += ffmpeg_net_args(source, headers)
    if start:
        cmd += ["-ss", f"{start:.2f}"]
    cmd += ["-i", source]
    if length:
        cmd += ["-t", f"{length:.2f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(AUDIO_RATE), "-f", "s16le", "-"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    chunk_bytes = AUDIO_RATE * 2  # one second of s16 mono
    values: list[float] = []
    buf = b""
    assert proc.stdout is not None
    while True:
        data = proc.stdout.read(chunk_bytes * 60)
        if not data:
            break
        buf += data
        n = len(buf) // chunk_bytes
        if n:
            arr = np.frombuffer(buf[: n * chunk_bytes], dtype=np.int16).astype(np.float32)
            arr = arr.reshape(n, AUDIO_RATE) / 32768.0
            rms = np.sqrt(np.mean(arr * arr, axis=1))
            values.extend((20 * np.log10(np.maximum(rms, 1e-5))).tolist())
            buf = buf[n * chunk_bytes:]
    _, err = proc.communicate()
    if proc.returncode != 0 and not values:
        raise RuntimeError(f"ffmpeg audio analysis failed: {err.decode(errors='replace').strip()[:500]}")
    out = np.array(values, dtype=float)
    if length:  # pad/trim so chunks line up exactly when concatenated
        want = int(round(length))
        out = np.pad(out, (0, max(0, want - out.size)), constant_values=out.min() if out.size else -100)[:want]
    return out


def audio_loudness(source: str, headers: dict | None = None, duration: float = 0,
                   workers: int = 8, chunk: int = 1800) -> np.ndarray:
    """Per-second RMS loudness in dBFS, streamed through ffmpeg.

    Long VODs are split into `chunk`-second pieces analysed in parallel.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if duration <= chunk:
        return _loudness_chunk(source, headers, None, None)
    starts = list(range(0, int(math.ceil(duration)), chunk))
    lengths = [min(chunk, duration - s) for s in starts]
    results: list[np.ndarray | None] = [None] * len(starts)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_loudness_chunk, source, headers, s, l): i
                for i, (s, l) in enumerate(zip(starts, lengths))}
        done = 0
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
            done += 1
            log.info("  audio: %d/%d chunks", done, len(starts))
    return np.concatenate(results)


MOTION_W, MOTION_H, MOTION_FPS = 64, 36, 2
MOTION_PIXEL_DELTA = 12      # grey-level change that counts as "moved"
GAMEPLAY_THRESHOLD = 0.15    # fraction of pixels moving for a second to count as gameplay


def _motion_chunk(source: str, headers: dict | None, start: float | None, length: float | None) -> np.ndarray:
    """Per-second fraction of pixels that change between frames (tiny greyscale frames)."""
    cmd = [require("ffmpeg"), "-nostdin", "-loglevel", "error"]
    cmd += ffmpeg_net_args(source, headers)
    if start:
        cmd += ["-ss", f"{start:.2f}"]
    cmd += ["-i", source]
    if length:
        cmd += ["-t", f"{length:.2f}"]
    cmd += ["-an", "-vf", f"fps={MOTION_FPS},scale={MOTION_W}:{MOTION_H},format=gray",
            "-f", "rawvideo", "-"]
    res = subprocess.run(cmd, capture_output=True)
    frame = MOTION_W * MOTION_H
    n = len(res.stdout) // frame
    if res.returncode != 0 and n == 0:
        raise RuntimeError(f"ffmpeg motion analysis failed: {res.stderr.decode(errors='replace').strip()[:500]}")
    frames = np.frombuffer(res.stdout[: n * frame], dtype=np.uint8).reshape(n, MOTION_H, MOTION_W).astype(np.int16)
    if n < 2:
        diffs = np.zeros(n)
    else:
        diffs = (np.abs(np.diff(frames, axis=0)) > MOTION_PIXEL_DELTA).mean(axis=(1, 2))
        diffs = np.concatenate([[diffs[0]], diffs])
    secs = int(math.ceil(n / MOTION_FPS))
    padded = np.pad(diffs, (0, secs * MOTION_FPS - n), mode="edge") if n else np.zeros(0)
    out = padded.reshape(secs, MOTION_FPS).mean(axis=1) if secs else np.zeros(0)
    if length:
        want = int(round(length))
        out = np.pad(out, (0, max(0, want - out.size)))[:want]
    return out


def video_motion(source: str, headers: dict | None = None, duration: float = 0,
                 workers: int = 8, chunk: int = 1800) -> np.ndarray:
    """Per-second motion for the whole VOD, analysed in parallel chunks."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if duration <= chunk:
        return _motion_chunk(source, headers, None, None)
    starts = list(range(0, int(math.ceil(duration)), chunk))
    lengths = [min(chunk, duration - s) for s in starts]
    results: list[np.ndarray | None] = [None] * len(starts)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_motion_chunk, source, headers, s, l): i
                for i, (s, l) in enumerate(zip(starts, lengths))}
        for done, fut in enumerate(as_completed(futs), 1):
            results[futs[fut]] = fut.result()
            log.info("  video: %d/%d chunks", done, len(starts))
    return np.concatenate(results)


def gameplay_fraction(motion: np.ndarray, length: int, clip_len: float, lead: float) -> np.ndarray:
    """For each candidate peak second, the share of its clip window that is gameplay."""
    return window_fraction(motion >= GAMEPLAY_THRESHOLD, length, clip_len, lead)


def window_fraction(flags: np.ndarray, length: int, clip_len: float, lead: float) -> np.ndarray:
    """For each candidate peak second, the share of its clip window where `flags` is true."""
    active = np.zeros(length)
    m = flags[:length]
    active[: m.size] = m
    csum = np.concatenate([[0.0], np.cumsum(active)])
    L = max(1, int(round(clip_len)))
    starts = np.clip(np.arange(length) - int(round(clip_len * lead)), 0, max(0, length - L))
    ends = np.minimum(starts + L, length)
    return (csum[ends] - csum[starts]) / np.maximum(1, ends - starts)


def chat_series(msgs: list[ChatMessage], length: int, keywords: list[str]):
    counts = np.zeros(length)
    hits = np.zeros(length)
    words_by_sec: dict[int, list[str]] = {}
    kw_regex = [(k[3:], re.compile(k[3:])) for k in keywords if k.startswith("re:")]
    plain = [k for k in keywords if not k.startswith("re:")]
    kw_single = {k for k in plain if " " not in k and k.isalnum()}
    kw_phrase = [k for k in plain if k not in kw_single]
    for m in msgs:
        s = int(m.t)
        if not 0 <= s < length:
            continue
        counts[s] += 1
        low = m.text.lower()
        tokens = re.findall(r"\w+", low)
        matched = [t for t in tokens if t in kw_single] + [p for p in kw_phrase if p in low]
        for _, rx in kw_regex:
            mm = rx.search(low)
            if mm:
                matched.append(re.sub(r"(.)\1{2,}", r"\1\1\1", mm.group(0)))
        if matched:
            hits[s] += 1
            words_by_sec.setdefault(s, []).extend(matched)
    return counts, hits, words_by_sec


def relative_loudness(loudness: np.ndarray, window: int = 300, silence_db: float = -60) -> np.ndarray:
    """Loudness minus a rolling median baseline, so yelling stands out in quiet and loud scenes alike.

    Muted stretches (Twitch mutes copyrighted audio) are treated as baseline.
    """
    x = loudness.astype(float).copy()
    voiced = x > silence_db
    if not voiced.any():
        return np.zeros_like(x)
    x[~voiced] = np.median(x[voiced])
    block = 30
    n = int(math.ceil(x.size / block))
    meds = np.array([np.median(x[i * block:(i + 1) * block]) for i in range(n)])
    k = max(1, window // block)
    base_blocks = np.array([np.median(meds[max(0, i - k // 2): i + k // 2 + 1]) for i in range(n)])
    base = np.repeat(base_blocks, block)[: x.size]
    return np.where(voiced, x - base, 0.0)


def score_series(
    length: int,
    chat_counts: np.ndarray | None,
    kw_hits: np.ndarray | None,
    loudness: np.ndarray | None,
    chat_delay: int,
    window: int,
    w_chat: float,
    w_kw: float,
    w_audio: float,
) -> np.ndarray:
    parts = []
    if chat_counts is not None and chat_counts.sum() > 0 and w_chat > 0:
        parts.append(w_chat * robust_z(smooth(shift(chat_counts, chat_delay), window)))
    if kw_hits is not None and kw_hits.sum() > 0 and w_kw > 0:
        parts.append(w_kw * robust_z(smooth(shift(kw_hits, chat_delay), window)))
    if loudness is not None and loudness.size and w_audio > 0:
        loud = np.full(length, np.median(loudness))
        loud[: min(length, loudness.size)] = loudness[:length]
        parts.append(w_audio * robust_z(smooth(loud, window)))
    if not parts:
        return np.zeros(length)
    return np.sum(parts, axis=0)


def pick_peaks(score: np.ndarray, n: int, min_gap: int, edge: int) -> list[int]:
    """Greedy non-maximum suppression: take the best second, blank out ±min_gap."""
    s = score.copy()
    if edge > 0 and s.size > 2 * edge:
        s[:edge] = -np.inf
        s[-edge:] = -np.inf
    peaks = []
    while len(peaks) < n:
        i = int(np.argmax(s))
        if not np.isfinite(s[i]):
            break
        # Capped z-scores form flat tops; centre the peak on the plateau.
        j = i
        while j + 1 < s.size and s[j + 1] >= s[i] - 1e-6:
            j += 1
        i = (i + j) // 2
        peaks.append(i)
        s[max(0, i - min_gap): i + min_gap + 1] = -np.inf
    return peaks


def build_highlights(
    peaks: list[int],
    score: np.ndarray,
    duration: float,
    clip_len: float,
    lead: float,
    chat_counts: np.ndarray | None,
    kw_hits: np.ndarray | None,
    words_by_sec: dict[int, list[str]],
    loudness: np.ndarray | None,
    chat_delay: int,
) -> list[Highlight]:
    out = []
    for rank, p in enumerate(peaks, 1):
        start = max(0.0, p - clip_len * lead)
        end = min(duration, start + clip_len)
        start = max(0.0, end - clip_len)
        # Chat for this moment arrives chat_delay seconds later.
        c0, c1 = int(start + chat_delay), int(end + chat_delay)
        words: dict[str, int] = {}
        for s in range(c0, c1):
            for w in words_by_sec.get(s, []):
                words[w] = words.get(w, 0) + 1
        top = sorted(words, key=words.get, reverse=True)[:5]
        loud = float(np.max(loudness[int(start):int(end)])) if loudness is not None and loudness.size > int(start) else float("nan")
        out.append(Highlight(
            rank=rank,
            peak=float(p),
            start=round(start, 1),
            end=round(end, 1),
            score=round(float(score[p]), 3),
            chat_msgs=int(chat_counts[c0:c1].sum()) if chat_counts is not None else 0,
            keyword_hits=int(kw_hits[c0:c1].sum()) if kw_hits is not None else 0,
            loudness_db=round(loud, 1) if not math.isnan(loud) else loud,
            top_words=top,
        ))
    return out


# --------------------------------------------------------------------------- #
# Cutting
# --------------------------------------------------------------------------- #
# overlay x/y expressions; {m} is the margin in pixels
_L, _C, _R = "{m}", "(main_w-overlay_w)/2", "main_w-overlay_w-{m}"
_T, _M, _B = "{m}", "(main_h-overlay_h)/2", "main_h-overlay_h-{m}"
LOGO_POSITIONS = {
    "top-left": (_L, _T), "top-center": (_C, _T), "top-right": (_R, _T),
    "center": (_C, _M),
    "bottom-left": (_L, _B), "bottom-center": (_C, _B), "bottom-right": (_R, _B),
}


@dataclass
class LogoSpec:
    path: Path
    position: str = "bottom-center"
    width: float = 0.30     # fraction of video width
    opacity: float = 1.0
    margin: float = 0.03    # fraction of video height
    video_width: int = 0    # filled in once the source is known


def logo_filter(spec: LogoSpec, video_width: int) -> str:
    """filter_complex that scales input 1 to spec.width of the video and overlays it on input 0.

    Width is passed in pixels rather than via scale2ref, whose variable
    meanings differ between ffmpeg versions.
    """
    m = f"main_h*{spec.margin}"
    x, y = (e.replace("{m}", m) for e in LOGO_POSITIONS[spec.position])
    w = max(2, int(video_width * spec.width) // 2 * 2)
    alpha = f",colorchannelmixer=aa={spec.opacity}" if spec.opacity < 1 else ""
    return (
        f"[1:v]format=rgba,scale={w}:-2:flags=lanczos{alpha}[lg];"
        f"[0:v][lg]overlay=x='{x}':y='{y}':format=auto[v]"
    )


def probe_width(path: str) -> int:
    res = subprocess.run(
        [require("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    try:
        return int(res.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        raise RuntimeError(f"Could not read video width of {path}")


def fetch_logo(logo: str, out_dir: Path) -> Path:
    """Accept a local path or an http(s) URL; returns a local file."""
    if not logo.startswith(("http://", "https://")):
        path = Path(logo)
        if not path.is_file():
            raise SystemExit(f"Logo not found: {logo}")
        return path
    suffix = Path(logo.split("?")[0]).suffix or ".png"
    dest = out_dir / f"logo{suffix}"
    if not dest.exists():
        log.info("Downloading logo ...")
        req = urllib.request.Request(logo, headers={"User-Agent": "clip_agent"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            dest.write_bytes(resp.read())
    return dest


ENCODE_ARGS = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k"]


def cut_clip(source: str, start: float, end: float, dest: Path, headers: dict | None,
             copy: bool, logo: LogoSpec | None = None) -> None:
    cmd = [require("ffmpeg"), "-nostdin", "-loglevel", "error", "-y"]
    cmd += ffmpeg_net_args(source, headers)
    cmd += ["-ss", f"{start:.2f}", "-i", source]
    if logo:
        cmd += ["-i", str(logo.path), "-filter_complex", logo_filter(logo, logo.video_width),
                "-map", "[v]", "-map", "0:a?"]
    cmd += ["-t", f"{end - start:.2f}"]
    if copy and not logo:
        cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    else:
        cmd += ENCODE_ARGS
    cmd += ["-movflags", "+faststart", str(dest)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {dest.name}: {res.stderr.strip()[:500]}")


def brand_clip(src: Path, dest: Path, logo: LogoSpec) -> None:
    """Overlay a logo on an already-cut clip."""
    cmd = [require("ffmpeg"), "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
           "-i", str(logo.path), "-filter_complex", logo_filter(logo, probe_width(str(src))),
           "-map", "[v]", "-map", "0:a?", *ENCODE_ARGS[:6], "-c:a", "copy",
           "-movflags", "+faststart", str(dest)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {dest.name}: {res.stderr.strip()[:500]}")


def write_reports(out_dir: Path, meta: dict, highlights: list[Highlight]) -> None:
    rows = [{**asdict(h), "timestamp": h.timestamp} for h in highlights]
    (out_dir / "highlights.json").write_text(
        json.dumps({"vod": meta, "highlights": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with open(out_dir / "highlights.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "timestamp", "start", "end", "score", "chat_msgs",
                    "keyword_hits", "loudness_db", "gameplay", "top_words", "file", "vod_link"])
        for h in highlights:
            link = (f"https://www.twitch.tv/videos/{meta['id']}?t={int(h.start)}s"
                    if str(meta.get("id", "")).isdigit() else "")
            w.writerow([h.rank, h.timestamp, h.start, h.end, h.score, h.chat_msgs,
                        h.keyword_hits, h.loudness_db, "" if h.gameplay is None else h.gameplay,
                        " ".join(h.top_words), h.file or "", link])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find and cut highlight clips from a Twitch VOD.")
    src = p.add_argument_group("source")
    src.add_argument("--vod", help="Twitch VOD url or id, e.g. https://www.twitch.tv/videos/123")
    src.add_argument("--channel", help="Channel name for the output folder; defaults to the VOD's channel")
    src.add_argument("--video-file", type=Path, help="Use a local video file instead of streaming the VOD")
    src.add_argument("--chat-file", type=Path, help="Use a saved chat JSON instead of downloading it")
    src.add_argument("--save-chat", action="store_true", help="Save downloaded chat to chat.json in the output folder")

    sel = p.add_argument_group("selection")
    sel.add_argument("--num-clips", type=int, default=10, help="Number of clips to make (default 10)")
    sel.add_argument("--clip-length", type=float, default=30, help="Clip length in seconds (default 30)")
    sel.add_argument("--lead", type=float, default=0.65, help="Fraction of the clip placed before the peak (default 0.65)")
    sel.add_argument("--min-gap", type=float, default=90, help="Minimum seconds between clip peaks (default 90)")
    sel.add_argument("--chat-delay", type=int, default=6, help="Seconds chat lags behind the stream (default 6)")
    sel.add_argument("--skip-edges", type=int, default=60, help="Ignore the first/last N seconds (default 60)")
    sel.add_argument("--min-score", type=float, default=0.5,
                     help="Drop peaks scoring below this (combined z-score, default 0.5)")
    sel.add_argument("--style", choices=sorted(STYLES), default="hype",
                     help="What to look for: 'hype' (big chat reactions) or 'rage' (streamer losing it)")
    sel.add_argument("--keywords", help="Comma-separated keywords (replaces the style's list; prefix re: for regex)")
    sel.add_argument("--weight-chat", type=float, help="Weight of chat volume (default from --style)")
    sel.add_argument("--weight-keywords", type=float, help="Weight of keyword hits (default from --style)")
    sel.add_argument("--weight-audio", type=float, help="Weight of loudness (default from --style)")
    sel.add_argument("--no-audio", action="store_true", help="Skip audio analysis (chat only)")
    sel.add_argument("--no-chat", action="store_true", help="Skip chat (audio only)")
    sel.add_argument("--max-muted", type=float, default=0.1,
                     help="Skip moments where more than this share of the clip's audio is muted by Twitch (default 0.1)")
    sel.add_argument("--min-gameplay", type=float, default=0.7,
                     help="Skip moments where less than this share of the clip is moving gameplay "
                          "(menus, loading and pause screens are static). 0 disables (default 0.7)")

    out = p.add_argument_group("output")
    out.add_argument("--output", type=Path, default=Path("clips"), help="Output root (default ./clips)")
    out.add_argument("--quality", default="best", help="'best' or max height like 720p (default best)")
    out.add_argument("--fast-copy", action="store_true", help="Stream-copy instead of re-encoding (faster, less exact cuts)")
    out.add_argument("--logo", help="Image (path or URL) to overlay on every clip; PNG/WebP transparency is kept")
    out.add_argument("--logo-position", default="bottom-center", choices=sorted(LOGO_POSITIONS))
    out.add_argument("--logo-width", type=float, default=0.30, help="Logo width as a fraction of video width (default 0.30)")
    out.add_argument("--logo-opacity", type=float, default=1.0, help="0-1 (default 1)")
    out.add_argument("--dry-run", action="store_true", help="Score and report only; don't cut clips")
    out.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    style = STYLES[args.style]
    for name, default in zip(("weight_chat", "weight_keywords", "weight_audio"), style["weights"]):
        if getattr(args, name) is None:
            setattr(args, name, default)
    args.relative_audio = style["relative_audio"]

    if not args.vod and not args.video_file:
        p.error("give --vod or --video-file")
    if args.no_audio and (args.no_chat or not (args.vod or args.chat_file)):
        p.error("nothing to score: audio is disabled and there is no chat source")
    if not 0 < args.logo_width <= 1 or not 0 <= args.logo_opacity <= 1:
        p.error("--logo-width must be in (0, 1] and --logo-opacity in [0, 1]")
    if not 0 <= args.lead <= 1:
        p.error("--lead must be between 0 and 1")
    return args


def run(args: argparse.Namespace) -> list[Highlight]:
    require("ffmpeg")
    vod_id = parse_vod_id(args.vod) if args.vod else None
    headers: dict = {}

    # 1. Source ---------------------------------------------------------------
    if args.video_file:
        video_src = audio_src = motion_src = str(args.video_file)
        meta = {"id": vod_id or args.video_file.stem, "title": args.video_file.name,
                "channel": args.channel or "local", "duration": probe_duration(video_src)}
    else:
        url = args.vod if args.vod.startswith("http") else f"https://www.twitch.tv/videos/{vod_id}"
        log.info("Resolving VOD %s ...", vod_id)
        info = resolve_vod(url, args.quality)
        if args.channel and info["uploader_id"] and info["uploader_id"].lower() != args.channel.lower():
            log.warning("VOD belongs to '%s', not '%s' - continuing anyway",
                        info["uploader_id"], args.channel)
        video_src, audio_src, headers = info["video_url"], info["audio_url"], info["http_headers"]
        motion_src = info["motion_url"]
        meta = {"id": info["id"] or vod_id, "title": info["title"], "channel": info["uploader_id"] or args.channel,
                "duration": info["duration"], "video_height": info["video_height"],
                "video_width": info["video_width"]}
        log.info("  '%s' by %s, %s long, cutting at %sp",
                 meta["title"], meta["channel"], hms(meta["duration"]), info["video_height"])

    channel = re.sub(r"[^\w.-]", "_", (args.channel or meta["channel"] or "unknown").lower())
    out_dir = args.output / channel / str(meta["id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logo = None
    if args.logo and not args.dry_run:
        logo = LogoSpec(fetch_logo(args.logo, out_dir), args.logo_position,
                        args.logo_width, args.logo_opacity)
        logo.video_width = meta.get("video_width") or probe_width(video_src)
        if args.fast_copy:
            log.warning("--logo needs re-encoding; ignoring --fast-copy")

    # 2. Chat -----------------------------------------------------------------
    msgs: list[ChatMessage] = []
    if not args.no_chat:
        if args.chat_file:
            msgs = load_chat_file(args.chat_file)
            log.info("Loaded %d chat messages from %s", len(msgs), args.chat_file)
        elif vod_id:
            log.info("Downloading chat replay ...")
            try:
                msgs = fetch_chat(vod_id, meta["duration"])
                log.info("  %d chat messages", len(msgs))
            except Exception as e:  # chat is optional; audio can still carry it
                log.warning("Chat download failed (%s); falling back to audio only", e)
        if msgs and args.save_chat:
            (out_dir / "chat.json").write_text(
                json.dumps([asdict(m) for m in msgs], ensure_ascii=False), encoding="utf-8")

    # 3. Audio ----------------------------------------------------------------
    cache = out_dir / "signals.npz"
    cached = dict(np.load(cache)) if cache.exists() else {}
    loudness = cached.get("loudness")
    if loudness is not None:
        log.info("Using cached audio analysis from %s", cache)
    if loudness is None and not args.no_audio:
        log.info("Analysing audio loudness ...")
        try:
            loudness = audio_loudness(audio_src, headers, meta["duration"])
            log.info("  %s of audio analysed", hms(loudness.size))
            cached["loudness"] = loudness
            np.savez(cache, **cached)
        except Exception as e:
            if not msgs:
                raise
            log.warning("Audio analysis failed (%s); using chat only", e)

    duration = meta["duration"] or (loudness.size if loudness is not None else 0) \
        or (max(m.t for m in msgs) + 1 if msgs else 0)
    length = int(math.ceil(duration))
    if length <= 0:
        raise SystemExit("Could not determine VOD duration")

    gameplay = None
    if args.min_gameplay > 0:
        log.info("Analysing on-screen motion (to skip menus and loading screens) ...")
        try:
            motion = cached.get("motion")
            if motion is None:
                motion = video_motion(motion_src, headers, duration)
                cached["motion"] = motion
                np.savez(cache, **cached)
            else:
                log.info("  using cached motion analysis")
            gameplay = gameplay_fraction(motion, length, args.clip_length, args.lead)
            log.info("  %.0f%% of the VOD looks like gameplay",
                     100 * float((motion[:length] >= GAMEPLAY_THRESHOLD).mean()))
        except Exception as e:
            log.warning("Motion analysis failed (%s); not filtering for gameplay", e)

    # 4. Score ----------------------------------------------------------------
    keywords = ([k.strip() if k.strip().startswith("re:") else k.strip().lower() for k in args.keywords.split(",")]
                if args.keywords else STYLES[args.style]["keywords"])
    counts = hits = None
    words_by_sec: dict[int, list[str]] = {}
    if msgs:
        counts, hits, words_by_sec = chat_series(msgs, length + args.chat_delay, keywords)
    window = max(3, int(args.clip_length / 3))
    score = combined_score(length, counts, hits, loudness, args, window)

    edge = args.skip_edges if length > 4 * args.skip_edges else 0
    pick_from = score
    if loudness is not None and args.max_muted < 1:
        # Twitch mutes copyrighted audio; a silent clip is useless.
        muted = window_fraction(loudness < -60, length, args.clip_length, args.lead)
        pick_from = np.where(muted <= args.max_muted, pick_from, -np.inf)
    if gameplay is not None:
        pick_from = np.where(gameplay >= args.min_gameplay, pick_from, -np.inf)
        if not np.isfinite(pick_from).any():
            log.warning("No moment has %.0f%% gameplay; ignoring --min-gameplay", 100 * args.min_gameplay)
            pick_from = score
    peaks = [p for p in pick_peaks(pick_from, args.num_clips, int(args.min_gap), edge)
             if score[p] >= args.min_score]
    if not peaks:
        log.warning("No moment scored above --min-score %.2f", args.min_score)
    highlights = build_highlights(peaks, score, duration, args.clip_length, args.lead,
                                  counts, hits, words_by_sec, loudness, args.chat_delay)
    if gameplay is not None:
        for h in highlights:
            h.gameplay = round(float(gameplay[int(h.peak)]), 2)

    # 5. Cut ------------------------------------------------------------------
    if not args.dry_run:
        for h in highlights:
            name = f"{h.rank:02d}_{h.timestamp.replace(':', '-')}.mp4"
            log.info("Cutting #%d  %s-%s  (score %.2f)", h.rank, hms(h.start), hms(h.end), h.score)
            try:
                cut_clip(video_src, h.start, h.end, out_dir / name, headers, args.fast_copy, logo)
                h.file = name
            except RuntimeError as e:
                log.error("  %s", e)

    write_reports(out_dir, meta, highlights)

    # 6. Summary --------------------------------------------------------------
    print(f"\n{len(highlights)} highlights for {meta['channel']} - {meta['title']}")
    print(f"{'#':>3}  {'time':8}  {'score':>6}  {'chat':>5}  {'hype':>5}  {'dB':>6}  {'game':>5}  words")
    for h in highlights:
        print(f"{h.rank:>3}  {h.timestamp:8}  {h.score:6.2f}  {h.chat_msgs:5d}  {h.keyword_hits:5d}  "
              f"{h.loudness_db:6.1f}  {'' if h.gameplay is None else format(h.gameplay, '.0%'):>5}  "
              f"{' '.join(h.top_words)}")
    print(f"\nOutput: {out_dir.resolve()}")
    return highlights


def combined_score(length, counts, hits, loudness, args, window) -> np.ndarray:
    """Sum of chat, keyword and audio z-scores, one value per second of VOD."""
    total = np.zeros(length)
    used = False
    if counts is not None and counts.sum() > 0:
        # Chat arrays are padded by chat_delay so shifting doesn't lose the tail.
        total += score_series(counts.size, counts, hits, None, args.chat_delay, window,
                              args.weight_chat, args.weight_keywords, 0)[:length]
        used = True
    if loudness is not None and loudness.size and args.weight_audio > 0:
        loud = relative_loudness(loudness) if getattr(args, "relative_audio", False) else loudness
        total += score_series(length, None, None, loud, 0, window, 0, 0, args.weight_audio)
        used = True
    if not used:
        raise SystemExit("No usable signal (no chat and no audio) - nothing to score")
    return total


def probe_duration(path: str) -> float:
    res = subprocess.run(
        [require("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    try:
        return float(res.stdout.strip())
    except ValueError:
        return 0.0


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    try:
        run(args)
    except KeyboardInterrupt:
        log.error("Interrupted")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
