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
    spread = max(np.median(np.abs(x - med)) * 1.4826, 0.25 * x.std())
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
        "audio_url": audio["url"],
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


def fetch_chat(vod_id: str, duration: float = 0) -> list[ChatMessage]:
    """Download the full chat replay for a VOD via Twitch GQL."""
    msgs: list[ChatMessage] = []
    cursor = None
    last_report = 0.0
    while True:
        variables = {"videoID": vod_id}
        if cursor:
            variables["cursor"] = cursor
        else:
            variables["contentOffsetSeconds"] = 0
        resp = _gql([{
            "operationName": "VideoCommentsByOffsetOrCursor",
            "variables": variables,
            "extensions": {"persistedQuery": {"version": 1, "sha256Hash": COMMENTS_QUERY_HASH}},
        }])
        item = resp[0] if isinstance(resp, list) else resp
        if item.get("errors"):
            raise RuntimeError(f"Twitch GQL error: {item['errors']}")
        video = (item.get("data") or {}).get("video")
        if not video or not video.get("comments"):
            break
        comments = video["comments"]
        edges = comments.get("edges") or []
        for edge in edges:
            node = edge.get("node") or {}
            frags = (node.get("message") or {}).get("fragments") or []
            text = "".join(f.get("text", "") for f in frags)
            msgs.append(ChatMessage(float(node.get("contentOffsetSeconds", 0)), text))
        if msgs and duration and msgs[-1].t - last_report > 600:
            last_report = msgs[-1].t
            log.info("  chat: %s / %s (%d msgs)", hms(msgs[-1].t), hms(duration), len(msgs))
        if not edges or not (comments.get("pageInfo") or {}).get("hasNextPage"):
            break
        cursor = edges[-1].get("cursor")
        if not cursor:
            break
    return msgs


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
def audio_loudness(source: str, headers: dict | None = None) -> np.ndarray:
    """Per-second RMS loudness in dBFS, streamed through ffmpeg."""
    cmd = [require("ffmpeg"), "-nostdin", "-loglevel", "error"]
    if headers:
        cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
    cmd += ["-i", source, "-vn", "-ac", "1", "-ar", str(AUDIO_RATE), "-f", "s16le", "-"]

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
            if len(values) % 1800 < 60:
                log.info("  audio: %s analysed", hms(len(values)))
    proc.wait()
    if proc.returncode != 0 and not values:
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        raise RuntimeError(f"ffmpeg audio analysis failed: {err.strip()[:500]}")
    return np.array(values, dtype=float)


def chat_series(msgs: list[ChatMessage], length: int, keywords: list[str]):
    counts = np.zeros(length)
    hits = np.zeros(length)
    words_by_sec: dict[int, list[str]] = {}
    kw_single = {k for k in keywords if " " not in k and k.isalnum()}
    kw_phrase = [k for k in keywords if k not in kw_single]
    for m in msgs:
        s = int(m.t)
        if not 0 <= s < length:
            continue
        counts[s] += 1
        low = m.text.lower()
        tokens = re.findall(r"\w+", low)
        matched = [t for t in tokens if t in kw_single] + [p for p in kw_phrase if p in low]
        if matched:
            hits[s] += 1
            words_by_sec.setdefault(s, []).extend(matched)
    return counts, hits, words_by_sec


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
def cut_clip(source: str, start: float, end: float, dest: Path, headers: dict | None, copy: bool) -> None:
    cmd = [require("ffmpeg"), "-nostdin", "-loglevel", "error", "-y"]
    if headers:
        cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
    cmd += ["-ss", f"{start:.2f}", "-i", source, "-t", f"{end - start:.2f}"]
    if copy:
        cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-b:a", "160k"]
    cmd += ["-movflags", "+faststart", str(dest)]
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
                    "keyword_hits", "loudness_db", "top_words", "file", "vod_link"])
        for h in highlights:
            link = (f"https://www.twitch.tv/videos/{meta['id']}?t={int(h.start)}s"
                    if str(meta.get("id", "")).isdigit() else "")
            w.writerow([h.rank, h.timestamp, h.start, h.end, h.score, h.chat_msgs,
                        h.keyword_hits, h.loudness_db, " ".join(h.top_words), h.file or "", link])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find and cut highlight clips from a Twitch VOD.")
    src = p.add_argument_group("source")
    src.add_argument("--vod", help="Twitch VOD url or id, e.g. https://www.twitch.tv/videos/123")
    src.add_argument("--channel", required=True, help="Channel name (used for output folder and a sanity check)")
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
    sel.add_argument("--keywords", help="Comma-separated hype keywords (replaces the defaults)")
    sel.add_argument("--weight-chat", type=float, default=1.0)
    sel.add_argument("--weight-keywords", type=float, default=1.0)
    sel.add_argument("--weight-audio", type=float, default=0.7)
    sel.add_argument("--no-audio", action="store_true", help="Skip audio analysis (chat only)")
    sel.add_argument("--no-chat", action="store_true", help="Skip chat (audio only)")

    out = p.add_argument_group("output")
    out.add_argument("--output", type=Path, default=Path("clips"), help="Output root (default ./clips)")
    out.add_argument("--quality", default="best", help="'best' or max height like 720p (default best)")
    out.add_argument("--fast-copy", action="store_true", help="Stream-copy instead of re-encoding (faster, less exact cuts)")
    out.add_argument("--dry-run", action="store_true", help="Score and report only; don't cut clips")
    out.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    if not args.vod and not args.video_file:
        p.error("give --vod or --video-file")
    if args.no_audio and (args.no_chat or not (args.vod or args.chat_file)):
        p.error("nothing to score: audio is disabled and there is no chat source")
    if not 0 <= args.lead <= 1:
        p.error("--lead must be between 0 and 1")
    return args


def run(args: argparse.Namespace) -> list[Highlight]:
    require("ffmpeg")
    channel = re.sub(r"[^\w.-]", "_", args.channel.lower())
    vod_id = parse_vod_id(args.vod) if args.vod else None
    headers: dict = {}

    # 1. Source ---------------------------------------------------------------
    if args.video_file:
        video_src = audio_src = str(args.video_file)
        meta = {"id": vod_id or args.video_file.stem, "title": args.video_file.name,
                "channel": args.channel, "duration": probe_duration(video_src)}
    else:
        url = args.vod if args.vod.startswith("http") else f"https://www.twitch.tv/videos/{vod_id}"
        log.info("Resolving VOD %s ...", vod_id)
        info = resolve_vod(url, args.quality)
        if info["uploader_id"] and info["uploader_id"].lower() != args.channel.lower():
            log.warning("VOD belongs to '%s', not '%s' - continuing anyway",
                        info["uploader_id"], args.channel)
        video_src, audio_src, headers = info["video_url"], info["audio_url"], info["http_headers"]
        meta = {"id": info["id"] or vod_id, "title": info["title"], "channel": info["uploader_id"] or args.channel,
                "duration": info["duration"], "video_height": info["video_height"]}
        log.info("  '%s' by %s, %s long, cutting at %sp",
                 meta["title"], meta["channel"], hms(meta["duration"]), info["video_height"])

    out_dir = args.output / channel / str(meta["id"])
    out_dir.mkdir(parents=True, exist_ok=True)

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
    loudness = None
    if not args.no_audio:
        log.info("Analysing audio loudness ...")
        try:
            loudness = audio_loudness(audio_src, headers)
            log.info("  %s of audio analysed", hms(loudness.size))
        except Exception as e:
            if not msgs:
                raise
            log.warning("Audio analysis failed (%s); using chat only", e)

    duration = meta["duration"] or (loudness.size if loudness is not None else 0) \
        or (max(m.t for m in msgs) + 1 if msgs else 0)
    length = int(math.ceil(duration))
    if length <= 0:
        raise SystemExit("Could not determine VOD duration")

    # 4. Score ----------------------------------------------------------------
    keywords = [k.strip().lower() for k in args.keywords.split(",")] if args.keywords else DEFAULT_KEYWORDS
    counts = hits = None
    words_by_sec: dict[int, list[str]] = {}
    if msgs:
        counts, hits, words_by_sec = chat_series(msgs, length + args.chat_delay, keywords)
    window = max(3, int(args.clip_length / 3))
    score = combined_score(length, counts, hits, loudness, args, window)

    edge = args.skip_edges if length > 4 * args.skip_edges else 0
    peaks = [p for p in pick_peaks(score, args.num_clips, int(args.min_gap), edge)
             if score[p] >= args.min_score]
    if not peaks:
        log.warning("No moment scored above --min-score %.2f", args.min_score)
    highlights = build_highlights(peaks, score, duration, args.clip_length, args.lead,
                                  counts, hits, words_by_sec, loudness, args.chat_delay)

    # 5. Cut ------------------------------------------------------------------
    if not args.dry_run:
        for h in highlights:
            name = f"{h.rank:02d}_{h.timestamp.replace(':', '-')}.mp4"
            log.info("Cutting #%d  %s-%s  (score %.2f)", h.rank, hms(h.start), hms(h.end), h.score)
            try:
                cut_clip(video_src, h.start, h.end, out_dir / name, headers, args.fast_copy)
                h.file = name
            except RuntimeError as e:
                log.error("  %s", e)

    write_reports(out_dir, meta, highlights)

    # 6. Summary --------------------------------------------------------------
    print(f"\n{len(highlights)} highlights for {meta['channel']} - {meta['title']}")
    print(f"{'#':>3}  {'time':8}  {'score':>6}  {'chat':>5}  {'hype':>5}  {'dB':>6}  words")
    for h in highlights:
        print(f"{h.rank:>3}  {h.timestamp:8}  {h.score:6.2f}  {h.chat_msgs:5d}  {h.keyword_hits:5d}  "
              f"{h.loudness_db:6.1f}  {' '.join(h.top_words)}")
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
        total += score_series(length, None, None, loudness, 0, window, 0, 0, args.weight_audio)
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
