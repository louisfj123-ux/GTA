#!/usr/bin/env python3
"""caption_clip.py - turn a Twitch clip (or any video) into a captioned, branded clip.

  python caption_clip.py --clip https://www.twitch.tv/<channel>/clip/<slug> \
      --logo logo.webp --logo-position top-right

Steps
  1. Download the clip with yt-dlp (or use a local file).
  2. Transcribe with faster-whisper, keeping word timings -> <name>.words.json
  3. Group words into short lines and burn them in as bold, word-highlighted captions.
  4. Optionally blank out the stream's own live captions (--cover x,y,w,h).
  5. Overlay the logo.

Whisper can mishear shouting. Edit <name>.words.json and re-run with
--words <name>.words.json to re-render without transcribing again. Add
"break": true to a word to start a new caption line there.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import clip_agent as ca

STYLE = {
    "font": "Montserrat Black",
    "size": 0.085,          # font size as a fraction of video height
    "primary": "&H00FFFFFF",   # white (ASS colours are &HAABBGGRR)
    "highlight": "&H0000E1FF", # yellow
    "outline": "&H00000000",
    "outline_w": 0.0065,    # outline width as a fraction of video height
    "margin_v": 0.20,       # distance from bottom as a fraction of video height
}


def download_clip(url: str, out_dir: Path) -> Path:
    import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {"quiet": True, "no_warnings": True, "noprogress": True, "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
            "format": "best[ext=mp4]/best"}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info))


def transcribe(video: Path, model_name: str) -> list[dict]:
    from faster_whisper import WhisperModel

    raw = subprocess.run([ca.require("ffmpeg"), "-nostdin", "-loglevel", "error", "-i", str(video),
                          "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], capture_output=True).stdout
    audio = np.frombuffer(raw, np.float32)
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    # condition_on_previous_text=False stops the repetition loops whisper falls
    # into on repeated shouting.
    segments, _ = model.transcribe(audio, word_timestamps=True, beam_size=5,
                                   condition_on_previous_text=False)
    words = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append({"start": round(w.start, 2), "end": round(w.end, 2), "word": text})
    return words


def chunk_words(words: list[dict], max_words: int, max_gap: float) -> list[list[dict]]:
    """Group words into short caption lines, breaking on pauses and punctuation."""
    chunks: list[list[dict]] = []
    for w in words:
        if (not chunks or w.get("break") or len(chunks[-1]) >= max_words
                or w["start"] - chunks[-1][-1]["end"] > max_gap
                or re.search(r"[.!?]$", chunks[-1][-1]["word"])):
            chunks.append([w])
        else:
            chunks[-1].append(w)
    return chunks


def ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def build_ass(words: list[dict], width: int, height: int, duration: float,
              max_words: int = 3, max_gap: float = 0.35, hold: float = 0.6, upper: bool = True) -> str:
    size = int(height * STYLE["size"])
    outline = max(2, int(height * STYLE["outline_w"]))
    margin_v = int(height * STYLE["margin_v"])
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{STYLE['font']},{size},{STYLE['primary']},{STYLE['primary']},{STYLE['outline']},&H80000000,0,0,0,0,100,100,1,0,1,{outline},{max(1, outline // 2)},2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    chunks = chunk_words(words, max_words, max_gap)
    for ci, chunk in enumerate(chunks):
        next_start = chunks[ci + 1][0]["start"] if ci + 1 < len(chunks) else duration
        chunk_end = min(next_start, chunk[-1]["end"] + hold)
        texts = [ass_escape(w["word"].upper() if upper else w["word"]) for w in chunk]
        for wi, w in enumerate(chunk):
            start = chunk[0]["start"] if wi == 0 else w["start"]
            end = chunk[wi + 1]["start"] if wi + 1 < len(chunk) else chunk_end
            if end <= start:
                continue
            parts = [f"{{\\c{STYLE['highlight']}}}{t}{{\\c{STYLE['primary']}}}" if i == wi else t
                     for i, t in enumerate(texts)]
            # Pop the line in when it first appears.
            pop = "{\\fscx112\\fscy112\\t(0,90,\\fscx100\\fscy100)}" if wi == 0 else ""
            lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Cap,,0,0,0,,{pop}{' '.join(parts)}")
    return head + "\n".join(lines) + "\n"


def probe(video: Path) -> tuple[int, int, float]:
    out = subprocess.run([ca.require("ffprobe"), "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height:format=duration", "-of", "json", str(video)],
                         capture_output=True, text=True).stdout
    d = json.loads(out)
    return int(d["streams"][0]["width"]), int(d["streams"][0]["height"]), float(d["format"]["duration"])


def render(video: Path, ass_path: Path, dest: Path, logo: ca.LogoSpec | None,
           cover: tuple[int, int, int, int] | None) -> None:
    vf = []
    if cover:
        x, y, w, h = cover
        vf.append(f"delogo=x={x}:y={y}:w={w}:h={h}")
    # ass filter path: escape characters that are special in filtergraphs.
    esc = str(ass_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    vf.append(f"ass='{esc}'")
    cmd = [ca.require("ffmpeg"), "-nostdin", "-loglevel", "error", "-y", "-i", str(video)]
    if logo:
        logo_fc = ca.logo_filter(logo, logo.video_width)  # uses [0:v] and [1:v], outputs [v]
        cmd += ["-i", str(logo.path), "-filter_complex",
                f"[0:v]{','.join(vf)}[sub];" + logo_fc.replace("[0:v]", "[sub]"),
                "-map", "[v]", "-map", "0:a?"]
    else:
        cmd += ["-vf", ",".join(vf)]
    cmd += [*ca.ENCODE_ARGS, "-movflags", "+faststart", str(dest)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip()[:800])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Caption and brand a clip.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--clip", help="Twitch clip URL")
    src.add_argument("--video", type=Path, help="Local video file")
    p.add_argument("--output", type=Path, default=Path("clips/captioned"), help="Output folder")
    p.add_argument("--words", type=Path, help="Use this words JSON instead of transcribing")
    p.add_argument("--model", default="medium.en", help="faster-whisper model (default medium.en)")
    p.add_argument("--max-words", type=int, default=3, help="Words per caption line (default 3)")
    p.add_argument("--cover", help="x,y,w,h of burned-in captions to blank out (e.g. a stream's live captions)")
    p.add_argument("--logo", help="Logo image path or URL")
    p.add_argument("--logo-position", default="top-right", choices=sorted(ca.LOGO_POSITIONS))
    p.add_argument("--logo-width", type=float, default=0.30)
    p.add_argument("--logo-opacity", type=float, default=1.0)
    args = p.parse_args(argv)
    ca.logging.basicConfig(level=ca.logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                           datefmt="%H:%M:%S")
    log = ca.log

    args.output.mkdir(parents=True, exist_ok=True)
    if args.clip:
        log.info("Downloading clip ...")
        video = download_clip(args.clip, args.output / "source")
    else:
        video = args.video
    width, height, duration = probe(video)
    stem = video.stem

    words_path = args.words or args.output / f"{stem}.words.json"
    if args.words:
        words = json.loads(args.words.read_text(encoding="utf-8"))
    else:
        log.info("Transcribing with %s ...", args.model)
        words = transcribe(video, args.model)
        words_path.write_text(json.dumps(words, indent=1, ensure_ascii=False), encoding="utf-8")
        log.info("  %d words -> %s (edit and re-run with --words to fix mistakes)", len(words), words_path)

    ass_path = args.output / f"{stem}.ass"
    ass_path.write_text(build_ass(words, width, height, duration, args.max_words), encoding="utf-8")

    logo = None
    if args.logo:
        logo = ca.LogoSpec(ca.fetch_logo(args.logo, args.output), args.logo_position,
                           args.logo_width, args.logo_opacity, video_width=width)
    cover = tuple(int(v) for v in args.cover.split(",")) if args.cover else None
    dest = args.output / f"{stem}_captioned.mp4"
    log.info("Rendering %s ...", dest.name)
    render(video, ass_path, dest, logo, cover)
    print(f"\n{dest.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
