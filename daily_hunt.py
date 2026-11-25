#!/usr/bin/env python3
"""daily_hunt.py - find trending Twitch clips and turn them into TikToks.

Three steps, with a human (or Claude) review between prepare and render:

  python daily_hunt.py discover --work work/2026-11-26 --history history.json
      Top clips for a game over the last day, ranked by view velocity,
      English only, not already used. -> work/candidates.json

  python daily_hunt.py prepare --work work/2026-11-26
      For each candidate: download, transcribe (words.json), detect the
      facecam, and write review images:
        <slug>/sheet.jpg    one frame per second, to check content
        <slug>/layout.jpg   one frame of the vertical layout, to check the facecam crop
        <slug>/grid.jpg     a full frame with yellow lines every 120 px (1920x1080 coords),
                            to measure a facecam box by eye when detection is wrong
      -> work/review.json (transcript text, facecam box, paths)

  python daily_hunt.py render --work work/2026-11-26 --picks picks.json
      picks.json: [{"slug": ..., "hook": "...", "post_caption": "...",
                    "facecam": [x,y,w,h] | null, "start": s, "end": s}, ...]
      Renders each pick as a captioned, branded 9:16 video, compressed to
      fit the delivery page. -> work/final/*.mp4 + work/final/manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
import sys
from pathlib import Path

import clip_agent as ca

ROOT = Path(__file__).resolve().parent
LOGO = ROOT / "assets" / "logo.webp"
MAX_UPLOAD_BYTES = 14 * 1024 * 1024   # artifact assets accept up to 15 MB


# --------------------------------------------------------------------------- #
# discover
# --------------------------------------------------------------------------- #
CLIPS_QUERY = """query($game: String!, $first: Int!, $period: ClipsPeriod!) {
  game(name: $game) {
    clips(first: $first, criteria: {period: $period, sort: VIEWS_DESC}) {
      edges { node {
        slug title viewCount createdAt durationSeconds language url
        broadcaster { login displayName }
        video { id } videoOffsetSeconds
      } }
    }
  }
}"""


def gql_query(query: str, variables: dict) -> dict:
    resp = ca._gql({"query": query, "variables": variables})
    if resp.get("errors"):
        raise RuntimeError(f"Twitch GQL error: {resp['errors']}")
    return resp["data"]


def discover(args) -> None:
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    history = set()
    if args.history and Path(args.history).exists():
        history = set(json.loads(Path(args.history).read_text()))

    data = gql_query(CLIPS_QUERY, {"game": args.game, "first": 100, "period": args.period})
    edges = ((data.get("game") or {}).get("clips") or {}).get("edges") or []
    now = dt.datetime.now(dt.timezone.utc)
    langs = {l.strip().upper() for l in args.languages.split(",")}
    out, per_streamer = [], {}
    for e in edges:
        c = e["node"]
        if c["slug"] in history or (c.get("language") or "").upper() not in langs:
            continue
        if not args.min_secs <= (c.get("durationSeconds") or 0) <= args.max_secs:
            continue
        created = dt.datetime.fromisoformat(c["createdAt"].replace("Z", "+00:00"))
        hours = max(0.5, (now - created).total_seconds() / 3600)
        # Views per hour, softened so a very fresh clip with few views doesn't win.
        c["velocity"] = round(c["viewCount"] / hours ** 0.8, 1)
        c["ageHours"] = round(hours, 1)
        out.append(c)
    out.sort(key=lambda c: c["velocity"], reverse=True)
    picked = []
    for c in out:
        login = c["broadcaster"]["login"]
        if per_streamer.get(login, 0) >= args.per_streamer:
            continue
        per_streamer[login] = per_streamer.get(login, 0) + 1
        picked.append(c)
        if len(picked) >= args.n:
            break
    (work / "candidates.json").write_text(json.dumps(picked, indent=1, ensure_ascii=False))
    print(f"{len(picked)} candidates -> {work / 'candidates.json'}")
    for c in picked:
        print(f"  {c['viewCount']:>7} views  {c['velocity']:>7}/h^0.8  {c['durationSeconds']:>3}s  "
              f"{c['broadcaster']['login']:<16} {c['slug']}  | {c['title'][:50]}")


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #
def run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {res.stderr.strip()[:400]}")


def prepare(args) -> None:
    import caption_clip as cc
    import facecam

    work = Path(args.work)
    cands = json.loads((work / "candidates.json").read_text())
    review = []
    for c in cands:
        d = work / c["slug"]
        d.mkdir(exist_ok=True)
        clip = d / "clip.mp4"
        try:
            if not clip.exists():
                print(f"[{c['slug']}] downloading")
                src = cc.download_clip(c["url"], d / "dl")
                src.rename(clip)
            print(f"[{c['slug']}] transcribing")
            words_path = d / "words.json"
            if not words_path.exists():
                words_path.write_text(json.dumps(cc.transcribe(clip, args.model), indent=1, ensure_ascii=False))
            words = json.loads(words_path.read_text())
            box = facecam.detect(str(clip))
            # One frame per second for content review.
            dur = cc.probe(clip)[2]
            cols = 6
            rows = max(1, math.ceil(dur / cols))
            run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(clip), "-vf",
                 f"fps=1,scale=320:-2,tile={cols}x{rows}", "-frames:v", "1", str(d / "sheet.jpg")])
            # A full frame with a grid every 120 source px, for measuring a facecam by eye.
            run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{min(dur / 2, 5):.1f}",
                 "-i", str(clip), "-vf", "scale=1920:1080,drawgrid=w=120:h=120:t=2:c=yellow@0.7,scale=960:-2",
                 "-frames:v", "1", str(d / "grid.jpg")])
            # One frame of the vertical layout, to check the facecam crop.
            layout, _ = cc.vertical_layout(box, fill="blur")
            run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{min(dur / 2, 5):.1f}",
                 "-i", str(clip), "-filter_complex", layout.replace("[stack]", ",scale=540:-2[o]"),
                 "-map", "[o]", "-frames:v", "1", str(d / "layout.jpg")])
            review.append({
                "slug": c["slug"], "streamer": c["broadcaster"]["displayName"], "title": c["title"],
                "views": c["viewCount"], "ageHours": c["ageHours"], "seconds": round(dur, 1),
                "url": c["url"], "facecam": list(box) if box else None,
                "transcript": " ".join(w["word"] for w in words),
                "sheet": str(d / "sheet.jpg"), "layout": str(d / "layout.jpg"), "grid": str(d / "grid.jpg"),
                "words": str(words_path),
            })
        except Exception as e:  # one bad clip must not stop the run
            print(f"[{c['slug']}] skipped: {e}")
    (work / "review.json").write_text(json.dumps(review, indent=1, ensure_ascii=False))
    print(f"{len(review)} prepared -> {work / 'review.json'}")


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def fit_size(src: Path, dest: Path, limit: int) -> None:
    """Copy, or re-encode with 2-pass x264 so the file fits under `limit` bytes."""
    if src.stat().st_size <= limit:
        dest.write_bytes(src.read_bytes())
        return
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", str(src)], capture_output=True, text=True).stdout)
    vbit = int((limit * 0.95 * 8 / dur - 128_000) / 1000)
    log = dest.with_suffix(".passlog")
    run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-c:v", "libx264",
         "-preset", "slow", "-b:v", f"{vbit}k", "-pass", "1", "-passlogfile", str(log), "-an", "-f", "null", "-"])
    run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-c:v", "libx264",
         "-preset", "slow", "-b:v", f"{vbit}k", "-pass", "2", "-passlogfile", str(log),
         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(dest)])
    for f in dest.parent.glob(log.name + "*"):
        f.unlink()


def render(args) -> None:
    import caption_clip as cc

    work = Path(args.work)
    picks = json.loads(Path(args.picks).read_text())
    cands = {c["slug"]: c for c in json.loads((work / "candidates.json").read_text())}
    final = work / "final"
    final.mkdir(exist_ok=True)
    manifest = []
    for i, p in enumerate(picks, 1):
        c = cands[p["slug"]]
        d = work / p["slug"]
        name = f"{i}_{c['broadcaster']['login']}"
        argv = ["--video", str(d / "clip.mp4"), "--words", str(d / "words.json"), "--vertical",
                "--censor", "--output", str(d / "out"), "--name", name, "--hook", p["hook"],
                "--logo", str(LOGO), "--logo-position", "top-right", "--max-words", "3"]
        if p.get("facecam"):
            argv += ["--facecam", ",".join(str(int(v)) for v in p["facecam"])]
        else:
            argv += ["--fill", "blur"]
        if p.get("start") is not None:
            argv += ["--start", str(p["start"])]
        if p.get("end") is not None:
            argv += ["--end", str(p["end"])]
        print(f"[{p['slug']}] rendering {name}")
        cc.main(argv)
        out = d / "out" / f"{name}_captioned.mp4"
        dest = final / f"{name}.mp4"
        fit_size(out, dest, MAX_UPLOAD_BYTES)
        manifest.append({
            "n": i, "file": dest.name, "path": str(dest), "bytes": dest.stat().st_size,
            "slug": p["slug"], "streamer": c["broadcaster"]["displayName"], "title": p["hook"],
            "post_caption": p.get("post_caption", ""), "source_url": c["url"],
            "views": c["viewCount"], "seconds": round(cc.probe(dest)[2]),
        })
    (final / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"{len(manifest)} rendered -> {final / 'manifest.json'}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--work", required=True)
    d.add_argument("--history", help="JSON list of clip slugs already used")
    d.add_argument("--game", default="Grand Theft Auto VI")
    d.add_argument("--period", default="LAST_DAY", choices=["LAST_DAY", "LAST_WEEK", "LAST_MONTH", "ALL_TIME"])
    d.add_argument("--languages", default="EN")
    d.add_argument("--n", type=int, default=8, help="Candidates to prepare (default 8)")
    d.add_argument("--per-streamer", type=int, default=2)
    d.add_argument("--min-secs", type=int, default=10)
    d.add_argument("--max-secs", type=int, default=60)
    pr = sub.add_parser("prepare")
    pr.add_argument("--work", required=True)
    pr.add_argument("--model", default="medium.en")
    r = sub.add_parser("render")
    r.add_argument("--work", required=True)
    r.add_argument("--picks", required=True)
    args = p.parse_args(argv)
    ca.logging.basicConfig(level=ca.logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                           datefmt="%H:%M:%S")
    {"discover": discover, "prepare": prepare, "render": render}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
