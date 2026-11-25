#!/usr/bin/env python3
"""daily_page.py - build the "Daily GTA 6 TikToks" delivery page.

The page embeds its own state as JSON (<script id="state">): the last few
days of clips and the slugs of every clip ever used. Each run:

  python daily_page.py extract --html current.html --out state.json
      Pull the state out of the currently published page (after an Artifact read).
      --html may also be a file holding just the state JSON.

  python daily_page.py build --state state.json --manifest work/final/manifest.json \
      --urls urls.json --date 2026-11-26 --out page.html --keep-days 3
      Add today's clips (urls.json maps each file name to its uploaded /_blob/ url),
      drop days beyond --keep-days, and write the page. Prints the asset ids of
      dropped days so they can be deleted, and writes history.json for discover.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "templates" / "daily_page.html"
STATE_RE = re.compile(r'<script type="application/json" id="state">(.*?)</script>', re.S)


def extract(args) -> None:
    text = Path(args.html).read_text(encoding="utf-8").strip()
    state = {"days": [], "history": []}
    if text.startswith("{"):          # just the state JSON, copied out of the page
        state = json.loads(text)
    else:
        m = STATE_RE.search(text)
        if m and m.group(1).strip() and m.group(1).strip() != "/*STATE*/":
            state = json.loads(m.group(1))
    Path(args.out).write_text(json.dumps(state, indent=1, ensure_ascii=False))
    Path(args.out).with_name("history.json").write_text(json.dumps(state.get("history", [])))
    print(f"{len(state.get('days', []))} days, {len(state.get('history', []))} used clips")


def build(args) -> None:
    state = json.loads(Path(args.state).read_text()) if args.state and Path(args.state).exists() \
        else {"days": [], "history": []}
    manifest = json.loads(Path(args.manifest).read_text())
    urls = json.loads(Path(args.urls).read_text())
    clips = []
    for m in manifest:
        url = urls[m["file"]]
        if not url.startswith("/_blob/"):
            raise SystemExit(f"unexpected asset url for {m['file']}: {url}")
        clips.append({k: m[k] for k in ("n", "file", "title", "streamer", "seconds", "views",
                                        "post_caption", "source_url", "slug")} | {"url": url})
    days = [d for d in state.get("days", []) if d["date"] != args.date]
    days.insert(0, {"date": args.date, "clips": clips})
    keep, dropped = days[: args.keep_days], days[args.keep_days:]
    history = list(dict.fromkeys(state.get("history", []) + [c["slug"] for c in clips]))[-2000:]
    now = dt.datetime.now(dt.timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    new_state = {"updated": now, "days": keep, "history": history}
    blob = json.dumps(new_state, ensure_ascii=False).replace("<", "\\u003c")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*STATE*/", blob, 1)
    Path(args.out).write_text(html, encoding="utf-8")
    Path(args.out).with_name("history.json").write_text(json.dumps(history))
    stale = [c["url"].split("/_blob/")[1] for d in dropped for c in d["clips"]]
    print(json.dumps({"page": args.out, "days": [d["date"] for d in keep], "delete_asset_ids": stale}))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--html", required=True)
    e.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--state")
    b.add_argument("--manifest", required=True)
    b.add_argument("--urls", required=True)
    b.add_argument("--date", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--keep-days", type=int, default=3)
    args = p.parse_args(argv)
    {"extract": extract, "build": build}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
