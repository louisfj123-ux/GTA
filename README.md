# clip_agent

Finds the best moments in a Twitch VOD and cuts them into clips.

```bash
pip install -r requirements.txt      # also needs ffmpeg on PATH
python clip_agent.py --vod https://www.twitch.tv/videos/2932568279    # --channel is optional
```

## How it works

1. **Resolve** the VOD's HLS streams with yt-dlp. Nothing is fully downloaded.
2. **Chat**: downloads the chat replay from Twitch's GQL API.
3. **Audio**: streams the lowest-bitrate audio through ffmpeg and measures loudness every second.
4. **Score**: turns each signal into a robust z-score (median/MAD, capped at 10), smooths it and adds them up:
   - chat messages per second, shifted back by `--chat-delay` because chat reacts late
   - messages containing hype keywords (KEKW, POG, "clip it", LUL, ...)
   - audio loudness
5. **Gameplay check**: measures on-screen motion from the lowest-quality stream, so moments spent mostly in menus, pause or loading screens are skipped.
6. **Pick** the top `--num-clips` peaks, at least `--min-gap` seconds apart and scoring above `--min-score`.
7. **Cut** each clip with ffmpeg straight from the stream. Clips are re-encoded for exact cuts, or stream-copied with `--fast-copy`.

If the chat download fails, the agent carries on with audio only. If audio fails, it uses chat only.

## Output

```
clips/<channel>/<vod_id>/
  01_01-23-45.mp4
  02_02-10-07.mp4
  ...
  highlights.json   # metadata + every highlight with scores
  highlights.csv    # rank, timestamp, scores, top words, link to that point in the VOD
  chat.json         # only with --save-chat
```

## Options

| flag | default | |
|---|---|---|
| `--num-clips` | 10 | number of clips |
| `--clip-length` | 30 | seconds per clip |
| `--lead` | 0.65 | fraction of the clip placed before the peak |
| `--min-gap` | 90 | minimum seconds between clip peaks |
| `--chat-delay` | 6 | how many seconds chat lags behind the stream |
| `--skip-edges` | 60 | ignore the first and last N seconds (intro and outro) |
| `--min-score` | 0.5 | drop weak peaks |
| `--style` | hype | `funny` finds chat laughing (LOL, LMAO, 😂, 💀, KEKW) rather than cheering. `hype` finds big chat reactions. `rage` finds the streamer losing it: chat telling him to chill or calm down, "NOOOO" spam, NotLikeThis, and yelling that is loud compared with his recent baseline |
| `--keywords` | built-in list | comma-separated list that replaces the built-in keywords |
| `--weight-chat` / `--weight-keywords` / `--weight-audio` | set by `--style` (hype: 1 / 1 / 0.7; rage: 0.3 / 1.5 / 1) | how much each signal counts |
| `--max-muted` | 0.1 | skip moments where more than this share of the audio was muted by Twitch for copyright |
| `--min-gameplay` | 0.7 | skip moments where less than this share of the clip is moving gameplay (menus, pause and loading screens are static); 0 turns it off |
| `--no-chat` / `--no-audio` | | turn off a signal |
| `--quality` | best | `best`, or a maximum height such as `720p` |
| `--fast-copy` | off | stream-copy instead of re-encoding |
| `--logo` | none | image path or URL to overlay on every clip (PNG/WebP transparency is kept) |
| `--logo-position` | bottom-center | top-left, top-center, top-right, center, bottom-left, bottom-center or bottom-right |
| `--logo-width` | 0.30 | logo width as a fraction of the video width |
| `--logo-opacity` | 1.0 | 0 to 1 |
| `--dry-run` | off | score and write the reports, but don't cut clips |
| `--output` | `clips` | output root |

### Branding

```bash
python clip_agent.py --vod <url> --channel <name> --logo https://example.com/logo.webp
```

To brand clips you've already cut, without cutting them again:

```python
import clip_agent as c
from pathlib import Path
c.brand_clip(Path("clips/x/123/01_00-10-00.mp4"), Path("branded.mp4"), c.LogoSpec(Path("logo.webp")))
```

### Offline use

```bash
python clip_agent.py --video-file stream.mp4 --chat-file chat.json --channel me
```

`--chat-file` accepts this tool's own `chat.json` (`[{"t": 12.3, "text": "..."}]`) or a TwitchDownloader JSON export.

## Captioned clips (`caption_clip.py`)

Turns a Twitch clip, or any video, into a captioned clip with the logo, using bold word-by-word captions in the short-form style.

```bash
python caption_clip.py --clip https://www.twitch.tv/<channel>/clip/<slug> \
    --logo logo.webp --logo-position top-right --cover 500,940,920,130
```

- The clip is transcribed with faster-whisper (`--model`, default `medium.en`). The words and their timings are saved to `clips/captioned/<id>.words.json`.
- Whisper can mishear shouting. Fix the words in that file and re-run with `--words <file>` to re-render without transcribing again. Add `"break": true` to a word to start a new caption line there.
- `--vertical --facecam x,y,w,h` makes a 9:16 TikTok/Shorts video: the facecam (given in 1920×1080 source pixels) across the top, the gameplay below, and captions in the upper middle, clear of TikTok's buttons and description.
- `--hook "text"` adds a boxed title. `--start`/`--end` trim the clip. `--censor` stars out swearing and slurs in the captions; the audio is unchanged.
- `--cover x,y,w,h` blanks out burned-in captions, such as a streamer's live captions, so the video doesn't show two sets.
- Captions use Montserrat Black. Install it with `apt-get install fonts-montserrat`; without it, a fallback font is used.

Example TikTok render:

```bash
python caption_clip.py --video clip.mp4 --start 13 --end 40 --vertical --facecam 0,432,392,220 \
    --hook "Asked a stranger to take our picture..." --censor --logo logo.webp --logo-position top-right
```

Analysis results are cached in `signals.npz` in each VOD's output folder, so re-running `clip_agent.py` with different settings skips the slow audio and video passes.

## Daily automation (`daily_hunt.py`, `daily_page.py`)

A scheduled cloud session runs [`DAILY_RUN.md`](DAILY_RUN.md) every morning:

1. `daily_hunt.py discover` finds the day's top English GTA 6 Twitch clips through Twitch's public GraphQL API (no API keys needed). They are ranked by views per hour, and clips already used are skipped.
2. `daily_hunt.py prepare` downloads and transcribes each clip, finds the facecam (`facecam.py`), and writes review images.
3. Claude reviews the candidates, rejecting nudity, non-gameplay, slur-heavy or dead clips. It picks 3 and writes their hooks and post captions.
4. `daily_hunt.py render` makes the 9:16 videos. Clips without a reliable facecam use a blurred-background layout.
5. `daily_page.py` rebuilds the private delivery page with save buttons, keeping the last 3 days.

## Network

The agent needs to reach `www.twitch.tv`, `gql.twitch.tv`, `usher.ttvnw.net` and Twitch's VOD CDN hosts (`*.cloudfront.net` / `*.ttvnw.net`).

## Tests

```bash
pip install pytest && python -m pytest tests
```
