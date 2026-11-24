# clip_agent

Finds the best moments in a Twitch VOD and cuts them into clips.

```bash
pip install -r requirements.txt      # also needs ffmpeg on PATH
python clip_agent.py --vod https://www.twitch.tv/videos/2932568279 --channel streamername
```

## How it works

1. **Resolve** the VOD's HLS streams with yt-dlp. Nothing is fully downloaded.
2. **Chat**: downloads the chat replay from Twitch's GQL API.
3. **Audio**: streams the lowest-bitrate audio through ffmpeg and measures loudness every second.
4. **Score**: turns each signal into a robust z-score (median/MAD, capped at 10), smooths it and adds them up:
   - chat messages per second, shifted back by `--chat-delay` because chat reacts late
   - messages containing hype keywords (KEKW, POG, "clip it", LUL, ...)
   - audio loudness
5. **Pick** the top `--num-clips` peaks, at least `--min-gap` seconds apart and scoring above `--min-score`.
6. **Cut** each clip with ffmpeg straight from the stream. Clips are re-encoded for exact cuts, or stream-copied with `--fast-copy`.

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
| `--keywords` | built-in list | comma-separated list that replaces the built-in keywords |
| `--weight-chat` / `--weight-keywords` / `--weight-audio` | 1 / 1 / 0.7 | how much each signal counts |
| `--no-chat` / `--no-audio` | | turn off a signal |
| `--quality` | best | `best`, or a maximum height such as `720p` |
| `--fast-copy` | off | stream-copy instead of re-encoding |
| `--dry-run` | off | score and write the reports, but don't cut clips |
| `--output` | `clips` | output root |

### Offline use

```bash
python clip_agent.py --video-file stream.mp4 --chat-file chat.json --channel me
```

`--chat-file` accepts this tool's own `chat.json` (`[{"t": 12.3, "text": "..."}]`) or a TwitchDownloader JSON export.

## Network

The agent needs to reach `www.twitch.tv`, `gql.twitch.tv`, `usher.ttvnw.net` and Twitch's VOD CDN hosts (`*.cloudfront.net` / `*.ttvnw.net`).

## Tests

```bash
pip install pytest && python -m pytest tests
```
