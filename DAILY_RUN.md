# Daily TikTok run

The steps a scheduled Claude session follows each morning to turn the day's
top GTA 6 Twitch clips into 3 ready-to-post TikToks on the delivery page.

- Delivery page: https://claude.ai/artifact/DKw2g9riKUkdLTCGWKW9zA
- Work folder: `work/<date>` inside the repo checkout (it is git-ignored).
- Tools: `daily_hunt.py` (discover, prepare, render), `daily_page.py` (page state and build).
- Set `HF_HUB_DISABLE_XET=1` for every command; the speech model downloads on first use.

## 0. Set up (fresh machine)

Scheduled sessions may start without the repository or its tools. Check first:

```bash
[ -f GTA/daily_hunt.py ] || [ -f daily_hunt.py ] || git clone -b claude/clip-agent-vod-processing-ca11gh https://github.com/louisfj123-ux/GTA
cd GTA 2>/dev/null || true
python3 -c "import yt_dlp, faster_whisper, cv2" 2>/dev/null || pip install -q -r requirements.txt
fc-list | grep -qi montserrat || (apt-get update -q && apt-get install -y -q fonts-montserrat) >/dev/null 2>&1
```

Run every later command from the repository root.

## 1. Load yesterday's state

1. Read the delivery page with the Artifact tool (`action: "read"`, the URL above).
2. Copy the JSON inside `<script type="application/json" id="state">…</script>`
   into `work/<date>/state.json`. If the tag holds `/*STATE*/`, write `{"days": [], "history": []}`.
3. `python daily_page.py extract --html work/<date>/state.json --out work/<date>/state.json`
   (this also writes `work/<date>/history.json`, the clips already used).

## 2. Find and prepare candidates

```bash
python daily_hunt.py discover --work work/<date> --history work/<date>/history.json --n 8
python daily_hunt.py prepare  --work work/<date>
```

**Celebrity mode** (when the task's prompt asks for it): add
`--channels watchlists/celebrities.txt --period LAST_WEEK` to discover. It looks only at
famous streamers and celebrities with Twitch channels; clips already used are still skipped.
In hooks and post captions, name the celebrity (e.g. "xQc can't believe this", `#xqc`).

If discover finds fewer than 4 candidates, run it again with `--period LAST_WEEK`.
Discover retries Twitch's occasional "service error" by itself; if it still fails, wait a minute and run it again.

## 3. Review (the judgement step)

For every entry in `work/<date>/review.json`, read the transcript and look at
`sheet.jpg` (one frame per second) and `layout.jpg` (the vertical layout).

Reject a clip if any of these is true:
- any nudity, or sexual or strongly suggestive content (a strip club, a sex scene,
  lingering close-ups of bodies). Bikinis and shirtless men are fine.
- it is not gameplay (Just Chatting, a full-screen camera, mostly menus or maps)
- slurs are a big part of the audio (captions are censored, the audio is not)
- the audio is silent or muted, or nothing actually happens
- it is not in English

Choose the 3 best of the rest: a clear funny or dramatic payoff, a visible
reaction, and different streamers where possible. Fewer than 3 is fine if
the rest fail review; never pad with weak clips.

For each pick, write an entry in `work/<date>/picks.json`:

```json
{"slug": "...", "hook": "Who did that?!", "post_caption": "GTA 6 traffic is undefeated 💀 #gta6 #kaicenat",
 "facecam": [1620, 348, 300, 192], "cover": null, "start": null, "end": null}
```

- `hook`: at most 7 words, sparks curiosity, no emoji (the caption font cannot draw them).
- `post_caption`: one short line plus 2–4 hashtags, always `#gta6` and the streamer.
- `facecam`: take it from review.json if `layout.jpg` shows the streamer's camera
  cropped correctly on top. If the crop is wrong, measure the box on `grid.jpg`
  (yellow lines every 120 px in 1920×1080 coordinates), or use `null` for the
  blurred-background layout, which is always safe.
- `cover` (optional, `[x, y, w, h]`): if the stream shows its own live captions
  (usually white or highlighted words at the bottom centre), measure their area on
  `grid.jpg` and set `cover` so they are blanked out. Otherwise the video shows two
  sets of captions. Leave it out when there are none.
- `start`/`end` (seconds): trim dead air, such as a map or menu at the end.

Fix clear transcription mistakes in the clip's `words.json`, such as names
(Jason, Lucia) and slang ("L driver"). If a word is unclear, delete it rather
than guess.

## 4. Render and check

```bash
python daily_hunt.py render --work work/<date> --picks work/<date>/picks.json
```

Look at one frame from the middle of each `work/<date>/final/*.mp4` to confirm
the captions, the logo (top right) and the facecam crop.

## 5. Publish

1. Upload the videos to the delivery page: Artifact publish with `url` (above),
   `asset: true` and `file_paths` set to the files in `work/<date>/final/`.
2. Write `work/<date>/urls.json` mapping each file name to the `/_blob/...` url it got.
3. Build the page:
   `python daily_page.py build --state work/<date>/state.json --manifest work/<date>/final/manifest.json --urls work/<date>/urls.json --date <date> --out work/<date>/page.html`
4. Publish `work/<date>/page.html` to the delivery page URL (Artifact publish with `url`
   and `file_path`). Leave `capabilities` out so the page keeps them.
5. The build prints `delete_asset_ids`: videos from days that dropped off the page.
   Delete each one (Artifact `action: "delete"`, the page URL, `path` = the id).
   If a delete is refused, leave it; it only costs storage.

## 6. Report

Finish with a short summary: the clips posted to the page (streamer, hook,
Twitch views), what was rejected and why, and any step that failed.
If nothing passed review, say so and leave the page unchanged.
