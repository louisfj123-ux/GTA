import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clip_agent as ca  # noqa: E402


@pytest.mark.parametrize("vod,expected", [
    ("https://www.twitch.tv/videos/2932568279", "2932568279"),
    ("https://www.twitch.tv/videos/2932568279?t=1h2m3s", "2932568279"),
    ("v2932568279", "2932568279"),
    ("2932568279", "2932568279"),
])
def test_parse_vod_id(vod, expected):
    assert ca.parse_vod_id(vod) == expected


def test_parse_vod_id_rejects_garbage():
    with pytest.raises(SystemExit):
        ca.parse_vod_id("https://www.twitch.tv/somechannel")


def test_hms():
    assert ca.hms(3725.9) == "01:02:05"


def test_chat_series_counts_and_keywords():
    msgs = [ca.ChatMessage(1.2, "KEKW that was insane"), ca.ChatMessage(1.9, "hello"),
            ca.ChatMessage(3.0, "someone clip it")]
    counts, hits, words = ca.chat_series(msgs, 5, ["kekw", "insane", "clip it"])
    assert counts.tolist() == [0, 2, 0, 1, 0]
    assert hits.tolist() == [0, 1, 0, 1, 0]
    assert sorted(words[1]) == ["insane", "kekw"]
    assert words[3] == ["clip it"]


def test_pick_peaks_respects_gap_and_centres_plateaus():
    s = np.zeros(1000)
    s[100:111] = 5.0     # plateau centred on 105
    s[130] = 4.0         # too close to the first peak
    s[600] = 3.0
    assert ca.pick_peaks(s, 3, min_gap=50, edge=0)[:2] == [105, 600]


def test_robust_z_is_capped_and_handles_constant_input():
    assert np.all(ca.robust_z(np.ones(50)) == 0)
    x = np.zeros(100)
    x[50] = 1e6
    assert ca.robust_z(x).max() == ca.Z_CAP


def test_load_twitchdownloader_format(tmp_path):
    f = tmp_path / "chat.json"
    f.write_text('{"comments": [{"content_offset_seconds": 12.5, "message": {"body": "POG"}}]}')
    msgs = ca.load_chat_file(f)
    assert msgs[0].t == 12.5 and msgs[0].text == "POG"
