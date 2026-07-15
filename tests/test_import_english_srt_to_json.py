import tempfile
import unittest
from pathlib import Path

from import_english_srt_to_json import import_english_sdh_srt


class ImportEnglishSrtToJsonTests(unittest.TestCase):
    def test_drops_effects_and_keeps_offscreen_foreign_speech_as_dialogue(self):
        payload = """1
00:00:01,000 --> 00:00:02,000
{\\an8}(LOW BUZZ)

2
00:00:03,000 --> 00:00:04,000
- (MUSIC PLAYING)
- Hello there.

3
00:00:05,000 --> 00:00:06,000
<i>Salvete quicumque estis;</i>

4
00:00:07,000 --> 00:00:08,000
NAREN: I am still here.

5
00:00:09,000 --> 00:00:10,000
(SOBBING): Standard?

6
00:00:11,000 --> 00:00:12,000
Turn it off. - BOBBY: Okay.

7
00:00:13,000 --> 00:00:14,000
FEMALE VOICE in Cantonese: Gok wai hou maa?
"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.srt"
            source.write_text(payload, encoding="utf-8")
            data = import_english_sdh_srt(source)

        self.assertEqual(data["counts"]["source_cues"], 7)
        self.assertEqual(data["counts"]["segments"], 6)
        self.assertEqual(
            [segment["text"] for segment in data["segments"]],
            [
                "Hello there.",
                "Salvete quicumque estis;",
                "I am still here.",
                "Standard?",
                "Turn it off. — Okay.",
                "Gok wai hou maa?",
            ],
        )
        self.assertTrue(all(segment["kind"] == "dialogue" for segment in data["segments"]))
        self.assertTrue(all(not segment["is_music"] for segment in data["segments"]))
        self.assertTrue(all(not segment["is_chant"] for segment in data["segments"]))


if __name__ == "__main__":
    unittest.main()
