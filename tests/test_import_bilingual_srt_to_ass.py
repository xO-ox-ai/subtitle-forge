import unittest

from import_bilingual_srt_to_ass import parse_bilingual_srt, render_ass


class ImportBilingualSrtTests(unittest.TestCase):
    def test_parses_cues_without_blank_separators(self):
        source = """1
00:00:01,000 --> 00:00:03,000
Hello.
你好。
2
00:00:03,000 --> 00:00:05,000
Goodbye.
再见。
"""

        cues = parse_bilingual_srt(source)

        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0].en, "Hello.")
        self.assertEqual(cues[0].zh, "你好。")
        self.assertEqual(cues[1].start, 3.0)

    def test_music_pair_uses_current_music_style(self):
        source = """1
00:00:01,000 --> 00:00:03,000
♪ Sing ♪
♪ 歌唱 ♪
"""

        cue = parse_bilingual_srt(source)[0]

        self.assertEqual(cue.style, "BILINGUAL_MUSIC")
        self.assertIn(",BILINGUAL_MUSIC,", render_ass([cue]))

    def test_missing_chinese_uses_translation_placeholder(self):
        source = """1
00:00:01,000 --> 00:00:03,000
English only.
"""

        rendered = render_ass(parse_bilingual_srt(source))

        self.assertIn(r"…\N{\fs34}English only.", rendered)


if __name__ == "__main__":
    unittest.main()
